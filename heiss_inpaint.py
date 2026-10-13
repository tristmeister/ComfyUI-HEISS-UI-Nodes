"""
HEISS Inpaint Stitch: lays an inpainted crop back onto the original, lined up and colour matched first.

Edit models (Qwen-Image Edit, Flux.2 / Klein with a reference) re-render the whole crop. They often shift or zoom it a
little and shift its exposure or tint, so a plain composite shows doubled contours where the mask edge crosses a line
and a patch that is a shade off. This node measures both on the ring around the mask, where the result should still
show the original, and corrects them before blending:

  align        an affine fit (Lucas-Kanade, Gauss-Newton with a gain and a bias so a colour drift does not pull the
               geometry, Huber weights, on a pyramid at most 512 px on the long side). Kept only when it is plausible
               (scale within 8 %, shear under 0.03, shift under 5 % of the box) and lowers the ring difference by 15 %
               or more: a fit that locked onto the wrong detail gains a few per cent at best, a real one far more. A flat ring (nothing to fit on) or a ring too small is left alone.
  color_match  per-channel mean and spread of the result moved onto the original's, both measured on the ring. Kept
               only when the drift is the size of a model's drift (means within 0.1, spread within 0.75-1.33); a
               larger difference is taken as meant and left alone.

Models that sample only inside a latent mask (SDXL, Flux.1, Z-Image ...) keep the ring as it was: both steps find
nothing to correct and the node is a plain composite.

The idea follows Inpaint Canvas (DenRakEiw, ComfyUI-InpaintCanvas: `_align_patch`, `_color_match`); this is a separate
implementation in torch, with no OpenCV.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.utils

WORK = 512
LIMITS = {"scale": 0.08, "shear": 0.03, "shift": 0.05, "gain": 0.85, "min_ring": 2000, "min_ring_share": 0.02}
COLOR = {"mean": 0.1, "spread": (0.75, 1.0 / 0.75), "min_weight": 500}
# The ring starts this many full-size pixels past the last trace of the mask.
RING_GAP = 4


def report(data):
    """Tells the client that queued the run (HEISS) what the stitch did; a prompt without a client gets the console line only."""
    print(f"[HEISS Inpaint Stitch] {data.get('message', '')}")
    try:
        from server import PromptServer

        server = PromptServer.instance
        client = getattr(server, "client_id", None)
        if client:
            server.send_sync("heiss.inpaint", {**data, "prompt_id": getattr(server, "last_prompt_id", None)}, client)
    except Exception:
        pass


# ---- small image helpers ([H, W] or [C, H, W] float tensors) ----------------------------------------------------------

def gray(rgb):
    """Luma of [H, W, 3] in 0..1."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def shrink(plane, h, w):
    """Area average of [H, W] or [H, W, C] down to h x w."""
    if plane.shape[0] == h and plane.shape[1] == w:
        return plane.clone()
    x = plane[None, None] if plane.dim() == 2 else plane.permute(2, 0, 1)[None]
    out = F.interpolate(x, size=(h, w), mode="area")[0]
    return out[0] if plane.dim() == 2 else out.permute(1, 2, 0)


def dilate(mask, px):
    """Square max filter of [H, W] by px pixels."""
    px = int(px)
    if px <= 0:
        return mask
    k = 2 * px + 1
    out = F.max_pool2d(mask[None, None], kernel_size=(1, k), stride=1, padding=(0, px))
    return F.max_pool2d(out, kernel_size=(k, 1), stride=1, padding=(px, 0))[0, 0]


def blur121(plane):
    """A 1-2-1 blur each way, edges repeated: takes the aliasing out of the gradients."""
    k = torch.tensor([0.25, 0.5, 0.25], dtype=plane.dtype)
    x = F.pad(plane[None, None], (1, 1, 1, 1), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, 3))
    return F.conv2d(x, k.view(1, 1, 3, 1))[0, 0]


def gradients(plane):
    """Central differences of [H, W] (one-sided at the border)."""
    gx = torch.zeros_like(plane)
    gy = torch.zeros_like(plane)
    gx[:, 1:-1] = (plane[:, 2:] - plane[:, :-2]) * 0.5
    gy[1:-1, :] = (plane[2:, :] - plane[:-2, :]) * 0.5
    return gx, gy


def sample(planes, X, Y, w, h):
    """Bilinear samples of [C, H, W] at pixel coordinates X, Y ([N] each), edges repeated."""
    gx = X / max(1, w - 1) * 2 - 1
    gy = Y / max(1, h - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2).to(planes.dtype)
    return F.grid_sample(planes[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]


def half(level):
    T, I, R = level["T"], level["I"], level["R"]
    h, w = T.shape[0] // 2, T.shape[1] // 2
    return {
        "T": F.avg_pool2d(T[None, None], 2)[0, 0],
        "I": F.avg_pool2d(I[None, None], 2)[0, 0],
        # a ring pixel at the coarser level only where all four below are ring
        "R": F.avg_pool2d(R[None, None].float(), 2)[0, 0] > 0.99,
    }


# ---- the fit ----------------------------------------------------------------------------------------------------------

def photometric(a, b, weight):
    """Least-squares gain and bias with gain * a + bias ~ b over weight."""
    n = weight.sum()
    ma, mb = (a * weight).sum() / n, (b * weight).sum() / n
    var = (((a - ma) ** 2) * weight).sum() / n
    gain = (((a - ma) * (b - mb)) * weight).sum() / n / var if var > 1e-8 else torch.tensor(1.0, dtype=a.dtype)
    gain = gain.clamp(0.5, 2.0)
    return gain, mb - gain * ma


def ring_difference(T, I, ring, A=None, t=None, c=None):
    """Mean absolute difference on the ring after the best gain and bias, with I read through the warp (A, t about c)."""
    h, w = T.shape
    ys, xs = torch.nonzero(ring, as_tuple=True)
    if ys.numel() < 16:
        return None
    x, y = xs.to(T.dtype), ys.to(T.dtype)
    if A is None:
        warped = I[ys, xs]
        inside = torch.ones_like(warped, dtype=torch.bool)
    else:
        X = A[0, 0] * (x - c[0]) + A[0, 1] * (y - c[1]) + c[0] + t[0]
        Y = A[1, 0] * (x - c[0]) + A[1, 1] * (y - c[1]) + c[1] + t[1]
        inside = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
        warped = sample(I[None], X, Y, w, h)[0]
    base = T[ys, xs]
    weight = inside.to(T.dtype)
    if weight.sum() < 16:
        return None
    gain, bias = photometric(warped, base, weight)
    return float(((gain * warped + bias - base).abs() * weight).sum() / weight.sum())


def iterate(level, p, c, active, iterations):
    """Gauss-Newton steps on one pyramid level. p = [a00, a01, a10, a11, tx, ty, gain, bias] (tensor, updated in place)."""
    T, I, R = level["T"], level["I"], level["R"]
    h, w = T.shape
    ys, xs = torch.nonzero(R, as_tuple=True)
    if ys.numel() < 32:
        return False
    x, y = xs.to(torch.float64) - c[0], ys.to(torch.float64) - c[1]
    # both sides blurred alike, so the residual compares like with like
    base = blur121(T)[ys, xs].to(torch.float64)
    Is = blur121(I)
    gx, gy = gradients(Is)
    planes = torch.stack([Is, gx, gy]).to(torch.float32)
    on = torch.tensor(active, dtype=torch.bool)
    # the affine terms are per pixel of distance: scaled to the level's size so the normal equations stay balanced
    L = max(w, h) / 2.0
    for _ in range(iterations):
        X = p[0] * x + p[1] * y + c[0] + p[4]
        Y = p[2] * x + p[3] * y + c[1] + p[5]
        inside = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
        if inside.sum() < 32:
            return False
        v = sample(planes, X.to(torch.float32), Y.to(torch.float32), w, h).to(torch.float64)
        Iw, Ix, Iy = v[0][inside], v[1][inside], v[2][inside]
        xi, yi, bi = x[inside], y[inside], base[inside]
        g = p[6]
        r = g * Iw + p[7] - bi
        J = torch.stack([g * Ix * xi / L, g * Ix * yi / L, g * Iy * xi / L, g * Iy * yi / L, g * Ix, g * Iy, Iw, torch.ones_like(Iw)], dim=1)
        J = J[:, on]
        # Huber weights: what the model changed on the ring (a shadow it moved) does not drag the fit
        scale = 1.4826 * (r - r.median()).abs().median().clamp_min(1e-4)
        k = 1.345 * scale
        wts = torch.where(r.abs() <= k, torch.ones_like(r), k / r.abs())
        H = (J * wts[:, None]).T @ J
        b = (J * wts[:, None]).T @ r
        H = H + torch.eye(H.shape[0], dtype=H.dtype) * (1e-9 + 1e-6 * H.diagonal().mean())
        try:
            delta = -torch.linalg.solve(H, b)
        except RuntimeError:
            return False
        full = torch.zeros(8, dtype=torch.float64)
        full[on] = delta
        full[:4] /= L
        p += full
        if not torch.isfinite(p).all():
            return False
        if full[4:6].abs().max() < 1e-3 and full[:4].abs().max() < 1e-6:
            break
    return True


def fit(base_rgb, patch_rgb, ring_full):
    """
    The affine drift of the patch against the base on the ring, both [H, W, 3] in 0..1 at full size, ring [H, W] bool.
    -> dict with aligned (bool), reason, and for a fit the full-size A (2 x 2), t (2) and centre c (2) for
    out(x) = patch(A (x - c) + c + t), scale, shear, shift, before, after.
    """
    H, W = ring_full.shape
    ring_count = int(ring_full.sum())
    if ring_count < LIMITS["min_ring"] or ring_count < LIMITS["min_ring_share"] * W * H:
        return {"aligned": False, "reason": "ring too small"}
    s = min(1.0, WORK / max(W, H))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    T = shrink(gray(base_rgb.float()), h, w)
    I = shrink(gray(patch_rgb.float()), h, w)
    R = shrink(ring_full.float(), h, w) > 0.99
    # flat: no contrast, or gradients in one direction only (a single straight edge cannot be fitted)
    Tb = blur121(T)
    gx, gy = gradients(Tb)
    on = R.clone()
    on[0, :] = on[-1, :] = False
    on[:, 0] = on[:, -1] = False
    n = int(on.sum())
    if n < 32:
        return {"aligned": False, "reason": "ring too small"}
    vals = T[on]
    sxx, syy, sxy = float((gx[on] ** 2).mean()), float((gy[on] ** 2).mean()), float((gx[on] * gy[on]).mean())
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lmin = tr / 2 - math.sqrt(max(0.0, tr * tr / 4 - det))
    if float(vals.std()) < 0.02 or lmin < 2e-6:
        return {"aligned": False, "reason": "flat area"}

    levels = [{"T": T, "I": I, "R": R}]
    while len(levels) < 3 and min(levels[-1]["T"].shape) >= 192:
        levels.append(half(levels[-1]))
    c0 = ((w - 1) / 2.0, (h - 1) / 2.0)
    p = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float64)
    for k in range(len(levels) - 1, -1, -1):
        f = 2 ** k
        c = ((c0[0] + 0.5) / f - 0.5, (c0[1] + 0.5) / f - 0.5)
        p[4] /= f
        p[5] /= f
        ok = True
        if k == len(levels) - 1:
            ok = iterate(levels[k], p, c, [0, 0, 0, 0, 1, 1, 1, 1], 30)
        if ok:
            ok = iterate(levels[k], p, c, [1] * 8, 40 if k else 60)
        p[4] *= f
        p[5] *= f
        if not ok:
            return {"aligned": False, "reason": "no fit found"}

    A = p[:4].view(2, 2)
    cf = ((c0[0] + 0.5) / s - 0.5, (c0[1] + 0.5) / s - 0.5)
    tf = (float(p[4]) / s, float(p[5]) / s)
    sx, sy = float(torch.hypot(A[0, 0], A[1, 0])), float(torch.hypot(A[0, 1], A[1, 1]))
    shear = abs(float(A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]))
    out = {"scale": [round(sx, 4), round(sy, 4)], "shear": round(shear, 4), "shift": [round(tf[0], 1), round(tf[1], 1)],
           "A": A.to(torch.float32), "t": tf, "c": cf}
    if (abs(sx - 1) > LIMITS["scale"] or abs(sy - 1) > LIMITS["scale"] or shear > LIMITS["shear"]
            or abs(tf[0]) > LIMITS["shift"] * W or abs(tf[1]) > LIMITS["shift"] * H):
        return {"aligned": False, "reason": "fit out of range", **out}
    before = ring_difference(T, I, R)
    after = ring_difference(T, I, R, A.to(T.dtype), (float(p[4]), float(p[5])), c0)
    if before is None or after is None:
        return {"aligned": False, "reason": "ring too small", **out}
    out["before"], out["after"] = round(before * 255, 2), round(after * 255, 2)
    if after > before * LIMITS["gain"]:
        return {"aligned": False, "reason": "no gain", **out}
    return {"aligned": True, **out}


def warp(rgb, A, t, c):
    """[H, W, 3] read through out(x) = rgb(A (x - c) + c + t), bilinear, edges repeated."""
    H, W = rgb.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    X = A[0, 0] * (xs - c[0]) + A[0, 1] * (ys - c[1]) + c[0] + t[0]
    Y = A[1, 0] * (xs - c[0]) + A[1, 1] * (ys - c[1]) + c[1] + t[1]
    grid = torch.stack([X / max(1, W - 1) * 2 - 1, Y / max(1, H - 1) * 2 - 1], dim=-1)[None]
    out = F.grid_sample(rgb.permute(2, 0, 1)[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0].permute(1, 2, 0).clamp(0, 1)


def color_match(patch, base, keep):
    """
    The patch's per-channel mean and spread moved onto the base's, both measured where keep (0..1) is high.
    -> (patch, report). Measured at most WORK px on the long side; the correction applies at full size.
    """
    H, W = keep.shape
    s = min(1.0, WORK / max(W, H))
    h, w = max(1, round(H * s)), max(1, round(W * s))
    k = shrink(keep.float(), h, w).clamp(0, 1)
    total = float(k.sum())
    if total < COLOR["min_weight"]:
        return patch, {"matched": False, "reason": "ring too small"}
    P, B = shrink(patch.float(), h, w), shrink(base.float(), h, w)
    k3 = k[..., None]

    def stats(img):
        mean = (img * k3).sum(dim=(0, 1)) / total
        var = (((img - mean) ** 2) * k3).sum(dim=(0, 1)) / total
        return mean, var.clamp_min(1e-6).sqrt()

    mb, sb = stats(B)
    mp, sp = stats(P)
    gain = (sb / sp).clamp(0.5, 2.0)
    shift = (mb - mp).abs().max()
    info = {"mean": [round(float(v) * 255, 1) for v in (mb - mp)], "spread": [round(float(v), 3) for v in gain]}
    lo, hi = COLOR["spread"]
    if float(shift) > COLOR["mean"] or float(gain.min()) < lo or float(gain.max()) > hi:
        return patch, {"matched": False, "reason": "difference looks meant", **info}
    if float(shift) < 1.0 / 255 and float((gain - 1).abs().max()) < 0.01:
        return patch, {"matched": False, "reason": "no drift", **info}
    return ((patch - mp) * gain + mb).clamp(0, 1), {"matched": True, **info}


def stitch(original, patch, mask, x, y, align=True, match=True):
    """
    One picture: patch ([h, w, 3], already at the box's size) onto original ([H, W, 3]) at (x, y) through mask ([h, w],
    1 = the patch). -> (picture, report)
    """
    h, w = mask.shape
    region = original[y:y + h, x:x + w]
    if region.shape[0] != h or region.shape[1] != w:
        raise ValueError(f"HEISS Inpaint Stitch: the box {w}x{h} at {x},{y} does not fit a {original.shape[1]}x{original.shape[0]} picture.")
    # the ring: where not even the softest trace of the mask reaches, a few pixels clear of it
    ring = dilate((mask > 1.0 / 255).float(), RING_GAP) < 0.5
    result = {"align": {"aligned": False, "reason": "off"}, "color": {"matched": False, "reason": "off"}}
    if align:
        found = fit(region, patch, ring)
        if found.get("aligned"):
            patch = warp(patch, found["A"], found["t"], found["c"])
        result["align"] = {key: value for key, value in found.items() if key not in ("A", "t", "c")}
    if match:
        patch, result["color"] = color_match(patch, region, (1.0 - mask).clamp(0, 1) * ring.float())
    out = original.clone()
    m = mask[..., None]
    out[y:y + h, x:x + w] = region * (1.0 - m) + patch * m
    return out, result


def describe(result):
    a, c = result["align"], result["color"]
    parts = []
    if a.get("aligned"):
        parts.append(f"lined up (shift {a['shift'][0]}, {a['shift'][1]} px, scale {a['scale'][0]}, ring {a['before']} -> {a['after']})")
    else:
        parts.append(f"not moved ({a.get('reason')})")
    if c.get("matched"):
        parts.append(f"colour matched (means {c['mean']})")
    else:
        parts.append(f"colour kept ({c.get('reason')})")
    return ", ".join(parts)


class HeissInpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE", {"tooltip": "The whole picture the crop was cut from."}),
            "image": ("IMAGE", {"tooltip": "The inpainted crop, any size: it is scaled to the mask's size."}),
            "mask": ("MASK", {"tooltip": "Where the crop goes over the original, at the box's own size (1 = the crop, soft edge included)."}),
            "x": ("INT", {"default": 0, "min": 0, "max": 65536}),
            "y": ("INT", {"default": 0, "min": 0, "max": 65536}),
            "align": ("BOOLEAN", {"default": True, "tooltip": "Undo a small shift or zoom of the crop, measured on the ring around the mask."}),
            "color_match": ("BOOLEAN", {"default": True, "tooltip": "Undo a small exposure or colour drift of the crop, measured on the ring around the mask."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "HEISS UI/inpaint"
    DESCRIPTION = ("Puts an inpainted crop back onto the picture. Edit models shift, zoom and tint what they re-render; this "
                   "lines the crop up and matches its colour on the ring around the mask first, so the seam does not show.")

    def stitch(self, original, image, mask, x, y, align=True, color_match=True):
        if mask.dim() == 3:
            mask = mask[0]
        h, w = mask.shape
        crops = image[..., :3].float()
        if crops.shape[1] != h or crops.shape[2] != w:
            crops = comfy.utils.common_upscale(crops.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
        mask = mask.float().cpu().clamp(0, 1)
        outputs, results = [], []
        for index in range(crops.shape[0]):
            base = original[min(index, original.shape[0] - 1), ..., :3].float().cpu()
            picture, result = stitch(base, crops[index].cpu(), mask, int(x), int(y), bool(align), bool(color_match))
            outputs.append(picture)
            results.append(result)
        report({"results": results, "message": "; ".join(describe(r) for r in results)})
        return (torch.stack(outputs).to(original.dtype),)


NODE_CLASS_MAPPINGS = {"HeissInpaintStitch": HeissInpaintStitch}
NODE_DISPLAY_NAME_MAPPINGS = {"HeissInpaintStitch": "HEISS Inpaint Stitch"}
