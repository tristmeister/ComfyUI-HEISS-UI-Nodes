"""
HEISS Rapid: the noisy first part of a run on a half-size latent, the rest at full size.

While the picture is still mostly noise, the steps only settle layout, pose, light and colour; nothing there needs the
full grid. Rapid samples those steps on a latent half as wide and half as tall, then grows it to full size in its
cosine spectrum before the fine detail is drawn, filling the new high frequencies with fresh noise at the current noise
level. Same steps, same peak VRAM; the early steps cost about a quarter as much. A seed frames a little differently
with it on.

Based on SPEED, "Spectral Progressive Diffusion for Efficient Image and Video Generation" (Xiao, Chao, Yariv and
Wetzstein, 2026, MIT), with lessons from LC Speed Boost (lonecatone23, ComfyUI_LC123_nodes, MIT) and the SwarmNeo
fork of ComfyUI-SPEED (aoleg, MIT).

The jump: after the grow the noise level is realigned upwards (0.70 -> 0.82 at half size), and the paper's own code,
LC Speed Boost and ComfyUI-SPEED then step straight from there to the schedule's next level. That one long Euler step
costs the low frequencies about 6-9 % of their strength, at any step count (tests/test_rapid.py measures it with an
exact denoiser). `smooth_switch` adds one full-size step halfway through that jump, which brings them back to within
about 2 % of a run without Rapid, for the price of one step.

The grow, per kind of model (r = full size / small size, from the areas):
  flow  (x = (1 - t) * picture + t * s * noise; CONST, s = its noise_scale): padding leaves the picture 1/r as strong
        as the noise, so the state is scaled by k = r / (1 + (r - 1) * t) and sampling carries on at k * t.
  sigma (x = picture + sigma * noise; EPS, V_PREDICTION, EDM, Cosmos RFlow): the state is scaled by r and sampling
        carries on at r * sigma. Its noise level compares on the flow scale, t = sigma / (1 + sigma).
"""

from __future__ import annotations

import math

import torch

from comfy.samplers import KSAMPLER

MIN_START = 0.9  # a run that starts below this (on the flow scale) is image to image / upscale: its layout is given
MAX_SMALL_SHARE = 0.8  # never more than this share of the steps small, whatever the switch point says
SMOOTH_FROM_STEPS = 8  # below this the smoothing step costs more than the start saves (a 4-step run would be slower)

_DCT = {}


def _dct(n, device):
    """Orthonormal DCT-II matrix (n x n): coefficients = D @ signal, signal = D.T @ coefficients."""
    key = (n, str(device))
    if key not in _DCT:
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
        i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
        d[0] /= math.sqrt(2.0)
        _DCT[key] = d.to(device=device, dtype=torch.float32)
    return _DCT[key]


def spectrum(x):
    """Cosine spectrum over the last two (height, width) axes, in float32. Works for image and video latents alike."""
    h, w = x.shape[-2:]
    return _dct(h, x.device) @ x.float() @ _dct(w, x.device).T


def unspectrum(c):
    h, w = c.shape[-2:]
    return _dct(h, c.device).T @ c @ _dct(w, c.device)


def shrink(x, h, w):
    """Full size -> h x w by keeping the low corner of the spectrum. White noise stays white at the same strength; a
    picture comes out r times as strong (the same energy on fewer pixels)."""
    return unspectrum(spectrum(x)[..., :h, :w].contiguous())


def grow(y, H, W, t, seed, kind="flow", noise_scale=1.0):
    """Small state at noise level t (flow scale for "flow", sigma for "sigma") -> (full size state, its new noise
    level). The new high frequencies get fresh noise as strong as the noise already in the state."""
    h, w = y.shape[-2:]
    c = torch.zeros(*y.shape[:-2], H, W, device=y.device, dtype=torch.float32)
    c[..., :h, :w] = spectrum(y)
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    fresh = torch.randn(*y.shape[:-2], H, W, generator=g, dtype=torch.float32).to(y.device)
    fresh[..., :h, :w] = 0.0  # white noise in this basis: only where the small grid had nothing
    c += (t * noise_scale) * fresh
    r = math.sqrt((H * W) / float(h * w))
    k = r if kind == "sigma" else r / (1.0 + (r - 1.0) * t)
    return (unspectrum(c) * k).to(y.dtype), k * t


def _patcher(model):
    m = model
    for _ in range(8):
        if hasattr(m, "model_patcher"):
            return m.model_patcher
        m = getattr(m, "inner_model", None)
        if m is None:
            return None
    return None


def model_kind(model):
    """("flow" | "sigma" | None, noise_scale) from the model's own model_sampling."""
    import comfy.model_sampling as ms

    p = _patcher(model)
    sampling = p.get_model_object("model_sampling") if p is not None else getattr(model, "heiss_model_sampling", None)
    if sampling is None:
        return None, 1.0
    # These start from the picture itself, not from noise.
    if isinstance(sampling, tuple(getattr(ms, name) for name in ("IMG_TO_IMG", "IMG_TO_IMG_FLOW") if hasattr(ms, name))):
        return None, 1.0
    if isinstance(sampling, ms.CONST):
        return "flow", float(getattr(sampling, "noise_scale", 1.0) or 1.0)
    sigma_kinds = tuple(getattr(ms, name) for name in ("EPS", "V_PREDICTION", "COSMOS_RFLOW") if hasattr(ms, name))
    if isinstance(sampling, sigma_kinds):
        return "sigma", 1.0
    return None, 1.0


def flow_t(s, kind):
    """A noise level on the 0..1 flow scale (sigma s carries the same mix as flow t = s / (1 + s))."""
    return s if kind == "flow" else s / (1.0 + s)


def plan(sigmas, kind, switch_at, min_full_steps=2):
    """The step where the picture grows to full size, counted from the start of this pass, or (None, why).
    The step nearest the switch point, with the last steps at full size (only the last one when the pass hands over
    to another sampler above noise 0) and no more than MAX_SMALL_SHARE of the steps small."""
    n = len(sigmas) - 1
    ts = [flow_t(float(v), kind) for v in sigmas]
    if ts[0] < MIN_START:
        return None, f"the run starts at noise {ts[0]:.2f}, too little to start small (image to image or upscale)"
    keep = 1 if ts[-1] > 0 else max(1, int(min_full_steps))
    limit = min(n - keep, int(math.floor(MAX_SMALL_SHARE * n)))
    if limit < 1:
        return None, f"too few steps ({n})"
    j = min(range(1, n), key=lambda i: (abs(ts[i] - switch_at), i))
    return max(1, min(j, limit)), None


def small_size(H, W, scale):
    """The small grid, even on both sides (DiTs patch in 2s), never the full one."""
    h = max(2, int(math.floor(H * scale / 2.0 + 0.5)) * 2)
    w = max(2, int(math.floor(W * scale / 2.0 + 0.5)) * 2)
    return min(h, H - 2 if H > 2 else H), min(w, W - 2 if W > 2 else W)


def shrink_start(model, x, s0, h, w, kind, noise_scale):
    """The starting state at the small size. Noise shrinks as noise; the picture in it (if any) shrinks as a picture,
    which comes out r times too strong and is put back to its own strength."""
    lat = getattr(model, "latent_image", None)
    small = shrink(x, h, w)
    if lat is None or lat.shape != x.shape or not torch.count_nonzero(lat):
        return small.to(x.dtype)
    r = math.sqrt((x.shape[-2] * x.shape[-1]) / float(h * w))
    picture = (1.0 - s0) * lat if kind == "flow" else lat
    return (small - shrink(picture.to(x.device), h, w) * (1.0 - 1.0 / r)).to(x.dtype)


def report(data):
    """Tells the client that queued this run (HEISS) what Rapid did with it. A prompt queued without a client
    gets the console line only, never a broadcast to everyone connected."""
    print(f"[HEISS Rapid] {data.get('message', '')}")
    try:
        from server import PromptServer

        server = PromptServer.instance
        client = getattr(server, "client_id", None)
        if client:
            server.send_sync("heiss.rapid", {**data, "prompt_id": getattr(server, "last_prompt_id", None)}, client)
    except Exception:
        pass


def spatial_conds(model):
    """Why the conditioning can't follow a smaller latent, or None: images concatenated to the latent (inpaint models,
    image-to-video, Fill, InstructPix2Pix), masked or area conditioning. They are made at full size before sampling."""
    guider = getattr(model, "inner_model", None)
    conds = getattr(guider, "conds", None) or {}
    for items in conds.values():
        for cond in items or []:
            if not isinstance(cond, dict):
                continue
            if cond.get("mask") is not None or cond.get("area") is not None:
                return "masked or area conditioning"
            model_conds = cond.get("model_conds") or {}
            if any(key in model_conds for key in ("c_concat", "concat_latent_image", "noise_concat", "concat_mask")):
                return "the model reads a full-size image next to the latent (inpaint, image to video, Fill)"
    return None


def reset_shape_caches(model):
    """Caches a model keeps for one sampling run, keyed by the latent size: they get a second entry once the picture
    grows, and ComfyUI's Qwen-Image 2.1 prefix cache can't hold two (its slot lookup compares tensors with ==).
    Starting it afresh at the grow is what a run at the new size would have."""
    patcher = _patcher(model)
    diffusion = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    reset = getattr(diffusion, "reset_prefix_cache", None)
    if callable(reset):
        try:
            reset(bool(getattr(diffusion, "prefix_cache_enabled", False)))
        except Exception:
            pass


def after_grow(sigmas, j, t2, kind, smooth):
    """The schedule once the picture is full size: from the realigned level t2 on, optionally with one extra step in
    the middle of the jump back to the schedule (geometric for sigma schedules, which space their steps that way)."""
    nxt = float(sigmas[j + 1])
    head = [t2]
    if smooth and nxt > 0:
        head.append(math.sqrt(t2 * nxt) if kind == "sigma" else (t2 + nxt) / 2.0)
    return torch.cat([torch.tensor(head, dtype=sigmas.dtype, device=sigmas.device), sigmas[j + 1:]])


def sample_rapid(model, x, sigmas, extra_args=None, callback=None, disable=None, *,
                 heiss_inner, heiss_switch_at, heiss_scale, heiss_min_full, heiss_smooth=True, **kwargs):
    extra_args = dict(extra_args or {})
    inner_fn, inner_opts = heiss_inner.sampler_function, dict(heiss_inner.extra_options)

    def run(xx, ss, offset, extra=0):
        # Progress counts the run's own steps: an added step shows as the step it splits.
        cb = None
        if callback is not None:
            cb = lambda d: callback({**d, "i": offset + max(0, d["i"] - extra)})
        return inner_fn(model, xx, ss, extra_args=extra_args, callback=cb, disable=disable, **inner_opts)

    def full(why):
        report({"active": False, "reason": why, "message": f"Off for this run: {why}. Sampling at full size."})
        return run(x, sigmas, 0)

    n = len(sigmas) - 1
    if x.ndim not in (4, 5):
        return full(f"a {x.ndim}D latent (it works on pictures and video)")
    if extra_args.get("denoise_mask") is not None:
        return full("inpaint mask (it only works on the whole picture)")
    spatial = spatial_conds(model)
    if spatial:
        return full(spatial)
    kind, noise_scale = model_kind(model)
    if kind is None:
        return full("this kind of model isn't supported (flow models and SD-family models are)")
    if n < 2:
        return full(f"too few steps ({n})")
    j, why = plan(sigmas, kind, float(heiss_switch_at), heiss_min_full)
    if j is None:
        return full(why)
    H, W = x.shape[-2:]
    if H < 8 or W < 8:
        return full("the picture is too small to start smaller")
    h, w = small_size(H, W, float(heiss_scale))
    if h >= H or w >= W:
        return full("the picture is too small to start smaller")

    xs = shrink_start(model, x, float(sigmas[0]), h, w, kind, noise_scale)
    y = run(xs, sigmas[: j + 1], 0)
    reset_shape_caches(model)
    t = float(sigmas[j])
    X, t2 = grow(y, H, W, t, int(extra_args.get("seed", 0) or 0) + 7, kind, noise_scale)
    # One more full-size step is worth it on longer runs; on a few-step distill it costs more than it saves.
    rest = after_grow(sigmas, j, t2, kind, bool(heiss_smooth) and n >= SMOOTH_FROM_STEPS)
    extra = len(rest) - 1 - (n - j)
    report({
        "active": True, "small_steps": j, "full_steps": n - j, "switch_noise": round(flow_t(t, kind), 4),
        "extra_steps": extra, "small": [w, h], "full": [W, H],
        "message": f"{j} of {n} steps small (latent {w}x{h} of {W}x{H}), grew at noise {flow_t(t, kind):.3f}, "
                   f"{n - j} at full size{' plus one to smooth the switch' if extra else ''}.",
    })
    return run(X, rest, j, extra)


class HeissRapid:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER", {"tooltip": "The sampler to speed up (from KSamplerSelect). For a two-pass setup, the first (high-noise) pass only."}),
            "switch_at": ("FLOAT", {"default": 0.7, "min": 0.3, "max": 0.99, "step": 0.01,
                                    "tooltip": "Noise level where the picture grows to full size (1 = pure noise, 0 = finished; "
                                               "SD-family models on the same scale, sigma / (1 + sigma)). Later is faster, earlier is safer "
                                               "for small detail such as text."}),
            "scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 0.9, "step": 0.05,
                                "tooltip": "Size of the start, per side."}),
            "min_full_steps": ("INT", {"default": 2, "min": 1, "max": 50,
                                       "tooltip": "Steps at the end that always run at full size."}),
            "smooth_switch": ("BOOLEAN", {"default": True,
                                          "tooltip": "One extra full-size step right after the switch. Keeps the picture's contrast "
                                                     "and colour true to a run without Rapid, for the price of one step."}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "wrap"
    CATEGORY = "HEISS UI/sampling"
    DESCRIPTION = ("Starts the picture at half size and grows it to full size partway through: about 1.4x to 2x faster, "
                   "same VRAM, detail drawn at full size. A seed frames a little differently with it on. Steps aside for "
                   "image to image, upscale and inpaint passes. Based on SPEED (Xiao, Chao, Yariv and Wetzstein, 2026).")

    def wrap(self, sampler, switch_at, scale, min_full_steps, smooth_switch=True):
        return (KSAMPLER(sample_rapid, extra_options={
            "heiss_inner": sampler, "heiss_switch_at": float(switch_at), "heiss_scale": float(scale), "heiss_min_full": int(min_full_steps),
            "heiss_smooth": bool(smooth_switch),
        }, inpaint_options=getattr(sampler, "inpaint_options", {})),)


NODE_CLASS_MAPPINGS = {"HeissRapid": HeissRapid}
NODE_DISPLAY_NAME_MAPPINGS = {"HeissRapid": "HEISS Rapid"}
