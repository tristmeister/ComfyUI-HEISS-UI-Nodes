"""HEISS Inpaint Stitch: alignment and colour match on known drifts. Run like test_rapid.py."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import heiss_inpaint as inpaint  # noqa: E402


def texture(W, H, seed=1):
    """An opaque RGB picture with detail at several sizes and some hard-edged discs, in 0..1."""
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    v = torch.full((H, W), 0.5)
    tint = torch.zeros((H, W))
    for _ in range(14):
        a = float(torch.rand(1, generator=g)) * 2 * math.pi
        f = 0.015 + float(torch.rand(1, generator=g)) * 0.2
        wave = torch.sin(math.cos(a) * f * xs + math.sin(a) * f * ys + float(torch.rand(1, generator=g)) * 6.28)
        amp = (8 + float(torch.rand(1, generator=g)) * 14) / 255
        v += wave * amp
        tint += wave * amp * (float(torch.rand(1, generator=g)) - 0.5)
    for _ in range(25):
        cx, cy = float(torch.rand(1, generator=g)) * W, float(torch.rand(1, generator=g)) * H
        r = 6 + float(torch.rand(1, generator=g)) * 40
        v += ((xs - cx) ** 2 + (ys - cy) ** 2 < r * r).float() * (float(torch.rand(1, generator=g)) - 0.5) * 0.45
    return torch.stack([v, v + tint, v - tint], dim=-1).clamp(0, 1)


def ellipse_mask(W, H, rx, ry, soft=6.0):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    d = torch.sqrt(((xs - (W - 1) / 2) / rx) ** 2 + ((ys - (H - 1) / 2) / ry) ** 2)
    return ((1 - d) * min(rx, ry) / soft + 0.5).clamp(0, 1)


def drifted(base, B, tb, mask, new=None, gain=1.0, offset=0.0):
    """What an edit model hands back: the picture read through B (x - c) + c + tb, new content inside the mask, a colour drift."""
    H, W = base.shape[:2]
    c = ((W - 1) / 2, (H - 1) / 2)
    out = inpaint.warp(base, torch.tensor(B, dtype=torch.float32), tb, c)
    if new is not None:
        m = (mask > 0.01).float()[..., None]
        out = out * (1 - m) + new * m
    return (out * gain + offset).clamp(0, 1)


def corner_error(found, B, tb, W, H):
    """Largest distance at the box's corners between the fitted correction and the exact inverse of the drift."""
    A, t, c = found["A"].double(), found["t"], found["c"]
    Bm = torch.tensor(B, dtype=torch.float64)
    worst = 0.0
    for x, y in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        X = A @ torch.tensor([x - c[0], y - c[1]], dtype=torch.float64) + torch.tensor(c, dtype=torch.float64) + torch.tensor(t, dtype=torch.float64)
        # where the drift sends that point back: B (X - c) + c + tb must be (x, y)
        back = Bm @ (X - torch.tensor(c, dtype=torch.float64)) + torch.tensor(c, dtype=torch.float64) + torch.tensor(tb, dtype=torch.float64)
        worst = max(worst, float(torch.hypot(back[0] - x, back[1] - y)))
    return worst


class Align(unittest.TestCase):
    W, H = 640, 520

    def setUp(self):
        self.base = texture(self.W, self.H)
        self.mask = ellipse_mask(self.W, self.H, 150, 120)
        self.ring = inpaint.dilate((self.mask > 1 / 255).float(), inpaint.RING_GAP) < 0.5
        self.new = texture(self.W, self.H, seed=7).flip(0)

    def check_recovers(self, B, tb, **drift):
        patch = drifted(self.base, B, tb, self.mask, self.new, **drift)
        found = inpaint.fit(self.base, patch, self.ring)
        self.assertTrue(found["aligned"], found)
        self.assertLess(corner_error(found, B, tb, self.W, self.H), 1.0, found)
        self.assertLess(found["after"], found["before"])
        return found

    def test_a_shift_comes_back(self):
        self.check_recovers([[1, 0], [0, 1]], (9.0, -6.0))

    def test_a_zoom_and_shift_come_back(self):
        self.check_recovers([[1.035, 0], [0, 1.035]], (-7.0, 4.0))

    def test_a_small_rotation_comes_back(self):
        a = math.radians(1.2)
        self.check_recovers([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], (3.0, 2.0))

    def test_a_colour_drift_does_not_pull_the_geometry(self):
        self.check_recovers([[1.02, 0], [0, 1.02]], (5.0, 5.0), gain=0.9, offset=0.06)

    def test_under_half_a_pixel_is_left_alone(self):
        # what masked-sampling models come back with: a fit there, but no line could double, so no resampling
        patch = drifted(self.base, [[1, 0], [0, 1]], (0.3, 0.2), self.mask, self.new)
        found = inpaint.fit(self.base, patch, self.ring)
        self.assertFalse(found["aligned"])
        self.assertEqual(found["reason"], "under half a pixel", found)
        self.assertLess(found["move"], 0.5)

    def test_no_drift_finds_no_gain(self):
        patch = drifted(self.base, [[1, 0], [0, 1]], (0.0, 0.0), self.mask, self.new)
        found = inpaint.fit(self.base, patch, self.ring)
        self.assertFalse(found["aligned"])

    def test_a_drift_past_the_limits_is_refused(self):
        for share in (0.06, 0.09, 0.12):
            patch = drifted(self.base, [[1, 0], [0, 1]], (share * self.W, 0.0), self.mask, self.new)
            found = inpaint.fit(self.base, patch, self.ring)
            self.assertFalse(found["aligned"], (share, found))

    def test_a_flat_ring_is_refused(self):
        flat = torch.full((self.H, self.W, 3), 0.4)
        found = inpaint.fit(flat, drifted(flat, [[1, 0], [0, 1]], (4.0, 0.0), self.mask, self.new), self.ring)
        self.assertEqual(found["reason"], "flat area")

    def test_a_ring_too_small_is_refused(self):
        ring = torch.zeros((self.H, self.W), dtype=torch.bool)
        ring[:10, :10] = True
        self.assertEqual(inpaint.fit(self.base, self.base, ring)["reason"], "ring too small")


class Warp(unittest.TestCase):
    def test_no_move_is_exact(self):
        rgb = texture(120, 90)
        out = inpaint.warp(rgb, torch.eye(2), (0.0, 0.0), (59.5, 44.5))
        self.assertLess(float((out - rgb).abs().max()), 1e-4)

    def test_a_fraction_of_a_pixel_keeps_the_detail(self):
        # a 0.4 px correction, as on a masked-sampling model: bicubic keeps more fine detail than bilinear did
        rgb = texture(200, 160)
        out = inpaint.warp(rgb, torch.eye(2), (0.4, 0.4), (99.5, 79.5))
        H, W = rgb.shape[:2]
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        grid = torch.stack([(xs + 0.4) / (W - 1) * 2 - 1, (ys + 0.4) / (H - 1) * 2 - 1], dim=-1)[None]
        bilinear = torch.nn.functional.grid_sample(rgb.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 0)
        detail = lambda img: float((img[1:, :] - img[:-1, :]).abs().mean() + (img[:, 1:] - img[:, :-1]).abs().mean())
        self.assertGreater(detail(out), detail(bilinear) * 1.02)


class ColorMatch(unittest.TestCase):
    def setUp(self):
        self.base = texture(400, 300)
        self.mask = ellipse_mask(400, 300, 90, 70)
        ring = inpaint.dilate((self.mask > 1 / 255).float(), inpaint.RING_GAP) < 0.5
        # as the node measures it: the ring clear of the mask
        self.keep = (1 - self.mask) * ring.float()

    def ring_mean_error(self, patch):
        k = (self.keep > 0.99).float()[..., None]
        return float((((patch - self.base) * k).sum(dim=(0, 1)) / k.sum()).abs().max())

    def test_a_small_drift_is_undone(self):
        patch = (self.base * torch.tensor([1.04, 0.97, 0.95]) + 0.03).clamp(0, 1)
        out, info = inpaint.color_match(patch, self.base, self.keep)
        self.assertTrue(info["matched"], info)
        self.assertLess(self.ring_mean_error(out), 0.25 * self.ring_mean_error(patch))

    def test_a_large_difference_is_left_as_meant(self):
        patch = (self.base * 0.45).clamp(0, 1)
        out, info = inpaint.color_match(patch, self.base, self.keep)
        self.assertFalse(info["matched"])
        self.assertTrue(torch.equal(out, patch))

    def test_a_change_bleeding_onto_the_ring_is_left_as_meant(self):
        # Flux.2 Klein asked for a green curtain over a flag: the ring came back 24, 9 and 20 levels off.
        # Matching that tinted the curtain grey and the jacket next to it purple.
        patch = (self.base - torch.tensor([24.0, 9.0, 20.0]) / 255).clamp(0, 1)
        out, info = inpaint.color_match(patch, self.base, self.keep)
        self.assertFalse(info["matched"], info)
        self.assertTrue(torch.equal(out, patch))

    def test_new_content_inside_the_mask_does_not_count(self):
        new = self.base.clone()
        m = (self.mask > 0.01)[..., None]
        new = torch.where(m, torch.tensor([0.9, 0.1, 0.1]), new)
        out, info = inpaint.color_match(new, self.base, self.keep)
        self.assertFalse(info["matched"], info)


class Node(unittest.TestCase):
    def test_outside_the_mask_the_original_stays_exact(self):
        original = texture(900, 700)[None]
        mask = ellipse_mask(320, 256, 100, 80)
        crop = drifted(original[0, 100:356, 200:520], [[1.03, 0], [0, 1.03]], (4.0, -3.0), mask, texture(320, 256, 9))
        # a batch of two at the model's working size, which the node scales to the box
        small = torch.nn.functional.interpolate(crop.permute(2, 0, 1)[None], size=(512, 640), mode="bilinear").permute(0, 2, 3, 1)
        out = inpaint.HeissInpaintStitch().stitch(original, small.repeat(2, 1, 1, 1), mask[None], 200, 100)[0]
        self.assertEqual(tuple(out.shape), (2, 700, 900, 3))
        outside = torch.ones((700, 900), dtype=torch.bool)
        outside[100:356, 200:520] = mask <= 0
        self.assertTrue(torch.equal(out[0][outside], original[0][outside]))
        self.assertTrue(torch.equal(out[0], out[1]))

    def test_off_is_a_plain_composite(self):
        original = texture(300, 200)[None]
        mask = ellipse_mask(100, 80, 30, 25)
        crop = texture(100, 80, 3)[None]
        out = inpaint.HeissInpaintStitch().stitch(original, crop, mask[None], 50, 60, align=False, color_match=False)[0][0]
        region = original[0, 60:140, 50:150]
        expected = region * (1 - mask[..., None]) + crop[0] * mask[..., None]
        self.assertTrue(torch.allclose(out[60:140, 50:150], expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
