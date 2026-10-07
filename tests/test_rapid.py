"""
Tests for HEISS Rapid. They need ComfyUI's source for `comfy.*`:

    COMFYUI_PATH=/path/to/ComfyUI /path/to/ComfyUI/python -m unittest discover -s tests -v

The central check samples Gaussian data with its exact denoiser. For pictures whose cosine coefficients are
independent with variances P, E[picture | state] has a closed form at every noise level and every grid size, so the
sampler is "a perfect model". Sampling with Rapid has to land on the same distribution as sampling at full size: the
same variance in every frequency band. A wrong rescale at the grow (the paper's kappa) shows up as the wrong variance,
which the negative control proves.
"""

import math
import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_PATH", "")
if COMFY:
    sys.path.insert(0, COMFY)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

try:
    import comfy.model_sampling as ms  # noqa: E402
    import comfy.k_diffusion.sampling as kds  # noqa: E402
    import heiss_rapid as rapid  # noqa: E402
except ImportError as error:  # pragma: no cover
    raise SystemExit(f"Set COMFYUI_PATH to a ComfyUI checkout ({error}).")

from comfy.samplers import KSAMPLER  # noqa: E402

H = W = 32
BATCH = 384


def power(h=H, w=W):
    """A picture spectrum that falls off with frequency, like a VAE latent's."""
    i = torch.arange(h, dtype=torch.float32).view(-1, 1)
    j = torch.arange(w, dtype=torch.float32).view(1, -1)
    return 6.0 / (1.0 + (i ** 2 + j ** 2) / 9.0) ** 1.4 + 0.02


class Oracle:
    """The exact denoiser for that Gaussian data, at any grid size (a small grid's picture is the low corner of the
    full one's spectrum, 1/r as strong per coefficient)."""

    def __init__(self, kind, noise_scale=1.0):
        self.kind = kind
        self.noise_scale = noise_scale
        self.P = power()
        # The same mixes ComfyUI builds for a flow model and an SD-family one (model_base.model_sampling).
        cls = type("Sampling", (ms.ModelSamplingDiscreteFlow, ms.CONST), {}) if kind == "flow" else type("Sampling", (ms.ModelSamplingDiscrete, ms.EPS), {})
        sampling = cls(None)
        if kind == "flow" and noise_scale != 1.0:
            sampling.noise_scale = noise_scale
        self.use(sampling)
        self.latent_image = None
        self.sizes = []

    def use(self, sampling):
        """The same chain ComfyUI hands a sampler: KSamplerX0Inpaint -> guider (model_patcher) -> model."""
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=sampling),
                                           model_patcher=SimpleNamespace(get_model_object=lambda name: sampling))

    def __call__(self, x, sigma, **kwargs):
        h, w = x.shape[-2:]
        self.sizes.append((h, w))
        P = self.P[:h, :w] / ((H * W) / float(h * w))
        s = sigma.float().view(-1, *([1] * (x.ndim - 1)))
        c = rapid.spectrum(x)
        if self.kind == "flow":
            gain = (1 - s) * P / ((1 - s) ** 2 * P + (s * self.noise_scale) ** 2)
        else:
            gain = P / (P + s ** 2)
        return rapid.unspectrum(c * gain).to(x.dtype)


def flow_sigmas(n=20, shift=3.0):
    u = torch.linspace(1, 0, n + 1)
    return shift * u / (1 + (shift - 1) * u)


def sdxl_sigmas(n=25):
    return kds.get_sigmas_karras(n, 0.0292, 14.6146)


def wrapped(switch_at, scale=0.5, min_full=2, inner="euler", smooth=True):
    sampler = KSAMPLER(getattr(kds, f"sample_{inner}"))
    return rapid.HeissRapid().wrap(sampler, switch_at, scale, min_full, smooth)[0]


def sample(model, sigmas, sampler=None, seed=1, start=None):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(BATCH, 1, H, W, generator=g)
    x = start if start is not None else noise * (model.noise_scale if model.kind == "flow" else 1.0) * float(sigmas[0])
    model.latent_image = torch.zeros_like(x)
    fn = sampler.sampler_function if sampler else kds.sample_euler
    opts = sampler.extra_options if sampler else {}
    return fn(model, x, sigmas, extra_args={"seed": seed}, disable=True, **opts)


def band_variance(x):
    """Variance per radial band of the cosine spectrum."""
    c = rapid.spectrum(x)
    var = c.pow(2).mean(dim=(0, 1))
    i = torch.arange(H).view(-1, 1)
    j = torch.arange(W).view(1, -1)
    radius = torch.sqrt((i ** 2 + j ** 2).float())
    edges = [0, 1, 2, 4, 8, 16, 46]
    return [float(var[(radius >= a) & (radius < b)].mean()) for a, b in zip(edges, edges[1:])]


class Quiet(unittest.TestCase):
    def setUp(self):
        self._report = rapid.report
        self.reports = []
        rapid.report = self.reports.append

    def tearDown(self):
        rapid.report = self._report


def gains(kind, sigmas, sampler=None, noise_scale=1.0, batch=4):
    """How strongly each cosine coefficient of the starting noise ends up in the picture, relative to the data's own
    strength sqrt(P): 1 is exact. In this Gaussian case every coefficient evolves on its own and linearly, so this is
    exact per coefficient, with no sampling noise. Averaged per radial band, the top (fresh-noise) band left out."""
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(batch, 1, H, W, generator=g)
    model = Oracle(kind, noise_scale)
    x = noise * (noise_scale if kind == "flow" else 1.0) * float(sigmas[0])
    model.latent_image = torch.zeros_like(x)
    fn = sampler.sampler_function if sampler else kds.sample_euler
    out = fn(model, x, sigmas, extra_args={"seed": 1}, disable=True, **(sampler.extra_options if sampler else {}))
    gain = (rapid.spectrum(out) / rapid.spectrum(noise))[0, 0] / power().sqrt()
    i = torch.arange(H).view(-1, 1)
    j = torch.arange(W).view(1, -1)
    radius = torch.sqrt((i ** 2 + j ** 2).float())
    edges = [0, 1, 2, 4, 8, 16]
    return [float(gain[(radius >= a) & (radius < b)].mean()) for a, b in zip(edges, edges[1:])], model


class GaussianOracle(Quiet):
    """Rapid against the same run at full size, with a perfect model."""

    def compare(self, kind, sigmas, switch_at, smooth=True, noise_scale=1.0):
        off, _ = gains(kind, sigmas, noise_scale=noise_scale)
        on, model = gains(kind, sigmas, wrapped(switch_at, smooth=smooth), noise_scale=noise_scale)
        self.assertTrue(self.reports[-1]["active"], self.reports[-1])
        self.assertIn((H // 2, W // 2), model.sizes, "never sampled small")
        self.assertEqual(model.sizes[-1], (H, W), "didn't finish at full size")
        return [b / a for a, b in zip(off, on)]

    def assertWithin(self, ratios, tolerance, what):
        for band, ratio in enumerate(ratios):
            self.assertLess(abs(ratio - 1), tolerance, f"{what}: band {band} at {ratio:.3f} of full size ({[round(r, 3) for r in ratios]})")

    def test_flow_smooth_switch_matches_full_size(self):
        for steps in (8, 10, 20, 30):
            with self.subTest(steps=steps):
                self.assertWithin(self.compare("flow", flow_sigmas(steps), 0.7), 0.035, f"flow, {steps} steps")

    def test_flow_with_noise_scale(self):
        self.assertWithin(self.compare("flow", flow_sigmas(20), 0.7, noise_scale=2.0), 0.035, "flow, noise scale 2")

    def test_sigma_smooth_switch_matches_full_size(self):
        for steps in (20, 25, 30):
            with self.subTest(steps=steps):
                self.assertWithin(self.compare("sigma", sdxl_sigmas(steps), 0.6), 0.035, f"sigma, {steps} steps")

    def test_plain_switch_loses_a_little_low_frequency(self):
        """The paper's plain switch: close, but the low frequencies come out several percent weak, at any step count.
        This is what smooth_switch is for; if this ever reads exact, the extra step can go."""
        for steps in (10, 40, 160):
            with self.subTest(steps=steps):
                ratios = self.compare("flow", flow_sigmas(steps), 0.7, smooth=False)
                self.assertWithin(ratios, 0.12, f"plain switch, {steps} steps")
                self.assertLess(ratios[0], 0.97)

    def test_full_size_matches_the_data(self):
        """The baseline itself converges on the data: the oracle is right."""
        off, _ = gains("flow", flow_sigmas(400))
        self.assertWithin(off, 0.01, "full size, 400 steps")

    def test_wrong_rescale_is_caught(self):
        """Negative control: without the paper's rescale the picture comes out far too weak, and the check sees it."""
        good = rapid.grow

        def no_rescale(y, H_, W_, t, seed, kind="flow", noise_scale=1.0):
            X, _ = good(y, H_, W_, t, seed, kind, noise_scale)
            r = math.sqrt((H_ * W_) / float(y.shape[-2] * y.shape[-1]))
            k = r if kind == "sigma" else r / (1.0 + (r - 1.0) * t)
            return X / k, t

        rapid.grow = no_rescale
        try:
            ratios = self.compare("flow", flow_sigmas(20), 0.7)
        finally:
            rapid.grow = good
        self.assertGreater(max(abs(r - 1) for r in ratios), 0.15)


class Plan(unittest.TestCase):
    def test_nearest_step_to_the_switch_point(self):
        sigmas = torch.tensor([1.0, 0.95, 0.88, 0.80, 0.706, 0.60, 0.45, 0.30, 0.12, 0.0])  # a 9-step Turbo run
        self.assertEqual(rapid.plan(sigmas, "flow", 0.7), (4, None))

    def test_last_steps_stay_full(self):
        sigmas = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.0])
        j, _ = rapid.plan(sigmas, "flow", 0.1, min_full_steps=2)
        self.assertLessEqual(j, 2)

    def test_small_share_is_capped(self):
        sigmas = torch.linspace(1, 0, 41)
        j, _ = rapid.plan(sigmas, "flow", 0.02)
        self.assertLessEqual(j, 32)

    def test_hand_over_pass_keeps_one_full_step(self):
        sigmas = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])  # ends above 0: a low pass follows
        self.assertEqual(rapid.plan(sigmas, "flow", 0.55)[0], 4)

    def test_image_to_image_steps_aside(self):
        j, why = rapid.plan(torch.linspace(0.6, 0, 11), "flow", 0.5)
        self.assertIsNone(j)
        self.assertIn("image to image", why)

    def test_sigma_runs_compare_on_the_flow_scale(self):
        sigmas = sdxl_sigmas(25)
        j, _ = rapid.plan(sigmas, "sigma", 0.6)
        t = [float(s) / (1 + float(s)) for s in sigmas]
        self.assertTrue(abs(t[j] - 0.6) <= min(abs(v - 0.6) for v in t[1:-1]) + 1e-9)

    def test_too_few_steps(self):
        self.assertIsNone(rapid.plan(torch.tensor([1.0, 0.0]), "flow", 0.7)[0])


class Spectrum(unittest.TestCase):
    def test_white_noise_stays_white(self):
        x = torch.randn(64, 4, 64, 64)
        small = rapid.shrink(x, 32, 32)
        self.assertAlmostEqual(float(small.std()), 1.0, delta=0.02)

    def test_picture_comes_out_r_times_stronger(self):
        smooth = torch.nn.functional.interpolate(torch.randn(8, 4, 8, 8), size=(64, 64), mode="bicubic", align_corners=False)
        self.assertAlmostEqual(float(rapid.shrink(smooth, 32, 32).std() / smooth.std()), 2.0, delta=0.05)

    def test_round_trip_keeps_low_frequencies(self):
        x = torch.randn(2, 4, 40, 24)
        X, _ = rapid.grow(rapid.shrink(x, 20, 12), 40, 24, 0.0, seed=1)
        r = math.sqrt((40 * 24) / (20 * 12))
        low = rapid.spectrum(X)[..., :20, :12] / (r / (1 + (r - 1) * 0.0))
        self.assertTrue(torch.allclose(low, rapid.spectrum(x)[..., :20, :12], atol=1e-4))

    def test_video_latents(self):
        x = torch.randn(1, 16, 5, 30, 52)  # Wan: batch, channels, frames, height, width
        small = rapid.shrink(x, 16, 26)
        self.assertEqual(tuple(small.shape), (1, 16, 5, 16, 26))
        X, t = rapid.grow(small, 30, 52, 0.7, seed=3)
        self.assertEqual(tuple(X.shape), tuple(x.shape))

    def test_small_size_is_even(self):
        self.assertEqual(rapid.small_size(152, 104, 0.5), (76, 52))
        self.assertEqual(rapid.small_size(150, 98, 0.5), (76, 50))
        self.assertEqual(rapid.small_size(4, 4, 0.5), (2, 2))

    @unittest.skipUnless(torch.backends.mps.is_available(), "no Apple GPU")
    def test_runs_on_apple_gpu(self):
        x = torch.randn(1, 16, 64, 64, device="mps", dtype=torch.float16)
        X, _ = rapid.grow(rapid.shrink(x, 32, 32).to(x.dtype), 64, 64, 0.7, seed=1)
        self.assertEqual(X.device.type, "mps")
        self.assertEqual(X.dtype, torch.float16)
        self.assertTrue(torch.isfinite(X).all())


class StepsAside(Quiet):
    def test_image_to_image_shrinks_the_picture_as_a_picture(self):
        """A start picture at full denoise: the small start holds the same picture at its natural strength."""
        smooth = torch.nn.functional.interpolate(torch.randn(1, 1, 8, 8), size=(H, W), mode="bicubic", align_corners=False)
        model = Oracle("flow")
        model.latent_image = smooth
        small = rapid.shrink_start(model, 0.05 * torch.randn(1, 1, H, W) + 0.95 * smooth, 0.05, H // 2, W // 2, "flow", 1.0)
        plain = rapid.shrink(smooth, H // 2, W // 2) / 2.0
        self.assertLess(float((small - 0.95 * plain).std() / (0.95 * plain).std()), 0.15)

    def test_inpaint_mask_samples_full_size(self):
        model = Oracle("flow")
        sigmas = flow_sigmas(10)
        x = torch.randn(4, 1, H, W)
        model.latent_image = torch.zeros_like(x)
        s = wrapped(0.7)
        s.sampler_function(model, x, sigmas, extra_args={"seed": 1, "denoise_mask": torch.ones_like(x)}, disable=True, **s.extra_options)
        self.assertFalse(self.reports[-1]["active"])
        self.assertIn("inpaint", self.reports[-1]["reason"])
        self.assertEqual(set(model.sizes), {(H, W)})

    def test_unknown_model_kind(self):
        model = Oracle("flow")
        model.use(object())
        s = wrapped(0.7)
        x = torch.randn(2, 1, H, W)
        model.latent_image = torch.zeros_like(x)
        s.sampler_function(model, x, flow_sigmas(8), extra_args={}, disable=True, **s.extra_options)
        self.assertFalse(self.reports[-1]["active"])

    def test_same_seed_same_picture(self):
        a = sample(Oracle("flow"), flow_sigmas(10), wrapped(0.7), seed=5)
        b = sample(Oracle("flow"), flow_sigmas(10), wrapped(0.7), seed=5)
        self.assertTrue(torch.equal(a, b))

    def test_progress_counts_the_whole_run(self):
        model = Oracle("flow")
        seen = []
        s = wrapped(0.7)
        x = torch.randn(2, 1, H, W)
        model.latent_image = torch.zeros_like(x)
        s.sampler_function(model, x, flow_sigmas(10), extra_args={}, callback=lambda d: seen.append(d["i"]), disable=True, **s.extra_options)
        # 10 steps plus the smoothing one, which shows as the step it splits: never past the run's own count.
        self.assertEqual(len(seen), 11)
        self.assertEqual(sorted(set(seen)), list(range(10)))
        self.assertEqual(seen, sorted(seen))

    def test_multistep_and_ancestral_samplers(self):
        for inner in ("res_multistep", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "lcm"):
            with self.subTest(inner):
                fn = getattr(kds, f"sample_{inner}", None)
                if fn is None:
                    continue
                out = sample(Oracle("flow"), flow_sigmas(12), wrapped(0.7, inner=inner))
                self.assertEqual(tuple(out.shape), (BATCH, 1, H, W))
                self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
