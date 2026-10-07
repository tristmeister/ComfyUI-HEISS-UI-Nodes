# HEISS UI Nodes

These are the ComfyUI custom nodes that [HEISS UI](https://github.com/tristmeister/HEISS-UI) builds on. HEISS installs them for you when a feature needs them. They also work on their own in any ComfyUI workflow.

## HEISS Rapid

**About 1.4× to 2× faster pictures at the same VRAM, with fine detail still drawn at full size.**

In the first part of a run the picture is still mostly noise. Those steps only settle layout, pose, light and colour, and none of that needs the full grid. Rapid does them on a latent half as wide and half as tall. Before the fine detail starts, it grows the latent back to full size in its cosine spectrum and fills the new high frequencies with fresh noise at the current noise level. The step count stays the same and the early steps cost about a quarter as much. A given seed frames a little differently with Rapid on than off.

### Use

`KSamplerSelect → HEISS Rapid → SamplerCustomAdvanced`

For a two-pass (high/low) setup, put it on the first pass only.

| Input | Default | What it does |
|---|---|---|
| `switch_at` | 0.7 | Noise level where the picture grows to full size (1 = pure noise, 0 = finished). SD-family models use the same scale, σ/(1+σ). A lower value is faster; a higher value keeps small detail such as text safer. |
| `scale` | 0.5 | Size of the start, per side. |
| `min_full_steps` | 2 | Steps at the end that always run at full size. |
| `smooth_switch` | on | One extra full-size step right after the switch (see below). |

**Models:**
- Works on flow models: Krea 2, Z-Image, Flux.1, Flux.2 / Klein, Qwen-Image, Chroma, Wan and similar.
- Works on SD-family models: SDXL, Pony, Illustrious, and their v-prediction and EDM variants.
- Turns itself off, and says why in the console, for:
  - image-to-image and upscale passes (starting below noise 0.9)
  - inpaint masks
  - models it can't handle
- Runs on whatever device ComfyUI uses (CUDA, Apple MPS, CPU).
- Image and video latents.

### The smooth switch

At the switch, the paper realigns the noise level upwards (for a half-size start at 0.70, to 0.82). The original code, and the other ComfyUI ports, then take one long step from there straight to the schedule's next level. With an exact denoiser, that one step leaves the lowest frequencies, the picture's broad contrast and colour, about 6–9% weaker than a run without Rapid, at any step count. `smooth_switch` adds one full-size step in the middle of that jump, which brings them back to within about 2%. Turn it off for the most speed on few-step models.

### Tests

The central test samples Gaussian data with its exact denoiser. Each frequency's outcome is then known in closed form, so Rapid can be checked against the same run at full size, frequency by frequency. A negative control checks that a wrong rescale is caught.

```
COMFYUI_PATH=/path/to/ComfyUI /path/to/ComfyUI/python -m unittest discover -s tests -v
```

## HEISS Rapid Guidance

**The late detail steps at half the cost on models that use CFG.**

With CFG above 1, every step runs the model twice: once with the prompt and once without. The steps that decide what the picture shows need that push. The late steps only refine detail that's already there, and do nearly as well without it. HEISS Rapid Guidance keeps CFG while the noise level is at or above `cfg_until` (default 0.3, on the same 0..1 scale as Rapid). Below that it samples at CFG 1, where ComfyUI skips the second pass.

Use it in place of `CFGGuider`:

`HEISS Rapid Guidance → SamplerCustomAdvanced`

It stacks with HEISS Rapid: Rapid makes the early steps cheap, and this makes the late ones cheaper. At CFG 1 (Turbo and Lightning models) there's nothing to save, and it behaves like the plain guider.

### Credits

- **SPEED**: "Spectral Progressive Diffusion for Efficient Image and Video Generation", Howard Xiao, Brian Chao, Lior Yariv and Gordon Wetzstein, 2026 ([paper](https://arxiv.org/abs/2605.18736), [code](https://github.com/howardhx/speed), MIT).
- **[LC Speed Boost](https://github.com/lonecatone23/ComfyUI_LC123_nodes)** by lonecatone23 (MIT): the on-GPU sampler wrapper this node is adapted from, and the SDXL-family support.
- **[ComfyUI-SPEED, SwarmNeo fork](https://github.com/aoleg/ComfyUI-SPEED)** by aoleg (MIT), after [ruwwww/ComfyUI-SPEED](https://github.com/ruwwww/ComfyUI-SPEED): the measurements showing that the quality limit is a noise level, and the cap on the small share.

MIT licensed; see LICENSE.
