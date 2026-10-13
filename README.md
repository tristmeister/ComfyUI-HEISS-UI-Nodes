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

## HEISS RTX Video and HEISS RTX Upscale

**NVIDIA RTX Video Super Resolution and Frame Generation: videos upscaled in seconds, and smoother.** NVIDIA RTX cards only (20-series and newer; smoothing needs a 40-series or newer), on Windows or Linux.

Both run NVIDIA's own models through its `nvidia-vfx` package. Upscaling sharpens and cleans what is already in a frame: it doesn't invent new detail the way SeedVR2 does, which is why it takes seconds, not minutes.

- **HEISS RTX Video** (`VIDEO` → a saved MP4 with its sound): reads a video one frame at a time, optionally cleans it (deblur or denoise), optionally adds frames in between (`smooth` 2 turns 16 fps into 32 at the same length), upscales each frame and writes it straight to the encoder. Memory stays flat however long the video or large the output. Load Video's file is read straight from disk. It encodes with H.264 NVENC when ComfyUI's FFmpeg has it and the size fits (4096 per side), otherwise with libx264, without failing either way.
- **HEISS RTX Upscale** (`IMAGE` → `IMAGE`): pictures and short frame batches.

| Input | Default | What it does |
|---|---|---|
| `scale` | 2.0 | Output size per side, 1 to 4, rounded to a multiple of 8. |
| `quality` | `HIGHBITRATE_HIGH` (video), `ULTRA` (pictures) | `HIGHBITRATE_*` for clean sources such as fresh frames, `HIGH`/`ULTRA` for compressed files, `DENOISE_*`/`DEBLUR_*` keep the size. `ULTRA` costs much more VRAM for little on most video. |
| `cleanup` | none | A deblur or denoise pass at the source size first. Try `DEBLUR_MEDIUM` when the upscale alone looks like it did little. |
| `smooth` | 1 | Frame rate multiplier from RTX Frame Generation (2–4). |

`GET /heiss/rtx/status` says whether this machine can run them and, if not, why: the card, ComfyUI's PyTorch and CUDA, and whether NVIDIA's package is installed. With `?probe=1` it also runs a tiny upscale to prove it works with this driver.

**Install:** NVIDIA's package isn't in `requirements.txt`, so this pack still installs on a Mac or an AMD card. On an NVIDIA machine, with ComfyUI's own Python (about 0.5 GB, models included):

```
python -m pip install -r requirements-rtx.txt --extra-index-url https://pypi.nvidia.com
```

NVIDIA asks for driver 570 or newer. Comfy-Org's own nodes say PyTorch built for CUDA 13 (cu130) is required; nothing in these nodes depends on it, and the status check above reports what actually runs.

## For HEISS UI: the pack's version

`GET /heiss/nodes` returns `{ version, commit, nodes }`. `version` (`PACK_VERSION` in `heiss_info.py`) goes up by one whenever HEISS needs something new from the pack. HEISS asks for the version its features need and offers the update when the loaded copy is older.

### Credits

- **SPEED**: "Spectral Progressive Diffusion for Efficient Image and Video Generation", Howard Xiao, Brian Chao, Lior Yariv and Gordon Wetzstein, 2026 ([paper](https://arxiv.org/abs/2605.18736), [code](https://github.com/howardhx/speed), MIT).
- **[LC Speed Boost](https://github.com/lonecatone23/ComfyUI_LC123_nodes)** by lonecatone23 (MIT): the on-GPU sampler wrapper this node is adapted from, and the SDXL-family support.
- **[Comfy-Org/Nvidia_RTX_Nodes_ComfyUI](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI)** (Apache-2.0): how HEISS RTX drives NVIDIA's effects (frame layout, DLPack copies, batch sizes, the alpha channel).
- **[tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video](https://github.com/tritant/ComfyUI_NVIDIA_RTX_VSR_Long_Video)** (Apache-2.0): streaming a video through VSR with flat memory, reading Load Video's file directly, the cleanup pass and keeping the audio.
- **[ComfyUI-SPEED, SwarmNeo fork](https://github.com/aoleg/ComfyUI-SPEED)** by aoleg (MIT), after [ruwwww/ComfyUI-SPEED](https://github.com/ruwwww/ComfyUI-SPEED): the measurements showing that the quality limit is a noise level, and the cap on the small share.

## HEISS Inpaint Stitch

**Inpainted crops that go back in without a visible seam, even from edit models.**

Inpainting crops the painted area with some context around it, the model repaints the crop, and the crop goes back onto the picture through a soft mask. Edit models (Qwen-Image Edit, Flux.2 / Klein with a reference) re-render the whole crop, and they often shift or zoom it by a few pixels and make it a shade brighter, darker or warmer. Pasted back as it is, lines that cross the mask edge double up and the painted area looks a shade off.

HEISS Inpaint Stitch measures both on the ring around the mask, where the result should still show the original, and corrects them before blending:

- **Align** fits a small shift, zoom and rotation (an affine fit, with a gain and a bias so a colour drift doesn't pull it). It keeps the fit only when it's plausible (scale within 8%, shift under 5% of the box, little shear) and it lowers the ring's difference by 15% or more. A flat ring, with nothing to line up on, is left alone.
- **Color match** moves each channel's mean and spread onto the original's, measured on the same ring. A drift the size of a model's (means within 0.1, spread within 0.75–1.33) is corrected; a larger difference is taken as meant and left alone.

Models that sample only inside a latent mask (SDXL, Flux.1, Z-Image and similar) keep the ring as it was. For them both steps find nothing to correct, and the node is a plain composite.

### Use

`VAE Decode → HEISS Inpaint Stitch → Save Image`

| Input | What it is |
|---|---|
| `original` | The whole picture the crop was cut from. |
| `image` | The inpainted crop, any size. It's scaled (Lanczos) to the mask's size. A batch gives a batch. |
| `mask` | Where the crop goes over the original, at the box's own size, soft edge included. |
| `x`, `y` | Where the box sits in the original. |
| `align`, `color_match` | Both on by default. Off, the node is a plain composite. |

Everything outside the mask stays the original's exact pixels. What it did goes to the console, and to the client that queued the run as a `heiss.inpaint` message.

### Credits

The idea of lining up and colour matching on the ring comes from **[Inpaint Canvas](https://github.com/DenRakEiw/ComfyUI-InpaintCanvas)** by DenRakEiw (`_align_patch`, `_color_match`). This is a separate implementation in torch, with no OpenCV.

MIT licensed; see LICENSE. NVIDIA's `nvidia-vfx` package and its models are NVIDIA's, under NVIDIA's terms.
