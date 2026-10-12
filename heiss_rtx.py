"""
HEISS RTX: NVIDIA's RTX Video Super Resolution and Frame Generation, for pictures and for videos of any length.

Both run NVIDIA's own models through its `nvidia-vfx` package (the Video Effects SDK), on NVIDIA RTX cards only.
Upscaling sharpens and cleans what is already in a frame; it does not invent new detail the way SeedVR2 does, which is
why it can do in seconds what SeedVR2 needs minutes for.

- HEISS RTX Upscale (IMAGE -> IMAGE): pictures and short frame batches, after the official Comfy-Org node
  (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI, Apache-2.0).
- HEISS RTX Video (VIDEO -> saved MP4): a whole video, one frame at a time. Frames are read, made smoother (Frame
  Generation, optional), upscaled and written straight to the encoder, so memory stays flat however long the video
  or large the output, as in tritant's ComfyUI_NVIDIA_RTX_VSR_Long_Video (Apache-2.0). A 4x upscale of a 5 s clip
  kept in memory, as an IMAGE batch, would need about 6 GB of RAM; streamed it needs one frame. Its audio is kept.

Nothing NVIDIA is imported until a node runs: this pack also carries Rapid, which runs on every machine, and a Mac
or an AMD card must still load it. GET /heiss/rtx/status says whether this machine can run these nodes, and why not.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager, nullcontext
from fractions import Fraction

import torch

UPSCALE_QUALITIES = (
    "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_LOW",
    "HIGH", "ULTRA", "MEDIUM", "LOW", "STREAMING_MEDIUM", "STREAMING_ULTRA", "BICUBIC",
)
CLEANUP_QUALITIES = (
    "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
    "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA",
)
QUALITIES = UPSCALE_QUALITIES + CLEANUP_QUALITIES
SMOOTH_MODES = ("LOW", "MEDIUM", "HIGH")
MAX_SIDE = 8192
NVENC_MAX_SIDE = 4096  # H.264 NVENC's limit per side; above it libx264 encodes
FRAME_GENERATION_CAPABILITY = (8, 9)  # Ada (RTX 40) and newer
NVIDIA_INDEX = "https://pypi.nvidia.com"
IMAGE_CHUNK_PIXELS = 16 * 1024 * 1024  # output pixels moved to the GPU at once by HEISS RTX Upscale


class RTXUnavailable(RuntimeError):
    pass


def nvvfx_module():
    """NVIDIA's package, or an error that says what to install and where."""
    try:
        import nvvfx
    except ImportError as error:
        raise RTXUnavailable(
            "NVIDIA's RTX video package (nvidia-vfx) isn't installed in ComfyUI's Python. Install it with ComfyUI's "
            f"own Python: python -m pip install nvidia-vfx --extra-index-url {NVIDIA_INDEX}, then run this again. "
            f"({error})") from error
    return nvvfx


def effect_device():
    """ComfyUI's CUDA device and its index, or a plain reason this machine can't run the RTX effects."""
    import comfy.model_management

    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise RTXUnavailable("RTX upscaling runs only on an NVIDIA RTX card, and ComfyUI isn't using one.")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device, index


def _enum(nvvfx, owner, name):
    """nvidia-vfx 0.2 exports its enums at the top; 0.1 kept them on the effect class."""
    return getattr(nvvfx, name, None) or getattr(owner, name)


def output_size(width, height, scale, quality):
    """What VSR makes of a width x height frame: cleanup modes keep the size, upscales round to multiples of 8."""
    if quality in CLEANUP_QUALITIES:
        return width, height
    w = max(8, round(width * scale / 8) * 8)
    h = max(8, round(height * scale / 8) * 8)
    if max(w, h) > MAX_SIDE:
        raise ValueError(f"{w}x{h} is larger than the {MAX_SIDE} pixels per side RTX upscaling can make. Pick a smaller scale.")
    return w, h


def _load_vsr(nvvfx, device_index, quality, strength, in_w, in_h, out_w, out_h):
    cls = nvvfx.VideoSuperRes
    effect = cls(quality=_enum(nvvfx, cls, "QualityLevel")[quality], strength=float(strength), device=device_index,
                 image_encoding=_enum(nvvfx, cls, "ImageEncoding").RGB8)
    for name, value in (("input_width", in_w), ("input_height", in_h)):
        try:
            setattr(effect, name, value)  # 0.1 wants them; 0.2 reads them from each frame
        except AttributeError:
            pass
    effect.output_width = out_w
    effect.output_height = out_h
    effect.load()
    return effect


def _load_frame_generation(nvvfx, device_index, mode, multiplier, width, height):
    cls = nvvfx.VideoFrameGeneration
    try:
        effect = cls(mode=cls.Mode[mode], automatic_shot_change_detection_enabled=True, device=device_index,
                     image_encoding=_enum(nvvfx, cls, "ImageEncoding").RGB8)
        effect.input_width = width
        effect.input_height = height
        effect.frame_multiplier = multiplier
        effect.load()
    except Exception as error:
        raise RTXUnavailable(f"Smoothing (RTX Frame Generation) needs an RTX 40-series card or newer. ({error})") from error
    return effect


def _close(effect):
    try:
        if hasattr(effect, "close"):
            effect.close()
        elif hasattr(effect, "__exit__"):
            effect.__exit__(None, None, None)
    except Exception:
        pass


def _copy_out(result):
    """The SDK's output lives in its own memory and is overwritten by the next run: copy it at once."""
    return torch.from_dlpack(getattr(result, "image", result)).clone()


@contextmanager
def frame_effects(width, height, *, scale, quality, strength=1.0, cleanup="none", cleanup_strength=1.0,
                  smooth=1, smooth_mode="MEDIUM"):
    """
    Loads what a run needs, in the order frames pass through it: cleanup at the source size, then Frame Generation
    at the source size (cheaper there than after upscaling), then VSR. Yields (device, clean, process, interpolate,
    out_size): clean(chw) -> chw, process(chw) -> chw at out_size, interpolate(prev, cur) -> the smooth - 1 frames
    between them (None when smooth is 1). Frames are (3, H, W) floats in 0..1 on the device.
    """
    nvvfx = nvvfx_module()
    device, index = effect_device()
    out_w, out_h = output_size(width, height, scale, quality)
    with ExitStack() as stack, (torch.cuda.device(device) if device.type == "cuda" else nullcontext()):
        def keep(effect):
            stack.callback(_close, effect)
            return effect

        pre = keep(_load_vsr(nvvfx, index, cleanup, cleanup_strength, width, height, width, height)) if cleanup != "none" else None
        vsr = keep(_load_vsr(nvvfx, index, quality, strength, width, height, out_w, out_h))
        fg = keep(_load_frame_generation(nvvfx, index, smooth_mode, smooth, width, height)) if smooth > 1 else None

        def clean(chw):
            return _copy_out(pre.run(chw)) if pre is not None else chw

        def process(chw):
            return _copy_out(vsr.run(chw))

        def interpolate(prev, cur):
            return [_copy_out(fg.run(prev, cur, frame_index=i, shot_change=False)) for i in range(1, smooth)]

        yield device, clean, process, (interpolate if fg is not None else None), (out_w, out_h)


def stream_frames(frames, *, clean, process, interpolate, emit, check=None):
    """
    The pipeline, frame by frame: each source frame is cleaned, the frames between it and the one before are
    generated, and every frame goes through `process` to `emit` in display order. Only the previous source frame is
    held. Returns how many frames were emitted.
    """
    previous = None
    count = 0
    for frame in frames:
        if check is not None:
            check()
        current = clean(frame)
        if previous is not None and interpolate is not None:
            for between in interpolate(previous, current):
                emit(process(between))
                count += 1
        emit(process(current))
        count += 1
        previous = current
    return count


# --- reading and writing video -----------------------------------------------------------------------------------

def _file_source(video):
    """The file behind a native, untrimmed, uncropped Load Video, so it can be read frame by frame; else None."""
    if type(video).__name__ != "VideoFromFile":
        return None
    try:
        source = video.get_stream_source()
        trim = getattr(video, "get_active_trim_window", None)
        if callable(trim) and any(float(value) > 0 for value in trim()):
            return None
        import av
        with av.open(source if isinstance(source, str) else source, mode="r") as container:
            stream = container.streams.video[0]
            coded = (stream.codec_context.width, stream.codec_context.height)
        if tuple(video.get_dimensions()) != coded:
            return None  # cropped
        if not isinstance(source, str):
            source.seek(0)
        return source
    except Exception:
        return None


class VideoSource:
    """A VIDEO's frames as uint8 (H, W, 3) tensors, its frame rate, size, frame count and audio, without holding
    more than one decoded frame of a file in memory."""

    def __init__(self, video):
        self.file = _file_source(video)
        self.components = None
        if self.file is None:
            self.components = video.get_components()
            images = self.components.images
            self.count = int(images.shape[0])
            self.height, self.width = int(images.shape[1]), int(images.shape[2])
            self.fps = Fraction(self.components.frame_rate).limit_denominator(100000)
        else:
            import av
            with av.open(self.file, mode="r") as container:
                stream = container.streams.video[0]
                self.width, self.height = stream.codec_context.width, stream.codec_context.height
                self.fps = Fraction(stream.average_rate or stream.guessed_rate or 24).limit_denominator(100000)
                self.count = int(stream.frames or 0) or int(round(float(container.duration or 0) / 1e6 * float(self.fps)))
            if not isinstance(self.file, str):
                self.file.seek(0)

    def frames(self):
        if self.components is not None:
            for image in self.components.images:
                yield (image[..., :3].clamp(0, 1) * 255).round().to(torch.uint8)
            return
        import av
        with av.open(self.file, mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                yield torch.from_numpy(frame.to_ndarray(format="rgb24"))

    def audio(self):
        """('waveform', dict) from components, ('file', source) when the file has an audio track, else None."""
        if self.components is not None:
            audio = self.components.audio
            return ("waveform", audio) if audio else None
        import av
        with av.open(self.file, mode="r") as container:
            has = bool(container.streams.audio)
        if not isinstance(self.file, str):
            self.file.seek(0)
        return ("file", self.file) if has else None


def _pick_encoder(width, height, prefer_nvenc=True):
    """H.264 NVENC when this FFmpeg has it and the size fits, else libx264. Tried on one frame first, so a missing
    or busy NVENC falls back here rather than failing a long run halfway."""
    import io
    import av

    if prefer_nvenc and max(width, height) <= NVENC_MAX_SIDE and "h264_nvenc" in av.codecs_available:
        try:
            with av.open(io.BytesIO(), mode="w", format="mp4") as probe:
                stream = probe.add_stream("h264_nvenc", rate=24)
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                frame = av.VideoFrame(width, height, "yuv420p")
                for packet in stream.encode(frame):
                    probe.mux(packet)
                for packet in stream.encode(None):
                    probe.mux(packet)
            return "h264_nvenc", {"preset": "p5", "rc": "vbr", "cq": "18", "b": "0"}
        except Exception:
            pass
    return "libx264", {"crf": "17", "preset": "medium"}


def _color(target):
    """BT.709, limited range: what ComfyUI's own Save Video writes."""
    try:
        from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
        target.color_primaries = ColorPrimaries.BT709
        target.color_trc = ColorTrc.BT709
        target.colorspace = 1
        target.color_range = ColorRange.MPEG
    except Exception:
        pass


class Mp4Writer:
    """Frames in, an H.264 MP4 out, with the source's audio re-encoded as AAC (MP4 can't hold every codec). Every
    stream is added before the first frame: once the encoder has written packets the header is fixed."""

    def __init__(self, path, width, height, fps, *, audio=None, metadata=None, prefer_nvenc=True):
        import av

        self.path = path
        self.container = av.open(path, mode="w", format="mp4", options={"movflags": "+faststart"})
        for key, value in (metadata or {}).items():
            self.container.metadata[key] = json.dumps(value)
        self.codec, options = _pick_encoder(width, height, prefer_nvenc)
        self.stream = self.container.add_stream(self.codec, rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"
        self.stream.options = options
        _color(self.stream.codec_context)
        self.fps = fps
        self.frames = 0
        self.audio = audio
        self.audio_stream = None
        if audio is not None:
            rate, layout = self._audio_format(audio)
            self.audio_stream = self.container.add_stream("aac", rate=rate, layout=layout)
            self.audio_rate, self.audio_layout = rate, layout

    @staticmethod
    def _audio_format(audio):
        import av

        kind, value = audio
        if kind == "waveform":
            channels = int(value["waveform"].shape[1])
            return int(value["sample_rate"]), {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
        with av.open(value, mode="r") as source:
            context = source.streams.audio[0].codec_context
            layout = context.layout.name if context.layout and context.layout.name in ("mono", "stereo", "5.1") else "stereo"
            rate = context.sample_rate or 48000
        if not isinstance(value, str):
            value.seek(0)
        return rate, layout

    def write(self, rgb):
        import av

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p", dst_colorspace=1)
        _color(frame)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def _write_audio(self):
        """The sound, cut to the video's length (a smoothed video lasts as long as its source)."""
        import av
        import math

        if self.audio_stream is None:
            return
        stream, rate, layout = self.audio_stream, self.audio_rate, self.audio_layout
        limit = int(math.ceil(rate * self.frames / float(self.fps)))
        kind, value = self.audio
        if kind == "waveform":
            waveform = value["waveform"][0, :, :limit]
            frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
            frame.sample_rate, frame.pts = rate, 0
            for packet in stream.encode(frame):
                self.container.mux(packet)
        else:
            with av.open(value, mode="r") as source:
                resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=rate)
                written = 0
                for decoded in source.decode(source.streams.audio[0]):
                    for frame in resampler.resample(decoded):
                        if written >= limit:
                            break
                        frame.pts = written
                        written += frame.samples
                        for packet in stream.encode(frame):
                            self.container.mux(packet)
        for packet in stream.encode(None):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self._write_audio()
        self.container.close()

    def abort(self):
        try:
            self.container.close()
        except Exception:
            pass


# --- nodes -------------------------------------------------------------------------------------------------------

def _chw(frame_u8, device):
    return frame_u8.to(device=device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).contiguous()


def _hwc_u8(chw):
    return chw.clamp(0, 1).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


class HeissRTXVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Load Video's VIDEO, or any other: a generated video from Create Video too."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "Output size per side. Rounded to a multiple of 8."}),
                "quality": (QUALITIES, {"default": "HIGHBITRATE_HIGH",
                                        "tooltip": "HIGHBITRATE for clean sources such as freshly generated frames; HIGH/ULTRA for "
                                                   "compressed files; DENOISE/DEBLUR keep the size. ULTRA costs much more VRAM for little."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cleanup": (("none",) + CLEANUP_QUALITIES, {"default": "none",
                                                           "tooltip": "An optional first pass at the source size. Deblur helps soft "
                                                                      "AI video when the upscale alone looks like it did little."}),
                "smooth": ("INT", {"default": 1, "min": 1, "max": 4,
                                   "tooltip": "Frame rate multiplier from RTX Frame Generation: 2 turns 16 fps into 32. "
                                              "1 is off. Needs an RTX 40-series card or newer."}),
                "filename_prefix": ("STRING", {"default": "heiss-ui/rtx"}),
            },
            "optional": {
                "smooth_mode": (SMOOTH_MODES, {"default": "MEDIUM"}),
                "cleanup_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "HEISS UI/video"
    DESCRIPTION = ("Upscales a video with NVIDIA RTX Video Super Resolution and, optionally, makes it smoother with RTX "
                   "Frame Generation, one frame at a time, and saves it as an MP4 with its sound. NVIDIA RTX cards only.")

    def run(self, video, scale, quality, strength, cleanup, smooth, filename_prefix,
            smooth_mode="MEDIUM", cleanup_strength=1.0, prompt=None, extra_pnginfo=None):
        import comfy.model_management
        import comfy.utils
        import folder_paths

        started = time.monotonic()
        source = VideoSource(video)
        fps = source.fps * smooth
        expected = max(1, (source.count - 1) * smooth + 1) if source.count else 0
        out_w, out_h = output_size(source.width, source.height, scale, quality)
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), out_w, out_h)
        name = f"{filename}_{counter:05}_.mp4"
        final = os.path.join(folder, name)
        partial = final + ".partial.mp4"
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt
        if extra_pnginfo:
            metadata.update(extra_pnginfo)
        progress = comfy.utils.ProgressBar(expected) if expected else None

        writer = None
        try:
            with frame_effects(source.width, source.height, scale=scale, quality=quality, strength=strength,
                               cleanup=cleanup, cleanup_strength=cleanup_strength, smooth=smooth,
                               smooth_mode=smooth_mode) as (device, clean, process, interpolate, size):
                writer = Mp4Writer(partial, size[0], size[1], fps, audio=source.audio(), metadata=metadata)

                def emit(chw):
                    writer.write(_hwc_u8(chw))
                    if progress is not None:
                        progress.update(1)

                count = stream_frames((_chw(frame, device) for frame in source.frames()),
                                      clean=clean, process=process, interpolate=interpolate, emit=emit,
                                      check=comfy.model_management.throw_exception_if_processing_interrupted)
            if not count:
                raise RuntimeError("The video has no frames to upscale.")
            writer.close()
            os.replace(partial, final)
        except BaseException:
            if writer is not None:
                writer.abort()
            try:
                os.remove(partial)
            except OSError:
                pass
            raise

        report = {"frames": count, "width": out_w, "height": out_h, "fps": float(fps), "smooth": smooth,
                  "quality": quality, "cleanup": cleanup, "encoder": writer.codec,
                  "seconds": round(time.monotonic() - started, 2),
                  "source": {"width": source.width, "height": source.height, "fps": float(source.fps), "frames": source.count}}
        print(f"[HEISS RTX] {source.width}x{source.height} -> {out_w}x{out_h}, {count} frames at {float(fps):.3g} fps, "
              f"{writer.codec}, {report['seconds']} s")
        return {"ui": {"images": [{"filename": name, "subfolder": subfolder, "type": "output"}], "animated": (True,),
                       "heiss_rtx": [report]}}


class HeissRTXUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
            "quality": (QUALITIES, {"default": "ULTRA",
                                    "tooltip": "ULTRA for single pictures; HIGHBITRATE for clean frames; DENOISE/DEBLUR keep the size."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "HEISS UI/image"
    DESCRIPTION = "Upscales pictures with NVIDIA RTX Video Super Resolution, in about a second. NVIDIA RTX cards only."

    def upscale(self, images, scale, quality, strength):
        import torch.nn.functional as F

        alpha = images[..., 3:4] if images.shape[-1] > 3 else None
        rgb = images[..., :3]
        b, h, w, _ = rgb.shape
        with frame_effects(w, h, scale=scale, quality=quality, strength=strength) as (device, _clean, process, _i, size):
            out_w, out_h = size
            output = torch.empty((b, out_h, out_w, 3), dtype=rgb.dtype)
            chunk = max(1, IMAGE_CHUNK_PIXELS // (out_w * out_h))
            for start in range(0, b, chunk):
                batch = rgb[start:start + chunk].to(device=device, dtype=torch.float32).movedim(-1, 1).contiguous()
                for offset, frame in enumerate(batch):
                    output[start + offset].copy_(process(frame).clamp(0, 1).movedim(0, -1).cpu())
        if alpha is not None:
            alpha = F.interpolate(alpha.movedim(-1, 1), size=(out_h, out_w), mode="bilinear", align_corners=False).movedim(1, -1)
            output = torch.cat((output, alpha), dim=-1)
        return (output,)


# --- status ------------------------------------------------------------------------------------------------------

_probe_cache = {}


def rtx_status(probe=False):
    """What HEISS asks before it offers RTX upscaling: the card, ComfyUI's PyTorch, NVIDIA's package and, with
    probe, whether a small upscale really loads and runs here. Every field is a fact from this machine."""
    status = {"torch": torch.__version__, "cuda": torch.version.cuda, "nvvfx": None, "device": None,
              "frameGeneration": False, "ready": False, "reason": ""}
    try:
        import importlib.metadata
        status["nvvfx"] = importlib.metadata.version("nvidia-vfx")
    except Exception:
        status["nvvfx"] = None
    try:
        device, index = effect_device()
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        status["device"] = {"name": props.name, "capability": list(capability), "vram": props.total_memory}
        status["frameGeneration"] = capability >= FRAME_GENERATION_CAPABILITY
        if capability < (7, 5):
            status["reason"] = "RTX upscaling needs an RTX card (20-series or newer); this card is older."
            return status
    except Exception as error:
        status["reason"] = str(error)
        return status
    if not status["nvvfx"]:
        status["reason"] = "NVIDIA's RTX video package (nvidia-vfx) isn't installed yet."
        return status
    if not probe:
        status["ready"] = True
        return status
    key = (status["nvvfx"], index)
    if key not in _probe_cache:
        try:
            with frame_effects(64, 64, scale=2.0, quality="HIGH") as (dev, _clean, process, _i, _size):
                process(torch.zeros(3, 64, 64, device=dev))
                torch.cuda.synchronize(dev)
            _probe_cache[key] = ""
        except Exception as error:
            _probe_cache[key] = f"NVIDIA's RTX package is installed but didn't run: {error}. Updating the NVIDIA driver usually fixes this."
    status["reason"] = _probe_cache[key]
    status["ready"] = not _probe_cache[key]
    status["probed"] = True
    return status


def _register_route():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # loaded outside ComfyUI (tests)

    @PromptServer.instance.routes.get("/heiss/rtx/status")
    async def _status(request):
        probe = request.rel_url.query.get("probe") == "1"
        try:
            return web.json_response(rtx_status(probe=probe))
        except Exception as error:
            return web.json_response({"ready": False, "reason": str(error)})


_register_route()

NODE_CLASS_MAPPINGS = {"HeissRTXVideo": HeissRTXVideo, "HeissRTXUpscale": HeissRTXUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"HeissRTXVideo": "HEISS RTX Video", "HeissRTXUpscale": "HEISS RTX Upscale"}
