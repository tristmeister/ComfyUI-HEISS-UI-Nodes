"""HEISS RTX Video and Upscale: the pipeline, the files it writes and its guards, with a stand-in for NVIDIA's
package (nvvfx) so they run on any machine. Needs torch and av (both come with ComfyUI). Run like test_rapid.py."""

import enum
import math
import os
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = tempfile.mkdtemp(prefix="heiss-rtx-test-")
INTERRUPT = {"after": None, "seen": 0}


def _fake_comfy():
    """ComfyUI's own modules when COMFYUI_PATH has them (only the stop check is stood in for), stand-ins otherwise.
    Saving always goes to a temporary folder."""
    def interrupted():
        INTERRUPT["seen"] += 1
        if INTERRUPT["after"] is not None and INTERRUPT["seen"] > INTERRUPT["after"]:
            raise KeyboardInterrupt("stopped")

    try:
        import comfy.model_management as mm
        import comfy.utils  # noqa: F401
    except Exception:
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: torch.device("cpu")
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = lambda total: SimpleNamespace(update=lambda n: None)
        comfy.model_management, comfy.utils = mm, utils
        sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils})
    mm.throw_exception_if_processing_interrupted = interrupted
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: OUT

    def save_path(prefix, out, w, h):
        folder = os.path.join(out, os.path.dirname(prefix))
        os.makedirs(folder, exist_ok=True)
        count = len([n for n in os.listdir(folder) if n.endswith(".mp4")]) + 1
        return folder, os.path.basename(prefix), count, os.path.dirname(prefix), prefix
    fp.get_save_image_path = save_path
    sys.modules["folder_paths"] = fp


def _fake_nvvfx():
    """Enough of nvidia-vfx 0.2's surface: VSR resizes (nearest), Frame Generation blends the pair."""
    nv = types.ModuleType("nvvfx")
    import heiss_rtx
    nv.QualityLevel = enum.IntEnum("QualityLevel", list(heiss_rtx.QUALITIES))
    nv.ImageEncoding = enum.IntEnum("ImageEncoding", ["RGB8", "RGB10A2"])
    nv.loaded = []

    class VideoSuperRes:
        def __init__(self, quality, strength, device, image_encoding):
            self.quality = quality
            nv.loaded.append(("vsr", quality.name))

        def load(self):
            pass

        def run(self, chw):
            out = torch.nn.functional.interpolate(chw[None], size=(self.output_height, self.output_width), mode="nearest")[0]
            return SimpleNamespace(image=out)

        def close(self):
            nv.loaded.append(("closed", self.quality.name))

    class VideoFrameGeneration:
        Mode = enum.IntEnum("Mode", ["LOW", "MEDIUM", "HIGH"])

        def __init__(self, mode, automatic_shot_change_detection_enabled, device, image_encoding):
            nv.loaded.append(("fg", mode.name))

        def load(self):
            pass

        def run(self, prev, cur, frame_index, shot_change):
            t = frame_index / self.frame_multiplier
            return SimpleNamespace(image=prev * (1 - t) + cur * t)

        def close(self):
            pass

    nv.VideoSuperRes, nv.VideoFrameGeneration = VideoSuperRes, VideoFrameGeneration
    sys.modules["nvvfx"] = nv
    return nv


_fake_comfy()
import heiss_rtx  # noqa: E402


class Components(SimpleNamespace):
    pass


class GeneratedVideo:
    """Like ComfyUI's VideoFromComponents: frames already in memory (Create Video)."""

    def __init__(self, frames, fps=16, audio=None):
        self.components = Components(images=frames, audio=audio, frame_rate=Fraction(fps), metadata=None, alpha=None)

    def get_components(self):
        return self.components


class VideoFromFile:
    """Named like ComfyUI's, so the node streams it from disk; get_components must then never be called."""

    def __init__(self, path, size):
        self.path, self.size = path, size

    def get_stream_source(self):
        return self.path

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_dimensions(self):
        return self.size

    def get_components(self):
        raise AssertionError("a file must be streamed, not loaded whole")


def ramp(count, h=48, w=64):
    """Frames whose brightness tells their index apart."""
    return torch.stack([torch.full((h, w, 3), (i + 1) / (count + 1)) for i in range(count)])


def write_source(path, count=12, w=64, h=48, fps=12, audio=True):
    with av.open(path, mode="w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        astream = out.add_stream("aac", rate=44100, layout="stereo") if audio else None
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((h, w, 3), 20 * i % 255, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
        if audio:
            samples = int(44100 * count / fps)
            tone = np.sin(np.linspace(0, 440 * 2 * math.pi * count / fps, samples, dtype=np.float32))
            frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = 44100, 0
            for packet in astream.encode(frame):
                out.mux(packet)
            for packet in astream.encode(None):
                out.mux(packet)


def probe(path):
    with av.open(path) as container:
        video = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video))
    with av.open(path) as container:
        return {"frames": frames, "width": video.codec_context.width, "height": video.codec_context.height,
                "fps": float(container.streams.video[0].average_rate), "audio": bool(container.streams.audio),
                "seconds": float(container.duration or 0) / 1e6}


def saved(result):
    item = result["ui"]["images"][0]
    return os.path.join(OUT, item["subfolder"], item["filename"])


class Loading(unittest.TestCase):
    def test_the_pack_loads_without_nvidia(self):
        # Rapid lives in the same pack and must load on a Mac or an AMD card: nothing NVIDIA at import.
        sys.modules.pop("nvvfx", None)
        import importlib
        importlib.reload(heiss_rtx)
        self.assertNotIn("nvvfx", sys.modules)

    def test_missing_package_says_how_to_install_it(self):
        sys.modules.pop("nvvfx", None)
        real = sys.meta_path[:]
        blocker = type("Block", (), {"find_spec": staticmethod(lambda name, *a, **k: (_ for _ in ()).throw(ImportError("no")) if name == "nvvfx" else None)})
        sys.meta_path.insert(0, blocker)
        try:
            with self.assertRaises(heiss_rtx.RTXUnavailable) as caught:
                heiss_rtx.nvvfx_module()
            self.assertIn("pip install nvidia-vfx", str(caught.exception))
            self.assertIn("pypi.nvidia.com", str(caught.exception))
        finally:
            sys.meta_path[:] = real

    def test_status_without_a_cuda_card_says_why(self):
        status = heiss_rtx.rtx_status()
        self.assertFalse(status["ready"])
        self.assertIn("NVIDIA RTX", status["reason"])


class Sizes(unittest.TestCase):
    def test_rounds_to_eight(self):
        self.assertEqual(heiss_rtx.output_size(832, 480, 2.0, "HIGHBITRATE_HIGH"), (1664, 960))
        self.assertEqual(heiss_rtx.output_size(830, 470, 1.5, "HIGH"), (1248, 704))

    def test_cleanup_keeps_the_size(self):
        self.assertEqual(heiss_rtx.output_size(830, 470, 4.0, "DEBLUR_HIGH"), (830, 470))

    def test_refuses_beyond_8192(self):
        with self.assertRaises(ValueError):
            heiss_rtx.output_size(2400, 1350, 4.0, "HIGH")


class Pipeline(unittest.TestCase):
    def test_frames_come_out_in_display_order(self):
        out = []
        count = heiss_rtx.stream_frames(
            iter([0.0, 3.0, 6.0]), clean=lambda f: f, process=lambda f: f, emit=out.append,
            interpolate=lambda a, b: [a + (b - a) * i / 3 for i in (1, 2)])
        self.assertEqual(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(count, 7)

    def test_smoothing_works_on_the_cleaned_frames(self):
        out = []
        heiss_rtx.stream_frames(iter([1, 2]), clean=lambda f: f * 10, process=lambda f: f,
                                interpolate=lambda a, b: [(a + b) / 2], emit=out.append)
        self.assertEqual(out, [10, 15.0, 20])


class OnTheCard(unittest.TestCase):
    """The stand-in runs on the CPU, so ComfyUI's device check is stood in for too."""

    def setUp(self):
        self.nv = _fake_nvvfx()
        self._device = heiss_rtx.effect_device
        heiss_rtx.effect_device = lambda: (torch.device("cpu"), 0)
        INTERRUPT.update(after=None, seen=0)

    def tearDown(self):
        heiss_rtx.effect_device = self._device


class Video(OnTheCard):
    def run_node(self, video, **kw):
        args = dict(scale=2.0, quality="HIGHBITRATE_HIGH", strength=1.0, cleanup="none", smooth=1, filename_prefix="heiss-ui/rtx")
        args.update(kw)
        return heiss_rtx.HeissRTXVideo().run(video, **args)

    def test_generated_video_upscales_with_its_sound(self):
        audio = {"waveform": torch.zeros(1, 2, 44100), "sample_rate": 44100}
        # Long enough that the encoder writes video before the sound is added (its buffer is about 40 frames).
        result = self.run_node(GeneratedVideo(ramp(81), fps=16, audio=audio))
        info = probe(saved(result))
        self.assertEqual((info["width"], info["height"], info["frames"]), (128, 96, 81))
        self.assertAlmostEqual(info["fps"], 16, places=3)
        self.assertTrue(info["audio"])
        report = result["ui"]["heiss_rtx"][0]
        self.assertEqual(report["encoder"], "libx264")
        self.assertEqual(result["ui"]["animated"], (True,))

    def test_smooth_doubles_the_frame_rate_and_keeps_the_length(self):
        result = self.run_node(GeneratedVideo(ramp(9), fps=16), smooth=2)
        info = probe(saved(result))
        self.assertEqual(info["frames"], 17)  # (N - 1) * 2 + 1
        self.assertAlmostEqual(info["fps"], 32, places=3)
        self.assertIn(("fg", "MEDIUM"), self.nv.loaded)

    def test_a_file_is_streamed_and_keeps_its_audio(self):
        path = os.path.join(OUT, "source.mp4")
        write_source(path, count=72, fps=24)
        result = self.run_node(VideoFromFile(path, (64, 48)), quality="HIGH", cleanup="DEBLUR_MEDIUM", smooth=2)
        info = probe(saved(result))
        self.assertEqual((info["width"], info["height"], info["frames"]), (128, 96, 143))
        self.assertAlmostEqual(info["fps"], 48, places=3)
        self.assertTrue(info["audio"])
        self.assertEqual([q for kind, q in self.nv.loaded if kind == "vsr"], ["DEBLUR_MEDIUM", "HIGH"])

    def test_effects_are_released(self):
        self.run_node(GeneratedVideo(ramp(3)), cleanup="DENOISE_LOW")
        self.assertEqual(sorted(q for kind, q in self.nv.loaded if kind == "closed"), ["DENOISE_LOW", "HIGHBITRATE_HIGH"])

    def test_a_stopped_run_leaves_no_file(self):
        before = set(os.listdir(os.path.join(OUT, "heiss-ui"))) if os.path.isdir(os.path.join(OUT, "heiss-ui")) else set()
        INTERRUPT["after"] = 3
        with self.assertRaises(KeyboardInterrupt):
            self.run_node(GeneratedVideo(ramp(9)), filename_prefix="heiss-ui/stopped")
        self.assertEqual(set(os.listdir(os.path.join(OUT, "heiss-ui"))) - before, set())


class WithComfyUIsOwnVideos(OnTheCard):
    """ComfyUI's real VIDEO classes, when COMFYUI_PATH points at a checkout: Load Video's file and Create Video's
    frames must both go through, the file without being decoded whole."""

    def setUp(self):
        super().setUp()
        try:
            from comfy_api.latest._input_impl.video_types import VideoFromComponents, VideoFromFile
            from comfy_api.latest._util.video_types import VideoComponents
        except Exception as error:
            self.skipTest(f"needs a ComfyUI checkout ({error})")
        self.classes = VideoFromComponents, VideoFromFile, VideoComponents

    def test_load_video_is_streamed(self):
        _, VideoFromFile, _ = self.classes
        path = os.path.join(OUT, "load-video.mp4")
        write_source(path, count=48, fps=24)
        video = VideoFromFile(path)
        video.get_components = lambda: self.fail("Load Video's file must be streamed, not decoded whole")
        info = probe(saved(heiss_rtx.HeissRTXVideo().run(video, 2.0, "HIGH", 1.0, "none", 1, "heiss-ui/load")))
        self.assertEqual((info["width"], info["height"], info["frames"], info["audio"]), (128, 96, 48, True))

    def test_a_trimmed_load_video_uses_comfyuis_own_frames(self):
        _, VideoFromFile, _ = self.classes
        path = os.path.join(OUT, "trimmed.mp4")
        write_source(path, count=48, fps=24)
        video = VideoFromFile(path, start_time=1.0)
        self.assertIsNone(heiss_rtx._file_source(video))
        info = probe(saved(heiss_rtx.HeissRTXVideo().run(video, 2.0, "HIGH", 1.0, "none", 1, "heiss-ui/trim")))
        self.assertEqual(info["frames"], 24)

    def test_create_video_with_sound(self):
        VideoFromComponents, _, VideoComponents = self.classes
        audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
        video = VideoFromComponents(VideoComponents(images=ramp(50), audio=audio, frame_rate=Fraction(25)))
        info = probe(saved(heiss_rtx.HeissRTXVideo().run(video, 1.5, "HIGHBITRATE_HIGH", 1.0, "none", 2, "heiss-ui/create")))
        self.assertEqual((info["width"], info["height"], info["frames"], info["audio"]), (96, 72, 99, True))
        self.assertAlmostEqual(info["fps"], 50, places=3)


class Image(OnTheCard):
    def test_upscales_a_batch_and_its_alpha(self):
        images = torch.rand(3, 40, 56, 4)
        (out,) = heiss_rtx.HeissRTXUpscale().upscale(images, 2.0, "ULTRA", 1.0)
        self.assertEqual(tuple(out.shape), (3, 80, 112, 4))
        self.assertTrue(torch.allclose(out[0, ::2, ::2, :3], images[0, :, :, :3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
