"""
HEISS Prompt Enhance: a local vision LLM rewrites a short idea into a full prompt, or reads a picture into one.

It runs on ComfyUI's own text generation (the CLIP loader's LLM text encoders: Qwen3-VL, Qwen3.5, Gemma 4), so the
model shares ComfyUI's memory management with the image and video models instead of fighting them for VRAM. What it
adds over Generate Text:

- Qwen3-VL's own chat template. ComfyUI's default appends an empty think block, which the Instruct models never saw in
  training; they then talk about the prompt instead of writing one.
- streaming: the text so far goes out on the websocket as `heiss.enhance` while it is written, so HEISS can show it
  appear word by word.
- a clean result: no think block, no "Prompt:" label, no quotes or Markdown around it.
"""

from __future__ import annotations

import re
import time

STREAM_EVENT = "heiss.enhance"
STREAM_INTERVAL = 0.05  # seconds between partial texts on the websocket

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer
except ImportError:  # loaded outside ComfyUI (tests)
    PromptServer = None


def is_qwen3vl(clip):
    return "qwen3vl" in (getattr(clip.tokenizer, "clip", "") or "").lower()


def qwen3vl_chat(system, user, images=0):
    """Qwen3-VL Instruct's chat format: a system turn, a user turn with an image slot per picture, no think block."""
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * images
    text = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return text + f"<|im_start|>user\n{vision}{user}<|im_end|>\n<|im_start|>assistant\n"


_LABEL = re.compile(r"^(?:\*\*)?(?:(?:enhanced|final|rewritten|improved|image|video) )?prompt(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.I)
_PAIRS = {'"': '"', "'": "'", "*": "*", "`": "`", "“": "”", "«": "»"}


def clean(text):
    """The prompt alone: no reasoning, label, wrapping quotes, code fence or Markdown emphasis."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif text.lstrip().startswith("<think>"):
        return ""  # still reasoning: nothing to show yet
    text = re.sub(r"<\|?[a-z_]+\|?>", "", text).strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    text = _LABEL.sub("", text, count=1)
    while len(text) > 1 and _PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.strip()


def _sampler(clip):
    """The module whose sample_token() picks each token of generate()."""
    for module in clip.cond_stage_model.modules():
        if hasattr(module, "sample_token") and hasattr(module, "generate"):
            return module
    return None


class _Stream:
    """Sees each token as generate() samples it and sends the decoded text so far, a few times a second."""

    def __init__(self, clip, stream_id):
        self.clip, self.stream_id, self.ids, self.last = clip, stream_id, [], 0.0
        self.target = _sampler(clip) if stream_id and PromptServer is not None else None

    def __enter__(self):
        if self.target is not None:
            original = self.target.sample_token

            def sample_token(*args, **kwargs):
                token = original(*args, **kwargs)
                self.ids.append(int(token.reshape(-1)[0]))
                now = time.monotonic()
                if now - self.last >= STREAM_INTERVAL:
                    self.last = now
                    self.send(clean(self.clip.decode(self.ids)), False)
                return token

            self.target.sample_token = sample_token  # shadows the method on this instance only, until __exit__
        return self

    def send(self, text, done):
        if self.stream_id and PromptServer is not None:
            server = PromptServer.instance
            server.send_sync(STREAM_EVENT, {"id": self.stream_id, "text": text, "done": done}, server.client_id)

    def __exit__(self, *exc):
        if self.target is not None:
            del self.target.sample_token
        return False


class HeissPromptEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "A vision LLM from the CLIP loader: Qwen3-VL, Qwen3.5 or Gemma 4."}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 400, "min": 16, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "0 always picks the likeliest word."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Pictures the model looks at; a batch is several pictures."}),
                "thinking": ("BOOLEAN", {"default": False, "tooltip": "Reason first, on models that can. Slower."}),
                "stream_id": ("STRING", {"default": "", "tooltip": "Sends the text so far as heiss.enhance events with this id."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "enhance"
    OUTPUT_NODE = True
    CATEGORY = "HEISS UI/text"
    DESCRIPTION = ("Rewrites a short idea into a detailed prompt, or describes a picture as one, with a local vision LLM. "
                   "Streams the text while it is written.")

    def enhance(self, clip, system_prompt, prompt, max_tokens, temperature, seed, image=None, thinking=False, stream_id=""):
        user = prompt.strip() or ("Describe this picture." if image is not None else "Surprise me.")
        system = system_prompt.strip()
        started = time.monotonic()
        if is_qwen3vl(clip):
            tokens = clip.tokenize(qwen3vl_chat(system, user, 0 if image is None else image.shape[0]), image=image)
        else:
            # Qwen3.5 and Gemma 4 templates are right as they are; Gemma's tokenizer skips its own unless asked.
            tokens = clip.tokenize(user, image=image, thinking=thinking, skip_template=False, system_prompt=system)
        with _Stream(clip, stream_id) as stream:
            ids = clip.generate(tokens, do_sample=temperature > 0, max_length=max_tokens, temperature=max(temperature, 0.01),
                                top_k=20, top_p=0.9, min_p=0.0, repetition_penalty=1.05, seed=seed)
            # Nothing usable back keeps what the person wrote.
            text = clean(clip.decode(ids)) or prompt.strip()
            stream.send(text, True)
        seconds = time.monotonic() - started
        print(f"[HEISS Prompt Enhance] {len(ids)} tokens in {seconds:.1f}s")
        return {"ui": {"text": [text], "tokens": [len(ids)], "seconds": [round(seconds, 2)]}, "result": (text,)}


def register_route():
    """
    POST /heiss/models/rescan: forget ComfyUI's cached model lists, so a file HEISS just downloaded can be picked.
    ComfyUI only looks again when a folder's modified time changes, and exFAT and some network drives never change it.
    """
    if PromptServer is None:
        return

    @PromptServer.instance.routes.post("/heiss/models/rescan")
    async def rescan(request):
        folder_paths.filename_list_cache.clear()
        return web.json_response({"ok": True})


NODE_CLASS_MAPPINGS = {"HeissPromptEnhance": HeissPromptEnhance}
NODE_DISPLAY_NAME_MAPPINGS = {"HeissPromptEnhance": "HEISS Prompt Enhance"}
