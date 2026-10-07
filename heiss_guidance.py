"""
HEISS Rapid Guidance: prompt guidance (CFG) for the first part of a run, plain sampling for the rest.

With CFG above 1 every step runs the model twice, once with the prompt and once without. The steps that settle what the
picture shows need that push; the late steps only refine detail that is already decided, and do nearly as well without
it. This guider keeps CFG while the noise level is at or above `cfg_until` and samples at CFG 1 below it, where ComfyUI
skips the second pass: those steps cost half. It stacks with HEISS Rapid, which makes the early steps cheap; this makes
the late ones cheaper.
"""

from __future__ import annotations

import comfy.samplers

try:
    from .heiss_rapid import flow_t, model_kind
except ImportError:  # loaded on its own (tests)
    from heiss_rapid import flow_t, model_kind


class GuiderLateCFG(comfy.samplers.CFGGuider):
    def set_until(self, until):
        self.cfg_until = float(until)
        self._kind = None

    def _noise_level(self, timestep):
        if self._kind is None:
            kind, _ = model_kind(self)
            self._kind = kind or "flow"
        return flow_t(float(timestep.reshape(-1)[0]), self._kind)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cfg = self.cfg if self._noise_level(timestep) >= self.cfg_until else 1.0
        return comfy.samplers.sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), cfg, model_options=model_options, seed=seed)


class HeissRapidGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "cfg_until": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Noise level where guidance stops (1 = pure noise, 0 = finished; SD-family models on the "
                                               "same scale, sigma / (1 + sigma)). 0 keeps CFG for the whole run."}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "HEISS UI/sampling"
    DESCRIPTION = ("CFG for the steps that decide the picture, plain sampling for the late detail steps, which then cost half. "
                   "Use it in place of CFGGuider; it stacks with HEISS Rapid.")

    def get_guider(self, model, positive, negative, cfg, cfg_until):
        guider = GuiderLateCFG(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_until(cfg_until)
        return (guider,)


NODE_CLASS_MAPPINGS = {"HeissRapidGuidance": HeissRapidGuidance}
NODE_DISPLAY_NAME_MAPPINGS = {"HeissRapidGuidance": "HEISS Rapid Guidance"}
