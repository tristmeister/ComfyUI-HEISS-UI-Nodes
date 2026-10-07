"""HEISS Rapid Guidance: CFG while the noise is high, CFG 1 after. Run like test_rapid.py."""

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import comfy.model_sampling as ms  # noqa: E402
import comfy.samplers  # noqa: E402
import heiss_guidance as guidance  # noqa: E402


def guider(kind, cfg=5.0, until=0.3):
    cls = type("Sampling", (ms.ModelSamplingDiscreteFlow, ms.CONST), {}) if kind == "flow" else type("Sampling", (ms.ModelSamplingDiscrete, ms.EPS), {})
    sampling = cls(None)
    patcher = SimpleNamespace(model_options={}, get_model_object=lambda name: sampling, is_dynamic=lambda: False)
    g = guidance.GuiderLateCFG(patcher)
    g.cfg = cfg
    g.set_until(until)
    g.conds = {}
    g.inner_model = None
    return g


class LateCFG(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self._real = comfy.samplers.sampling_function
        comfy.samplers.sampling_function = lambda model, x, t, uncond, cond, scale, **kw: self.seen.append(scale) or x

    def tearDown(self):
        comfy.samplers.sampling_function = self._real

    def test_flow_keeps_cfg_while_the_noise_is_high(self):
        g = guider("flow", cfg=4.5, until=0.3)
        for sigma in (1.0, 0.7, 0.31, 0.29, 0.05):
            g.predict_noise(torch.zeros(1), torch.tensor([sigma]))
        self.assertEqual(self.seen, [4.5, 4.5, 4.5, 1.0, 1.0])

    def test_sd_family_compares_on_the_flow_scale(self):
        g = guider("sigma", cfg=7.0, until=0.3)
        for sigma in (14.6, 1.0, 0.43, 0.42, 0.03):  # t = sigma / (1 + sigma): 0.94, 0.5, 0.30, 0.296, 0.03
            g.predict_noise(torch.zeros(1), torch.tensor([sigma]))
        self.assertEqual(self.seen, [7.0, 7.0, 7.0, 1.0, 1.0])

    def test_zero_keeps_cfg_throughout(self):
        g = guider("flow", cfg=3.0, until=0.0)
        for sigma in (1.0, 0.5, 0.0):
            g.predict_noise(torch.zeros(1), torch.tensor([sigma]))
        self.assertEqual(self.seen, [3.0, 3.0, 3.0])

    def test_node_builds_the_guider(self):
        patcher = SimpleNamespace(model_options={}, get_model_object=lambda name: ms.CONST(), is_dynamic=lambda: False)
        g, = guidance.HeissRapidGuidance().get_guider(patcher, [], [], 4.0, 0.25)
        self.assertIsInstance(g, comfy.samplers.CFGGuider)
        self.assertEqual((g.cfg, g.cfg_until), (4.0, 0.25))


if __name__ == "__main__":
    unittest.main()
