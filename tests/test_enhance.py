"""HEISS Prompt Enhance: the template, the clean result, the stream. Run like test_rapid.py."""

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import heiss_enhance as enhance  # noqa: E402


class Template(unittest.TestCase):
    def test_qwen3vl_turns_have_no_think_block(self):
        text = enhance.qwen3vl_chat("Be brief.", "Idea: a fox", images=1)
        self.assertTrue(text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Idea: a fox"))
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", text)

    def test_no_system_turn_without_a_system_prompt(self):
        self.assertTrue(enhance.qwen3vl_chat("", "hi").startswith("<|im_start|>user\nhi"))

    def test_detects_qwen3vl_from_the_tokenizer(self):
        self.assertTrue(enhance.is_qwen3vl(SimpleNamespace(tokenizer=SimpleNamespace(clip="clip_qwen3vl_4b"))))
        self.assertFalse(enhance.is_qwen3vl(SimpleNamespace(tokenizer=SimpleNamespace(clip="clip_qwen35_4b"))))


class Clean(unittest.TestCase):
    def test_strips_reasoning_labels_quotes_and_markdown(self):
        self.assertEqual(enhance.clean("<think>plan</think>\n\n**Prompt:** \"A red fox, **sharp** focus.\""), "A red fox, sharp focus.")
        self.assertEqual(enhance.clean("```\nA fox\n```"), "A fox")
        self.assertEqual(enhance.clean("Enhanced prompt: “A fox”"), "A fox")

    def test_unfinished_reasoning_shows_nothing_yet(self):
        self.assertEqual(enhance.clean("<think>still planning"), "")

    def test_keeps_inner_quotes(self):
        self.assertEqual(enhance.clean('A sign that reads "OPEN" above the door'), 'A sign that reads "OPEN" above the door')


class Stream(unittest.TestCase):
    def test_sends_text_while_sampling_and_restores_the_method(self):
        sent = []

        class Generator(torch.nn.Module):
            def generate(self):
                return [self.sample_token(torch.tensor([[i]])) for i in (1, 2, 3)]

            def sample_token(self, token):
                return token

        model = Generator()
        clip = SimpleNamespace(cond_stage_model=torch.nn.Sequential(model), decode=lambda ids: " ".join(f"w{i}" for i in ids))
        server = SimpleNamespace(client_id="c", send_sync=lambda event, data, sid: sent.append((event, data)))
        interval, prompt_server = enhance.STREAM_INTERVAL, enhance.PromptServer
        enhance.STREAM_INTERVAL, enhance.PromptServer = 0, SimpleNamespace(instance=server)
        try:
            with enhance._Stream(clip, "abc") as stream:
                model.generate()
                stream.send("w1 w2 w3", True)
        finally:
            enhance.STREAM_INTERVAL, enhance.PromptServer = interval, prompt_server
        self.assertEqual([data["text"] for _, data in sent], ["w1", "w1 w2", "w1 w2 w3", "w1 w2 w3"])
        self.assertTrue(sent[-1][1]["done"])
        self.assertTrue(all(event == "heiss.enhance" and data["id"] == "abc" for event, data in sent))
        self.assertNotIn("sample_token", vars(model))


if __name__ == "__main__":
    unittest.main()
