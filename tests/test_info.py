"""GET /heiss/nodes: the pack's version, for HEISS UI. Run like test_rapid.py."""

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import heiss_info  # noqa: E402


class Info(unittest.TestCase):
    def test_reports_version_commit_and_nodes(self):
        info = heiss_info.pack_info({"HeissRapid": object, "HeissRTXVideo": object})
        self.assertEqual(info["version"], heiss_info.PACK_VERSION)
        self.assertEqual(info["nodes"], ["HeissRTXVideo", "HeissRapid"])
        self.assertRegex(info["commit"], r"^([0-9a-f]{40})?$")

    def test_heiss_can_read_the_version_from_the_file(self):
        # server/heiss-nodes.js reads it this way from copies ComfyUI hasn't loaded yet; keep the line simple.
        with open(os.path.join(ROOT, "heiss_info.py"), encoding="utf-8") as handle:
            found = re.search(r"^PACK_VERSION = (\d+)\s*$", handle.read(), re.M)
        self.assertEqual(int(found.group(1)), heiss_info.PACK_VERSION)


if __name__ == "__main__":
    unittest.main()
