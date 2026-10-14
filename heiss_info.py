"""
Which HEISS UI Nodes this is, for HEISS UI to ask: GET /heiss/nodes.

PACK_VERSION goes up by one whenever HEISS needs something new from the pack: a new node, or a change in what an
existing node does. HEISS asks for at least the version its features need (server/heiss-nodes.js) and offers the
update when the loaded pack is older. HEISS also reads this line from the file on disk, so it can tell "installed
but ComfyUI hasn't restarted yet" from "out of date", for copies installed with git, from a zip or by Manager alike.
Keep it a plain integer on one line.

  1  HEISS Rapid
  2  HEISS Rapid Guidance
  3  HEISS RTX Video and HEISS RTX Upscale; this route
  4  HEISS Inpaint Stitch
  5  HEISS Inpaint Stitch: colour match leaves a meant change alone (0.06), bicubic correction
  6  HEISS RTX: nvidia-vfx older than 0.2 is reported as too old, not ready
  7  HEISS Inpaint Stitch leaves corrections under half a pixel alone; HEISS RTX says when an old package is locked
  8  HEISS Prompt Enhance; POST /heiss/models/rescan
  9  HEISS Rapid grows with the run's own high-frequency noise, so a Rapid picture keeps closer to its seed
"""

from __future__ import annotations

import os

PACK_VERSION = 9


def _commit():
    """The checked-out commit, when the pack is a git checkout; read from .git, without running git."""
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(root, ".git", "HEAD"), encoding="utf-8") as handle:
            head = handle.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = os.path.join(root, ".git", *ref.split("/"))
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as handle:
                    return handle.read().strip()
            with open(os.path.join(root, ".git", "packed-refs"), encoding="utf-8") as handle:
                for line in handle:
                    if line.strip().endswith(" " + ref):
                        return line.split()[0]
            return ""
        return head
    except OSError:
        return ""


def pack_info(nodes):
    return {"pack": "heiss-ui-nodes", "version": PACK_VERSION, "commit": _commit(),
            "folder": os.path.basename(os.path.dirname(os.path.abspath(__file__))), "nodes": sorted(nodes)}


def register_route(nodes):
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # loaded outside ComfyUI (tests)

    @PromptServer.instance.routes.get("/heiss/nodes")
    async def _info(request):
        return web.json_response(pack_info(nodes))
