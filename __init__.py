"""ComfyUI-HEISS-UI-Nodes: the custom nodes HEISS UI builds on. https://github.com/tristmeister/ComfyUI-HEISS-UI-Nodes"""

from .heiss_guidance import NODE_CLASS_MAPPINGS as _guidance, NODE_DISPLAY_NAME_MAPPINGS as _guidance_names
from .heiss_rapid import NODE_CLASS_MAPPINGS as _rapid, NODE_DISPLAY_NAME_MAPPINGS as _rapid_names
# Imports nothing NVIDIA until one of its nodes runs, so the pack loads on every machine.
from .heiss_rtx import NODE_CLASS_MAPPINGS as _rtx, NODE_DISPLAY_NAME_MAPPINGS as _rtx_names

NODE_CLASS_MAPPINGS = {**_rapid, **_guidance, **_rtx}
NODE_DISPLAY_NAME_MAPPINGS = {**_rapid_names, **_guidance_names, **_rtx_names}

# GET /heiss/nodes: this pack's version and nodes, so HEISS can tell an older copy that needs updating.
from .heiss_info import PACK_VERSION, register_route as _register_info  # noqa: E402

_register_info(NODE_CLASS_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
