"""ComfyUI-HEISS-UI-Nodes: the custom nodes HEISS UI builds on. https://github.com/tristmeister/ComfyUI-HEISS-UI-Nodes"""

from .heiss_enhance import NODE_CLASS_MAPPINGS as _enhance, NODE_DISPLAY_NAME_MAPPINGS as _enhance_names, register_route as _register_rescan
from .heiss_guidance import NODE_CLASS_MAPPINGS as _guidance, NODE_DISPLAY_NAME_MAPPINGS as _guidance_names
from .heiss_inpaint import NODE_CLASS_MAPPINGS as _inpaint, NODE_DISPLAY_NAME_MAPPINGS as _inpaint_names
from .heiss_rapid import NODE_CLASS_MAPPINGS as _rapid, NODE_DISPLAY_NAME_MAPPINGS as _rapid_names
# Imports nothing NVIDIA until one of its nodes runs, so the pack loads on every machine.
from .heiss_rtx import NODE_CLASS_MAPPINGS as _rtx, NODE_DISPLAY_NAME_MAPPINGS as _rtx_names

NODE_CLASS_MAPPINGS = {**_rapid, **_guidance, **_rtx, **_inpaint, **_enhance}
NODE_DISPLAY_NAME_MAPPINGS = {**_rapid_names, **_guidance_names, **_rtx_names, **_inpaint_names, **_enhance_names}

# GET /heiss/nodes: this pack's version and nodes, so HEISS can tell an older copy that needs updating.
from .heiss_info import PACK_VERSION, register_route as _register_info  # noqa: E402

_register_info(NODE_CLASS_MAPPINGS)
# POST /heiss/models/rescan: lists a model HEISS just downloaded, on drives whose folders keep their modified time.
_register_rescan()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
