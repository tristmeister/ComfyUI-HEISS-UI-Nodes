"""ComfyUI-HEISS-UI-Nodes: the custom nodes HEISS UI builds on. https://github.com/tristmeister/ComfyUI-HEISS-UI-Nodes"""

from .heiss_rapid import NODE_CLASS_MAPPINGS as _rapid, NODE_DISPLAY_NAME_MAPPINGS as _rapid_names

NODE_CLASS_MAPPINGS = {**_rapid}
NODE_DISPLAY_NAME_MAPPINGS = {**_rapid_names}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
