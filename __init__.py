"""ComfyUI CopyProof: commercial copy acceptance and actionable region masks."""
from .nodes import CopyProofRapidOCR, CopyProofValidate

NODE_CLASS_MAPPINGS = {
    "CopyProofRapidOCR": CopyProofRapidOCR,
    "CopyProofValidate": CopyProofValidate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CopyProofRapidOCR": "CopyProof · Local OCR (中英)",
    "CopyProofValidate": "CopyProof · Validate Expected Copy",
}
__version__ = "0.1.1"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
