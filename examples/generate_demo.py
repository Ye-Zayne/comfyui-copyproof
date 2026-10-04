"""Generate reproducible demo assets and valid ComfyUI API prompts. No OCR."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("copyproof_demo_package", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from copyproof_demo_package.nodes import CopyProofValidate


def font(size):
    paths = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    # Pillow's own scalable font is available on supported recent versions.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


image = Image.new("RGB", (640, 360), "#101a28")
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((20, 20, 620, 340), radius=16, outline="#3d566e", width=2)
draw.text((42, 46), "ACME", fill="#55e9bd", font=font(42))
draw.text((42, 139), "LIMITED OFFER", fill="white", font=font(38))
draw.text((42, 227), "$1999", fill="#ffe099", font=font(56))
image.save(OUT / "copyproof_demo.png")

expected = {"fields": [
    {"id": "brand", "text": "ACME", "kind": "brand", "bbox": [35, 35, 250, 108]},
    {"id": "headline", "text": "LIMITED OFFER", "kind": "text", "bbox": [35, 130, 590, 197]},
    {"id": "price", "text": "$199", "kind": "price", "bbox": [35, 214, 430, 300]},
]}
ocr = {"width": 640, "height": 360, "detections": [
    {"id": "d0", "text": "ACME", "confidence": 0.99, "bbox": [42, 46, 170, 90]},
    {"id": "d1", "text": "LIMITED OFFER", "confidence": 0.99, "bbox": [42, 139, 365, 182]},
    {"id": "d2", "text": "$1999", "confidence": 0.99, "bbox": [42, 227, 235, 286]},
]}
tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).unsqueeze(0)
validation_inputs = {"expected_json": json.dumps(expected, ensure_ascii=False), "min_confidence": 0.75, "match_threshold": 0.55, "ignore_whitespace": True, "ignore_punctuation": False, "ignore_case": False, "missing_policy": "review", "unexpected_policy": "review", "mask_padding": 0}
mask, overlay, report, verdict = CopyProofValidate().validate(tensor, ocr_json=json.dumps(ocr), **validation_inputs)["result"]
Image.fromarray(np.round(overlay[0].numpy() * 255).astype(np.uint8)).save(OUT / "copyproof_overlay.png")
Image.fromarray(np.round(mask[0].numpy() * 255).astype(np.uint8)).save(OUT / "copyproof_mask.png")
(OUT / "copyproof_report.json").write_text(report + "\n", encoding="utf-8")
(OUT / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "synthetic_ocr.json").write_text(json.dumps(ocr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

external_prompt = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "copyproof_demo.png"}},
    "2": {"class_type": "CopyProofValidate", "inputs": {"images": ["1", 0], "ocr_json": json.dumps(ocr), **validation_inputs}},
    "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 1], "filename_prefix": "CopyProof/overlay"}},
    "4": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 0]}},
    "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "CopyProof/error_mask"}},
}
local_prompt = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "copyproof_demo.png"}},
    "2": {"class_type": "CopyProofRapidOCR", "inputs": {"images": ["1", 0], "accelerator": "cpu", "capture_confidence": 0.05, "model_root": ""}},
    "3": {"class_type": "CopyProofValidate", "inputs": {"images": ["1", 0], "ocr_json": ["2", 0], **validation_inputs}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 1], "filename_prefix": "CopyProof/local_ocr_overlay"}},
    "5": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
    "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "CopyProof/local_ocr_error_mask"}},
}
for filename, prompt in (("api_external_ocr.json", external_prompt), ("api_local_ocr.json", local_prompt)):
    (OUT / filename).write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Demo verdict: {verdict} (FAIL=0). Assets written to {OUT}")
