"""Optional real OCR smoke test, against explicitly provisioned model weights."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("copyproof_smoke_package", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from copyproof_smoke_package.nodes import CopyProofRapidOCR, CopyProofValidate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-root", required=True, help="Existing RapidOCR 3.x model directory (no implicit model download intended)")
parser.add_argument("--output-dir", default=str(EXAMPLES / "real_ocr_run"))
args = parser.parse_args()
model_root = Path(args.model_root).expanduser().resolve()
if not model_root.is_dir() or not list(model_root.glob("*.onnx")):
    parser.error("--model-root must point to an existing directory containing the matching ONNX weights")
out = Path(args.output_dir).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
image = Image.open(EXAMPLES / "copyproof_demo.png").convert("RGB")
images = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).unsqueeze(0)
ocr_json = CopyProofRapidOCR().recognize(images, model_root=str(model_root))[0]
expected_json = (EXAMPLES / "expected.json").read_text(encoding="utf-8")
mask, overlay, report_json, verdict = CopyProofValidate().validate(images, expected_json, ocr_json)["result"]
(out / "ocr.json").write_text(ocr_json + "\n", encoding="utf-8")
(out / "report.json").write_text(report_json + "\n", encoding="utf-8")
Image.fromarray(np.round(overlay[0].numpy() * 255).astype(np.uint8)).save(out / "overlay.png")
Image.fromarray(np.round(mask[0].numpy() * 255).astype(np.uint8)).save(out / "mask.png")
report = json.loads(report_json)
print(json.dumps({"real_ocr": True, "observed": [d["text"] for d in json.loads(ocr_json)["images"][0]["detections"]], "verdict": report["verdict"], "verdict_code": verdict, "output_dir": str(out)}, ensure_ascii=False))
assert verdict == 0, "The deliberately wrong price in the demo should FAIL"
price = next(field for field in report["images"][0]["fields"] if field["id"] == "price")
assert price["reason"] == "text_mismatch", "The price should be a real recognized mismatch, not a missing-field failure"
assert mask.sum().item() > 0, "A detected price error must generate a localized repair mask"
