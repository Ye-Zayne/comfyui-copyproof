"""ComfyUI V1 nodes. RapidOCR imports and model loading happen only on execution."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import Options, proof_batch

_ENGINE_LOCK = threading.RLock()
_ENGINES: dict[tuple, tuple[Any, str]] = {}

_EXPECTED_EXAMPLE = '{"fields":[{"id":"title","text":"限时优惠","kind":"text"},{"id":"price","text":"¥199","kind":"price"}]}'


def image_array(images: torch.Tensor) -> np.ndarray:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("CopyProof expects IMAGE tensor [batch, height, width, 3 or 4]")
    array = images.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("CopyProof IMAGE values must be finite floats between 0 and 1")
    return array


def render_report(array: np.ndarray, report: dict, padding: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole OCR regions only; never pretend character string offsets are pixels."""
    batch_size, height, width, channels = array.shape
    masks, overlays = [], []
    for index in range(batch_size):
        rgb = np.round(array[index, :, :, :3] * 255).astype(np.uint8)
        canvas = Image.fromarray(rgb, "RGB")
        mask = Image.new("L", (width, height), 0)
        draw, mask_draw = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
        for region_index, region in enumerate(report["images"][index]["regions"]):
            status = region["status"]
            color = {"FAIL": (240, 62, 72), "REVIEW": (255, 176, 32), "PASS": (35, 193, 111)}[status]
            x1, y1, x2, y2 = region["bbox"]
            box = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))), min(width - 1, int(np.ceil(x2)) - 1), min(height - 1, int(np.ceil(y2)) - 1))
            polygon = region.get("polygon")
            if polygon:
                points = [(min(width - 1, max(0, round(p[0]))), min(height - 1, max(0, round(p[1])))) for p in polygon]
                draw.line(points + [points[0]], fill=color, width=2)
            else:
                draw.rectangle(box, outline=color, width=2)
            # ASCII labels work without a platform-specific Chinese font.
            draw.text((box[0], max(0, box[1] - 12)), f"{region_index + 1}:{status}", fill=color)
            if status != "PASS":
                if polygon and padding == 0:
                    mask_draw.polygon(points, fill=255)
                else:
                    padded_box = (max(0, box[0] - padding), max(0, box[1] - padding), min(width - 1, box[2] + padding), min(height - 1, box[3] + padding))
                    mask_draw.rectangle(padded_box, fill=255)
        result = np.asarray(canvas, dtype=np.float32) / 255.0
        if channels == 4:
            result = np.concatenate((result, array[index, :, :, 3:4]), axis=-1)
        overlays.append(result)
        masks.append(np.asarray(mask, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(masks)), torch.from_numpy(np.stack(overlays))


def _model_directory(model_root: str) -> str:
    if model_root.strip():
        path = Path(model_root).expanduser().resolve()
    else:
        try:
            import folder_paths
        except ModuleNotFoundError:
            # Useful outside ComfyUI for a real OCR smoke test.
            path = Path(__file__).resolve().parent / "models" / "rapidocr"
        else:
            path = Path(folder_paths.models_dir) / "copyproof" / "rapidocr"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _engine(accelerator: str, model_root: str, capture_confidence: float) -> tuple[Any, str]:
    if accelerator not in {"cpu", "cuda"}:
        raise ValueError("accelerator must be cpu or cuda")
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise RuntimeError("Local OCR requires optional dependencies. In ComfyUI's Python run: python -m pip install -r requirements-ocr.txt. External JSON validation works without OCR packages.") from exc
    if accelerator == "cuda" and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable. Select cpu, or install a compatible onnxruntime-gpu in ComfyUI's Python environment.")
    root = _model_directory(model_root)
    key = (accelerator, root, capture_confidence)
    if key not in _ENGINES:
        try:
            import rapidocr
        except ModuleNotFoundError as exc:
            if exc.name != "rapidocr":
                raise RuntimeError(f"RapidOCR dependency is missing: {exc.name}. Reinstall requirements-ocr.txt in ComfyUI's Python.") from exc
            try:
                import rapidocr_onnxruntime
            except ModuleNotFoundError as legacy_exc:
                raise RuntimeError("RapidOCR is not installed. In ComfyUI's Python run: python -m pip install -r requirements-ocr.txt. Or connect an external OCR JSON to CopyProof Validate Expected Copy.") from legacy_exc
            engine = rapidocr_onnxruntime.RapidOCR(text_score=capture_confidence, det_use_cuda=accelerator == "cuda", cls_use_cuda=accelerator == "cuda", rec_use_cuda=accelerator == "cuda")
            _ENGINES[key] = (engine, "rapidocr_onnxruntime")
        else:
            engine = rapidocr.RapidOCR(params={"Global.text_score": capture_confidence, "Global.model_root_dir": root, "EngineConfig.onnxruntime.use_cuda": accelerator == "cuda"})
            _ENGINES[key] = (engine, "rapidocr")
    return _ENGINES[key]


def rapidocr_detections(result: Any) -> list[dict]:
    """Accept the documented RapidOCROutput and the legacy (lines, timings)."""
    if hasattr(result, "boxes"):
        boxes, texts, scores = result.boxes, getattr(result, "txts", None), getattr(result, "scores", None)
        if boxes is None and texts is None:
            return []
        if boxes is None or texts is None or scores is None or len(boxes) != len(texts) or len(scores) != len(texts):
            raise RuntimeError("RapidOCR returned incomplete recognition data; expected boxes, txts and scores")
        lines = zip(boxes, texts, scores)
    elif isinstance(result, tuple) and len(result) == 2:
        lines = result[0] if result[0] is not None else []
    else:
        raise RuntimeError("Unsupported RapidOCR result. Install rapidocr>=3.0,<4 or use canonical external OCR JSON.")
    detections = []
    for index, line in enumerate(lines):
        if len(line) != 3:
            raise RuntimeError("Invalid RapidOCR detection; expected [polygon, text, confidence]")
        points, text, score = line
        polygon = np.asarray(points, dtype=float).tolist()
        confidence = float(score)
        if not isinstance(text, str) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RuntimeError("RapidOCR returned an invalid text or confidence value")
        detections.append({"id": f"d{index}", "text": text, "confidence": confidence, "polygon": polygon})
    return detections


class CopyProofRapidOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "accelerator": (["cpu", "cuda"], {"default": "cpu"}),
            "capture_confidence": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "model_root": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ocr_json",)
    FUNCTION = "recognize"
    CATEGORY = "CopyProof"
    DESCRIPTION = "Local Chinese/English OCR. Optional RapidOCR imports only when run. Low confidence detections are retained for REVIEW. First use may download OCR weights."

    def recognize(self, images, accelerator="cpu", capture_confidence=0.05, model_root=""):
        if not 0 <= capture_confidence <= 1:
            raise ValueError("capture_confidence must be between 0 and 1")
        array = image_array(images)
        output = []
        # RapidOCR changes internal thresholds during __call__. A shared lock
        # keeps cached engines safe across concurrent ComfyUI executions.
        with _ENGINE_LOCK:
            engine, backend = _engine(accelerator, model_root, capture_confidence)
            for index, image in enumerate(array):
                # RapidOCR treats NumPy arrays as OpenCV BGR; IMAGE is RGB.
                bgr = np.ascontiguousarray(np.round(image[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1])
                result = engine(bgr)
                detections = rapidocr_detections(result)
                for detection in detections:
                    detection["id"] = f"b{index}_{detection['id']}"
                    # Detector corners can differ by subpixel rounding at an
                    # image edge. Clamp engine output only, never external JSON.
                    detection["polygon"] = [[float(np.clip(x, 0, array.shape[2])), float(np.clip(y, 0, array.shape[1]))] for x, y in detection["polygon"]]
                output.append({"batch_index": index, "width": array.shape[2], "height": array.shape[1], "coordinate_space": "pixels", "detections": detections})
        return (json.dumps({"schema_version": "1.0", "backend": backend, "accelerator_requested": accelerator, "capture_confidence": capture_confidence, "images": output}, ensure_ascii=False, allow_nan=False),)


class CopyProofValidate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "expected_json": ("STRING", {"default": _EXPECTED_EXAMPLE, "multiline": True}),
            "ocr_json": ("STRING", {"default": '{"detections":[]}', "multiline": True}),
            "min_confidence": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            "match_threshold": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
            "ignore_whitespace": ("BOOLEAN", {"default": True}),
            "ignore_punctuation": ("BOOLEAN", {"default": False}),
            "ignore_case": ("BOOLEAN", {"default": False}),
            "missing_policy": (["review", "fail"], {"default": "review"}),
            "unexpected_policy": (["review", "ignore", "fail"], {"default": "review"}),
            "mask_padding": ("INT", {"default": 0, "min": 0, "max": 128}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("error_mask", "overlay", "report_json", "verdict")
    FUNCTION = "validate"
    CATEGORY = "CopyProof"
    OUTPUT_NODE = True
    DESCRIPTION = "Compare expected fields to OCR. FAIL=0, REVIEW=1, PASS=2. Low/unknown confidence is REVIEW. Missing text uses supplied bbox only. Brands, numbers, prices and specifications stay strict."

    def validate(self, images, expected_json, ocr_json, min_confidence=0.75, match_threshold=0.55, ignore_whitespace=True, ignore_punctuation=False, ignore_case=False, missing_policy="review", unexpected_policy="review", mask_padding=0):
        if isinstance(mask_padding, bool) or not isinstance(mask_padding, int) or not 0 <= mask_padding <= 128:
            raise ValueError("mask_padding must be an integer between 0 and 128")
        array = image_array(images)
        options = Options(min_confidence, match_threshold, ignore_whitespace, ignore_punctuation, ignore_case, missing_policy, unexpected_policy)
        report = proof_batch(expected_json, ocr_json, array.shape[0], array.shape[2], array.shape[1], options)
        report["mask_padding"] = mask_padding
        report["mask_semantics"] = "1 selects FAIL or REVIEW regions. PASS and unlocated missing text remain 0."
        mask, overlay = render_report(array, report, mask_padding)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        return {"ui": {"text": [f"CopyProof: {report['verdict']} ({report['verdict_code']})", serialized]}, "result": (mask, overlay, serialized, report["verdict_code"])}
