"""Deterministic copy proofing. No OCR engine or ComfyUI imports here."""
from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

VERSION = "0.1.0"
VERDICT_CODES = {"FAIL": 0, "REVIEW": 1, "PASS": 2}


def _json(value: str | Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc.msg} at character {exc.pos}") from exc


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _bbox(value: Any, width: int, height: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be [x1, y1, x2, y2] in source-image pixels")
    x1, y1, x2, y2 = [_finite(v, name) for v in value]
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{name} must have positive width and height")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"{name} is outside the {width} x {height} source image")
    return [x1, y1, x2, y2]


def _polygon(value: Any, width: int, height: int, name: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{name} must contain at least three [x, y] points")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} points must be [x, y]")
        x, y = [_finite(v, name) for v in point]
        # OCR engines may report points exactly on the far edge.
        if x < 0 or y < 0 or x > width or y > height:
            raise ValueError(f"{name} is outside the {width} x {height} source image")
        result.append([x, y])
    xs, ys = zip(*result)
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        raise ValueError(f"{name} must cover a nonempty region")
    return result


def _check_size(record: dict, width: int, height: int, name: str) -> None:
    for key, expected in (("width", width), ("height", height)):
        if key in record and _finite(record[key], f"{name}.{key}") != expected:
            raise ValueError(f"{name}.{key} does not match the source image; rescale coordinates explicitly")
    if record.get("coordinate_space", "pixels") != "pixels":
        raise ValueError(f"{name} only supports source-image pixel coordinates")


def _records(value: Any, batch_size: int, name: str, shared: bool) -> list[Any]:
    """Only expected fields may broadcast. OCR results must map every image."""
    if isinstance(value, dict) and "images" in value:
        records = value["images"]
        if not isinstance(records, list) or len(records) != batch_size:
            raise ValueError(f"{name}.images must have exactly {batch_size} entries in IMAGE batch order")
        return records
    if batch_size != 1 and not shared:
        raise ValueError(f"{name} for a batch must use {{\"images\": [...]}}; OCR never broadcasts")
    return [value] * batch_size


def parse_expected(value: str | Any, batch_size: int, width: int, height: int) -> list[list[dict]]:
    value = _json(value, "expected_json")
    output = []
    for batch_index, record in enumerate(_records(value, batch_size, "expected_json", shared=True)):
        if isinstance(record, dict):
            _check_size(record, width, height, f"expected_json.images[{batch_index}]")
            fields = record.get("fields")
        else:
            fields = record
        if not isinstance(fields, list) or not fields:
            raise ValueError("Each expected record requires a nonempty fields list")
        parsed, ids = [], set()
        for index, field in enumerate(fields):
            if isinstance(field, str):
                field = {"text": field}
            if not isinstance(field, dict) or not isinstance(field.get("text"), str) or not field["text"].strip():
                raise ValueError(f"Expected field {index} requires nonempty text")
            field_id = field.get("id", f"field_{index + 1}")
            if not isinstance(field_id, str) or not field_id.strip() or field_id in ids:
                raise ValueError("Expected field IDs must be nonempty unique strings per image")
            ids.add(field_id)
            kind = field.get("kind", "text")
            if kind not in {"text", "brand", "number", "price", "spec"}:
                raise ValueError(f"Unsupported field kind: {kind}")
            box = _bbox(field["bbox"], width, height, f"field {field_id}.bbox") if "bbox" in field else None
            parsed.append({"id": field_id, "text": field["text"], "kind": kind, "bbox": box})
        output.append(parsed)
    return output


def parse_ocr(value: str | Any, batch_size: int, width: int, height: int) -> list[list[dict]]:
    value = _json(value, "ocr_json")
    output = []
    for batch_index, record in enumerate(_records(value, batch_size, "ocr_json", shared=False)):
        if isinstance(record, dict):
            _check_size(record, width, height, f"ocr_json.images[{batch_index}]")
            detections = record.get("detections")
        else:
            detections = record
        if not isinstance(detections, list):
            raise ValueError("Each OCR record requires a detections list; use [] when no text is detected")
        parsed, ids = [], set()
        for index, detection in enumerate(detections):
            # Also accepts legacy RapidOCR [polygon, text, confidence] entries.
            if isinstance(detection, (list, tuple)) and len(detection) == 3:
                detection = {"polygon": detection[0], "text": detection[1], "confidence": detection[2]}
            if not isinstance(detection, dict) or not isinstance(detection.get("text"), str):
                raise ValueError(f"OCR detection {index} requires a text string")
            text = detection["text"]
            if not text.strip():
                continue
            detection_id = detection.get("id", f"b{batch_index}_d{index}")
            if not isinstance(detection_id, str) or not detection_id or detection_id in ids:
                raise ValueError("OCR detection IDs must be unique nonempty strings per image")
            ids.add(detection_id)
            confidence = detection.get("confidence", detection.get("score"))
            if confidence is not None:
                confidence = _finite(confidence, f"detection {detection_id}.confidence")
                if not 0 <= confidence <= 1:
                    raise ValueError("OCR confidence must be between 0 and 1")
            polygon = None
            if "polygon" in detection or "points" in detection:
                polygon = _polygon(detection.get("polygon", detection.get("points")), width, height, f"detection {detection_id}.polygon")
                xs, ys = zip(*polygon)
                box = [min(xs), min(ys), max(xs), max(ys)]
            elif "bbox" in detection:
                box = _bbox(detection["bbox"], width, height, f"detection {detection_id}.bbox")
            else:
                box = None
            parsed.append({"id": detection_id, "text": text, "confidence": confidence, "bbox": box, "polygon": polygon})
        output.append(parsed)
    return output


@dataclass(frozen=True)
class Options:
    min_confidence: float = 0.75
    match_threshold: float = 0.55
    ignore_whitespace: bool = True
    ignore_punctuation: bool = False
    ignore_case: bool = False
    missing_policy: str = "review"
    unexpected_policy: str = "review"

    def __post_init__(self):
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.match_threshold <= 1:
            raise ValueError("Confidence and association thresholds must be between 0 and 1")
        if self.missing_policy not in {"review", "fail"} or self.unexpected_policy not in {"ignore", "review", "fail"}:
            raise ValueError("Invalid missing_policy or unexpected_policy")


def normalize(text: str, field: dict, options: Options) -> str:
    # Strict fields keep decimal points, currency signs, brand punctuation, case,
    # units and spaces even when a permissive policy is used for prose.
    if field["kind"] in {"brand", "number", "price", "spec"}:
        return text
    if options.ignore_whitespace:
        text = "".join(ch for ch in text if not ch.isspace())
    if options.ignore_punctuation:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    if options.ignore_case:
        text = text.casefold()
    return text


def numeric_signature(text: str) -> list[str]:
    # Do not let prose punctuation normalization turn 19.9 into 199.
    return re.findall(r"\d+(?:[.,:/\-]\d+)*", text)


def _equivalent(expected: str, observed: str, field: dict, options: Options) -> bool:
    return normalize(expected, field, options) == normalize(observed, field, options) and numeric_signature(expected) == numeric_signature(observed)


def text_diff(expected: str, observed: str) -> list[dict]:
    """Character offsets in strings; these are deliberately NOT image locations."""
    changes = []
    for op, a, b, c, d in difflib.SequenceMatcher(None, expected, observed, autojunk=False).get_opcodes():
        if op != "equal":
            changes.append({"operation": op, "expected_span": [a, b], "observed_span": [c, d], "expected": expected[a:b], "observed": observed[c:d]})
    return changes


def _reading_order(detections: list[dict]) -> list[dict]:
    """Cluster horizontal lines by center-height, then order left to right."""
    remaining = sorted(detections, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0]))
    rows: list[list[dict]] = []
    for item in remaining:
        center = (item["bbox"][1] + item["bbox"][3]) / 2
        item_height = item["bbox"][3] - item["bbox"][1]
        for row in rows:
            row_center = sum((d["bbox"][1] + d["bbox"][3]) / 2 for d in row) / len(row)
            row_height = min(d["bbox"][3] - d["bbox"][1] for d in row)
            if abs(center - row_center) <= min(item_height, row_height) * 0.5:
                row.append(item)
                break
        else:
            rows.append([item])
    return [d for row in rows for d in sorted(row, key=lambda item: item["bbox"][0])]


def _candidate(items: list[dict]) -> dict:
    boxes = [d["bbox"] for d in items if d["bbox"] is not None]
    confidence = None if any(d["confidence"] is None for d in items) else min(d["confidence"] for d in items)
    box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None
    return {"detections": items, "text": "\n".join(d["text"] for d in items), "confidence": confidence, "bbox": box}


def _candidates(field: dict, detections: list[dict]) -> list[dict]:
    box = field["bbox"]
    if box is None:
        return [_candidate([item]) for item in detections]
    x1, y1, x2, y2 = box
    items = [d for d in detections if d["bbox"] is not None and x1 <= (d["bbox"][0] + d["bbox"][2]) / 2 < x2 and y1 <= (d["bbox"][1] + d["bbox"][3]) / 2 < y2]
    return [_candidate(_reading_order(items))] if items else []


def _worst(statuses: list[str]) -> str:
    return "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"


def proof_image(fields: list[dict], detections: list[dict], options: Options) -> dict:
    candidates: dict[int, list[dict]] = {}
    pairs = []
    for field_index, field in enumerate(fields):
        if not normalize(field["text"], field, options):
            raise ValueError(f"Field {field['id']} becomes empty under the chosen normalization policy")
        candidates[field_index] = _candidates(field, detections)
        for candidate_index, candidate in enumerate(candidates[field_index]):
            exact = _equivalent(field["text"], candidate["text"], field, options)
            similarity = difflib.SequenceMatcher(None, normalize(field["text"], field, options), normalize(candidate["text"], field, options), autojunk=False).ratio()
            # A user-anchored field associates to its region regardless of text
            # similarity; unanchored fields never invent a spatial match.
            if field["bbox"] is not None or exact or similarity >= options.match_threshold:
                pairs.append((int(exact), similarity, candidate["confidence"] or 0.0, field_index, candidate_index))
    assignments, used = {}, set()
    # Reserve exact matches before fuzzy associations so a typo cannot steal
    # another field's correct detection. Stable tie-break: input field order.
    pairs.sort(key=lambda p: (-p[0], -p[1], -p[2], p[3], p[4]))
    for exact, similarity, confidence, field_index, candidate_index in pairs:
        candidate = candidates[field_index][candidate_index]
        candidate_ids = {d["id"] for d in candidate["detections"]}
        if field_index not in assignments and not candidate_ids.intersection(used):
            assignments[field_index] = (candidate, bool(exact), similarity)
            used.update(candidate_ids)

    field_results, regions = [], []
    for field_index, field in enumerate(fields):
        assigned = assignments.get(field_index)
        if assigned is None:
            status = "FAIL" if options.missing_policy == "fail" else "REVIEW"
            result = {"id": field["id"], "kind": field["kind"], "expected": field["text"], "observed": None, "status": status, "reason": "not_detected", "confidence": None, "missing": True, "detection_ids": [], "expected_bbox": field["bbox"], "located": field["bbox"] is not None, "location_source": "expected_bbox" if field["bbox"] else None, "changes": []}
            if field["bbox"]:
                regions.append({"status": status, "field_id": field["id"], "bbox": field["bbox"], "polygon": None, "source": "expected_bbox"})
        else:
            candidate, exact, similarity = assigned
            confidence = candidate["confidence"]
            if confidence is None or confidence < options.min_confidence:
                status, reason = "REVIEW", "confidence_unknown" if confidence is None else "low_confidence"
            elif exact:
                status, reason = "PASS", "matched"
            elif len(candidate["detections"]) > 1 and "".join(field["text"].split()) == "".join(candidate["text"].split()):
                # We insert line delimiters when grouping detections, so a
                # strict mismatch attributable only to OCR segmentation is
                # insufficient evidence of an actual copy error.
                status, reason = "REVIEW", "ambiguous_segmentation_whitespace"
            else:
                status, reason = "FAIL", "text_mismatch"
            result = {"id": field["id"], "kind": field["kind"], "expected": field["text"], "observed": candidate["text"], "status": status, "reason": reason, "confidence": confidence, "similarity": similarity, "missing": False, "detection_ids": [d["id"] for d in candidate["detections"]], "expected_bbox": field["bbox"], "located": candidate["bbox"] is not None or field["bbox"] is not None, "location_source": "ocr_region" if candidate["bbox"] else "expected_bbox" if field["bbox"] else None, "changes": [] if exact else text_diff(field["text"], candidate["text"]), "numeric_expected": numeric_signature(field["text"]), "numeric_observed": numeric_signature(candidate["text"])}
            located_detections = [d for d in candidate["detections"] if d["bbox"]]
            for detection in located_detections:
                regions.append({"status": status, "field_id": field["id"], "bbox": detection["bbox"], "polygon": detection["polygon"], "source": "ocr_region"})
            if not located_detections and field["bbox"]:
                regions.append({"status": status, "field_id": field["id"], "bbox": field["bbox"], "polygon": None, "source": "expected_bbox"})
        field_results.append(result)

    extras = []
    for detection in detections:
        if detection["id"] in used:
            continue
        if options.unexpected_policy == "ignore":
            status = "IGNORED"
        elif detection["confidence"] is None or detection["confidence"] < options.min_confidence:
            status = "REVIEW"
        else:
            status = "FAIL" if options.unexpected_policy == "fail" else "REVIEW"
        extras.append({"id": detection["id"], "text": detection["text"], "confidence": detection["confidence"], "bbox": detection["bbox"], "status": status, "reason": "unexpected_text"})
        if status != "IGNORED" and detection["bbox"]:
            regions.append({"status": status, "field_id": detection["id"], "bbox": detection["bbox"], "polygon": detection["polygon"], "source": "ocr_region"})
    verdict = _worst([f["status"] for f in field_results] + [e["status"] for e in extras])
    return {"verdict": verdict, "verdict_code": VERDICT_CODES[verdict], "fields": field_results, "unexpected": extras, "regions": regions}


def proof_batch(expected_json: str | Any, ocr_json: str | Any, batch_size: int, width: int, height: int, options: Options) -> dict:
    if batch_size < 1 or width < 1 or height < 1:
        raise ValueError("An IMAGE batch must contain at least one nonempty image")
    fields = parse_expected(expected_json, batch_size, width, height)
    detections = parse_ocr(ocr_json, batch_size, width, height)
    images = []
    for index, (field_list, detection_list) in enumerate(zip(fields, detections)):
        item = proof_image(field_list, detection_list, options)
        item.update({"batch_index": index, "width": width, "height": height})
        images.append(item)
    verdict = _worst([item["verdict"] for item in images])
    return {"schema_version": "1.0", "package_version": VERSION, "verdict": verdict, "verdict_code": VERDICT_CODES[verdict], "verdict_codes": VERDICT_CODES, "policy": vars(options), "images": images, "notes": ["Character diff offsets refer to strings, not image pixels.", "A missing OCR detection is not proof that text is absent; default missing policy is REVIEW.", "Error masks cover detected text regions or supplied expected bboxes, never inferred individual glyphs."]}
