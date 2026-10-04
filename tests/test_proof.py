import json

import pytest
import torch

from copyproof_test_package.core import Options, proof_batch
from copyproof_test_package.nodes import CopyProofValidate


def detection(text, confidence=0.99, bbox=None):
    return {"text": text, "confidence": confidence, "bbox": bbox or [10, 10, 100, 30]}


def proof(fields, detections, **options):
    return proof_batch({"fields": fields}, {"detections": detections}, 1, 160, 120, Options(**options))


def test_chinese_exact_copy_passes():
    report = proof([{"text": "限时优惠"}], [detection("限时优惠")])
    assert report["verdict_code"] == 2
    assert report["images"][0]["fields"][0]["reason"] == "matched"


def test_chinese_substitution_is_failure_with_text_offsets_only():
    report = proof([{"id": "headline", "text": "限时优惠"}], [detection("限时优患")])
    result = report["images"][0]["fields"][0]
    assert report["verdict_code"] == 0
    assert result["changes"] == [{"operation": "replace", "expected_span": [3, 4], "observed_span": [3, 4], "expected": "惠", "observed": "患"}]
    assert result["location_source"] == "ocr_region"


@pytest.mark.parametrize("kind", ["brand", "number", "price", "spec"])
def test_strict_fields_ignore_no_global_normalization(kind):
    report = proof([{"text": "ACME-199.0 ml", "kind": kind}], [detection("acme1990ml")], ignore_case=True, ignore_whitespace=True, ignore_punctuation=True, match_threshold=0.1)
    assert report["verdict"] == "FAIL"


def test_prose_whitespace_punctuation_and_case_can_be_ignored():
    report = proof([{"text": "Hello, World!"}], [detection("hello world")], ignore_case=True, ignore_punctuation=True)
    assert report["verdict"] == "PASS"


def test_numeric_decimal_is_never_erased_by_prose_punctuation_policy():
    report = proof([{"text": "只需19.9元"}], [detection("只需199元")], ignore_punctuation=True)
    assert report["verdict"] == "FAIL"
    result = report["images"][0]["fields"][0]
    assert result["numeric_expected"] == ["19.9"]
    assert result["numeric_observed"] == ["199"]


@pytest.mark.parametrize("text", ["限时优惠", "限时优患"])
def test_low_confidence_correct_or_wrong_copy_is_review(text):
    report = proof([{"text": "限时优惠"}], [detection(text, 0.4)])
    assert report["verdict_code"] == 1
    assert report["images"][0]["fields"][0]["reason"] == "low_confidence"


def test_external_ocr_without_confidence_is_review():
    report = proof([{"text": "限时优惠"}], [{"text": "限时优惠", "bbox": [10, 10, 100, 30]}])
    assert report["verdict"] == "REVIEW"
    assert report["images"][0]["fields"][0]["reason"] == "confidence_unknown"


def test_missing_unanchored_text_has_no_fabricated_localization():
    report = proof([{"text": "限时优惠"}], [])
    assert report["verdict"] == "REVIEW"
    assert report["images"][0]["regions"] == []
    assert report["images"][0]["fields"][0]["located"] is False


def test_missing_anchored_text_uses_expected_bbox_and_optional_fail_policy():
    report = proof([{"text": "限时优惠", "bbox": [20, 20, 80, 40]}], [], missing_policy="fail")
    assert report["verdict"] == "FAIL"
    assert report["images"][0]["regions"][0]["source"] == "expected_bbox"


def test_exact_assignments_are_reserved_before_fuzzy_matches():
    report = proof([{"id": "typo", "text": "ABCE"}, {"id": "correct", "text": "ABCD"}], [detection("ABCD")], unexpected_policy="ignore")
    results = report["images"][0]["fields"]
    assert results[0]["missing"] is True
    assert results[1]["status"] == "PASS"


def test_duplicate_expected_fields_cannot_reuse_one_detection():
    report = proof([{"id": "one", "text": "ACME"}, {"id": "two", "text": "ACME"}], [detection("ACME")])
    results = report["images"][0]["fields"]
    assert sum(result["status"] == "PASS" for result in results) == 1
    assert sum(result["missing"] for result in results) == 1


def test_roi_groups_split_text_in_reading_order():
    fields = [{"text": "限时优惠", "bbox": [0, 0, 130, 60]}]
    detections = [detection("优惠", bbox=[70, 11, 110, 31]), detection("限时", bbox=[10, 10, 50, 30])]
    report = proof(fields, detections)
    assert report["verdict"] == "PASS"
    assert report["images"][0]["fields"][0]["observed"] == "限时\n优惠"


def test_strict_brand_ocr_segmentation_alone_requires_review():
    report = proof([{"text": "ACME", "kind": "brand", "bbox": [0, 0, 130, 60]}], [detection("AC", bbox=[10, 10, 50, 30]), detection("ME", bbox=[70, 10, 110, 30])])
    assert report["verdict"] == "REVIEW"
    assert report["images"][0]["fields"][0]["reason"] == "ambiguous_segmentation_whitespace"


def test_punctuation_normalization_cannot_make_both_strings_empty_and_pass():
    with pytest.raises(ValueError, match="becomes empty"):
        proof([{"text": "!!!"}], [detection("???")], ignore_punctuation=True)


def test_batch_maps_different_copy_without_broadcasting_ocr():
    expected = {"images": [{"fields": [{"text": "AAA"}]}, {"fields": [{"text": "BBB"}]}]}
    ocr = {"images": [{"detections": [detection("AAA")]}, {"detections": [detection("BBC")]}]}
    report = proof_batch(expected, ocr, 2, 160, 120, Options())
    assert [item["verdict"] for item in report["images"]] == ["PASS", "FAIL"]
    assert report["verdict_code"] == 0
    with pytest.raises(ValueError, match="never broadcasts"):
        proof_batch(expected, {"detections": [detection("AAA")]}, 2, 160, 120, Options())


def test_shared_expected_fields_broadcast_to_an_explicit_ocr_batch():
    report = proof_batch({"fields": [{"text": "AAA"}]}, {"images": [{"detections": [detection("AAA")]}, {"detections": [detection("AAA")]}]}, 2, 160, 120, Options())
    assert report["verdict"] == "PASS"


@pytest.mark.parametrize("bad_ocr", [
    {"detections": [{"text": "A", "confidence": float("nan")} ]},
    {"detections": [{"text": "A", "confidence": 1.1}]},
    {"detections": [{"text": "A", "bbox": [-1, 0, 10, 10]}]},
    {"width": 100, "detections": []},
    {"coordinate_space": "normalized", "detections": []},
])
def test_invalid_ocr_is_rejected_before_acceptance(bad_ocr):
    with pytest.raises(ValueError):
        proof_batch({"fields": [{"text": "A"}]}, bad_ocr, 1, 160, 120, Options())


def test_duplicate_expected_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        proof([{"id": "same", "text": "AAA"}, {"id": "same", "text": "BBB"}], [])


def test_low_confidence_unexpected_text_never_becomes_definite_failure():
    report = proof([{"text": "ACME"}], [detection("ACME"), detection("Ghost", 0.1, [10, 50, 100, 80])], unexpected_policy="fail")
    assert report["verdict"] == "REVIEW"


def test_node_mask_shape_and_exact_missing_bbox():
    images = torch.zeros((2, 120, 160, 3))
    expected = json.dumps({"fields": [{"text": "ACME", "bbox": [20, 20, 80, 40]}]})
    ocr = json.dumps({"images": [{"detections": []}, {"detections": [detection("ACME", bbox=[25, 23, 70, 37])]}]})
    result = CopyProofValidate().validate(images, expected, ocr)
    masks, overlays, report, verdict = result["result"]
    assert masks.shape == (2, 120, 160)
    assert overlays.shape == images.shape
    assert masks[0].sum().item() == 60 * 20
    assert masks[0, 20:40, 20:80].min().item() == 1
    assert masks[1].sum().item() == 0
    assert verdict == 1
    assert json.loads(report)["images"][0]["fields"][0]["location_source"] == "expected_bbox"


def test_empty_mask_does_not_mean_pass_for_unlocated_review():
    result = CopyProofValidate().validate(torch.zeros((1, 120, 160, 3)), '{"fields":[{"text":"missing"}]}', '{"detections":[]}')
    assert result["result"][0].sum().item() == 0
    assert result["result"][3] == 1


def test_overlay_preserves_alpha_channel():
    images = torch.zeros((1, 120, 160, 4))
    images[..., 3] = 0.4
    result = CopyProofValidate().validate(images, '{"fields":[{"text":"ACME"}]}', json.dumps({"detections": [detection("ACME")]}))
    assert torch.equal(result["result"][1][..., 3], images[..., 3])


def test_nan_image_is_rejected():
    images = torch.zeros((1, 10, 10, 3))
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        CopyProofValidate().validate(images, '["ACME"]', '[]')
