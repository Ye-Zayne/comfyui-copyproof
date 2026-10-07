from types import SimpleNamespace
import sys

import numpy as np
import pytest

from copyproof_test_package import nodes


def test_modern_rapidocr_output_and_legacy_output_map_identically():
    polygon = [[1, 2], [30, 2], [30, 20], [1, 20]]
    modern = SimpleNamespace(boxes=np.array([polygon]), txts=("优惠",), scores=(0.9,))
    legacy = ([[polygon, "优惠", 0.9]], [0.1, 0.2, 0.3])
    assert nodes.rapidocr_detections(modern) == nodes.rapidocr_detections(legacy)


def test_empty_modern_and_legacy_ocr_are_valid():
    assert nodes.rapidocr_detections(SimpleNamespace(boxes=None, txts=None, scores=None)) == []
    assert nodes.rapidocr_detections((None, None)) == []


def test_incomplete_output_fails_instead_of_silent_pass():
    with pytest.raises(RuntimeError, match="incomplete"):
        nodes.rapidocr_detections(SimpleNamespace(boxes=np.zeros((1, 4, 2)), txts=("ACME",), scores=None))


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="requirements-ocr.txt"):
        nodes._engine("cpu", "", 0.05)


def test_requesting_unavailable_cuda_does_not_silently_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        nodes._engine("cuda", "", 0.05)


def test_modern_rapidocr_engine_parameters_and_cache_are_preserved(monkeypatch, tmp_path):
    calls = []
    engine = object()

    def create(**kwargs):
        calls.append(kwargs)
        return engine

    monkeypatch.setattr(nodes, "_ENGINES", {})
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=create))
    first = nodes._engine("cpu", str(tmp_path), 0.05)
    assert first == (engine, "rapidocr")
    assert nodes._engine("cpu", str(tmp_path), 0.05) == first
    assert calls == [{"params": {"Global.text_score": 0.05, "Global.model_root_dir": str(tmp_path), "EngineConfig.onnxruntime.use_cuda": False}}]


def test_missing_modern_rapidocr_preserves_legacy_fallback(monkeypatch, tmp_path):
    calls = []
    engine = object()

    def create(**kwargs):
        calls.append(kwargs)
        return engine

    monkeypatch.setattr(nodes, "_ENGINES", {})
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=create))
    assert nodes._engine("cpu", str(tmp_path), 0.1) == (engine, "rapidocr_onnxruntime")
    assert calls == [{"text_score": 0.1, "det_use_cuda": False, "cls_use_cuda": False, "rec_use_cuda": False}]


def test_missing_both_rapidocr_packages_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_ENGINES", {})
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(RuntimeError, match="requirements-ocr.txt"):
        nodes._engine("cpu", str(tmp_path), 0.05)


def test_node_converts_rgb_to_bgr_and_preserves_each_batch(monkeypatch):
    import json
    import torch

    seen = []

    def engine(image):
        seen.append(image.copy())
        return ([], None)

    monkeypatch.setattr(nodes, "_engine", lambda *args: (engine, "fake_for_adapter_test"))
    images = torch.zeros((2, 30, 40, 3))
    images[0, ..., 0] = 1
    images[1, ..., 2] = 1
    payload = json.loads(nodes.CopyProofRapidOCR().recognize(images)[0])
    assert seen[0][0, 0].tolist() == [0, 0, 255]
    assert seen[1][0, 0].tolist() == [255, 0, 0]
    assert len(payload["images"]) == 2
    assert payload["images"][1]["batch_index"] == 1


def test_comfyui_package_startup_without_optional_ocr_packages():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc, importlib.util, sys
from pathlib import Path
class NoOCR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rapidocr', 'rapidocr_onnxruntime', 'onnxruntime'}:
            raise ModuleNotFoundError('optional OCR deliberately absent', name=fullname)
sys.meta_path.insert(0, NoOCR())
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('startup_copyproof', root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert set(module.NODE_CLASS_MAPPINGS) == {'CopyProofRapidOCR', 'CopyProofValidate'}
'''
    result = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
