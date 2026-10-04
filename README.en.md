# ComfyUI CopyProof

Validate image copy against explicit expected fields and output actionable text-region masks. Chinese/English local OCR is optional. Validation with external OCR JSON requires no OCR model, cloud service, API key or LLM.

## Installation

Place the folder in `ComfyUI/custom_nodes/comfyui-copyproof` and restart. Validation only needs ComfyUI's existing Torch, NumPy and Pillow.

For actual local OCR, run with **ComfyUI's Python**, including `python_embeded/python.exe` for Windows portable:

```bash
python -m pip install -r custom_nodes/comfyui-copyproof/requirements-ocr.txt
```

Missing OCR packages do not prevent ComfyUI startup. They are imported only when the Local OCR node runs. Modern `rapidocr` 3.x is recommended; existing `rapidocr_onnxruntime` legacy installations are also adapted. Default recognition is Chinese and English. CPU works on Windows, macOS and Linux. CUDA requires a compatible `onnxruntime-gpu`; an unavailable CUDA provider raises an actionable error instead of silently falling back.

Modern OCR weights default to `ComfyUI/models/copyproof/rapidocr`; set `model_root` to reuse an existing model directory. Depending on the installed RapidOCR version, default model weights may be bundled or downloaded by RapidOCR on first execution. Pre-provision the matching weights for offline use. The legacy package uses its own model configuration and ignores `model_root`.

## Nodes

| Class ID | Inputs | Outputs |
| --- | --- | --- |
| `CopyProofRapidOCR` | IMAGE batch, accelerator, capture_confidence, model_root | OCR JSON STRING |
| `CopyProofValidate` | IMAGE, expected_json, ocr_json, policies | error MASK, annotated IMAGE, JSON report STRING, verdict INT |

**Verdict: FAIL=0, REVIEW=1, PASS=2.** A batch fails if any image fails; otherwise it needs review if any image needs review. Individual results remain in the report. Validation is an OUTPUT_NODE and can execute by itself; connect its outputs to other nodes as needed. Frontend text display depends on ComfyUI version; the STRING output is always available.

## Expected fields and OCR format

```json
{"fields":[
  {"id":"headline","text":"限时优惠","kind":"text","bbox":[20,20,300,90]},
  {"id":"brand","text":"ACME","kind":"brand"},
  {"id":"price","text":"¥199","kind":"price","bbox":[20,110,240,180]}
]}
```

Field IDs may be omitted but must be unique per image. `bbox` is optional and uses source-image pixel coordinates `[x1,y1,x2,y2]`, with exclusive right/bottom edges. Adapt example coordinates to your image. No implicit resizing or normalized coordinates. Fields with boxes combine OCR detections whose centers fall inside their ROI in horizontal reading order. Without boxes, each field associates with one OCR detection, exact matches first, then fuzzy association. One detection cannot satisfy two fields. Overlapping field ROIs may require review. Complex vertical/curved layouts are not automatically inferred.

Shared fields broadcast across a batch. Different copy per image requires `{"images":[{"fields":[...]},{"fields":[...]}]}` with exactly one entry per IMAGE item, in matching order.

External OCR:

```json
{"detections":[
  {"id":"d0","text":"限时优患","confidence":0.98,"bbox":[20,20,300,70]},
  {"id":"d1","text":"¥1999","confidence":0.96,"polygon":[[20,110],[240,110],[240,170],[20,170]]}
]}
```

Also accepts `score`, `points`, and legacy RapidOCR `[polygon,text,confidence]` entries. Missing confidence means REVIEW. Detections without geometry can be compared as strings but cannot produce masks. Optional OCR `width`/`height` must match the source image. **OCR never broadcasts**: batch OCR requires `{"images":[{"detections":[...]},{"detections":[...]}]}` with exactly one entry per source image. Coordinates, duplicate IDs, finite numbers and confidence ranges are checked before acceptance.

## Acceptance and mask policy

- `min_confidence=0.75`: low or unknown confidence means REVIEW, regardless of whether recognized text matches. Local OCR defaults to capture threshold 0.05 to retain uncertain candidates.
- `match_threshold=0.55` controls association only; fuzzy similarity never means PASS. Use boxes for short text, prices and fixed layouts.
- `kind=text` supports optional whitespace/punctuation/case normalization. `brand`, `number`, `price`, `spec` remain strict, including case, decimal separators, units, currency symbols and spaces. Numeric signatures prevent prose punctuation normalization from treating `19.9` as `199`. A mismatch caused solely by whitespace introduced while grouping multiple OCR detections becomes REVIEW because segmentation cannot prove incorrect image typography.
- Missing detections default to `missing_policy=review`: OCR may have missed existing text. Set `fail` explicitly for a business policy requiring every field to be read.
- `unexpected_policy=review/ignore/fail` controls unmatched detections. Low-confidence extra text stays REVIEW, including under `fail` policy.
- Error MASK contains **both FAIL and REVIEW** regions. PASS regions are black. Padding 0 retains OCR polygons; positive padding expands their bounding rectangles.
- Errors are located to whole OCR lines/regions. Character diff offsets refer to strings, **not pixel coordinates**. Undetected copy is localized only with a supplied expected bbox. Without one, `located=false` and its mask is empty. **An empty mask is not a PASS. Always check verdict.**
- Overlays: red FAIL, amber REVIEW, green PASS. ASCII labels index the report's region list. Full Unicode text and reasons remain in JSON. Images are only annotated; no automatic repair or content replacement occurs.

## Examples and tests

Copy `examples/copyproof_demo.png` into ComfyUI's `input/`. Send `examples/api_external_ocr.json` or `examples/api_local_ocr.json` as `{"prompt": <example JSON>}` to `POST /prompt`. These are API prompts, not canvas UI workflow files.

Expected price is `$199`; the demo image contains `$1999`. The external-OCR example uses synthetic detections for reproducible validation; it is explicitly not evidence of real recognition. The local-OCR example performs actual recognition.

```bash
python -m pytest -q tests --rootdir=.. --import-mode=importlib
python examples/generate_demo.py
```

The demo generator creates PNG input/overlay/mask, a report and API prompts without OCR execution or model downloads. Tests exercise Chinese typos, strict brand/numeric comparisons, confidence review, missing-region constraints, duplicate reservation, batch mapping, geometry rejection, mask coverage and modern/legacy OCR output compatibility.

The explicit pytest root avoids importing a hyphenated custom-node directory as a top-level `__init__`. To verify actual OCR with provisioned weights: `python examples/smoke_local_ocr.py --model-root /existing/rapidocr/models`. It asserts the demo price is a real recognized mismatch and saves evidence to `examples/real_ocr_run`. Ordinary unit tests do not execute OCR or download weights.

ComfyUI V1 API is supported; ComfyUI core is not modified. Small, stylized, embossed, heavily perspective-distorted, vertical, occluded or rotated text can require human review. OCR confidence is not a calibrated correctness probability. External JSON is user-supplied evidence whose authenticity cannot be established here. Wire verdict/report into your own save and repair logic; CopyProof does not guarantee OCR correctness or repair success.
