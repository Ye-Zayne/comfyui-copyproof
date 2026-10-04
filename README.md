# ComfyUI CopyProof

**按给定文案验收商业图片，定位需要人工复核或局部返工的文字区域。** 本地中英 OCR 可选；外部 OCR JSON 验收不需要 OCR 模型、API 或 LLM 密钥。

![演示：价格 199 被错误生成成 1999](examples/copyproof_overlay.png)

这是验收工具，不承诺 OCR 绝对正确、自动修复成功或识别视觉上的每一个字形。English documentation: [README.en.md](README.en.md).

## 安装

把整个目录放到 `ComfyUI/custom_nodes/comfyui-copyproof`，重启 ComfyUI。基础验收节点使用 ComfyUI 自带的 `torch`、`numpy`、`Pillow`，没有额外必装依赖。

需要直接读图片里的中文字或英文时，在 **ComfyUI 的 Python 环境**安装：

```bash
python -m pip install -r custom_nodes/comfyui-copyproof/requirements-ocr.txt
```

Windows portable 使用其 `python_embeded/python.exe`。没有安装 RapidOCR 时，ComfyUI 仍能启动，外部 JSON 验收仍能使用；只有运行 Local OCR 节点才提示依赖安装步骤。支持现代 `rapidocr` 3.x，以及已安装的旧 `rapidocr_onnxruntime` 输出适配。首版推荐 CPU，兼容 Windows、macOS、Linux；CUDA 需要用户自行配置匹配的 `onnxruntime-gpu`，缺少 CUDA provider 时会明确报错。

RapidOCR 默认识别中文及英文。现代 RapidOCR 的模型目录默认是 `ComfyUI/models/copyproof/rapidocr`，也可在 `model_root` 指定已有模型目录。模型是否已随安装包提供取决于 RapidOCR 版本；首次运行可能由 RapidOCR 下载其默认 OCR 权重。离线环境请预先准备与安装版本匹配的完整模型文件。旧 `rapidocr_onnxruntime` 使用其自身的模型设置，不读取 `model_root`。

## 两个节点

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| `CopyProofRapidOCR` / Local OCR (中英) | IMAGE 批次、cpu/cuda、capture_confidence、model_root | 标准 `ocr_json` STRING |
| `CopyProofValidate` / Validate Expected Copy | IMAGE、expected_json、ocr_json、验收策略 | `error_mask` MASK、`overlay` IMAGE、`report_json` STRING、`verdict` INT |

`verdict`：**0 = FAIL，1 = REVIEW，2 = PASS**。批次总体只要任一图片 FAIL 就 FAIL，否则只要任一图片 REVIEW 就 REVIEW；每张图的结果在报告中。验收节点本身是 OUTPUT_NODE，可单独运行，也可接 Save Image。前端文本报告展示依 ComfyUI 版本而异，STRING 输出始终可用。

## 期望文案格式

单张图片或整个批次共享同一文案：

```json
{
  "fields": [
    {"id": "title", "text": "限时优惠", "kind": "text", "bbox": [20, 20, 300, 90]},
    {"id": "brand", "text": "ACME", "kind": "brand"},
    {"id": "price", "text": "¥199", "kind": "price", "bbox": [20, 110, 240, 180]}
  ]
}
```

也支持 `["限时优惠", "新品上市"]` 或字段对象列表。`id` 可省略，但同图内不能重复。`bbox` 可省略；有 bbox 时，将中心点位于框内的 OCR 区域按从上到下、从左到右合并，再验收字段。没有 bbox 时，一个字段对应一个 OCR 检测区域；不会猜测多行文案、竖排、弧形字的复杂排版。

坐标均为**输入图片像素**，`[x1,y1,x2,y2]`，右/下边界不包含，不能超出图片；没有隐式缩放或归一化。以上框需要按自己的图片调整。多张图使用不同文案时：

```json
{"images":[
  {"fields":[{"id":"sku","text":"A001","kind":"brand"}]},
  {"fields":[{"id":"sku","text":"A002","kind":"brand"}]}
]}
```

`images` 必须与 IMAGE 批次数量完全一致、顺序相同。字段没有空间约束时，先保留精确匹配，再进行模糊关联；同一个 OCR 检测不能被两个字段重复消费。重叠字段框可能产生待复核结果，建议给每个字段不重叠的区域。

## 外部 OCR JSON

接已有 OCR 节点时，把其数据适配成以下格式即可，不必再安装或运行 RapidOCR：

```json
{"detections":[
  {"id":"line_1","text":"限时优患","confidence":0.98,"bbox":[20,20,300,70]},
  {"id":"line_2","text":"¥1999","confidence":0.96,"polygon":[[20,110],[240,110],[240,170],[20,170]]}
]}
```

支持 `confidence` / `score`，`polygon` / `points`，以及旧 RapidOCR 的 `[polygon,text,confidence]` 条目。没有置信度时必须 REVIEW。没有几何位置的检测仍可做文字验收，但不能生成对应区域掩膜。非空文本、有限数字、置信度 0–1、有效坐标会在验收前检查。

**OCR 结果永不广播到多张图**，批次请提供：

```json
{"images":[
  {"width":640,"height":360,"detections":[]},
  {"width":640,"height":360,"detections":[]}
]}
```

可选 `width`/`height` 必须与输入图片一致，防止把缩略图坐标误用到原图。

## 策略与返工

- `min_confidence=0.75`：低于阈值或没有置信度，正确或错误文字都 REVIEW。Local OCR 的 `capture_confidence=0.05` 用来保留低置信度候选，宜低于验收阈值。
- `match_threshold=0.55` **只控制无 bbox 时把哪一行关联到哪个字段**，模糊匹配绝不代表通过。提供 bbox 更适合短字、价格和固定版式。
- `kind=text` 可选择忽略空白、标点或英文大小写；`brand`、`number`、`price`、`spec` 总是严格比较，包括品牌大小写、数字小数点、单位、货币符号及空格。即使普通文字忽略标点，也不会把 `19.9` 当成 `199`。多检测区域合并产生的空白如果是唯一差异，返回 REVIEW，因为 OCR 分段不能证明图片排版错误。
- 未检测到字段默认 `missing_policy=review`：OCR 漏检不能证明图片真没字。若业务要求“未读到即拒绝”，可显式设置 `fail`。
- 多余文字按 `unexpected_policy=review/ignore/fail` 处理；低置信度多余文字不会被判定为确定失败。
- `error_mask` 白色选中 FAIL **及 REVIEW** 区域，接 inpaint 前请阅读报告。PASS 区域黑色。`mask_padding` 在错误区域外扩像素；设 0 时保留 OCR polygon，否则使用外扩的包围框。
- 错字位置仅定位至 OCR **整行/区域**；报告中字符 diff 的 offsets 是字符串索引，不是图像像素。未检测到字段只能用用户提供的 bbox；没 bbox 时该字段 `located=false`，掩膜为空。**空掩膜不代表 PASS**，必须同时检查 verdict。
- 红框 FAIL、黄框 REVIEW、绿框 PASS。图上 ASCII 编号按报告 `regions` 的顺序，文字内容和理由保留在 JSON。框选掩膜只提供返工位置，不会自动改动图片内容。

## 示例与验证

`examples/api_external_ocr.json` 是无需 OCR 的 ComfyUI API prompt；`examples/api_local_ocr.json` 从真实图片运行 RapidOCR。先把 `examples/copyproof_demo.png` 复制进 ComfyUI 的 `input/`，再通过 `POST /prompt` 发送 `{"prompt": <示例 JSON 内容>}`。API prompt 与画布 UI workflow 是不同格式，不适合直接当画布工作流拖入。

演示期望价格 `$199`，图片与外部检测是 `$1999`，应 FAIL；例子中的外部检测是合成数据，**用于可复现的验收演示，不能代替真实 OCR 结果**。Local OCR 示例才会实际识别输入图片。

```bash
python -m pytest -q tests --rootdir=.. --import-mode=importlib
python examples/generate_demo.py
```

测试覆盖中文错字、数字/品牌严格性、低置信度、漏检定位、重复字段防复用、多图映射、非法坐标、掩膜面积及 OCR 新旧输出适配。`generate_demo.py` 生成演示图、掩膜、报告和两个 API 示例，不运行 OCR 或下载模型。

测试使用 `--rootdir=..` 是为了避免 pytest 将含连字符的节点目录误当成顶层 `__init__` 导入。真实 OCR 冒烟验证可在准备好 ONNX 模型后执行 `python examples/smoke_local_ocr.py --model-root /已有的/rapidocr/models`；它验证演示图中的价格错误被真实识别并定位。该命令保存实际识别证据到 `examples/real_ocr_run`，普通单元测试不运行 OCR 或下载权重。

## 范围

此包兼容 ComfyUI V1 custom node API，不修改 ComfyUI core。适合可读的商业标题、Logo 文案、价格、规格及 SKU 标签；浮雕、严重透视、花体、小字号、竖排、旋转和遮挡可能需要人工复核。OCR confidence 不是经过校准的“正确概率”。外部 JSON 是用户提供的识别证据，无法验证其真实性。任何自动保存/返工流程都应按报告与 verdict 连接自己的逻辑节点。
