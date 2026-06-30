# 黑背景检测模块设计 (`dark_bg_detector`)

> 实现代码: [inferencer/dark_bg_detector.py](../../inferencer/dark_bg_detector.py) | 模块文档: [docs/inferencer/dark_bg_detector.md](../inferencer/dark_bg_detector.md)

## 1. 目的

独立的人脸图片黑背景检测模块，对外提供 HTTP 端点 `/dark_bg_check`，用于判断拍摄环境背景是否过暗。

## 2. 检测流程（两阶段）

```
输入图片 (BGR)
       │
       ▼
┌──────────────────────────────┐
│  Stage 1: 光照质量模型推理     │
│  (weights-finetune-1-40--A)  │
│                              │
│  输出 4 类分数 → 检查 class 2 │
│  ┌─ argmax ≠ 2 ──────► 非黑背景 │
│  └─ argmax = 2, 分数 ≥ 0.95 │
│         ↓ (进入 Stage 2)      │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Stage 2: 背景暗像素占比       │
│                              │
│  1. 获取人脸 bbox (YOLO)      │
│     └─ 无人脸 → resp=300      │
│                              │
│  2. 扩展 bbox + clip 边界     │
│     (上10%/其他0%)            │
│                              │
│  3. 背景区域 = 全图 - 扩展bbox │
│     (脸部下方区域排除，避免衣物) │
│     └─ 背景像素=0?            │
│        → 返回 sentinel -1.0    │
│           → 按 Stage1 结果输出  │
│     └─ 暗像素占比 > 75%?      │
│        ↓ 是/否                │
│  输出: is_dark_bg             │
└──────────────────────────────┘

最终判定:
is_dark_bg = (无人脸 → False, resp=300)
           | (stage1未通过 → False)
           | (bg_ratio == -1.0 → stage1结果) [sentinel: 无背景]
           | (否则 → stage2结果)
```

## 3. 各个阶段详述

### Stage 1: 模型推理

- **模型**: `weights-finetune-1-40--A.onnx`（已有 `ImgQuality` 封装）
- **输入**: 整张图片, resize 到 320×240
- **输出**: 4 个类别的分数分布
- **判定条件**: `argmax == 2` 且 `class_2_score >= 0.95`
- **作用**: 粗筛——模型认为画面整体偏暗才有必要进入 Stage 2。大量正常光照图在此阶段直接排除,节省算力

### Stage 2: 背景暗像素占比

**背景区域定义**: 整张图减去（扩展后的）人脸 bbox 覆盖区域。

**最终判定**: `is_dark_bg = Stage1 通过 AND Stage2 通过`。Stage2 各分支逻辑见下。

- **步骤 1 - 获取人脸区域**: 复用 YOLO face detector（不跑 dlib alignment,只拿 bbox）。无人脸时返回 `is_dark_bg=False, resp_code=300`
- **步骤 2 - 扩展人脸区域**: 将 bbox 各方向扩展，避免把头发、额头误判为背景：
  - 上方向：扩展 10%
  - 左/右/下方向：扩展 0%
  - 扩展后将坐标 clip 到图片边界内（避免越界）
- **步骤 3 - 暗像素统计**:
  - 图像转灰度
  - 背景区域构造：mask 中全图=255，人脸 bbox 区域=0，脸部下方区域=0（排除衣物干扰）
  - 背景区域（全图 — 扩展后的人脸 bbox — 脸部下方）中，亮度 < `dark_pixel_thresh` 的像素计为"暗像素"
  - 若背景像素数为 0（人脸 bbox 覆盖全图）→ 无背景可判，返回 sentinel -1.0，`is_dark_bg` 按 Stage1 结果输出
  - 若暗像素数 / 背景区域总像素数 > `dark_ratio_thresh`，判定为黑背景
- **作用**: 精确检查——确认「画面暗」不是因为「人脸本身暗」（如肤色深、逆光人脸），而是因为「背景区域黑」

### 判定逻辑总结

```
if 无人脸:
    is_dark_bg = False, resp_code = 300
elif Stage1 未通过:
    is_dark_bg = False, resp_code = 100
elif bg_ratio == -1.0 (sentinel, 背景像素数 == 0):
    is_dark_bg = True, resp_code = 100   (依赖 Stage1 已通过)
else:
    is_dark_bg = Stage2 结果, resp_code = 100
```

## 4. 配置项 (`config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dark_bg.model_threshold` | 0.95 | Stage1 模型 class 2 最低置信度 |
| `dark_bg.dark_pixel_thresh` | 50 | 灰度值阈值 (0-255)，低于此值视为暗像素 |
| `dark_bg.dark_ratio_thresh` | 0.75 | 背景暗像素占比阈值 |
| `dark_bg.bbox_expand_ratio` | 0.0 | 人脸 bbox 左/右/下方向扩展比例 |
| `dark_bg.bbox_expand_up_ratio` | 0.1 | 人脸 bbox 上方向扩展比例 |

## 5. 接口规格

### POST `/dark_bg_check`

**请求:**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `recordID` | string | 是 | 请求记录 ID |
| `sessionID` | string | 是 | 会话 ID |
| `msgID` | string | 是 | 消息 ID |
| `imgData` | string | 是 | Base64 编码图片 |

**响应:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `recordID` | string | 回显 |
| `sessionID` | string | 回显 |
| `msgID` | string | 回显 |
| `resp_code` | int | 100=检测正常完成, 300=无人脸, 999=服务异常 |
| `is_dark_bg` | bool | 是否黑背景 |
| `dark_score` | float | Stage1 模型 class 2 的得分 |
| `bg_ratio` | float | Stage2 背景暗像素占比 |

## 6. 边界情况处理

| 场景 | 行为 |
|------|------|
| Stage1 argmax ≠ 2 | 直接返 is_dark_bg=False, resp_code=100 |
| Stage1 class 2 分数 < 0.95 | 直接返 is_dark_bg=False, resp_code=100 |
| 图片中无人脸 | 返 is_dark_bg=False, resp_code=300 |
| 有人脸但背景像素数为 0（人脸 bbox 覆盖全图）| bg_ratio=-1.0 (sentinel), is_dark_bg 按 Stage1 结果输出 |
| 图片解码失败/异常 | resp_code=999,返回 error_msg |

## 7. 模块依赖

- `inferencer/image_quality.py` — `ImgQuality` 类（Stage1 模型推理）
- `inferencer/face_detector.py` — `Detector.get_face_bboxes()` 方法（Stage2 人脸定位,需新增）
- `inferencer/face_pipeline.py` — `StatusCode` 枚举（复用 `StatusCode.NOFACE`）
- `config.yaml` — 阈值配置读取

## 8. 文件改动概要

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `inferencer/face_detector.py` | 新增方法 | `get_face_bboxes()` — 只跑 YOLO 返回人脸坐标 |
| `inferencer/dark_bg_detector.py` | 新增文件 | 两阶段检测逻辑 |
| `server.py` | 新增端点 | `/dark_bg_check` HTTP 接口 |
| `config.yaml` | 新增配置 | `dark_bg.*` 阈值参数 |

## 9. 与 face_pipeline 的关系

- `/dark_bg_check` 是完全独立的 HTTP 端点，不被 `face_pipeline` 内部调用，也不消费 `face_register_` / `head_detection_` 的任何结果
- 复用 `ImgQuality` 和 `Detector` 两个底层类，但各持独立实例（不共享 `face_pipeline` 的模块级单例）
- 无人脸场景下复用 `StatusCode.NOFACE`（300），仅此一处符号级依赖

## 10. 与原有代码的隔离

- 此模块不修改 `face_pipeline.py`、`face_register_`、`head_detection_` 的任何逻辑
- 使用独立的模型实例和配置命名空间
