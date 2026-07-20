# 权益确认人脸识别服务 API 文档

## 1. 基本信息

| 项目 | 说明 |
|------|------|
| 协议 | HTTP/1.1 |
| Content-Type | `application/json` |
| 请求方式 | 除 `/version/` 为 GET 外，其余均为 **POST** |
| 端口 | dev: **37709** / stg: **80** / pro: **80** |
| 图片编码 | 所有图片字段为 **Base64 编码字符串**，不含 data URI 前缀 |

---

## 2. 公共请求字段

所有 POST 接口的 JSON 请求体中均需包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `recordID` | String | 是 | 业务流水号，唯一标识本次请求，原样返回 |
| `sessionID` | String | 是 | 会话标识，同一个用户的同一个流程（注册→检测）应保持一致 |
| `msgID` | String | 是 | 消息标识，用于链路追踪 |
| `imgData` | String | 是 | 图片的 Base64 编码（不含 `data:image/xxx;base64,` 前缀） |
| `imgType` | String | 否 | 图片格式，如 `"JPG"`、`"PNG"`（服务端自动识别，不依赖此字段） |
| `faceFeature` | String | 视接口 | **仅 `/head_detection` 必填**，为人脸特征向量字符串，来自 `/face_register` 的返回值 |

---

## 3. 公共响应字段

所有 POST 接口的成功响应中均包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `recordID` | String | 与请求一致，原样返回 |
| `sessionID` | String | 与请求一致，原样返回 |
| `msgID` | String | 与请求一致，原样返回 |
| `resp_code` | Integer | 状态码，详见 [状态码表](#4-状态码表) |

异常时返回：

| 字段 | 类型 | 说明 |
|------|------|------|
| `resp_code` | Integer | 固定为 `999` |
| `error_msg` | String | 异常描述 |

---

## 4. 状态码表

| 状态码 | 常量名 | 含义 |
|:------:|--------|------|
| **100** | `SUCCESS` | 成功 |
| **200** | `BLURRED` | 图片模糊，提示"不要晃动手机" |
| **201** | `BACKLIGHT` | 逆光，提示"摄像头不要正对灯光" |
| **221** | `BLURRED_BACKLIGHT` | 模糊 + 逆光 |
| **300** | `NOFACE` | 未检测到人脸，提示"看不到您的脸" |
| **400** | `MULTIFACE` | 检测到多张人脸 |
| **500** | `NO_FEATURE` | 未提供人脸特征，或特征长度不足 |
| **600** | `FEATURE_EXTRACT_ERROR` | 特征提取失败 |
| **999** | `ERROR` | 服务内部异常 |

---

## 5. 接口详情

### 5.1 人脸注册 — `/face_register`

**接口用途：** 上传一张包含单张正面人脸的图片，提取人脸特征向量，供后续人脸比对使用。

**业务流程：** 用户拍照上传 → 提取人脸特征 → 返回特征向量。

#### 请求

```
POST /face_register
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `recordID` | String | 是 | 业务流水号 |
| `sessionID` | String | 是 | 会话标识 |
| `msgID` | String | 是 | 消息标识 |
| `imgData` | String | 是 | 人脸图片 Base64（建议单张正面清晰人脸） |
| `imgType` | String | 否 | 图片格式，如 `"JPG"` |

#### 成功响应（resp_code = 100）

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 100,
  "imgQualityScore": 0.95,
  "faceFeature": "[0.123, -0.456, 0.789, ...]"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `imgQualityScore` | Float | 图片质量评分，范围 0~1，越高越好 |
| `faceFeature` | String | **人脸特征向量**（JSON 数组的字符串形式），长度为 512 维。后续调用 `/head_detection` 时原样传入 |

#### 异常响应

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 300
}
```

- 未提取到人脸特征时，`resp_code` 非 100，且响应中 **不含** `faceFeature` 字段。

---

### 5.2 人脸对比 — `/head_detection`

**接口用途：** 用摄像头实时帧与已注册的人脸特征做比对，判断是否为本人，并返回人脸检测结果和提示信息。

**前置条件：** 必须先调用 `/face_register` 获取 `faceFeature`。

**调用频率：** 每次调用传入一帧图片，客户端通常每秒连续调用 1~3 次做实时检测。

#### 请求

```
POST /head_detection
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `recordID` | String | 是 | 业务流水号 |
| `sessionID` | String | 是 | 会话标识，应与注册时一致 |
| `msgID` | String | 是 | 消息标识 |
| `imgData` | String | 是 | 摄像头帧图片 Base64 |
| `imgType` | String | 否 | 图片格式，如 `"JPG"` |
| `faceFeature` | String | **是** | 来自 `/face_register` 的 faceFeature，原样传入 |

#### 成功响应（resp_code = 100）

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 100,
  "detectRes": 1,
  "top3Similarity": [0.92, 0.88, 0.85],
  "faceSimilarity": [0.92],
  "msg": ""
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `detectRes` | Integer | 检测到的人脸数量，正常为 `1` |
| `top3Similarity` | Float[] | 检测到的每张人脸的置信度，数组长度 = detectRes（通常取第一个值） |
| `faceSimilarity` | Float[] | **注册人脸与当前帧人脸的相似度**，范围 0~1。值越高越像。`0` 表示不匹配。`[-1]` 表示特征缺失、无法比对。取第一个元素 `faceSimilarity[0]` 即为主比对结果 |
| `msg` | String | 提示文案，正常检测通过时为空字符串 `""` |

#### 非成功响应示例

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 200,
  "detectRes": 0,
  "top3Similarity": [],
  "faceSimilarity": [-1],
  "msg": "不要晃动手机"
}
```

| resp_code | msg | 含义 |
|:---------:|-----|------|
| 200 | `"不要晃动手机"` | 图片模糊 |
| 201 | `"摄像头不要正对灯光"` | 逆光 |
| 221 | — | 既模糊又逆光 |
| 300 | `"看不到您的脸"` | 未检测到人脸 |
| 400 | — | 检测到多张人脸 |
| 500 | — | 未提供 faceFeature 或长度不足 |
| 600 | `"看不到您的脸"` | 特征提取失败 |

> **注意：** 当 `resp_code` 非 100 时，`faceSimilarity` 为 `[-1]`，表示本次检测未产生有效的比对结果，业务方应根据 `resp_code` 和 `msg` 提示用户重新拍摄。

---

### 5.3 暗背景检测 — `/dark_bg_check`

**接口用途：** 检测用户拍照时背景是否过暗，用于在注册环节提示用户改善拍摄环境。

#### 请求

```
POST /dark_bg_check
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `recordID` | String | 是 | 业务流水号 |
| `sessionID` | String | 是 | 会话标识 |
| `msgID` | String | 是 | 消息标识 |
| `imgData` | String | 是 | 人脸图片 Base64 |
| `imgType` | String | 否 | 图片格式，如 `"JPG"` |

#### 成功响应（resp_code = 100）

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 100,
  "is_dark_bg": true,
  "dark_score": 0.8542,
  "bg_ratio": 0.9580
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_dark_bg` | Boolean | **是否为暗背景**。`true` = 背景过暗，建议提醒用户换到明亮处；`false` = 背景正常 |
| `dark_score` | Float | 最终背景昏暗程度评分（0~1，保留 4 位小数），越高越暗。仅当 `is_dark_bg=true` 时有意义，基于背景非暗区的灰度均值取反计算；该字段不是 Stage 1 模型置信度 |
| `bg_ratio` | Float | 背景区域中暗像素（灰度 < 30）的占比（0~1，保留 4 位小数） |

#### 正常非暗背景响应（resp_code = 100）

若 Stage 1 模型认为图片不是暗场景，将不再执行人脸检测和 Stage 2 分析，直接返回：

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 100,
  "is_dark_bg": false,
  "dark_score": 0.0,
  "bg_ratio": 0.0
}
```

#### 未检测到人脸响应（resp_code = 300）

仅当 Stage 1 模型判定为暗场景、但 Stage 2 未检测到人脸时返回 `300`：

```json
{
  "recordID": "test-20250424test",
  "sessionID": "test-HJ5Loo5546xxxxxx",
  "msgID": "test-hshgsuhguhbjh2356vjijh",
  "resp_code": 300,
  "is_dark_bg": false,
  "dark_score": 0.0,
  "bg_ratio": 0.0
}
```

#### 服务异常响应（resp_code = 999）

请求字段缺失、图片 Base64 无效、图片解码失败或服务内部处理异常时返回：

```json
{
  "resp_code": 999,
  "error_msg": "异常描述",
  "is_dark_bg": false
}
```

异常响应不保证包含 `recordID`、`sessionID`、`msgID`、`dark_score` 和 `bg_ratio`。

#### 检测逻辑说明

检测分两阶段：

1. **模型判断**（Stage 1）：用深度学习模型判断整体画面是否为暗场景。若模型认为非暗场景，直接返回 `resp_code=100, is_dark_bg=false`，不执行人脸检测。
2. **CV 连通域分析**（Stage 2）：若模型判定为暗场景，先定位人脸；未检测到人脸时返回 `resp_code=300, is_dark_bg=false`。检测到人脸后，对背景像素做连通域分析——将灰度 > 60 的像素按面积分为两类：大面积连通域（≥500px，如玻璃、座椅）和小光斑（<500px，如反光点）。需同时满足**三个条件**才最终判定为暗背景：① 暗像素占比 > 90% ② 大面积亮区域占比 < 2%（拦截白天车内玻璃/灰座椅场景）③ 小光斑占比 < 10%（容忍夜间反光点）。通过后再基于背景非暗区灰度均值计算 `dark_score`。

---

### 5.4 版本查询 — `/version/`

**接口用途：** 查询服务的版本信息、部署时间、进程数，用于运维监控和健康检查。

#### 请求

```
GET /version/
```
无需请求体。

#### 响应

响应为纯文本（`text/plain`），格式如下：

```
[IP]: 10.x.x.x: [APP]: Wed Jun 25 16:00:00 2026: [CODE]: v2.3.1
: [gunicorn]: 6
```

| 字段 | 说明 |
|------|------|
| `[IP]` | 服务器内网 IP 地址 |
| `[APP]` | 服务启动时间 |
| `[CODE]` | 代码版本号，对应 `version` 文件内容 |
| `[gunicorn]` | 当前 Gunicorn Worker 进程数 |

---

## 6. 典型调用流程

```
┌─────────┐   ┌────────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ 客户端   │   │ /dark_bg_check │   │ /face_register  │   │ /head_detection  │
└────┬────┘   └───────┬────────┘   └───────┬─────────┘   └────────┬─────────┘
     │                │                    │                        │
     │ 1.(可选)拍照   │                    │                        │
     │───────────────>│                    │                        │
     │                │                    │                        │
     │ 2.is_dark_bg   │                    │                        │
     │   若true→提示  │                    │                        │
     │<───────────────│                    │                        │
     │                │                    │                        │
     │ 3.用户拍照     │                    │                        │
     │────────────────────────────────────>│                        │
     │                │                    │                        │
     │ 4.faceFeature  │                    │                        │
     │<────────────────────────────────────│                        │
     │                │                    │                        │
     │ 5.faceFeature  │                    │                        │
     │   +实时帧      │                    │                        │
     │──────────────────────────────────────────────────────────────>│
     │                │                    │                        │
     │ 6.faceSimilarity│                   │                        │
     │   +msg         │                    │                        │
     │<──────────────────────────────────────────────────────────────│
     │                │                    │                        │
     │ 7.重复5-6，    │                    │                        │
     │   直到通过/超时│                    │                        │
     │                │                    │                        │
```

1. **（可选）暗背景检测：** 调用 `/dark_bg_check` 检查拍摄环境亮度，如 `is_dark_bg = true` 则提示用户改善光照后重新拍摄。
2. **注册：** 调用 `/face_register`，传入用户正面照片，获取 `faceFeature`。
3. **检测：** 连续调用 `/head_detection`，每帧传入 `faceFeature` + 摄像头实时帧，判断 `faceSimilarity` 是否达标，同时关注 `msg` 引导用户调整动作。
4. **判断通过：** 当 `resp_code = 100` 且 `faceSimilarity[0]` 大于业务阈值时，视为人脸对比通过。

---

## 7. 注意事项

### 图片要求
- 图片需包含**清晰完整的人脸**，正面、无遮挡。
- 建议分辨率不低于 640×480。
- Base64 编码**不含** `data:image/xxx;base64,` 前缀。

### faceFeature 传递
- `/face_register` 返回的 `faceFeature` 是 JSON 数组的字符串形式，如 `"[0.1, 0.2, ...]"`。
- 调用 `/head_detection` 时**原样传入**，不要做任何编码或格式转换。

### 超时与重试
- 服务端 Gunicorn 配置了 1000s 超时（`-t 1000`），但建议客户端设置 10~15s 的连接/读取超时。
- 遇到 `resp_code = 999` 或网络错误时可重试，但建议间隔 1s 以上。

### 并发
- Gunicorn Worker 数量：dev=2, stg=4, pro=6。请根据环境合理控制并发请求数。

### msg 字段
- `/head_detection` 的 `msg` 字段可直接展示给用户，用于引导用户调整拍摄姿势。
- 仅在 `resp_code = 100` 时 `msg` 为空字符串。
