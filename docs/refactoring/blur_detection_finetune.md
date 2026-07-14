# Blur Detection Finetune 计划

**目标**：在自己的数据上微调现有 ONNX 模糊检测模型，提升特定场景准确率。

---

## 第一步：还原模型

- 用 Netron 打开 `blur_0727.onnx`，查看网络结构，确认 backbone（大概率是 MobileNetV2/V3 或 ResNet18）
- 确认输入层：`(1, 3, 240, 320)` NCHW、归一化参数是否匹配代码中的 `mean=0.5, std=0.5`
- 输出层：2 类 logits，无 softmax（代码中自行计算 softmax）
- 在 PyTorch 中搭建完全一致的结构，确保 forward 与 ONNX 逐层对应

## 第二步：准备数据

- 正样本（sharp）：高清图 + 业务中正确判定为清晰的图像
- 负样本（blur）：三种方式混合——高斯模糊、运动模糊、失焦模糊，辅以业务中误判的模糊图像
- 比例 1:1，建议 5000+ 张
- 使用与推理完全一致的预处理 pipeline（`Resize(320,240)` → `ToTensor` → `Normalize(0.5, 0.5)`）

## 第三步：训练策略

1. 恢复权重：如能获取原始 PyTorch 权重直接加载；否则从 ImageNet 预训练加载
2. 第一阶段：冻结 backbone，仅训练分类头 5-10 epoch，lr 1e-4，避免初期破坏已学特征
3. 第二阶段：解冻全量，用小学习率 1e-6 再训 10-20 epoch，适配目标数据分布
4. 验证：留 20% 数据做 validation set，监控准确率，防止过拟合

## 第四步：导出 ONNX

- `model.eval()` 后执行 `torch.onnx.export`
- 输入/输出 name 与现网 ONNX 保持一致（`input` / `output`），确保推理代码无需改动
- opset 版本使用 11，兼顾兼容性

## 第五步：回灌测试

- 用新 ONNX 替换 `blur_0727.onnx`
- 选取一批典型图像跑一遍，对比新旧模型的分数分布，确认方向正确后再全量上线

## 关键注意事项

- 预处理不可变：Resize 尺寸、归一化参数、PIL 转换逻辑必须与当前推理完全一致
- 数据类型：ONNX 输入为 float32，导出时注意 dummy input 的 dtype
- 若无原始 PyTorch 权重，直接从零训练亦可，但数据量需更大（1w+），backbone 选用 MobileNetV2 即可满足需求
