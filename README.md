# onnxquant — 离线 ONNX 权重量化库

面向边缘设备的离线 INT8 权重量化工具：缩小模型文件，同时保持 FP32 的
输入、激活与输出语义不变。由上层优化任务以本地文件形式调用。

## 功能

- 输入：opset 17 的单层 ONNX 计算图，仅含 `MatMul`、`Gemm`、`Identity`、
  `Transpose`、`Add`、`Relu`，所有张量数据内嵌（`raw_data` / `float_data`）。
- 仅量化 `MatMul` / `Gemm` 的右操作数；操作数可直接来自初始化权重，
  也可经过任意长度的 `Identity` / `Transpose` 别名链。
- 按输出通道做对称 INT8 量化，零点固定为 0：
  - `MatMul` 以实际右矩阵的列为通道；`Gemm` 以应用 `transB` 后的矩阵列为通道；
  - 每通道尺度 = 通道最大绝对值 / 127（FP32 计算），下限为最小正规 FP32
    正数；全零通道尺度固定为 1；
  - 商按 FP32 计算，舍入到最近整数（居中取偶），截断到 [-127, 127]。
- 量化语义：先用 INT8 值乘对应尺度恢复 FP32 权重（图中插入
  `DequantizeLinear`），再沿用原计算。公开输入输出的名字、类型、形状
  声明不变；`Gemm` 的 `transB` / `alpha` / `beta` 属性原样保留。
- 仍向图输出或非目标使用点供给数据的原始 FP32 权重及其别名路径保持
  原值，逐位不变。
- 只服务于已量化使用点的 FP32 权重副本与失去用途的别名节点会被移除
  （图输出也算使用点）。
- 同时列在图输入中的权重属于调用方可覆盖的默认值，其所有使用点都不
  会被量化；其他动态右操作数保持原样。

## 环境

- Python 3.13+，仅依赖 `numpy`（运行库本身）。
- 测试额外需要 `onnxruntime`（CPU）。
- 不依赖 `onnx` 包：库内置了 ONNX protobuf 线格式的最小编解码器
  （`onnxquant/proto.py`）。

## 用法

命令行：

```bash
PYTHONPATH=. .venv/bin/python -m onnxquant input.onnx output.onnx
```

Python API：

```python
from onnxquant import quantize_file

report = quantize_file("input.onnx", "output.onnx")
print(report.quantized_use_points)   # 已量化的节点名
print(report.removed_initializers)   # 被移除的 FP32 权重
print(report.removed_nodes)          # 被移除的别名节点
```

输出模型通过内置结构检查（`onnxquant/checker.py`），并可直接由
onnxruntime 在 CPU 上执行：

```python
import onnxruntime as ort
sess = ort.InferenceSession("output.onnx", providers=["CPUExecutionProvider"])
```

## 代码结构

- `onnxquant/proto.py` — ONNX protobuf 线格式最小编解码（无 `onnx` 依赖）
- `onnxquant/quantize.py` — 每通道对称 INT8 量化数值规则（FP32 计算）
- `onnxquant/transform.py` — 图分析、量化改写与死代码清理
- `onnxquant/checker.py` — 受支持子集的结构校验
- `onnxquant/cli.py` — 命令行入口
- `tests/test_quantizer.py` — 自动化测试

## 运行测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

覆盖场景：

- 同一权重经多层别名链，以两个通道方向参与 `MatMul` 和 `Gemm`，同时
  作为图输出：量化结果与按数值规则独立计算的参考一致，原始权重输出
  逐位不变；
- 全零通道、正负半整数舍入（居中取偶）、可覆盖默认权重：检查量化整数
  与尺度，调用方替换默认权重后输出反映替换值；
- 三维动态批次 `MatMul`、动态右操作数保持原样、权重完全消费后的
  FP32 副本与别名节点清理、非法模型拒绝等。
