# onnx-weight-quant

离线 ONNX 权重量化库：把计算图中 MatMul / Gemm 的右操作数权重按
**每个输出通道独立的对称 INT8** 量化，缩小模型文件，同时保持
FP32 的输入、激活与输出语义不变。

## 适用范围

- 输入：opset 17 的单层 ONNX 计算图，仅含 `MatMul`、`Gemm`、`Identity`、
  `Transpose`、`Add`、`Relu`，所有张量数据内嵌（raw_data 或 typed data）。
- 待量化权重：数值有限的非空二维 FP32 矩阵。
- 量化对象仅为 `MatMul` / `Gemm` 的右操作数。它可以直接来自初始化权重，
  也可以经过任意长度的 `Identity` / `Transpose` 链。满足条件的使用点
  全部转换；动态右操作数保持原样。
- 若源权重同时列在图输入中（调用方可覆盖的默认值），其所有使用点都
  不量化。

## 量化规则

- MatMul 以实际右矩阵的列为通道；Gemm 以应用 `transB` 后的矩阵列为通道。
- 每通道尺度 `scale = max(|w|) / 127`（FP32 计算），下限为最小正规
  FP32 正数；全零通道尺度固定为 1。
- 商按 FP32 计算，舍入到最近整数（恰好居中取偶数），截到 [-127, 127]，
  零点固定为 0。
- 量化后语义：先以 `int8 * scale` 恢复 FP32 权重（图中插入
  `DequantizeLinear(axis=1)` 节点），再沿用原计算。Gemm 的
  `transB` / `alpha` / `beta` 属性原样保留。
- 公开输入输出的名字、类型、形状声明不变；仍向图输出或非目标使用点
  供给数据的 FP32 权重及其别名路径逐位保持原值。
- 只服务于已量化使用点的 FP32 权重副本和失去用途的别名节点会从输出
  文件中移除（图输出也算使用点）。

## 使用

```python
from onnx_weight_quant import quantize_model_file

report = quantize_model_file("model.onnx", "model.int8.onnx")
for use in report.quantized_uses:
    print(use.consumer_node, use.op_type, use.source_initializer)
print("removed:", report.removed_initializers, report.removed_nodes)
```

也支持字节级 API：`quantize_model_bytes(model_bytes) -> (bytes, report)`。

输出模型可直接由 onnxruntime 在 CPU 上执行：

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.int8.onnx", providers=["CPUExecutionProvider"])
```

## 依赖与测试

- 运行依赖：Python >= 3.10、numpy。库本身自包含 ONNX 编解码
  （`onnx_weight_quant/_codec.py`），不强制依赖 `onnx` 包。
- 测试依赖：pytest、onnxruntime（CPU）。若环境中安装了 `onnx` 包，
  测试会额外执行 `onnx.checker.check_model` 格式检查，否则自动跳过。

```bash
python -m pytest tests/ -q
```

## 代码结构

- `onnx_weight_quant/_wire.py` — protobuf 线格式原语（varint/fixed/长度分隔）。
- `onnx_weight_quant/_model.py` — ONNX 子集的内存表示（dataclass）。
- `onnx_weight_quant/_codec.py` — ModelProto 子集的解码/编码。
- `onnx_weight_quant/_quantize.py` — 每通道对称 INT8 量化数值逻辑。
- `onnx_weight_quant/_transform.py` — 图分析（别名链回溯）与重写、死代码回收。
- `tests/` — 数值规则、共享权重多向使用、全零通道/半整数舍入/
  可覆盖默认权重等场景的自动化测试。
