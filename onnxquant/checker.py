"""Structural validation for the supported ONNX subset (opset 17)."""

from __future__ import annotations

from onnxquant.proto import FLOAT, INT8, Model, Tensor

ALLOWED_OPS = frozenset(
    {"MatMul", "Gemm", "Identity", "Transpose", "Add", "Relu"}
)
QUANTIZED_OPS = ALLOWED_OPS | {"DequantizeLinear"}

_DTYPE_SIZE = {FLOAT: 4, INT8: 1, 6: 4, 7: 8}


def _check_tensor(tensor: Tensor) -> None:
    if not tensor.name:
        raise ValueError("initializer with empty name")
    if any(d <= 0 for d in tensor.dims):
        raise ValueError(f"initializer {tensor.name!r} has non-positive dims")
    count = 1
    for d in tensor.dims:
        count *= d
    if tensor.raw_data is not None:
        expected = count * _DTYPE_SIZE.get(tensor.data_type, 0)
        if expected and len(tensor.raw_data) != expected:
            raise ValueError(
                f"initializer {tensor.name!r}: raw_data has {len(tensor.raw_data)} "
                f"bytes, expected {expected}"
            )
    elif tensor.float_data is not None:
        if len(tensor.float_data) != count:
            raise ValueError(
                f"initializer {tensor.name!r}: float_data has "
                f"{len(tensor.float_data)} elements, expected {count}"
            )
    else:
        raise ValueError(f"initializer {tensor.name!r} carries no data")


def check_model(model: Model, *, quantized: bool = False) -> None:
    """Raise ValueError if the model violates the supported subset."""
    opsets = dict(model.opset_imports)
    default_opset = opsets.get("", opsets.get("ai.onnx"))
    if default_opset != 17:
        raise ValueError(f"expected default-domain opset 17, got {default_opset}")

    graph = model.graph
    if graph is None:
        raise ValueError("model has no graph")
    if not graph.outputs:
        raise ValueError("graph has no outputs")

    allowed = QUANTIZED_OPS if quantized else ALLOWED_OPS
    for node in graph.nodes:
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"node {node.name!r} uses unsupported domain {node.domain!r}")
        if node.op_type not in allowed:
            raise ValueError(f"unsupported op {node.op_type!r}")

    init_names = set()
    for tensor in graph.initializers:
        if tensor.name in init_names:
            raise ValueError(f"duplicate initializer {tensor.name!r}")
        init_names.add(tensor.name)
        _check_tensor(tensor)

    available = set(init_names)
    for vi in graph.inputs:
        if vi.name in available:
            # An initializer listed as graph input is an overridable default.
            continue
        available.add(vi.name)
    for node in graph.nodes:
        for name in node.inputs:
            if name and name not in available:
                raise ValueError(
                    f"node {node.name or node.op_type!r}: input {name!r} is not "
                    "produced earlier nor declared"
                )
        for name in node.outputs:
            if name in available:
                raise ValueError(f"duplicate tensor name {name!r}")
            available.add(name)
    for vi in graph.outputs:
        if vi.name not in available:
            raise ValueError(f"graph output {vi.name!r} is never produced")
