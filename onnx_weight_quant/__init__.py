"""Offline ONNX weight quantization to per-channel symmetric INT8.

Reads an ONNX model (opset 17, MatMul/Gemm/Identity/Transpose/Add/Relu,
embedded tensors), quantizes the right operands of MatMul and Gemm nodes
that resolve to initializer weights, and writes a model that keeps FP32
inputs/activations/outputs while storing those weights as INT8 + scales.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._codec import decode_model, encode_model
from ._transform import QuantizedUse, TransformReport
from ._transform import transform_model

__all__ = ["quantize_model_file", "quantize_model_bytes", "QuantizeResult"]
__version__ = "0.1.0"


@dataclass
class QuantizeResult(TransformReport):
    """TransformReport plus the output model size in bytes."""

    output_size: int = 0


def quantize_model_bytes(model_bytes: bytes) -> tuple[bytes, QuantizeResult]:
    """Quantize a serialized ONNX model; return (new_bytes, report)."""
    model = decode_model(model_bytes)
    report = transform_model(model)
    out = encode_model(model)
    result = QuantizeResult(
        quantized_uses=report.quantized_uses,
        removed_initializers=report.removed_initializers,
        removed_nodes=report.removed_nodes,
        output_size=len(out),
    )
    return out, result


def quantize_model_file(input_path: str, output_path: str) -> QuantizeResult:
    """Quantize the model at input_path and write the result to output_path."""
    with open(input_path, "rb") as fh:
        model_bytes = fh.read()
    out, result = quantize_model_bytes(model_bytes)
    with open(output_path, "wb") as fh:
        fh.write(out)
    return result
