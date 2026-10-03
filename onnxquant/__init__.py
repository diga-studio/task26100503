"""Offline ONNX per-channel INT8 weight quantizer (opset 17).

Public API:
    quantize_file(input_path, output_path) -> QuantizeReport
"""

from onnxquant.transform import QuantizeReport, quantize_file

__all__ = ["quantize_file", "QuantizeReport"]
__version__ = "0.1.0"
