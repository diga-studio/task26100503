"""Command line interface: python -m onnxquant <input.onnx> <output.onnx>."""

from __future__ import annotations

import argparse
import sys

from onnxquant.transform import quantize_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="onnxquant",
        description="Offline per-channel symmetric INT8 weight quantization "
        "for single-graph opset-17 ONNX models (MatMul/Gemm right operands).",
    )
    parser.add_argument("input", help="path to the input .onnx model")
    parser.add_argument("output", help="path for the quantized .onnx model")
    args = parser.parse_args(argv)
    try:
        report = quantize_file(args.input, args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"quantized use points: {len(report.quantized_use_points)}")
    for name in report.quantized_use_points:
        print(f"  - {name}")
    if report.removed_initializers:
        print(f"removed initializers: {', '.join(report.removed_initializers)}")
    if report.removed_nodes:
        print(f"removed nodes: {', '.join(report.removed_nodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
