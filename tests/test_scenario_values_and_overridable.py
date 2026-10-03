"""Scenario 2: zero channels, tie rounding, overridable default weights."""

import numpy as np
import onnxruntime as ort
import pytest

from onnx_weight_quant import quantize_model_bytes, quantize_model_file
from onnx_weight_quant._codec import decode_model, encode_model, tensor_to_array
from onnx_weight_quant._model import TENSOR_FLOAT, TENSOR_INT8, Node

from conftest import (
    float_tensor,
    initializers_by_name,
    make_model,
    reference_quantize_per_column,
    run_model,
    vi,
)

# Column 0: all zeros -> scale 1, all-zero int8.
# Column 1: max_abs 254 -> scale 2.0, quotients hit exact +/-x.5 ties.
# Column 2: ordinary values.
W_VALUES = np.array(
    [
        [0.0, 1.0, 30.0],
        [0.0, -1.0, -60.0],
        [0.0, 3.0, 90.0],
        [0.0, -3.0, -120.0],
        [0.0, 5.0, 127.0],
        [0.0, -5.0, -127.0],
        [0.0, 254.0, 0.0],
    ],
    dtype=np.float32,
)


def build_matmul_model(weight, weight_name="W", weight_is_input=False):
    nodes = [Node(op_type="MatMul", inputs=["X", weight_name], outputs=["Y"],
                  name="mm")]
    inputs = [vi("X", ["batch", weight.shape[0]])]
    if weight_is_input:
        inputs.append(vi(weight_name, list(weight.shape)))
    return make_model(
        nodes=nodes,
        initializers=[float_tensor(weight_name, weight)],
        inputs=inputs,
        outputs=[vi("Y", ["batch", weight.shape[1]])],
    )


def test_zero_channel_and_tie_rounding_values():
    model = build_matmul_model(W_VALUES)
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert len(report.quantized_uses) == 1
    use = report.quantized_uses[0]

    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    q = tensor_to_array(inits[use.q_initializer])
    s = tensor_to_array(inits[use.scale_initializer])

    q_ref, s_ref = reference_quantize_per_column(W_VALUES)
    assert inits[use.q_initializer].dtype == TENSOR_INT8
    assert np.array_equal(q, q_ref)
    assert np.array_equal(s, s_ref)

    # Explicit expectations for the tricky columns.
    assert s[0] == 1.0 and (q[:, 0] == 0).all()
    assert s[1] == 2.0
    # +/-0.5 -> 0, +/-1.5 -> +/-2, +/-2.5 -> +/-2 (ties to even)
    assert q[:, 1].tolist() == [0, 0, 2, -2, 2, -2, 127]

    # The FP32 copy served only the quantized use: it must be gone.
    assert "W" not in inits
    assert report.removed_initializers == ["W"]

    # Runtime output matches the independent reference dequantization.
    x = np.arange(14, dtype=np.float32).reshape(2, 7) / 7.0
    (y,) = run_model(out_bytes, {"X": x})
    w_ref = q_ref.astype(np.float32) * s_ref
    np.testing.assert_allclose(y, x @ w_ref, rtol=1e-6, atol=1e-6)


def test_overridable_default_weight_is_never_quantized():
    model = build_matmul_model(W_VALUES, weight_is_input=True)
    out_bytes, report = quantize_model_bytes(encode_model(model))

    # No use point of an overridable default may be quantized.
    assert report.quantized_uses == []
    assert report.removed_initializers == []
    assert report.removed_nodes == []

    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    assert inits["W"].dtype == TENSOR_FLOAT
    assert tensor_to_array(inits["W"]).tobytes() == W_VALUES.tobytes()
    assert all(t.dtype != TENSOR_INT8 for t in out_model.graph.initializers)
    assert [n.op_type for n in out_model.graph.nodes] == ["MatMul"]

    # Default value is used when the caller does not override it.
    x = np.ones((2, 7), dtype=np.float32)
    (y_default,) = run_model(out_bytes, {"X": x})
    np.testing.assert_allclose(y_default, x @ W_VALUES, rtol=1e-6, atol=1e-6)

    # Replacing the default must be reflected in the output.
    replacement = np.full((7, 3), 2.0, dtype=np.float32)
    (y_override,) = run_model(out_bytes, {"X": x, "W": replacement})
    np.testing.assert_allclose(y_override, x @ replacement, rtol=1e-6, atol=1e-6)
    assert not np.allclose(y_override, y_default)


def test_overridable_default_through_alias_chain_not_quantized():
    nodes = [
        Node(op_type="Identity", inputs=["W"], outputs=["wid"], name="id"),
        Node(op_type="MatMul", inputs=["X", "wid"], outputs=["Y"], name="mm"),
    ]
    model = make_model(
        nodes=nodes,
        initializers=[float_tensor("W", W_VALUES)],
        inputs=[vi("X", ["batch", 7]), vi("W", [7, 3])],
        outputs=[vi("Y", ["batch", 3])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert report.quantized_uses == []
    out_model = decode_model(out_bytes)
    assert [n.op_type for n in out_model.graph.nodes] == ["Identity", "MatMul"]


def test_file_roundtrip(tmp_path):
    model = build_matmul_model(W_VALUES)
    src = tmp_path / "in.onnx"
    dst = tmp_path / "out.onnx"
    src.write_bytes(encode_model(model))
    report = quantize_model_file(str(src), str(dst))
    assert report.output_size == dst.stat().st_size
    ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    x = np.ones((1, 7), dtype=np.float32)
    (y,) = run_model(dst.read_bytes(), {"X": x})
    q_ref, s_ref = reference_quantize_per_column(W_VALUES)
    np.testing.assert_allclose(y, x @ (q_ref.astype(np.float32) * s_ref),
                               rtol=1e-6, atol=1e-6)
