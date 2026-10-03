"""Miscellaneous behavior: skipped operands, mixed uses, attribute fidelity."""

import numpy as np
import onnxruntime as ort
import pytest

from onnx_weight_quant import quantize_model_bytes
from onnx_weight_quant._codec import decode_model, encode_model, tensor_to_array
from onnx_weight_quant._model import (
    ATTR_FLOAT,
    ATTR_INT,
    ATTR_INTS,
    TENSOR_FLOAT,
    TENSOR_INT8,
    Attribute,
    Node,
)

from conftest import (
    float_tensor,
    initializers_by_name,
    make_model,
    reference_dequantize,
    reference_quantize_per_column,
    run_model,
    vi,
)


def test_dynamic_right_operand_untouched():
    model = make_model(
        nodes=[Node(op_type="MatMul", inputs=["X", "D"], outputs=["Y"], name="mm")],
        initializers=[],
        inputs=[vi("X", ["batch", 2]), vi("D", [2, 2])],
        outputs=[vi("Y", ["batch", 2])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert report.quantized_uses == []
    out_model = decode_model(out_bytes)
    assert [n.op_type for n in out_model.graph.nodes] == ["MatMul"]
    assert out_model.graph.initializers == []


def test_non_2d_initializer_right_operand_untouched():
    w3 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = make_model(
        nodes=[Node(op_type="MatMul", inputs=["X", "W3"], outputs=["Y"], name="mm")],
        initializers=[float_tensor("W3", w3)],
        inputs=[vi("X", [2, 3])],
        outputs=[vi("Y", [2, 2, 4])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert report.quantized_uses == []
    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    assert inits["W3"].dtype == TENSOR_FLOAT
    x = np.ones((2, 3), dtype=np.float32)
    (y,) = run_model(out_bytes, {"X": x})
    assert y.shape == (2, 2, 4)


def test_weight_shared_with_non_target_use_keeps_fp32():
    # W feeds both a MatMul (quantized) and an Add (non-target): the FP32
    # initializer must stay for the Add, bit-identical.
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    model = make_model(
        nodes=[
            Node(op_type="MatMul", inputs=["X", "W"], outputs=["mm"], name="mm"),
            Node(op_type="Add", inputs=["W", "W"], outputs=["S"], name="add"),
        ],
        initializers=[float_tensor("W", w)],
        inputs=[vi("X", ["batch", 2])],
        outputs=[vi("mm", ["batch", 2]), vi("S", [2, 2])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert len(report.quantized_uses) == 1
    assert report.removed_initializers == []
    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    assert inits["W"].dtype == TENSOR_FLOAT
    assert tensor_to_array(inits["W"]).tobytes() == w.tobytes()
    x = np.eye(2, dtype=np.float32)
    mm, s = run_model(out_bytes, {"X": x})
    assert s.tobytes() == (w + w).tobytes()
    q_ref, s_ref = reference_quantize_per_column(w)
    np.testing.assert_allclose(mm, x @ reference_dequantize(q_ref, s_ref),
                               rtol=1e-6, atol=1e-7)


def test_gemm_without_transpose_and_attributes_preserved():
    w = np.array([[2.0, 1.0], [0.5, -1.5]], dtype=np.float32)
    c = np.array([0.25, -0.25], dtype=np.float32)
    model = make_model(
        nodes=[Node(op_type="Gemm", inputs=["X", "W", "C"], outputs=["Y"],
                    name="gemm",
                    attributes={
                        "alpha": Attribute("alpha", ATTR_FLOAT, 2.5),
                        "beta": Attribute("beta", ATTR_FLOAT, 0.5),
                    })],
        initializers=[float_tensor("W", w), float_tensor("C", c)],
        inputs=[vi("X", ["batch", 2])],
        outputs=[vi("Y", ["batch", 2])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert len(report.quantized_uses) == 1
    out_model = decode_model(out_bytes)
    gemm = next(n for n in out_model.graph.nodes if n.op_type == "Gemm")
    assert gemm.attributes["alpha"].value == np.float32(2.5)
    assert gemm.attributes["beta"].value == np.float32(0.5)
    assert "transB" not in gemm.attributes
    x = np.array([[1.0, -1.0]], dtype=np.float32)
    (y,) = run_model(out_bytes, {"X": x})
    q_ref, s_ref = reference_quantize_per_column(w)
    w_ref = reference_dequantize(q_ref, s_ref)
    np.testing.assert_allclose(y, 2.5 * (x @ w_ref) + 0.5 * c,
                               rtol=1e-6, atol=1e-6)


def test_transpose_chain_without_perm_attribute():
    # A Transpose without perm reverses dims: for 2-D that is a transpose.
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    model = make_model(
        nodes=[
            Node(op_type="Transpose", inputs=["W"], outputs=["Wt"], name="t"),
            Node(op_type="MatMul", inputs=["X", "Wt"], outputs=["Y"], name="mm"),
        ],
        initializers=[float_tensor("W", w)],
        inputs=[vi("X", ["batch", 3])],
        outputs=[vi("Y", ["batch", 2])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert len(report.quantized_uses) == 1
    assert report.quantized_uses[0].transposed is True
    x = np.ones((2, 3), dtype=np.float32)
    (y,) = run_model(out_bytes, {"X": x})
    q_ref, s_ref = reference_quantize_per_column(w.T)
    np.testing.assert_allclose(y, x @ reference_dequantize(q_ref, s_ref),
                               rtol=1e-6, atol=1e-6)


def test_identity_and_relu_pipeline_runs():
    w = np.array([[0.5, -0.5], [1.0, 2.0]], dtype=np.float32)
    model = make_model(
        nodes=[
            Node(op_type="Identity", inputs=["W"], outputs=["wi"], name="id"),
            Node(op_type="MatMul", inputs=["X", "wi"], outputs=["mm"], name="mm"),
            Node(op_type="Relu", inputs=["mm"], outputs=["Y"], name="relu"),
        ],
        initializers=[float_tensor("W", w)],
        inputs=[vi("X", ["batch", 2])],
        outputs=[vi("Y", ["batch", 2])],
    )
    out_bytes, report = quantize_model_bytes(encode_model(model))
    assert len(report.quantized_uses) == 1
    # The alias chain lost its purpose and must be gone with the FP32 copy.
    assert report.removed_initializers == ["W"]
    assert report.removed_nodes == ["id"]
    out_model = decode_model(out_bytes)
    assert "W" not in initializers_by_name(out_model)
    x = np.array([[1.0, 1.0]], dtype=np.float32)
    (y,) = run_model(out_bytes, {"X": x})
    q_ref, s_ref = reference_quantize_per_column(w)
    expected = np.maximum(x @ reference_dequantize(q_ref, s_ref), 0)
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


def test_non_finite_weight_skipped():
    w = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float32)
    model = make_model(
        nodes=[Node(op_type="MatMul", inputs=["X", "W"], outputs=["Y"], name="mm")],
        initializers=[float_tensor("W", w)],
        inputs=[vi("X", ["batch", 2])],
        outputs=[vi("Y", ["batch", 2])],
    )
    _, report = quantize_model_bytes(encode_model(model))
    assert report.quantized_uses == []
