"""Scenario 1: one weight, alias chains, MatMul + Gemm, graph outputs.

Graph under test (W is a 3x4 FP32 initializer, not a graph input):

    W -- Identity(idA) -- Transpose(tA) --+--> MatMul(X1, tA_out) --> Y1
       |                    (perm 1,0)    |
       |                                  +--> graph output tA_out
       |
       +-- Transpose(tB) -- Identity(idB) --> Gemm(X2, idB_out, C,
       |        (perm 1,0)                     transB=1, alpha, beta) --> Y2
       |
       +-- graph output W_out

MatMul sees W.T (channels = columns of W.T); Gemm with transB=1 sees
(W.T).T == W (channels = columns of W). Both uses must be quantized, the
FP32 W and the alias path feeding graph outputs must stay bit-identical,
and the now-purposeless chain B nodes must be removed.
"""

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

W_VALUES = np.array(
    [
        [1.5, -2.5, 0.0, 4.0],
        [-0.75, 3.25, 2.0, -1.0],
        [0.5, 0.0, -3.5, 2.75],
    ],
    dtype=np.float32,
)  # 3x4
ALPHA = 0.75
BETA = 1.25
C_VALUES = np.array([0.25, -0.5, 1.0, 0.0], dtype=np.float32)


def build_model():
    nodes = [
        Node(op_type="Identity", inputs=["W"], outputs=["idA_out"], name="idA"),
        Node(op_type="Transpose", inputs=["idA_out"], outputs=["tA_out"],
             name="tA", attributes={"perm": Attribute("perm", ATTR_INTS, [1, 0])}),
        Node(op_type="Transpose", inputs=["W"], outputs=["tB_out"], name="tB",
             attributes={"perm": Attribute("perm", ATTR_INTS, [1, 0])}),
        Node(op_type="Identity", inputs=["tB_out"], outputs=["idB_out"], name="idB"),
        Node(op_type="MatMul", inputs=["X1", "tA_out"], outputs=["Y1"], name="mm"),
        Node(op_type="Gemm", inputs=["X2", "idB_out", "C"], outputs=["Y2"],
             name="gemm",
             attributes={
                 "transB": Attribute("transB", ATTR_INT, 1),
                 "alpha": Attribute("alpha", ATTR_FLOAT, ALPHA),
                 "beta": Attribute("beta", ATTR_FLOAT, BETA),
             }),
    ]
    return make_model(
        nodes=nodes,
        initializers=[float_tensor("W", W_VALUES), float_tensor("C", C_VALUES)],
        inputs=[vi("X1", ["batch", 2, 4]), vi("X2", ["batch", 3])],
        outputs=[
            vi("Y1", ["batch", 2, 3]),
            vi("Y2", ["batch", 4]),
            vi("W_out", [3, 4]),
            vi("tA_out", [4, 3]),
        ],
    )


@pytest.fixture(scope="module")
def quantized():
    model = build_model()
    # W_out is a graph output aliasing W: add an Identity so the output name
    # differs from the initializer name.
    model.graph.nodes.append(
        Node(op_type="Identity", inputs=["W"], outputs=["W_out"], name="w_out_id")
    )
    original_bytes = encode_model(model)
    out_bytes, report = quantize_model_bytes(original_bytes)
    return model, original_bytes, out_bytes, report


def test_both_use_points_quantized(quantized):
    _, _, _, report = quantized
    uses = {(u.consumer_node, u.op_type) for u in report.quantized_uses}
    assert uses == {("mm", "MatMul"), ("gemm", "Gemm")}


def test_public_signature_preserved(quantized):
    model, _, out_bytes, _ = quantized
    out_model = decode_model(out_bytes)
    assert [i.name for i in out_model.graph.inputs] == ["X1", "X2"]
    assert [o.name for o in out_model.graph.outputs] == ["Y1", "Y2", "W_out", "tA_out"]
    assert out_model.graph.inputs[0].shape == ["batch", 2, 4]
    assert out_model.graph.inputs[1].shape == ["batch", 3]
    assert out_model.graph.inputs[0].elem_type == TENSOR_FLOAT
    for out, shape in zip(out_model.graph.outputs,
                          [["batch", 2, 3], ["batch", 4], [3, 4], [4, 3]]):
        assert out.shape == shape
        assert out.elem_type == TENSOR_FLOAT


def test_fp32_weight_and_output_alias_path_kept(quantized):
    _, _, out_bytes, _ = quantized
    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    # W must remain as an FP32 initializer with identical values.
    assert "W" in inits
    assert inits["W"].dtype == TENSOR_FLOAT
    assert tensor_to_array(inits["W"]).tobytes() == W_VALUES.tobytes()
    # Chain A feeds graph output tA_out, so its alias nodes must stay.
    op_types = {n.name: n.op_type for n in out_model.graph.nodes}
    assert op_types["idA"] == "Identity"
    assert op_types["tA"] == "Transpose"
    assert op_types["w_out_id"] == "Identity"


def test_purposeless_alias_chain_removed(quantized):
    _, _, out_bytes, report = quantized
    out_model = decode_model(out_bytes)
    names = {n.name for n in out_model.graph.nodes}
    assert "tB" not in names and "idB" not in names
    assert set(report.removed_nodes) == {"tB", "idB"}
    # W is still used (graph output), so it must NOT be removed.
    assert report.removed_initializers == []


def test_quantized_tensors_and_scales_match_reference(quantized):
    _, _, out_bytes, report = quantized
    out_model = decode_model(out_bytes)
    inits = initializers_by_name(out_model)
    uses = {u.consumer_node: u for u in report.quantized_uses}

    # MatMul consumes W.T: channels are columns of W.T.
    q_ref, s_ref = reference_quantize_per_column(W_VALUES.T)
    mm = uses["mm"]
    assert inits[mm.q_initializer].dtype == TENSOR_INT8
    assert np.array_equal(tensor_to_array(inits[mm.q_initializer]), q_ref)
    assert np.array_equal(tensor_to_array(inits[mm.scale_initializer]), s_ref)

    # Gemm with transB=1 consumes (W.T).T == W: channels are columns of W.
    q_ref2, s_ref2 = reference_quantize_per_column(W_VALUES)
    gemm = uses["gemm"]
    assert np.array_equal(tensor_to_array(inits[gemm.q_initializer]), q_ref2)
    assert np.array_equal(tensor_to_array(inits[gemm.scale_initializer]), s_ref2)


def test_outputs_match_reference_and_originals_bitexact(quantized):
    _, original_bytes, out_bytes, _ = quantized
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=(5, 2, 4)).astype(np.float32)
    x2 = rng.normal(size=(5, 3)).astype(np.float32)
    feeds = {"X1": x1, "X2": x2}

    y1, y2, w_out, ta_out = run_model(out_bytes, feeds)

    # Independent reference: dequantize per the spec, then original math.
    q_mm, s_mm = reference_quantize_per_column(W_VALUES.T)
    w_mm = reference_dequantize(q_mm, s_mm)
    y1_ref = x1 @ w_mm
    q_g, s_g = reference_quantize_per_column(W_VALUES)
    w_g = reference_dequantize(q_g, s_g)
    y2_ref = ALPHA * (x2 @ w_g) + BETA * C_VALUES

    np.testing.assert_allclose(y1, y1_ref, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(y2, y2_ref, rtol=1e-5, atol=1e-6)

    # Original FP32 weight outputs must be bit-identical to the source model.
    assert w_out.tobytes() == W_VALUES.tobytes()
    assert ta_out.tobytes() == W_VALUES.T.tobytes()
    _, _, orig_w, orig_ta = run_model(original_bytes, feeds)
    assert w_out.tobytes() == orig_w.tobytes()
    assert ta_out.tobytes() == orig_ta.tobytes()


def test_dynamic_batch_dimension(quantized):
    _, _, out_bytes, _ = quantized
    for batch in (1, 3, 8):
        x1 = np.full((batch, 2, 4), 0.5, dtype=np.float32)
        x2 = np.full((batch, 3), 0.25, dtype=np.float32)
        y1, y2, _, _ = run_model(out_bytes, {"X1": x1, "X2": x2})
        assert y1.shape == (batch, 2, 3)
        assert y2.shape == (batch, 4)


def test_output_loads_in_onnxruntime_and_onnx_checker(quantized):
    _, _, out_bytes, _ = quantized
    # onnxruntime session creation is itself a strict format validation.
    ort.InferenceSession(out_bytes, providers=["CPUExecutionProvider"])
    onnx = pytest.importorskip("onnx", reason="onnx package not installed")
    onnx.checker.check_model(onnx.load_from_string(out_bytes))
