"""Automated tests for the onnxquant library.

Run from the repository root:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnxruntime as ort

from onnxquant import proto
from onnxquant.checker import check_model
from onnxquant.quantize import FLOAT32_MIN_NORMAL, quantize_matrix
from onnxquant.transform import quantize_file


def make_tensor(name: str, array: np.ndarray) -> proto.Tensor:
    return proto.Tensor(
        name=name,
        dims=list(array.shape),
        data_type=proto.FLOAT,
        raw_data=array.astype(np.float32).tobytes(),
    )


def vi(name: str, shape) -> proto.ValueInfo:
    return proto.ValueInfo(name=name, elem_type=proto.FLOAT, shape=list(shape))


def save_and_quantize(model: proto.Model, tmp: str):
    in_path = os.path.join(tmp, "in.onnx")
    out_path = os.path.join(tmp, "out.onnx")
    proto.save_model(model, in_path)
    report = quantize_file(in_path, out_path)
    return out_path, report


def run(path: str, feeds: dict, output_names=None):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(output_names, feeds)


def initializers_by_name(model: proto.Model):
    return {t.name: t for t in model.graph.initializers}


def tensor_array(tensor: proto.Tensor) -> np.ndarray:
    dtype = {proto.FLOAT: np.float32, proto.INT8: np.int8}[tensor.data_type]
    return np.frombuffer(tensor.raw_data, dtype=dtype).reshape(tensor.dims)


class AliasChainTest(unittest.TestCase):
    """One weight, multi-hop alias chain, two channel directions across
    MatMul and Gemm, and the raw weight also exposed as a graph output."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        self.W = (rng.standard_normal((2, 3)) * 5).astype(np.float32)
        self.X = rng.standard_normal((4, 2)).astype(np.float32)
        self.A = rng.standard_normal((4, 3)).astype(np.float32)
        self.C = rng.standard_normal((2,)).astype(np.float32)

        model = proto.Model(
            ir_version=8,
            producer_name="test",
            graph=proto.Graph(
                name="g",
                nodes=[
                    proto.Node(op_type="Identity", name="id1",
                               inputs=["W"], outputs=["Wa"]),
                    proto.Node(op_type="Transpose", name="tp",
                               inputs=["Wa"], outputs=["Wt"],
                               attributes=[proto.Attribute(
                                   name="perm", type=proto.ATTR_INTS,
                                   ints=[1, 0])]),
                    proto.Node(op_type="Identity", name="id2",
                               inputs=["Wt"], outputs=["Wta"]),
                    # channels = columns of W
                    proto.Node(op_type="MatMul", name="mm",
                               inputs=["X", "Wa"], outputs=["Y1"]),
                    # transB=0 -> channels = columns of Wt = rows of W
                    proto.Node(op_type="Gemm", name="gemm",
                               inputs=["A", "Wta", "C"], outputs=["Y2"],
                               attributes=[
                                   proto.Attribute(name="alpha",
                                                   type=proto.ATTR_FLOAT,
                                                   f=0.5),
                                   proto.Attribute(name="beta",
                                                   type=proto.ATTR_FLOAT,
                                                   f=2.0),
                               ]),
                ],
                initializers=[make_tensor("W", self.W),
                              make_tensor("C", self.C)],
                inputs=[vi("X", [4, 2]), vi("A", [4, 3])],
                outputs=[vi("Y1", [4, 3]), vi("Y2", [4, 2]),
                         vi("W", [2, 3])],
            ),
            opset_imports=[("", 17)],
        )
        self.out_path, self.report = save_and_quantize(model, self.tmp)
        self.out_model = proto.load_model(self.out_path)

    def test_both_use_points_quantized(self):
        self.assertEqual(sorted(self.report.quantized_use_points),
                         ["gemm", "mm"])
        dq_nodes = [n for n in self.out_model.graph.nodes
                    if n.op_type == "DequantizeLinear"]
        self.assertEqual(len(dq_nodes), 2)

    def test_fp32_weight_kept_for_graph_output(self):
        inits = initializers_by_name(self.out_model)
        self.assertIn("W", inits)
        self.assertEqual(inits["W"].data_type, proto.FLOAT)
        self.assertEqual(inits["W"].raw_data, self.W.tobytes())

    def test_dead_alias_nodes_removed(self):
        names = {n.name for n in self.out_model.graph.nodes}
        self.assertNotIn("id1", names)
        self.assertNotIn("tp", names)
        self.assertNotIn("id2", names)

    def test_gemm_attributes_preserved(self):
        gemm = next(n for n in self.out_model.graph.nodes
                    if n.op_type == "Gemm")
        self.assertEqual(gemm.get_attribute("alpha"), np.float32(0.5))
        self.assertEqual(gemm.get_attribute("beta"), np.float32(2.0))
        self.assertIsNone(gemm.get_attribute("transB"))

    def test_outputs_match_reference(self):
        y1, y2, w_out = run(self.out_path, {"X": self.X, "A": self.A})

        q1, s1 = quantize_matrix(self.W, channel_axis=1)
        ref1 = self.X @ (q1.astype(np.float32) * s1[None, :])
        np.testing.assert_allclose(y1, ref1, rtol=1e-5, atol=1e-5)

        w_t = np.ascontiguousarray(self.W.T)
        q2, s2 = quantize_matrix(w_t, channel_axis=1)
        deq2 = q2.astype(np.float32) * s2[None, :]
        ref2 = 0.5 * (self.A @ deq2) + 2.0 * self.C
        np.testing.assert_allclose(y2, ref2, rtol=1e-5, atol=1e-5)

    def test_graph_output_weight_bitwise_identical(self):
        _, _, w_out = run(self.out_path, {"X": self.X, "A": self.A})
        self.assertEqual(w_out.tobytes(), self.W.tobytes())

    def test_output_passes_checker(self):
        check_model(self.out_model, quantized=True)


class NumericsAndOverrideTest(unittest.TestCase):
    """All-zero channel, +/- half-integer rounding, overridable defaults."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        col0 = np.zeros(7, dtype=np.float32)
        col1 = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0],
                        dtype=np.float32)
        col2 = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32)
        self.W = np.stack([col0, col1, col2], axis=1)  # (7, 3)
        self.W_default = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.X = np.ones((2, 7), dtype=np.float32)
        self.X2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        model = proto.Model(
            ir_version=8,
            producer_name="test",
            graph=proto.Graph(
                name="g",
                nodes=[
                    proto.Node(op_type="MatMul", name="mm",
                               inputs=["X", "W"], outputs=["Y"]),
                    proto.Node(op_type="MatMul", name="mm_default",
                               inputs=["X2", "W_default"], outputs=["Y2"]),
                ],
                initializers=[make_tensor("W", self.W),
                              make_tensor("W_default", self.W_default)],
                inputs=[vi("X", [2, 7]), vi("X2", [2, 2]),
                        vi("W_default", [2, 2])],
                outputs=[vi("Y", [2, 3]), vi("Y2", [2, 2])],
            ),
            opset_imports=[("", 17)],
        )
        self.out_path, self.report = save_and_quantize(model, self.tmp)
        self.out_model = proto.load_model(self.out_path)

    def test_only_non_overridable_weight_quantized(self):
        self.assertEqual(self.report.quantized_use_points, ["mm"])
        dq = [n for n in self.out_model.graph.nodes
              if n.op_type == "DequantizeLinear"]
        self.assertEqual(len(dq), 1)

    def test_quantized_ints_and_scales(self):
        inits = initializers_by_name(self.out_model)
        q = tensor_array(inits["W__q"])
        s = tensor_array(inits["W__scale"])
        zp = tensor_array(inits["W__zp"])

        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(s.dtype, np.float32)
        # all-zero channel: scale fixed to 1, ints all zero
        self.assertEqual(s[0], np.float32(1.0))
        self.assertTrue(np.all(q[:, 0] == 0))
        # half-integer ties-to-even rounding with scale == 1
        self.assertEqual(s[1], np.float32(1.0))
        self.assertEqual(q[:, 1].tolist(), [0, 2, 2, 0, -2, -2, 127])
        # regular channel
        self.assertEqual(s[2], np.float32(np.float32(70.0) / np.float32(127.0)))
        self.assertEqual(q[6, 2], 127)
        self.assertTrue(np.all(zp == 0))

    def test_overridable_default_untouched(self):
        inits = initializers_by_name(self.out_model)
        self.assertIn("W_default", inits)
        self.assertEqual(inits["W_default"].data_type, proto.FLOAT)
        self.assertEqual(inits["W_default"].raw_data,
                         self.W_default.tobytes())
        mm_default = next(n for n in self.out_model.graph.nodes
                          if n.name == "mm_default")
        self.assertEqual(mm_default.inputs[1], "W_default")

    def test_caller_override_takes_effect(self):
        replacement = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        _, y2 = run(self.out_path,
                    {"X": self.X, "X2": self.X2, "W_default": replacement})
        np.testing.assert_array_equal(y2, self.X2 @ replacement)

    def test_default_value_used_without_override(self):
        _, y2 = run(self.out_path, {"X": self.X, "X2": self.X2})
        np.testing.assert_array_equal(y2, self.X2 @ self.W_default)

    def test_output_matches_reference(self):
        y, _ = run(self.out_path, {"X": self.X, "X2": self.X2})
        q, s = quantize_matrix(self.W, channel_axis=1)
        ref = self.X @ (q.astype(np.float32) * s[None, :])
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


class EdgeCasesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_3d_dynamic_batch_matmul(self):
        rng = np.random.default_rng(7)
        W = rng.standard_normal((3, 2)).astype(np.float32)
        model = proto.Model(
            ir_version=8, producer_name="test",
            graph=proto.Graph(
                name="g",
                nodes=[proto.Node(op_type="MatMul", name="mm",
                                  inputs=["X", "W"], outputs=["Y"])],
                initializers=[make_tensor("W", W)],
                inputs=[vi("X", ["batch", 4, 3])],
                outputs=[vi("Y", ["batch", 4, 2])],
            ),
            opset_imports=[("", 17)],
        )
        out_path, report = save_and_quantize(model, self.tmp)
        self.assertEqual(report.quantized_use_points, ["mm"])
        q, s = quantize_matrix(W, channel_axis=1)
        deq = q.astype(np.float32) * s[None, :]
        for batch in (1, 5):
            X = rng.standard_normal((batch, 4, 3)).astype(np.float32)
            (y,) = run(out_path, {"X": X})
            np.testing.assert_allclose(y, X @ deq, rtol=1e-5, atol=1e-5)

    def test_dynamic_right_operand_untouched(self):
        model = proto.Model(
            ir_version=8, producer_name="test",
            graph=proto.Graph(
                name="g",
                nodes=[proto.Node(op_type="MatMul", name="mm",
                                  inputs=["X", "B"], outputs=["Y"])],
                initializers=[],
                inputs=[vi("X", [2, 3]), vi("B", [3, 2])],
                outputs=[vi("Y", [2, 2])],
            ),
            opset_imports=[("", 17)],
        )
        out_path, report = save_and_quantize(model, self.tmp)
        self.assertEqual(report.quantized_use_points, [])
        out_model = proto.load_model(out_path)
        self.assertEqual([n.op_type for n in out_model.graph.nodes],
                         ["MatMul"])

    def test_fully_consumed_weight_removed(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((2, 2)).astype(np.float32)
        model = proto.Model(
            ir_version=8, producer_name="test",
            graph=proto.Graph(
                name="g",
                nodes=[
                    proto.Node(op_type="Identity", name="id1",
                               inputs=["W"], outputs=["Wa"]),
                    proto.Node(op_type="MatMul", name="mm",
                               inputs=["X", "Wa"], outputs=["Y"]),
                ],
                initializers=[make_tensor("W", W)],
                inputs=[vi("X", [2, 2])],
                outputs=[vi("Y", [2, 2])],
            ),
            opset_imports=[("", 17)],
        )
        out_path, report = save_and_quantize(model, self.tmp)
        self.assertIn("W", report.removed_initializers)
        self.assertIn("id1", report.removed_nodes)
        out_model = proto.load_model(out_path)
        self.assertNotIn("W", initializers_by_name(out_model))
        # file size shrinks: int8 + scales instead of fp32 copy
        X = rng.standard_normal((2, 2)).astype(np.float32)
        (y,) = run(out_path, {"X": X})
        q, s = quantize_matrix(W, channel_axis=1)
        np.testing.assert_allclose(y, X @ (q.astype(np.float32) * s[None, :]),
                                   rtol=1e-5, atol=1e-6)

    def test_tiny_channel_scale_floor(self):
        W = np.array([[1e-45, 1.0], [0.0, 2.0]], dtype=np.float32)
        q, s = quantize_matrix(W, channel_axis=1)
        self.assertEqual(s[0], FLOAT32_MIN_NORMAL)

    def test_checker_rejects_bad_models(self):
        base = dict(ir_version=8, producer_name="t",
                    graph=proto.Graph(name="g", nodes=[], initializers=[],
                                      inputs=[], outputs=[vi("Y", [1])]),
                    opset_imports=[("", 17)])
        bad_opset = proto.Model(opset_imports=[("", 16)],
                                **{k: v for k, v in base.items()
                                   if k != "opset_imports"})
        with self.assertRaises(ValueError):
            check_model(bad_opset)
        bad_op = proto.Model(
            **{**base,
               "graph": proto.Graph(
                   name="g",
                   nodes=[proto.Node(op_type="Conv", inputs=["X"],
                                     outputs=["Y"])],
                   initializers=[], inputs=[vi("X", [1])],
                   outputs=[vi("Y", [1])])})
        with self.assertRaises(ValueError):
            check_model(bad_op)


if __name__ == "__main__":
    unittest.main()
