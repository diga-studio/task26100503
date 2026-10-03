"""Shared helpers: model builders, onnxruntime runner, independent reference."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pytest

from onnx_weight_quant._codec import (
    array_to_tensor,
    decode_model,
    encode_model,
    tensor_to_array,
)
from onnx_weight_quant._model import (
    ATTR_FLOAT,
    ATTR_INT,
    ATTR_INTS,
    TENSOR_FLOAT,
    Attribute,
    Graph,
    Model,
    Node,
    ValueInfo,
)


def make_model(nodes, initializers, inputs, outputs, value_infos=None):
    graph = Graph(
        name="test_graph",
        nodes=nodes,
        initializers=initializers,
        inputs=inputs,
        outputs=outputs,
        value_infos=value_infos or [],
    )
    return Model(ir_version=8, producer_name="conftest", graph=graph,
                 opset_imports=[("", 17)])


def float_tensor(name, array):
    return array_to_tensor(name, np.asarray(array, dtype=np.float32))


def vi(name, shape, elem_type=TENSOR_FLOAT):
    return ValueInfo(name=name, elem_type=elem_type, shape=list(shape))


def run_model(model_bytes: bytes, feeds: Dict[str, np.ndarray],
              output_names: Optional[List[str]] = None) -> List[np.ndarray]:
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return sess.run(output_names, feeds)


def initializers_by_name(model: Model):
    return {t.name: t for t in model.graph.initializers}


def reference_quantize_per_column(weights: np.ndarray):
    """Independent re-implementation of the spec's quantization rules.

    Uses scalar float32 arithmetic and Python's round() (ties to even) so
    the library's vectorized path is checked by a different code path.
    """
    w = np.asarray(weights, dtype=np.float32)
    assert w.ndim == 2
    n_rows, n_cols = w.shape
    tiny = np.float32(np.finfo(np.float32).tiny)
    scales = np.empty(n_cols, dtype=np.float32)
    q = np.empty((n_rows, n_cols), dtype=np.int8)
    for j in range(n_cols):
        max_abs = np.float32(np.max(np.abs(w[:, j])))
        scale = np.float32(max_abs / np.float32(127.0))
        if scale < tiny:
            scale = tiny
        if max_abs == 0:
            scale = np.float32(1.0)
        scales[j] = scale
        for i in range(n_rows):
            quotient = np.float32(w[i, j] / scale)
            rounded = round(float(quotient))  # Python: ties to even
            q[i, j] = max(-127, min(127, rounded))
    return q, scales


def reference_dequantize(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    out = np.empty(q.shape, dtype=np.float32)
    for j in range(q.shape[1]):
        for i in range(q.shape[0]):
            out[i, j] = np.float32(np.float32(q[i, j]) * scales[j])
    return out
