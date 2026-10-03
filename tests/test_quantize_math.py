"""Unit tests for the per-column symmetric INT8 quantization rules."""

import numpy as np
import pytest

from onnx_weight_quant._quantize import dequantize_per_column, quantize_per_column

from conftest import reference_quantize_per_column


def test_scale_is_max_abs_over_127():
    w = np.array([[3.0, -4.0], [1.0, 2.0]], dtype=np.float32)
    q, s = quantize_per_column(w)
    assert s[0] == np.float32(3.0 / 127.0)
    assert s[1] == np.float32(4.0 / 127.0)
    assert q[0, 0] == 127 and q[0, 1] == -127


def test_all_zero_channel_has_scale_one():
    w = np.array([[0.0, 5.0], [0.0, -5.0]], dtype=np.float32)
    q, s = quantize_per_column(w)
    assert s[0] == 1.0
    assert (q[:, 0] == 0).all()


def test_scale_floored_at_min_normal_float32():
    tiny = np.float32(np.finfo(np.float32).tiny)
    w = np.array([[tiny]], dtype=np.float32)
    q, s = quantize_per_column(w)
    assert s[0] == tiny
    assert q[0, 0] == 1  # tiny / tiny


def test_ties_round_to_even_both_signs():
    # scale == 2.0 in every column (max_abs == 254)
    w = np.array(
        [
            [1.0, -1.0, 3.0, -3.0, 5.0, -5.0],
            [254.0, 254.0, 254.0, 254.0, 254.0, 254.0],
        ],
        dtype=np.float32,
    )
    q, s = quantize_per_column(w)
    assert (s == 2.0).all()
    # 0.5 -> 0, -0.5 -> 0, 1.5 -> 2, -1.5 -> -2, 2.5 -> 2, -2.5 -> -2
    assert q[0].tolist() == [0, 0, 2, -2, 2, -2]
    assert q[1].tolist() == [127] * 6


def test_clips_to_int8_range():
    w = np.array([[127.0, -127.0]], dtype=np.float32)
    q, s = quantize_per_column(w)
    assert q[0, 0] == 127 and q[0, 1] == -127


def test_matches_independent_reference_on_random_matrices():
    rng = np.random.default_rng(20261005)
    for _ in range(20):
        rows = int(rng.integers(1, 9))
        cols = int(rng.integers(1, 9))
        w = rng.normal(0.0, 10.0, size=(rows, cols)).astype(np.float32)
        if rng.random() < 0.5:
            w[:, 0] = 0.0  # sprinkle an all-zero channel
        q, s = quantize_per_column(w)
        q_ref, s_ref = reference_quantize_per_column(w)
        assert np.array_equal(q, q_ref)
        assert np.array_equal(s, s_ref)


def test_dequantize_broadcasts_scales_over_columns():
    q = np.array([[1, 2], [3, 4]], dtype=np.int8)
    s = np.array([0.5, 2.0], dtype=np.float32)
    out = dequantize_per_column(q, s)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.array([[0.5, 4.0], [1.5, 8.0]], dtype=np.float32))


def test_rejects_non_2d_or_empty():
    with pytest.raises(ValueError):
        quantize_per_column(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        quantize_per_column(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        quantize_per_column(np.array([[np.inf]], dtype=np.float32))
