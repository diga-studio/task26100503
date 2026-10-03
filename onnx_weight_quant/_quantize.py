"""Per-output-channel symmetric INT8 quantization math (FP32 scales)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

INT8_QMIN = -127
INT8_QMAX = 127


def quantize_per_column(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a 2-D FP32 matrix, one symmetric scale per column.

    Returns (q, scales): int8 matrix of the same shape and FP32 scales of
    shape (n_cols,). Scale per channel is max_abs/127 clamped below by the
    minimum normal positive FP32 value; all-zero channels get scale 1.
    Quotients are computed in FP32, rounded to nearest (ties to even) and
    clipped to [-127, 127].
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or 0 in w.shape:
        raise ValueError("weights must be a non-empty 2-D matrix")
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    max_abs = np.abs(w).max(axis=0)
    min_normal = np.float32(np.finfo(np.float32).tiny)
    scales = np.maximum(max_abs / np.float32(127.0), min_normal)
    scales = np.where(max_abs == 0, np.float32(1.0), scales).astype(np.float32)
    q = np.rint(w / scales)
    q = np.clip(q, INT8_QMIN, INT8_QMAX).astype(np.int8)
    return q, scales


def dequantize_per_column(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Rebuild FP32 weights as q * scale broadcast along columns."""
    return np.asarray(q, dtype=np.float32) * np.asarray(scales, dtype=np.float32)
