"""Per-output-channel symmetric INT8 quantization numerics (FP32 math)."""

from __future__ import annotations

import numpy as np

INT8_MIN = -127
INT8_MAX = 127
FLOAT32_MIN_NORMAL = np.float32(np.finfo(np.float32).tiny)


def channel_scales(matrix: np.ndarray, channel_axis: int) -> np.ndarray:
    """FP32 scale per output channel of a 2-D matrix.

    ``channel_axis`` selects which stored axis holds the channels. The scale
    is max(|channel|) / 127 computed in FP32, clamped below by the smallest
    positive normal FP32 value; an all-zero channel gets scale exactly 1.
    """
    if matrix.dtype != np.float32:
        raise TypeError("matrix must be float32")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    max_abs = np.max(np.abs(matrix), axis=1 - channel_axis).astype(np.float32)
    scales = np.maximum(max_abs / np.float32(127.0), FLOAT32_MIN_NORMAL).astype(np.float32)
    scales = np.where(max_abs == np.float32(0.0), np.float32(1.0), scales).astype(np.float32)
    return scales


def quantize_matrix(matrix: np.ndarray, channel_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a 2-D FP32 matrix per output channel.

    Returns (int8_data, fp32_scales). The quotient matrix / scale is computed
    in FP32, rounded to nearest with ties-to-even, then clipped to
    [-127, 127]. Zero point is fixed at 0.
    """
    scales = channel_scales(matrix, channel_axis)
    if channel_axis == 1:
        quotient = matrix / scales[np.newaxis, :]
    else:
        quotient = matrix / scales[:, np.newaxis]
    quotient = quotient.astype(np.float32)
    rounded = np.rint(quotient)  # round-half-to-even
    clipped = np.clip(rounded, INT8_MIN, INT8_MAX)
    return clipped.astype(np.int8), scales


def dequantize_matrix(qvalues: np.ndarray, scales: np.ndarray, channel_axis: int) -> np.ndarray:
    """Reference dequantization: int8 value times its channel scale in FP32."""
    restored = qvalues.astype(np.float32)
    if channel_axis == 1:
        return restored * scales[np.newaxis, :]
    return restored * scales[:, np.newaxis]
