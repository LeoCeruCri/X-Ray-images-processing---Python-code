"""Detector corrections applied before any attenuation calculation.

All functions return new arrays and never modify their inputs.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def subtract_dark(image: np.ndarray, dark: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Subtract the dark (detector offset) image and clip at ``floor``.

    Args:
        image: Single image (rows, cols) or stack (n_frames, rows, cols).
        dark: Time-averaged dark image (rows, cols), same detector settings
            (frame rate, VROI) as ``image``.
        floor: Lowest allowed value after subtraction; negative intensities
            are not physical.
    """
    if dark.shape != image.shape[-2:]:
        raise ValueError(f"Dark shape {dark.shape} does not match image {image.shape[-2:]}")
    return np.clip(image - dark, floor, None)


def interpolate_dead_pixels(image: np.ndarray, dead_mask: np.ndarray) -> np.ndarray:
    """Replace dead pixels by the mean of their valid 8-neighbours.

    Generic mask-based version. Pixels whose neighbours are all dead are
    filled iteratively from the outside in, so lines several pixels wide
    are also handled.

    Args:
        image: Image (rows, cols).
        dead_mask: Boolean array, True where the pixel is dead.
    """
    if dead_mask.shape != image.shape:
        raise ValueError("dead_mask must have the same shape as image")
    if dead_mask.all():
        raise ValueError("All pixels are marked dead")

    result = np.where(dead_mask, 0.0, image).astype(np.float64)
    remaining = dead_mask.copy()
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    while remaining.any():
        valid = ~remaining
        neighbour_sum = ndimage.convolve(result * valid, kernel, mode="nearest")
        neighbour_count = ndimage.convolve(valid.astype(float), kernel, mode="nearest")
        fillable = remaining & (neighbour_count > 0)
        result[fillable] = neighbour_sum[fillable] / neighbour_count[fillable]
        remaining &= ~fillable
    return result
