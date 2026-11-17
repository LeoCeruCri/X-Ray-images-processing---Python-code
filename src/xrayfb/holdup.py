"""Gas holdup from attenuation, using empty-bed and packed-bed references.

Two-point method (Beer-Lambert, monochromatic approximation)::

    holdup = ln(I / I_packed) / ln(I_empty / I_packed)

holdup = 0 in the packed (fixed) bed and 1 in the empty bed. It is the gas
fraction *relative to the packed bed*; converting to absolute voidage needs
the packed-bed porosity. Beam hardening makes ln(I) non-linear in path
length, so this result is the uncorrected baseline.
"""

from __future__ import annotations

import numpy as np

# Pixels where empty and packed references differ by less than this ratio carry
# no attenuation contrast (e.g. outside the column) and are returned as NaN.
MIN_CONTRAST_LOG_RATIO = 1e-3


def two_point_holdup(
    meas: np.ndarray,
    empty: np.ndarray,
    packed: np.ndarray,
    min_contrast: float = MIN_CONTRAST_LOG_RATIO,
    clip: bool = False,
) -> np.ndarray:
    """Holdup per pixel from dark-corrected intensities.

    Args:
        meas: Measurement image or stack (..., rows, cols), dark-corrected.
        empty: Empty-bed reference (rows, cols), dark-corrected.
        packed: Packed-bed reference (rows, cols), dark-corrected.
        min_contrast: Minimum ln(empty/packed); pixels below it become NaN.
        clip: If True, clip to [0, 1]. Off by default so that out-of-range
            values (noise, calibration errors) stay visible during validation.

    Returns:
        New float array with the shape of ``meas``. Inputs are not modified.
    """
    if empty.shape != packed.shape or meas.shape[-2:] != empty.shape:
        raise ValueError("meas, empty and packed must share the image shape")
    for name, arr in (("meas", meas), ("empty", empty), ("packed", packed)):
        if np.any(arr <= 0):
            raise ValueError(f"{name} has non-positive intensities; check dark correction")

    denominator = np.log(empty / packed)
    valid = denominator > min_contrast
    with np.errstate(invalid="ignore", divide="ignore"):
        holdup = np.where(valid, np.log(meas / packed) / denominator, np.nan)
    if clip:
        holdup = np.clip(holdup, 0.0, 1.0)
    return holdup
