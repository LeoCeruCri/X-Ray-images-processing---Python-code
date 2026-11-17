"""Synthetic data shared by the tests (Beer-Lambert, monochromatic)."""

import numpy as np
import pytest
import tifffile

SHAPE = (32, 48)
I0 = 20000.0  # unattenuated intensity, counts
MU_SOLID_L = 2.0  # attenuation of the packed bed along the beam (mu * L), dimensionless
DARK_LEVEL = 100.0  # detector offset, counts


@pytest.fixture
def references():
    """Dark-corrected empty and packed references with uniform path length."""
    empty = np.full(SHAPE, I0)
    packed = np.full(SHAPE, I0 * np.exp(-MU_SOLID_L))
    return empty, packed


def make_measurement(holdup: np.ndarray) -> np.ndarray:
    """Intensity of a bed where a fraction ``holdup`` of the packed solids is replaced by gas."""
    return I0 * np.exp(-MU_SOLID_L * (1 - holdup))


@pytest.fixture
def raw_measurement(tmp_path):
    """A fake raw folder: <raw>/meas/camera 1/img_<n>.tif stored as uint16."""
    raw_root = tmp_path / "raw"
    cam = raw_root / "meas" / "camera 1"
    cam.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for n in range(5):
        # Values above 32767 on purpose: they would wrap if cast to int16.
        img = rng.integers(30000, 60000, size=SHAPE, dtype=np.uint16)
        tifffile.imwrite(cam / f"img_{n}.tif", img)
    return raw_root, cam
