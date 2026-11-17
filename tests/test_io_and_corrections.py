import hashlib

import numpy as np
import pytest
import tifffile

from xrayfb import io
from xrayfb.corrections import interpolate_dead_pixels, subtract_dark


def _hash_folder(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_reading_keeps_values_above_int16(raw_measurement):
    _, cam = raw_measurement
    original = tifffile.imread(cam / "img_0.tif")
    np.testing.assert_array_equal(io.read_frame(cam / "img_0.tif"), original.astype(float))
    assert io.read_frame(cam / "img_0.tif").min() >= 30000


def test_reading_does_not_touch_raw_files(raw_measurement):
    _, cam = raw_measurement
    before = _hash_folder(cam)
    io.read_stack(cam, range(5))
    io.mean_image(cam, range(5))
    assert _hash_folder(cam) == before


def test_mean_image_matches_numpy(raw_measurement):
    _, cam = raw_measurement
    expected = io.read_stack(cam, range(5)).mean(axis=0)
    np.testing.assert_allclose(io.mean_image(cam, range(5)), expected)


def test_missing_frames_raise(raw_measurement):
    _, cam = raw_measurement
    with pytest.raises(KeyError):
        io.read_stack(cam, range(10))


def test_write_inside_raw_is_refused(raw_measurement, tmp_path):
    raw_root, cam = raw_measurement
    with pytest.raises(PermissionError):
        io.write_image(np.zeros((2, 2)), cam / "average.tif", raw_root)
    io.write_image(np.zeros((2, 2)), tmp_path / "out" / "a.tif", raw_root)
    assert (tmp_path / "out" / "a.tif").exists()


def test_subtract_dark_clips_and_checks_shape():
    img = np.array([[5.0, 50.0]])
    dark = np.array([[10.0, 10.0]])
    np.testing.assert_array_equal(subtract_dark(img, dark), [[0.0, 40.0]])
    with pytest.raises(ValueError):
        subtract_dark(img, np.zeros((2, 2)))


def test_dead_line_is_interpolated_from_a_smooth_image():
    # A linear ramp is reproduced exactly by neighbour averaging across a dead column.
    rows = np.mgrid[0:20, 0:20][0]
    image = 100.0 + 3.0 * rows
    mask = np.zeros_like(image, dtype=bool)
    mask[:, 7:9] = True  # two-pixel-wide dead column
    corrupted = image.copy()
    corrupted[mask] = 0.0
    fixed = interpolate_dead_pixels(corrupted, mask)
    np.testing.assert_allclose(fixed[1:-1, 7:9], image[1:-1, 7:9], rtol=1e-12)
    np.testing.assert_array_equal(corrupted[mask], 0.0)  # input untouched
