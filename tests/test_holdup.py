import numpy as np
import pytest
from conftest import SHAPE, make_measurement

from xrayfb.holdup import two_point_holdup


def test_empty_bed_gives_one(references):
    empty, packed = references
    np.testing.assert_allclose(two_point_holdup(empty, empty, packed), 1.0)


def test_packed_bed_gives_zero(references):
    empty, packed = references
    np.testing.assert_allclose(two_point_holdup(packed, empty, packed), 0.0, atol=1e-12)


def test_known_holdup_is_recovered(references):
    empty, packed = references
    truth = np.linspace(0, 1, SHAPE[0] * SHAPE[1]).reshape(SHAPE)
    np.testing.assert_allclose(two_point_holdup(make_measurement(truth), empty, packed), truth)


def test_stack_input(references):
    empty, packed = references
    stack = np.stack([make_measurement(np.full(SHAPE, h)) for h in (0.2, 0.5)])
    result = two_point_holdup(stack, empty, packed)
    np.testing.assert_allclose(result[:, 0, 0], [0.2, 0.5])


def test_inputs_are_not_modified(references):
    empty, packed = references
    meas = make_measurement(np.full(SHAPE, 0.3))
    copies = [a.copy() for a in (meas, empty, packed)]
    two_point_holdup(meas, empty, packed)
    for original, copy in zip((meas, empty, packed), copies):
        np.testing.assert_array_equal(original, copy)


def test_no_contrast_pixels_are_nan(references):
    empty, packed = references
    packed = packed.copy()
    packed[0, 0] = empty[0, 0]  # e.g. outside the column
    result = two_point_holdup(empty, empty, packed)
    assert np.isnan(result[0, 0])
    assert not np.isnan(result[1, 1])


def test_non_positive_intensity_raises(references):
    empty, packed = references
    with pytest.raises(ValueError, match="non-positive"):
        two_point_holdup(np.zeros(SHAPE), empty, packed)
