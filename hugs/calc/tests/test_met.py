"""Test the `met` module."""

from hugs.calc import get_wind_components, get_wind_dir, get_wind_speed
from hugs.calc import snell_angle

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)


def test_wind_component():
    """Test the wind component."""
    u, v = get_wind_components(100, 0)
    assert_almost_equal(u, 0)
    assert_almost_equal(v, -100)

    speeds = np.array([100, 100, 100, 100])
    directions = np.array([0, 90, 180, 275])
    u_expected = np.array([-0.000000e+00, -1.000000e+02, -1.224647e-14, 9.961947e+01])
    u, v = get_wind_components(speeds, directions)
    assert_array_almost_equal(u, u_expected)


def test_value_error():
    """Test value error."""
    with pytest.raises(ValueError):
        snell_angle(0, 0, 0)


def test_warning_error():
    """Test warning error."""
    with pytest.warns(UserWarning):
        get_wind_components(100, 400)
