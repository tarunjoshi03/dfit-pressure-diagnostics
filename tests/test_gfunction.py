"""Tests for the G-function and time-function module."""

import numpy as np
import pytest

from dfit import gfunction as gf


def test_G_zero_at_shutin():
    assert np.isclose(gf.G_function(np.array([0.0]))[0], 0.0)


def test_G_monotonic_increasing():
    dtd = np.linspace(0, 50, 500)
    G = gf.G_function(dtd)
    assert np.all(np.diff(G) > 0)


def test_g0_values():
    # high-leakoff bound g(0) = pi/2; low-leakoff bound g(0) = 4/3
    assert np.isclose(gf._g0(gf.ALPHA_HIGH_LEAKOFF), np.pi / 2)
    assert np.isclose(gf._g0(gf.ALPHA_LOW_LEAKOFF), 4.0 / 3.0)


def test_both_alpha_bounds_run():
    dtd = np.linspace(0, 10, 100)
    Ghi = gf.G_function(dtd, alpha=gf.ALPHA_HIGH_LEAKOFF)
    Glo = gf.G_function(dtd, alpha=gf.ALPHA_LOW_LEAKOFF)
    assert np.all(np.isfinite(Ghi))
    assert np.all(np.isfinite(Glo))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        gf.g_function(np.array([1.0]), alpha=0.75)


def test_negative_dtd_raises():
    with pytest.raises(ValueError):
        gf.g_function(np.array([-1.0]))


def test_G_time_roundtrip():
    """G_from_time and time_from_G should be mutual inverses."""
    t_pump = 5.0
    t = np.linspace(5.01, 200.0, 300)
    G = gf.G_from_time(t, t_pump)
    t_back = gf.time_from_G(G, t_pump)
    # round-trip within interpolation tolerance
    assert np.allclose(t, t_back, rtol=1e-3, atol=0.1)


def test_G_from_time_nan_during_injection():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    G = gf.G_from_time(t, t_pump=5.0)
    assert np.isnan(G[0]) and np.isnan(G[1])
    assert np.isfinite(G[3])


def test_superposition_time_positive():
    t = np.linspace(5.01, 100, 50)
    tau = gf.superposition_time(t, 5.0)
    assert np.all(tau[np.isfinite(tau)] > 0)


def test_sqrt_shutin_time():
    t = np.array([5.0, 6.0, 9.0])
    s = gf.sqrt_shutin_time(t, 5.0)
    assert np.isclose(s[1], 1.0)
    assert np.isclose(s[2], 2.0)
