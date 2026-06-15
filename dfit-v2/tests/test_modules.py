"""Tests for derivatives, leakoff classification, closure, and ISIP."""

import numpy as np
import pytest

from dfit.derivatives import bourdet_derivative, semilog_derivative, _moving_average
from dfit.leakoff import classify_leakoff
from dfit.closure import pick_closure, net_pressure
from dfit.isip import isip_log_extrapolation, wellbore_decompression_pressure
from dfit.synthetic import generate_dfit


# ---- derivatives ----------------------------------------------------------

def test_bourdet_derivative_of_log_line():
    """d/d(ln x) of y = a ln x + b should recover the slope a."""
    x = np.logspace(0, 3, 200)
    a, b = 2.5, 1.0
    y = a * np.log(x) + b
    d = bourdet_derivative(x, y, L=0.1)
    interior = d[10:-10]
    assert np.allclose(interior, a, rtol=0.05)


def test_bourdet_rejects_nonpositive_x():
    with pytest.raises(ValueError):
        bourdet_derivative(np.array([-1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_semilog_derivative_sign_convention():
    """For a declining falloff, G*dP/dG should be positive (industry sign)."""
    G = np.linspace(0.1, 10, 200)
    p = 8000 - 200 * G            # declining pressure
    sgd = semilog_derivative(G, p, smooth=0)
    assert np.all(sgd[5:-5] > 0)


def test_moving_average_preserves_length():
    a = np.random.randn(100)
    out = _moving_average(a, 3)
    assert len(out) == len(a)


# ---- leakoff --------------------------------------------------------------

@pytest.mark.parametrize(
    "regime", ["normal", "pressure_dependent", "height_recession", "tip_extension"]
)
def test_classify_leakoff_recovers_regime(regime):
    d = generate_dfit(regime=regime, closure_G=6.0, noise_psi=0.0, seed=2)
    G = d.G[np.isfinite(d.G)]
    p = d.pressure_psi[np.isfinite(d.G)]
    order = np.argsort(G)
    res = classify_leakoff(G[order], p[order], closure_G=6.0)
    assert res.regime == regime
    assert 0.0 <= res.confidence <= 1.0


def test_classify_leakoff_too_few_points():
    with pytest.raises(ValueError):
        classify_leakoff(np.array([1.0, 2.0]), np.array([1.0, 0.5]))


# ---- closure --------------------------------------------------------------

def test_pick_closure_returns_sensible_values():
    d = generate_dfit(regime="normal", closure_G=6.0, seed=3)
    G = d.G[np.isfinite(d.G)]
    p = d.pressure_psi[np.isfinite(d.G)]
    order = np.argsort(G)
    res = pick_closure(G[order], p[order])
    assert 3.0 < res.closure_G < 10.0
    assert res.closure_pressure > 0


def test_closure_gradient_with_depth():
    d = generate_dfit(regime="normal", closure_G=6.0, seed=3)
    G = d.G[np.isfinite(d.G)]
    p = d.pressure_psi[np.isfinite(d.G)]
    order = np.argsort(G)
    res = pick_closure(G[order], p[order], datum_depth_ft=10000.0)
    assert res.closure_gradient is not None
    assert 0.3 < res.closure_gradient < 1.2   # psi/ft, physically plausible


def test_net_pressure():
    assert net_pressure(8000, 6800) == 1200


# ---- ISIP -----------------------------------------------------------------

def test_isip_recovers_clean_value():
    d = generate_dfit(regime="normal", isip_psi=8000.0, noise_psi=0.0, seed=1)
    res = isip_log_extrapolation(d.time_min, d.pressure_psi, d.truth["t_pump_min"])
    assert abs(res.isip - 8000.0) < 80.0


def test_wellbore_decompression_decays():
    """Modeled wellbore pressure should decline monotonically from ISIP."""
    dt = np.linspace(0, 200, 201)
    p = wellbore_decompression_pressure(
        dt, isip=3640.0, q_shutin_bpm=4.0,
        tortuosity_factor=970.0, pressure_per_bbl=1170.0,
    )
    assert p[0] <= 3640.0 + 1e-6
    assert p[-1] < p[0]
    assert np.all(np.diff(p) <= 1e-6)
