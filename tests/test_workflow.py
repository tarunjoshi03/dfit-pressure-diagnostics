"""Tests for the synthetic generator and the end-to-end workflow."""

import numpy as np
import pytest

import dfit
from dfit.synthetic import generate_dfit
from dfit.workflow import analyze_dfit


REGIMES = ("normal", "pressure_dependent", "height_recession", "tip_extension")


@pytest.mark.parametrize("regime", REGIMES)
def test_synthetic_runs(regime):
    d = generate_dfit(regime=regime, seed=1)
    assert len(d.time_min) == len(d.pressure_psi) == len(d.rate_bpm)
    assert np.all(np.diff(d.time_min) > 0)        # time strictly increasing
    assert np.all(d.pressure_psi > 0)


@pytest.mark.parametrize("regime", REGIMES)
def test_synthetic_isip_endpoint(regime):
    """Falloff should start at (approximately) the specified ISIP."""
    d = generate_dfit(regime=regime, isip_psi=8000.0, noise_psi=0.0, seed=1)
    fall = np.isfinite(d.G)
    first_falloff_p = d.pressure_psi[fall][np.argmin(d.G[fall])]
    assert abs(first_falloff_p - 8000.0) < 50.0


@pytest.mark.parametrize("regime", REGIMES)
def test_synthetic_closure_pressure(regime):
    """Pressure at the truth closure_G should match the specified value."""
    d = generate_dfit(regime=regime, closure_G=6.0,
                      closure_pressure_psi=6800.0, noise_psi=0.0, seed=1)
    fall = np.isfinite(d.G)
    G = d.G[fall]
    p = d.pressure_psi[fall]
    idx = np.argmin(np.abs(G - 6.0))
    assert abs(p[idx] - 6800.0) < 30.0


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("seed", [1, 3, 7, 11])
def test_workflow_classifies_regime(regime, seed):
    """The full pipeline should recover the leakoff regime it was built from."""
    d = generate_dfit(regime=regime, closure_G=6.0, seed=seed)
    res = analyze_dfit(d.time_min, d.pressure_psi, d.rate_bpm)
    assert res.leakoff_regime == regime


@pytest.mark.parametrize("regime", REGIMES)
def test_workflow_isip_recovered(regime):
    d = generate_dfit(regime=regime, isip_psi=8000.0, seed=5)
    res = analyze_dfit(d.time_min, d.pressure_psi, d.rate_bpm)
    assert abs(res.isip_psi - 8000.0) < 100.0


def test_workflow_closure_in_range():
    """Closure pick should be within a sensible band around the truth."""
    d = generate_dfit(regime="normal", closure_G=6.0, seed=3)
    res = analyze_dfit(d.time_min, d.pressure_psi, d.rate_bpm)
    # picker is known to land slightly late; require within +/- 1.5 G
    assert abs(res.closure_G - 6.0) < 1.5


def test_workflow_pump_time_inferred():
    """If t_pump not given, it is inferred from the rate schedule."""
    d = generate_dfit(regime="normal", t_pump_min=5.0, seed=3)
    res = analyze_dfit(d.time_min, d.pressure_psi, d.rate_bpm)  # no t_pump
    assert res.isip_psi > 0


def test_to_dataframe():
    d = generate_dfit(regime="normal", seed=1)
    df = dfit.to_dataframe(d)
    assert list(df.columns) == ["time_min", "pressure_psi", "rate_bpm", "G"]
    assert len(df) == len(d.time_min)
