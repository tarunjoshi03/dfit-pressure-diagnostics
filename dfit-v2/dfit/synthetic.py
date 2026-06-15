"""
Synthetic DFIT generator.

Real DFIT data are proprietary, which is the single biggest barrier to building
and testing interpretation software. This module forward-models physically
reasonable pressure-time records for each of the four canonical leakoff regimes,
so the rest of the toolkit can be exercised, unit-tested, and demonstrated end
to end without any confidential field data.

Design
------
It is natural to specify the falloff in G-function space, because each leakoff
regime is defined by the shape of G * dP/dG relative to a line through the
origin. The generator therefore:

1. builds a target first derivative dP/dG over the pre-closure G range whose
   semilog derivative G * dP/dG reproduces the chosen regime,
2. integrates to obtain pressure versus G before closure,
3. appends a reservoir-dominated after-closure decline with a selectable flow
   regime (linear or radial), and
4. maps G back to physical time via the inverse G-function, optionally
   prepending an injection ramp and adding gauge noise.

The output mimics what a real bottomhole gauge records: time (min), pressure
(psi), and rate (bbl/min).

References
----------
Nolte, K.G. (1979) SPE-8341-MS.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gfunction import G_function, time_from_G, ALPHA_HIGH_LEAKOFF


@dataclass
class SyntheticDFIT:
    """A synthetic DFIT record with the ground-truth parameters used to build it."""

    time_min: np.ndarray          # time since start of injection (minutes)
    pressure_psi: np.ndarray      # pressure (psi)
    rate_bpm: np.ndarray          # injection rate (bbl/min)
    G: np.ndarray                 # G-function (NaN during injection)
    truth: dict                   # ground-truth parameters


def generate_dfit(
    regime: str = "normal",
    t_pump_min: float = 5.0,
    isip_psi: float = 8000.0,
    closure_pressure_psi: float = 6800.0,
    closure_G: float = 6.0,
    reservoir_pressure_psi: float = 5200.0,
    after_closure_regime: str = "linear",
    n_points: int = 1200,
    G_max: float = 18.0,
    rate_bpm: float = 8.0,
    noise_psi: float = 1.5,
    seed: int | None = 7,
    alpha: float = ALPHA_HIGH_LEAKOFF,
) -> SyntheticDFIT:
    """
    Generate one synthetic DFIT.

    Parameters
    ----------
    regime : {"normal", "pressure_dependent", "height_recession", "tip_extension"}
        Pre-closure leakoff signature to synthesise.
    t_pump_min : float
        Injection (pumping) time in minutes.
    isip_psi : float
        Instantaneous shut-in pressure.
    closure_pressure_psi : float
        Pressure at fracture closure (sets net pressure = ISIP - closure).
    closure_G : float
        G-function value at closure (ground truth for testing the picker).
    reservoir_pressure_psi : float
        Far-field pore pressure the after-closure decline tends toward.
    after_closure_regime : {"linear", "radial"}
        Flow regime that governs the post-closure decline.
    n_points : int
        Number of falloff samples.
    G_max : float
        Maximum G-function value recorded.
    rate_bpm : float
        Constant injection rate during pumping.
    noise_psi : float
        Standard deviation of Gaussian gauge noise added to pressure.
    seed : int, optional
        RNG seed for reproducibility.
    alpha : float
        Leakoff exponent for the G-function (0.5 or 1.0).

    Returns
    -------
    SyntheticDFIT
    """
    if regime not in {"normal", "pressure_dependent", "height_recession", "tip_extension"}:
        raise ValueError(f"Unknown regime: {regime!r}")
    if after_closure_regime not in {"linear", "radial"}:
        raise ValueError("after_closure_regime must be 'linear' or 'radial'.")

    rng = np.random.default_rng(seed)

    # G grid for the falloff (skip exact 0 to keep derivatives finite)
    G = np.linspace(1e-3, G_max, n_points)

    pre = G <= closure_G
    post = ~pre

    # ---- Pre-closure: build a positive dP/dG shape, then scale it ----------
    # We model the pressure DROP from ISIP, D(G) = ISIP - p(G), which increases
    # monotonically. Its derivative dD/dG = dP-drop/dG is positive, so the
    # semilog derivative G * dD/dG is positive and its shape (relative to a
    # line through the origin) maps directly onto the leakoff regimes.
    #
    # Each regime defines an UNSCALED positive shape s(G). We then scale the
    # whole shape by a single constant so that the integral of dD/dG over the
    # pre-closure window equals exactly (ISIP - closure_pressure). This fixes
    # the closure-pressure endpoint WITHOUT any shape-distorting correction.
    g = G / closure_G  # normalized pre-closure coordinate (1.0 at closure)

    if regime == "normal":
        # constant derivative -> G*dD/dG is a straight line through the origin
        shape = np.ones_like(G)

    elif regime == "pressure_dependent":
        # extra early leakoff: larger derivative near shut-in, relaxing to base
        # -> G*dD/dG humps ABOVE the reference line
        shape = 1.0 + 1.2 * np.exp(-(g ** 2) / (2 * 0.22 ** 2))

    elif regime == "height_recession":
        # reduced mid-range storage that recovers
        # -> G*dD/dG sags BELOW the reference line (belly)
        shape = 1.0 - 0.55 * np.exp(-((g - 0.5) ** 2) / (2 * 0.25 ** 2))

    elif regime == "tip_extension":
        # continued tip extension adds a 1/G term so G*dD/dG = a + b*G has a
        # POSITIVE intercept a
        shape = 1.0 + 0.45 * closure_G / np.maximum(G, 0.05)

    # Scale the shape so the pre-closure pressure drop integrates to the target.
    # Ensure the pre-closure window ends exactly at closure_G by inserting that
    # node, so the closure pressure endpoint is exact.
    if not np.any(np.isclose(G, closure_G)):
        insert_at = np.searchsorted(G, closure_G)
        G = np.insert(G, insert_at, closure_G)
        g = G / closure_G
        # rebuild shape at the new node set
        if regime == "normal":
            shape = np.ones_like(G)
        elif regime == "pressure_dependent":
            shape = 1.0 + 1.2 * np.exp(-(g ** 2) / (2 * 0.22 ** 2))
        elif regime == "height_recession":
            shape = 1.0 - 0.55 * np.exp(-((g - 0.5) ** 2) / (2 * 0.25 ** 2))
        elif regime == "tip_extension":
            shape = 1.0 + 0.45 * closure_G / np.maximum(G, 0.05)
        pre = G <= closure_G
        post = ~pre

    pre_idx = np.where(pre)[0]
    G_pre = G[pre_idx]
    target_drop = isip_psi - closure_pressure_psi
    raw_integral = np.trapezoid(shape[pre_idx], G_pre)
    k = target_drop / raw_integral
    dDdG = k * shape  # dP-drop / dG (positive)

    # Integrate to get the pressure drop, then pressure = ISIP - drop.
    p = np.empty_like(G)
    drop = np.zeros_like(G)
    drop[pre_idx] = _cumtrapz(dDdG[pre_idx], G_pre)
    p[pre_idx] = isip_psi - drop[pre_idx]

    # ---- After-closure reservoir-dominated decline ------------------------
    p_close = p[pre_idx[-1]] if len(pre_idx) else closure_pressure_psi
    Gc = G[pre_idx[-1]] if len(pre_idx) else closure_G
    dG_post = G[post] - Gc
    amp = p_close - reservoir_pressure_psi

    if after_closure_regime == "linear":
        # approaches p_res with a sqrt-like (slower) decay
        decline = amp * np.exp(-np.sqrt(dG_post) / np.sqrt(G_max - Gc) * 2.2)
    else:  # radial
        decline = amp * np.exp(-dG_post / (0.5 * (G_max - Gc)))

    p[post] = reservoir_pressure_psi + decline

    # ---- Map G back to physical time --------------------------------------
    t_falloff = time_from_G(G, t_pump_min, alpha)  # minutes since injection start

    # ---- Prepend the injection ramp ---------------------------------------
    n_inj = 60
    t_inj = np.linspace(0.0, t_pump_min, n_inj, endpoint=False)
    # simple breakdown-then-plateau pressure ramp up to ISIP
    p_inj = isip_psi * (0.55 + 0.45 * (t_inj / t_pump_min) ** 0.6)
    rate_inj = np.full(n_inj, rate_bpm)

    time_min = np.concatenate([t_inj, t_falloff])
    pressure = np.concatenate([p_inj, p])
    rate = np.concatenate([rate_inj, np.zeros_like(t_falloff)])
    G_full = np.concatenate([np.full(n_inj, np.nan), G])

    # gauge noise on the falloff only
    if noise_psi > 0:
        pressure[n_inj:] += rng.normal(0.0, noise_psi, size=len(t_falloff))

    truth = {
        "regime": regime,
        "t_pump_min": t_pump_min,
        "isip_psi": isip_psi,
        "closure_pressure_psi": closure_pressure_psi,
        "closure_G": closure_G,
        "closure_time_min": float(time_from_G(np.array([closure_G]), t_pump_min, alpha)[0]),
        "net_pressure_psi": isip_psi - closure_pressure_psi,
        "reservoir_pressure_psi": reservoir_pressure_psi,
        "after_closure_regime": after_closure_regime,
        "alpha": alpha,
    }

    return SyntheticDFIT(
        time_min=time_min,
        pressure_psi=pressure,
        rate_bpm=rate,
        G=G_full,
        truth=truth,
    )


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with a leading zero (length preserved)."""
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def to_dataframe(dfit: SyntheticDFIT):
    """Return the synthetic record as a pandas DataFrame (time, pressure, rate)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "time_min": dfit.time_min,
            "pressure_psi": dfit.pressure_psi,
            "rate_bpm": dfit.rate_bpm,
            "G": dfit.G,
        }
    )
