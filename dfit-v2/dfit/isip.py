"""
Instantaneous shut-in pressure (ISIP) estimation.

The ISIP anchors the entire DFIT interpretation: net pressure is ISIP minus
closure pressure, and the after-closure pressure-difference curve is built from
it. Yet ISIP is rarely the pressure at the instant the pump stops. Near-well
tortuosity and wellbore decompression keep injecting fluid into the fracture
for seconds to minutes after surface shut-in, so the apparent shut-in pressure
overstates the true fracture-extension pressure.

This module provides:

* a log-time extrapolation estimator (Barree et al., 2015, Fig. 8), and
* a wellbore-storage / tortuosity decompression model that reproduces the
  early falloff and lets the user separate wellbore effects from reservoir
  signal.

References
----------
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA, "Incorrect
    Instantaneous-Shut-in-Pressure (ISIP) Determination."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ISIPResult:
    """Container for an ISIP estimate and its supporting diagnostics."""

    isip: float
    method: str
    fit_slope: float | None = None
    fit_intercept: float | None = None
    note: str = ""


def isip_log_extrapolation(
    t: np.ndarray,
    p: np.ndarray,
    t_pump: float,
    fit_window=(0.3, 0.8),
) -> ISIPResult:
    """
    Estimate ISIP by extrapolating the pressure-vs-log(time) trend to shut-in.

    During the wellbore-dominated early falloff, pressure declines roughly
    linearly in log(shut-in time). Fitting that linear trend over a stable
    window and extrapolating back to shut-in gives an effective ISIP that
    removes the tortuosity / decompression overshoot (Barree et al., 2015).

    Parameters
    ----------
    t : array_like
        Time since start of injection.
    p : array_like
        Pressure.
    t_pump : float
        Pumping time (defines shut-in: dt = t - t_pump).
    fit_window : tuple(float, float)
        Fractional span (of the available falloff, in log10 of shut-in time)
        used for the linear fit. Default fits the middle of the early falloff.

    Returns
    -------
    ISIPResult
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    dt = t - t_pump
    mask = dt > 0
    dt = dt[mask]
    pf = p[mask]

    order = np.argsort(dt)
    dt, pf = dt[order], pf[order]

    logdt = np.log10(dt)
    lo, hi = fit_window
    span = logdt.max() - logdt.min()
    w_lo = logdt.min() + lo * span
    w_hi = logdt.min() + hi * span
    fitmask = (logdt >= w_lo) & (logdt <= w_hi)

    if fitmask.sum() < 3:
        raise ValueError("Not enough points in the fit window for extrapolation.")

    slope, intercept = np.polyfit(logdt[fitmask], pf[fitmask], 1)

    # Two candidate ISIPs:
    #   p_apparent : the earliest recorded falloff pressure (raw shut-in value)
    #   p_extrap   : the stable-trend line extrapolated to the first falloff time
    p_apparent = float(pf[0])
    p_extrap = float(slope * logdt[0] + intercept)

    # If the very early data sit ABOVE the extrapolated stable trend, that excess
    # is wellbore-storage / tortuosity decompression overshoot, and the effective
    # ISIP is the (lower) extrapolated value. If the early data already lie on the
    # trend, the apparent shut-in pressure is the ISIP.
    if p_apparent > p_extrap:
        isip = p_extrap
        note = ("Effective ISIP from stable-trend extrapolation; early data sit "
                "above the trend (wellbore decompression overshoot removed).")
    else:
        isip = p_apparent
        note = ("ISIP taken as the earliest falloff pressure; no wellbore-storage "
                "overshoot detected above the stable trend.")

    return ISIPResult(
        isip=float(isip),
        method="log-time extrapolation",
        fit_slope=float(slope),
        fit_intercept=float(intercept),
        note=note,
    )


def wellbore_decompression_pressure(
    dt_seconds: np.ndarray,
    isip: float,
    q_shutin_bpm: float,
    tortuosity_factor: float,
    pressure_per_bbl: float,
    dt_step: float = 1.0,
) -> np.ndarray:
    """
    Forward-model the early falloff driven by wellbore storage + tortuosity.

    Reproduces the Barree et al. (2015) calculation in which fluid keeps
    entering the fracture after shut-in. Near-well friction follows a
    square-root-of-rate law, friction drives fluid expansion, and the wellbore
    pressure / volume relationship converts the leaked volume back into a
    pressure decline. Integrating forward in time gives the characteristic fast
    early decay that should NOT be mistaken for reservoir leakoff.

    Parameters
    ----------
    dt_seconds : array_like
        Shut-in time grid, in seconds, at which to report pressure.
    isip : float
        Pressure at the instant of shut-in (psi).
    q_shutin_bpm : float
        Injection rate at shut-in (bbl/min).
    tortuosity_factor : float
        Near-well friction coefficient in psi / sqrt(bbl/min) (Barree's
        "tortuosity factor"; ~970 in the paper's worked example).
    pressure_per_bbl : float
        Wellbore stiffness: pressure change per barrel of volume change (psi/bbl)
        from combined fluid + pipe compressibility (~1170 in the example).
    dt_step : float
        Internal integration step (seconds).

    Returns
    -------
    p : ndarray
        Modeled wellbore pressure at the requested shut-in times.
    """
    dt_seconds = np.asarray(dt_seconds, dtype=float)
    t_end = float(dt_seconds.max())

    # March forward in small steps, tracking rate and pressure.
    n_steps = int(np.ceil(t_end / dt_step)) + 1
    tgrid = np.arange(n_steps) * dt_step

    p = np.empty(n_steps)
    q = q_shutin_bpm          # bbl/min
    pressure = isip
    for i in range(n_steps):
        p[i] = pressure
        # frictional pressure drop available to drive expansion
        p_fric = tortuosity_factor * np.sqrt(max(q, 0.0))
        # rate implied by remaining friction (square-root-of-rate relation)
        q = (p_fric / tortuosity_factor) ** 2 if tortuosity_factor > 0 else 0.0
        # volume injected this step (bbl): q[bbl/min] * step[s] / 60
        dV = q * dt_step / 60.0
        # pressure drop from that volume leaving the wellbore
        pressure = pressure - dV * pressure_per_bbl
        # rate decays as driving friction is consumed
        q = q * np.exp(-dt_step * pressure_per_bbl / max(tortuosity_factor, 1e-9))
        if pressure < 0:
            pressure = 0.0

    return np.interp(dt_seconds, tgrid, p)
