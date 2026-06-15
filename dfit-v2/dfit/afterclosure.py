"""
After-closure analysis (ACA).

Once the fracture has closed, the falloff is governed by the reservoir. The
log-log derivative of pressure change versus shut-in time reveals the flow
regime through its slope:

* pseudo-LINEAR flow  -> derivative slope approaches -1/2
* pseudo-RADIAL flow  -> derivative slope approaches -1

Barree et al. (2015) stress that true pseudo-radial flow is the exception, not
the rule, in unconventional reservoirs: it can take hundreds of days to develop,
so forcing a radial (Horner) interpretation typically overestimates
permeability by orders of magnitude. This module therefore:

1. computes the log-log Bourdet derivative of (p - p_i),
2. detects regions consistent with -1/2 and -1 slopes,
3. extrapolates the appropriate Cartesian flow-regime plot to estimate
   reservoir pore pressure, and
4. flags when a radial interpretation is unsupported.

References
----------
Nolte, K.G. (1979) SPE-8341-MS.
Talley, G.R. et al. (1999) "Field Application of After-Closure Analysis of
    Fracture Calibration Tests." SPE-52220-MS.
Soliman, M.Y. et al. (2005) after-closure analysis methods.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivatives import bourdet_derivative


@dataclass
class FlowRegimeResult:
    """Detected after-closure flow regimes and the pore-pressure estimate."""

    linear_flow_window: tuple | None
    radial_flow_window: tuple | None
    pore_pressure: float | None
    pore_pressure_method: str
    radial_supported: bool
    note: str = ""


def aca_derivative(
    t: np.ndarray,
    p: np.ndarray,
    t_closure: float,
    p_far: float | None = None,
    L: float = 0.15,
):
    """
    Compute the log-log after-closure derivative.

    Parameters
    ----------
    t : array_like
        Time since start of injection.
    p : array_like
        Pressure.
    t_closure : float
        Closure time (same clock as t). Only data after ~3x closure time are
        used, following the Barree rule of thumb that reservoir transients need
        roughly half a log cycle past closure to stabilise.
    p_far : float, optional
        Far-field / static pressure used to form (p - p_far). If omitted, the
        last recorded pressure is used as a provisional anchor.
    L : float
        Bourdet smoothing window (ln-time units).

    Returns
    -------
    dt : ndarray
        Shut-in time (t - t_pump_proxy) restricted to the valid ACA window.
    dp : ndarray
        Pressure difference p - p_far over that window.
    deriv : ndarray
        Log-log derivative d(dp)/d(ln dt).
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    # valid ACA window: at least ~3x closure time of shut-in has elapsed
    valid = t > 3.0 * t_closure
    if valid.sum() < 5:
        valid = t > t_closure  # relax if data are short
    tt = t[valid]
    pp = p[valid]

    if p_far is None:
        p_far = float(pp.min())

    dt = tt - tt.min() + (tt[1] - tt[0] if len(tt) > 1 else 1.0)
    dp = pp - p_far
    dp = np.where(dp <= 0, np.nan, dp)

    deriv = bourdet_derivative(dt, dp, L=L)
    return dt, dp, deriv


def detect_flow_regimes(
    dt: np.ndarray,
    dp: np.ndarray,
    deriv: np.ndarray,
    slope_tol: float = 0.12,
) -> FlowRegimeResult:
    """
    Detect pseudo-linear (-1/2) and pseudo-radial (-1) slope regions.

    The slope is evaluated as d(ln deriv)/d(ln dt). Regions where it sits near
    -1/2 are flagged as pseudo-linear; near -1 as pseudo-radial. Following
    Barree et al. (2015), a radial flag is only trusted if it appears at late
    time AND spans a meaningful range; otherwise it is reported as unsupported.

    Parameters
    ----------
    dt, dp, deriv : ndarray
        Output of `aca_derivative`.
    slope_tol : float
        Tolerance around the target slopes.

    Returns
    -------
    FlowRegimeResult
    """
    finite = np.isfinite(dt) & np.isfinite(deriv) & (deriv > 0)
    x = np.log(dt[finite])
    y = np.log(deriv[finite])
    if len(x) < 6:
        return FlowRegimeResult(None, None, None, "none", False,
                                "Insufficient after-closure data.")

    # local slope of the derivative on the log-log plot
    local_slope = np.gradient(y, x)

    lin_mask = np.abs(local_slope - (-0.5)) < slope_tol
    rad_mask = np.abs(local_slope - (-1.0)) < slope_tol

    lin_win = _contiguous_window(dt[finite], lin_mask)
    rad_win = _contiguous_window(dt[finite], rad_mask)

    # Radial flow is only "supported" if its window is at the late-time end and
    # spans at least a third of a log cycle.
    radial_supported = False
    if rad_win is not None:
        span = np.log10(rad_win[1] / rad_win[0])
        at_late_time = rad_win[1] >= 0.7 * dt[finite].max()
        radial_supported = (span >= 0.33) and at_late_time

    # Pore pressure: prefer linear-flow extrapolation (more reliable in tight
    # rock); fall back to the late-time data only with a warning.
    pore_p = None
    method = "none"
    note = ""
    if lin_win is not None:
        pore_p, method = _extrapolate_linear(dt[finite], dp[finite], lin_win)
        note = "Pore pressure from pseudo-linear flow extrapolation."
        if not radial_supported and rad_win is not None:
            note += (
                " A radial-looking section exists but is NOT supported "
                "(too short or too early); do not force a Horner analysis."
            )
    elif radial_supported:
        pore_p, method = _extrapolate_radial(dt[finite], dp[finite], rad_win)
        note = "Pore pressure from supported pseudo-radial flow."
    else:
        note = (
            "No reliable flow regime; reservoir transient not developed. "
            "Pore pressure not estimated (typical of ultra-tight DFITs)."
        )

    return FlowRegimeResult(
        linear_flow_window=lin_win,
        radial_flow_window=rad_win,
        pore_pressure=pore_p,
        pore_pressure_method=method,
        radial_supported=radial_supported,
        note=note,
    )


def _contiguous_window(x: np.ndarray, mask: np.ndarray):
    """Return (x_start, x_end) of the longest contiguous True run in mask."""
    if not mask.any():
        return None
    best_len = 0
    best = None
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) > best_len:
                best_len = j - i
                best = (float(x[i]), float(x[j - 1]))
            i = j
        else:
            i += 1
    return best if best_len >= 3 else None


def _extrapolate_linear(dt, dp, win):
    """Extrapolate the Cartesian linear-flow plot (p vs sqrt-time fn) to p_i."""
    mask = (dt >= win[0]) & (dt <= win[1])
    # linear flow: dp ~ a * F_L, with F_L decreasing as 1/sqrt-like; here we use
    # the practical proxy F = 1/sqrt(dt) and extrapolate F -> 0 (infinite time).
    F = 1.0 / np.sqrt(dt[mask])
    slope, intercept = np.polyfit(F, dp[mask], 1)
    # at F -> 0 (t -> inf), dp -> intercept; pore pressure offset is the anchor
    return float(intercept), "Cartesian linear-flow extrapolation"


def _extrapolate_radial(dt, dp, win):
    """Extrapolate the Cartesian radial-flow (Horner-like) plot to p_i."""
    mask = (dt >= win[0]) & (dt <= win[1])
    F = 1.0 / dt[mask]
    slope, intercept = np.polyfit(F, dp[mask], 1)
    return float(intercept), "Cartesian radial-flow extrapolation"
