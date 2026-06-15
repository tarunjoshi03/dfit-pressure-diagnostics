"""
Fracture-closure identification.

Closure is the point at which the fluid pressure in the fracture balances the
minimum in-situ stress (zero net stress on the fracture face). It is NOT a
mechanical sealing of the fracture. On the G-function plot, closure is where the
semilog derivative G * dP/dG departs from the straight line through the origin
that characterises normal leakoff (Barree et al., 2009).

This module locates that departure automatically by:

1. fitting the through-origin reference line to the early, straight pre-closure
   section of G * dP/dG, and
2. flagging closure at the G value where the data fall persistently below the
   line by more than a tolerance.

It also returns the closure pressure (BHP at closure) and the implied closure
stress gradient if a datum depth is supplied.

References
----------
Barree, R.D., Barree, V.L., Craig, D.P. (2009) SPE-107877-PA.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivatives import semilog_derivative


@dataclass
class ClosureResult:
    """Container for a fracture-closure pick."""

    closure_G: float
    closure_pressure: float
    closure_time: float | None
    reference_slope: float
    closure_gradient: float | None = None
    note: str = ""


def pick_closure(
    G: np.ndarray,
    p: np.ndarray,
    t: np.ndarray | None = None,
    early_fraction: float = 0.35,
    departure_tol: float = 0.04,
    persist: int = 4,
    smooth: int = 5,
    datum_depth_ft: float | None = None,
) -> ClosureResult:
    """
    Identify fracture closure on the G-function derivative.

    Parameters
    ----------
    G : array_like
        G-function values (increasing).
    p : array_like
        Pressure during falloff.
    t : array_like, optional
        Physical time aligned with G, used only to report closure time.
    early_fraction : float
        Fraction of the G range, from the start, assumed to be the straight
        pre-closure section used to fit the through-origin reference line.
    departure_tol : float
        Relative departure (data below line, normalized by the line value) that
        signals closure.
    persist : int
        Number of consecutive points that must satisfy the departure tolerance,
        to reject single-point noise.
    smooth : int
        Smoothing half-window for the semilog derivative.
    datum_depth_ft : float, optional
        Gauge / perforation datum depth (ft). If given, closure_gradient is
        returned in psi/ft.

    Returns
    -------
    ClosureResult
    """
    G = np.asarray(G, dtype=float)
    p = np.asarray(p, dtype=float)

    sgd = semilog_derivative(G, p, smooth=smooth)

    valid = (G > 0) & np.isfinite(sgd)
    Gv = G[valid]
    sv = sgd[valid]
    pv = p[valid]
    tv = None if t is None else np.asarray(t, dtype=float)[valid]

    if len(Gv) < 8:
        raise ValueError("Too few points to pick closure.")

    # Fit a through-origin line to the early straight section: slope = mean(sgd/G).
    gmax = Gv.max()
    early = Gv <= early_fraction * gmax
    if early.sum() < 3:
        early = np.arange(len(Gv)) < max(3, len(Gv) // 5)
    slope = float(np.sum(Gv[early] * sv[early]) / np.sum(Gv[early] ** 2))

    reference = slope * Gv
    # relative departure below the line
    rel_dev = (reference - sv) / (np.abs(reference) + 1e-9)

    # find first index (past the early section) with `persist` consecutive
    # points exceeding the tolerance
    start = int(np.argmax(Gv > early_fraction * gmax))
    closure_idx = None
    run = 0
    for i in range(start, len(Gv)):
        if rel_dev[i] > departure_tol:
            run += 1
            if run >= persist:
                closure_idx = i - persist + 1
                break
        else:
            run = 0

    if closure_idx is None:
        # fall back to the maximum of G*dP/dG, a common closure proxy
        closure_idx = int(np.argmax(sv))
        note = (
            "No clean departure found; closure set to the peak of G*dP/dG "
            "(fallback). Inspect manually."
        )
    else:
        note = "Closure at first persistent departure below the reference line."

    closure_G = float(Gv[closure_idx])
    closure_p = float(pv[closure_idx])
    closure_t = None if tv is None else float(tv[closure_idx])

    gradient = None
    if datum_depth_ft and datum_depth_ft > 0:
        gradient = closure_p / datum_depth_ft

    return ClosureResult(
        closure_G=closure_G,
        closure_pressure=closure_p,
        closure_time=closure_t,
        reference_slope=slope,
        closure_gradient=gradient,
        note=note,
    )


def net_pressure(isip: float, closure_pressure: float) -> float:
    """
    Net pressure = ISIP - closure pressure.

    A measure of the energy stored in the fracture / the rock's resistance to
    extension (process-zone stress). Drives fracture width and, with modulus,
    fracture compliance.
    """
    return float(isip - closure_pressure)
