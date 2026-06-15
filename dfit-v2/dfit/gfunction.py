"""
G-function and modified time functions for DFIT before-closure analysis.

The G-function (Nolte, 1979) is the workhorse diagnostic for fracture-injection
falloff. It transforms shut-in time into a dimensionless variable whose
semilog derivative, G * dP/dG, has characteristic shapes that reveal the
fracture leakoff mechanism and the moment of fracture closure.

References
----------
Nolte, K.G. (1979) "Determination of Fracture Parameters from Fracturing
    Pressure Decline." SPE-8341-MS.
Barree, R.D., Barree, V.L., Craig, D.P. (2009) "Holistic Fracture
    Diagnostics." SPE-107877-PA.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) "Diagnostic Fracture
    Injection Tests: Common Mistakes, Misfires, and Misdiagnoses."
    SPE Prod & Oper 30 (2): 84-98. SPE-169539-PA.
"""

from __future__ import annotations

import numpy as np


# Leakoff bounding exponents (Nolte). alpha = 1.0 corresponds to the
# low-efficiency (high-leakoff) limit; alpha = 0.5 to the high-efficiency limit.
ALPHA_HIGH_LEAKOFF = 1.0
ALPHA_LOW_LEAKOFF = 0.5


def _g0(alpha: float) -> float:
    """Return g(0), the g-function evaluated at dimensionless time zero."""
    if np.isclose(alpha, ALPHA_LOW_LEAKOFF):
        return 4.0 / 3.0
    if np.isclose(alpha, ALPHA_HIGH_LEAKOFF):
        return np.pi / 2.0
    raise ValueError(
        "alpha must be 0.5 (low leakoff) or 1.0 (high leakoff); "
        f"got {alpha}."
    )


def g_function(delta_td: np.ndarray, alpha: float = ALPHA_HIGH_LEAKOFF) -> np.ndarray:
    """
    Nolte g-function for the two bounding leakoff assumptions.

    Parameters
    ----------
    delta_td : array_like
        Dimensionless shut-in time, (t - tp) / tp, where tp is the pumping
        (injection) time and t is total time since start of injection.
    alpha : float
        Leakoff exponent. 1.0 (default) is the high-leakoff bound; 0.5 is the
        low-leakoff / high-efficiency bound. Real tests fall between the two.

    Returns
    -------
    g : ndarray
        Dimensionless g-function values.
    """
    dtd = np.asarray(delta_td, dtype=float)
    if np.any(dtd < 0):
        raise ValueError("delta_td must be non-negative.")

    if np.isclose(alpha, ALPHA_LOW_LEAKOFF):
        # High-efficiency bound (Nolte): g = 4/3 [ (1+dtD)^1.5 - dtD^1.5 ]
        return (4.0 / 3.0) * ((1.0 + dtd) ** 1.5 - dtd ** 1.5)

    if np.isclose(alpha, ALPHA_HIGH_LEAKOFF):
        # Low-efficiency bound (Nolte):
        # g = (1+dtD) arcsin[(1+dtD)^-0.5] + dtD^0.5
        return (1.0 + dtd) * np.arcsin((1.0 + dtd) ** -0.5) + np.sqrt(dtd)

    raise ValueError("alpha must be 0.5 or 1.0.")


def G_function(delta_td: np.ndarray, alpha: float = ALPHA_HIGH_LEAKOFF) -> np.ndarray:
    """
    Nolte G-function: G = (4 / pi) [ g(dtD) - g(0) ].

    G is zero at shut-in and increases monotonically with shut-in time. It is
    the standard abscissa for the before-closure diagnostic plot.

    Parameters
    ----------
    delta_td : array_like
        Dimensionless shut-in time (t - tp) / tp.
    alpha : float
        Leakoff exponent, 0.5 or 1.0.

    Returns
    -------
    G : ndarray
        G-function values.
    """
    return (4.0 / np.pi) * (g_function(delta_td, alpha) - _g0(alpha))


def G_from_time(
    t: np.ndarray,
    t_pump: float,
    alpha: float = ALPHA_HIGH_LEAKOFF,
) -> np.ndarray:
    """
    Convenience wrapper: compute G directly from physical time.

    Parameters
    ----------
    t : array_like
        Time since the start of injection (same units as t_pump). Only points
        with t >= t_pump (i.e. the falloff) are meaningful.
    t_pump : float
        Pumping (injection) time.
    alpha : float
        Leakoff exponent.

    Returns
    -------
    G : ndarray
        G-function values. Points during injection (t < t_pump) return NaN.
    """
    t = np.asarray(t, dtype=float)
    if t_pump <= 0:
        raise ValueError("t_pump must be positive.")
    delta_td = (t - t_pump) / t_pump
    G = np.full_like(t, np.nan, dtype=float)
    mask = delta_td >= 0
    G[mask] = G_function(delta_td[mask], alpha)
    return G


def time_from_G(
    G_target: np.ndarray,
    t_pump: float,
    alpha: float = ALPHA_HIGH_LEAKOFF,
) -> np.ndarray:
    """
    Invert the G-function to recover physical time from G values.

    Useful for synthetic-data generation, where it is natural to specify a
    pressure response as a function of G and then map back to a time series.
    The G-function is monotonic in shut-in time, so the inverse is unique.

    Parameters
    ----------
    G_target : array_like
        G-function values to invert (must be >= 0).
    t_pump : float
        Pumping time.
    alpha : float
        Leakoff exponent.

    Returns
    -------
    t : ndarray
        Physical time since start of injection.
    """
    G_target = np.asarray(G_target, dtype=float)
    if np.any(G_target < 0):
        raise ValueError("G_target must be non-negative.")

    # Build a dense monotonic lookup of G(delta_td) and invert by interpolation.
    dtd_grid = np.concatenate([
        np.linspace(0.0, 5.0, 4000),
        np.linspace(5.0, 5000.0, 4000),
    ])
    G_grid = G_function(dtd_grid, alpha)
    # numpy.interp requires increasing x; G_grid is monotonic increasing.
    dtd = np.interp(G_target, G_grid, dtd_grid)
    return t_pump + dtd * t_pump


def superposition_time(t: np.ndarray, t_pump: float) -> np.ndarray:
    """
    Radial superposition (Horner-like) time function for the falloff.

    Defined as ln[(t) / (t - t_pump)] for the shut-in period. A semilog plot of
    pressure against this function is the classic radial-flow diagnostic.

    Parameters
    ----------
    t : array_like
        Time since start of injection.
    t_pump : float
        Pumping time.

    Returns
    -------
    tau : ndarray
        Superposition time values; NaN during injection.
    """
    t = np.asarray(t, dtype=float)
    tau = np.full_like(t, np.nan, dtype=float)
    mask = t > t_pump
    tau[mask] = np.log(t[mask] / (t[mask] - t_pump))
    return tau


def sqrt_shutin_time(t: np.ndarray, t_pump: float) -> np.ndarray:
    """
    Square-root-of-shut-in-time function, sqrt(t - t_pump).

    The sqrt(t) plot is an independent before-closure method that cross-checks
    the G-function. Fracture closure appears as a departure from the straight
    line on the pressure-vs-sqrt(t) derivative.

    Parameters
    ----------
    t : array_like
        Time since start of injection.
    t_pump : float
        Pumping time.

    Returns
    -------
    s : ndarray
        sqrt of shut-in time; NaN during injection.
    """
    t = np.asarray(t, dtype=float)
    s = np.full_like(t, np.nan, dtype=float)
    mask = t >= t_pump
    s[mask] = np.sqrt(t[mask] - t_pump)
    return s
