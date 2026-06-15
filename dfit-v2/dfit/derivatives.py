"""
Derivative computation and smoothing for DFIT diagnostics.

DFIT interpretation lives or dies on the derivative of pressure with respect to
a time function. Field gauges are noisy, and a raw point-to-point derivative is
useless. This module provides robust derivative estimators:

* a windowed (Bourdet-style) log-derivative that smooths over a configurable
  span in the independent variable, and
* the semilog derivative G * dP/dG used for the before-closure G-function plot.

References
----------
Bourdet, D., Ayoub, J.A., Pirard, Y.M. (1989) "Use of Pressure Derivative in
    Well Test Interpretation." SPE-12777-PA.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

import numpy as np


def bourdet_derivative(
    x: np.ndarray,
    y: np.ndarray,
    L: float = 0.1,
) -> np.ndarray:
    """
    Bourdet windowed derivative dy/d(ln x) with smoothing span L.

    For each point i, the derivative uses the nearest points at least L apart
    (in ln x) on each side, then takes a weighted central difference. This is
    the standard pressure-transient derivative and rejects high-frequency gauge
    noise far better than a naive finite difference.

    Parameters
    ----------
    x : array_like
        Independent variable, strictly positive and increasing (e.g. shut-in
        time). Used in natural-log space.
    y : array_like
        Dependent variable (e.g. pressure change).
    L : float
        Smoothing window in ln(x) units. Typical 0.05-0.3. Larger is smoother.

    Returns
    -------
    deriv : ndarray
        dy/d(ln x) at each point. Endpoints fall back to one-sided differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if np.any(x <= 0):
        raise ValueError("x must be strictly positive for log derivative.")

    lnx = np.log(x)
    n = len(x)
    deriv = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # left point: furthest j < i with lnx[i] - lnx[j] >= L
        jl = i
        while jl > 0 and (lnx[i] - lnx[jl]) < L:
            jl -= 1
        # right point: nearest k > i with lnx[k] - lnx[i] >= L
        kr = i
        while kr < n - 1 and (lnx[kr] - lnx[i]) < L:
            kr += 1

        dl = lnx[i] - lnx[jl]
        dr = lnx[kr] - lnx[i]

        if dl > 0 and dr > 0:
            # Bourdet weighted central difference
            sl = (y[i] - y[jl]) / dl
            sr = (y[kr] - y[i]) / dr
            deriv[i] = (sl * dr + sr * dl) / (dl + dr)
        elif dr > 0:
            deriv[i] = (y[kr] - y[i]) / dr
        elif dl > 0:
            deriv[i] = (y[i] - y[jl]) / dl
    return deriv


def semilog_derivative(
    G: np.ndarray,
    p: np.ndarray,
    smooth: int = 5,
) -> np.ndarray:
    """
    Semilog G-function derivative G * dP/dG (industry sign convention).

    This is the single most important before-closure curve. Plotted against G,
    its shape diagnoses the leakoff mechanism, and the point where it departs
    from a straight line through the origin marks fracture closure.

    During a falloff the pressure DECLINES, so dP/dG is negative. By long-
    standing convention the G-function derivative is plotted as a POSITIVE
    quantity built from the pressure decline; this function therefore returns
    G * (-dP/dG) = -G * dP/dG, which is positive for a normal falloff and rises
    on a straight line through the origin.

    Parameters
    ----------
    G : array_like
        G-function values, increasing.
    p : array_like
        Bottomhole (or surface) pressure during falloff.
    smooth : int
        Half-window (in points) for a centered moving-average pre-smoothing of
        the raw dP/dG. Set 0 to disable.

    Returns
    -------
    sgd : ndarray
        G * dP/dG in the positive industry convention.
    """
    G = np.asarray(G, dtype=float)
    p = np.asarray(p, dtype=float)

    # central difference of p w.r.t. G (negative during a falloff)
    dpdG = np.gradient(p, G)

    if smooth and smooth > 0:
        dpdG = _moving_average(dpdG, smooth)

    return -G * dpdG


def first_derivative(G: np.ndarray, p: np.ndarray, smooth: int = 5) -> np.ndarray:
    """First derivative dP/dG with optional smoothing (negative during falloff)."""
    G = np.asarray(G, dtype=float)
    p = np.asarray(p, dtype=float)
    dpdG = np.gradient(p, G)
    if smooth and smooth > 0:
        dpdG = _moving_average(dpdG, smooth)
    return dpdG


def _moving_average(a: np.ndarray, half_window: int) -> np.ndarray:
    """Centered moving average with edge handling via reflection."""
    a = np.asarray(a, dtype=float)
    w = 2 * half_window + 1
    if w >= len(a):
        return np.full_like(a, np.nanmean(a))
    kernel = np.ones(w) / w
    padded = np.pad(a, half_window, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")
