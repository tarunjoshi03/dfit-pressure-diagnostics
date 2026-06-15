"""
Leakoff-regime classification from the G-function derivative signature.

This is the interpretive heart of a DFIT. Barree et al. (2009, 2015) showed that
the shape of the semilog derivative G * dP/dG, relative to a straight reference
line drawn through the origin, diagnoses the dominant leakoff mechanism. Getting
this right determines whether the closure pick - and therefore closure stress,
net pressure, and permeability - is meaningful.

Four canonical signatures are recognised:

1. NORMAL leakoff
   G * dP/dG rises on a straight line through the origin; closure is the point
   where it first departs (rolls over) below that line.

2. PRESSURE-DEPENDENT leakoff (PDL)
   The derivative bows ABOVE the reference line (a "hump") before closure,
   caused by secondary fractures / fissures opening while net pressure is high.
   Closure is taken after the hump, where the data return to and leave the line.

3. FRACTURE-HEIGHT RECESSION / TRANSVERSE STORAGE
   The derivative sags BELOW the reference line (a "belly") before the final
   straight-line section, reflecting variable fracture compliance / storage.

4. FRACTURE-TIP EXTENSION
   The straight-line section does NOT extrapolate through the origin but has a
   positive intercept, indicating continued tip extension after shut-in.

The classifier fits a reference line to the pre-closure straight section and
measures the signed area between the data and that line, plus the line's
intercept, to assign a regime with a confidence score.

References
----------
Barree, R.D., Barree, V.L., Craig, D.P. (2009) SPE-107877-PA.
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) SPE-169539-PA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .derivatives import semilog_derivative


REGIMES = (
    "normal",
    "pressure_dependent",
    "height_recession",
    "tip_extension",
)


@dataclass
class LeakoffResult:
    """Result of a leakoff-regime classification."""

    regime: str
    confidence: float
    reference_slope: float
    reference_intercept: float
    normalized_area: float
    intercept_ratio: float
    scores: dict = field(default_factory=dict)
    note: str = ""


def classify_leakoff(
    G: np.ndarray,
    p: np.ndarray,
    closure_G: float | None = None,
    fit_fraction=(0.4, 0.9),
    smooth: int = 5,
) -> LeakoffResult:
    """
    Classify the leakoff regime from a G-function falloff.

    Parameters
    ----------
    G : array_like
        G-function values (increasing, starting near 0).
    p : array_like
        Pressure during falloff aligned with G.
    closure_G : float, optional
        If known, restrict the analysis to G <= closure_G (the pre-closure
        window). If omitted, the whole curve is used and the late straight
        section is inferred from `fit_fraction`.
    fit_fraction : tuple(float, float)
        Fractional G-window used to fit the reference straight line, taken from
        the pre-closure portion. Defaults to the outer pre-closure section
        where normal leakoff is most linear.
    smooth : int
        Smoothing half-window passed to the semilog derivative.

    Returns
    -------
    LeakoffResult
    """
    G = np.asarray(G, dtype=float)
    p = np.asarray(p, dtype=float)

    sgd = semilog_derivative(G, p, smooth=smooth)

    # Restrict to the pre-closure window, staying safely INSIDE closure so the
    # sharp post-closure rollover does not contaminate the shape metrics.
    if closure_G is not None:
        inner = 0.9 * closure_G
        win = (G > 0) & (G <= inner)
    else:
        win = G > 0
    Gw = G[win]
    sw = sgd[win]
    finite = np.isfinite(Gw) & np.isfinite(sw)
    Gw, sw = Gw[finite], sw[finite]

    if len(Gw) < 5:
        raise ValueError("Too few finite points to classify leakoff.")

    # Fit reference line over the outer portion of the pre-closure window, where
    # normal leakoff is most linear.
    lo, hi = fit_fraction
    gmax = Gw.max()
    fitmask = (Gw >= lo * gmax) & (Gw <= hi * gmax)
    if fitmask.sum() < 3:
        fitmask = np.ones_like(Gw, dtype=bool)

    slope, intercept = np.polyfit(Gw[fitmask], sw[fitmask], 1)
    reference = slope * Gw  # line forced through the origin for area comparison

    # Signed, normalized area between data and the through-origin reference.
    # Positive -> data sits above the line (PDL hump);
    # negative -> data sits below the line (height recession belly).
    area = np.trapezoid(sw - reference, Gw)
    scale = np.trapezoid(np.abs(reference) + 1e-12, Gw)
    norm_area = float(area / scale)

    # Intercept of the fitted straight section, normalized by its mid value.
    mid_val = slope * np.median(Gw[fitmask]) + intercept
    intercept_ratio = float(intercept / (abs(mid_val) + 1e-12))

    scores = _regime_scores(norm_area, intercept_ratio)
    regime = max(scores, key=scores.get)
    confidence = _confidence(scores)

    note = _describe(regime, norm_area, intercept_ratio)

    return LeakoffResult(
        regime=regime,
        confidence=confidence,
        reference_slope=float(slope),
        reference_intercept=float(intercept),
        normalized_area=norm_area,
        intercept_ratio=intercept_ratio,
        scores=scores,
        note=note,
    )


def _regime_scores(norm_area: float, intercept_ratio: float) -> dict:
    """
    Convert the two shape metrics into soft scores for each regime.

    Calibrated against the synthetic archetypes:

      regime              norm_area   intercept_ratio
      normal               ~  0.14       ~ 0.00
      pressure_dependent   ~ +0.56       ~ +0.19
      height_recession     ~ -0.35       ~ -0.91
      tip_extension        ~ +1.42       ~ +0.41

    Discrimination logic:
      * height recession is the only regime with a clearly NEGATIVE area
        (belly below the line);
      * tip extension has the largest positive area AND a clear positive
        intercept;
      * pressure-dependent has a moderate positive area;
      * normal has a small (near-zero) area and near-zero intercept.
    """
    # Belly below the line -> height recession (negative area is its signature).
    below = _sigmoid(-norm_area, center=0.12, width=0.08)

    # Tip extension: LARGE positive area together with a clear positive intercept.
    # Lowered area center so a tip area near 1.0 fires strongly.
    tip = _sigmoid(norm_area, center=0.72, width=0.14) * \
        _sigmoid(intercept_ratio, center=0.34, width=0.07)

    # Pressure-dependent: MODERATE positive area (a bell, so it does not also
    # capture the much larger area of tip extension), with a modest intercept.
    pdl = _bell((norm_area - 0.50), width=0.22) * (1.0 - tip)

    # Normal: small area near zero, small intercept (wide bells).
    normal = _bell(norm_area, width=0.26) * _bell(intercept_ratio, width=0.22)

    # Height recession should not also fire as normal/pdl when area is negative.
    not_below = 1.0 - below
    scores = {
        "height_recession": below,
        "tip_extension": tip * not_below,
        "pressure_dependent": pdl * not_below,
        "normal": normal * not_below,
    }
    total = sum(scores.values()) + 1e-12
    return {k: float(v / total) for k, v in scores.items()}


def _confidence(scores: dict) -> float:
    """Confidence = top score minus runner-up, scaled to [0, 1]."""
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return float(vals[0])
    return float(np.clip((vals[0] - vals[1]) / (vals[0] + 1e-12), 0.0, 1.0))


def _sigmoid(x: float, center: float, width: float) -> float:
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def _bell(x: float, width: float) -> float:
    return float(np.exp(-0.5 * (x / width) ** 2))


def _describe(regime: str, area: float, intercept_ratio: float) -> str:
    text = {
        "normal": (
            "G*dP/dG tracks a straight line through the origin; standard "
            "Carter leakoff. Closure is the departure below the line."
        ),
        "pressure_dependent": (
            "G*dP/dG bows above the reference line (hump): pressure-dependent "
            "leakoff from secondary fractures / fissures opening at high net "
            "pressure. Pick closure after the hump."
        ),
        "height_recession": (
            "G*dP/dG sags below the reference line (belly): variable fracture "
            "compliance / transverse storage or fracture-height recession."
        ),
        "tip_extension": (
            "Straight section has a positive intercept: continued fracture-tip "
            "extension after shut-in. Closure stress may be overestimated if "
            "the through-origin assumption is used."
        ),
    }[regime]
    return f"{text} (norm_area={area:+.3f}, intercept_ratio={intercept_ratio:+.3f})"
