"""
End-to-end DFIT interpretation workflow.

Ties the individual modules into a single call that takes raw time / pressure /
rate arrays and returns a complete interpretation: ISIP, leakoff regime,
fracture closure, net pressure, and after-closure flow regime with a pore
pressure estimate.

This mirrors how an analyst actually works a DFIT:

    1. find ISIP (correct for wellbore decompression),
    2. build the G-function and its derivative,
    3. classify the leakoff mechanism,
    4. pick closure,
    5. work the after-closure period for reservoir properties.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .gfunction import G_from_time, ALPHA_HIGH_LEAKOFF
from .isip import isip_log_extrapolation
from .leakoff import classify_leakoff
from .closure import pick_closure, net_pressure
from .afterclosure import aca_derivative, detect_flow_regimes


@dataclass
class DFITAnalysis:
    """Full interpretation result."""

    isip_psi: float
    leakoff_regime: str
    leakoff_confidence: float
    closure_G: float
    closure_pressure_psi: float
    closure_time_min: float | None
    net_pressure_psi: float
    after_closure_regime: str
    radial_supported: bool
    pore_pressure_psi: float | None
    notes: dict

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        lines = [
            "DFIT INTERPRETATION SUMMARY",
            "=" * 40,
            f"ISIP                 : {self.isip_psi:,.0f} psi",
            f"Leakoff regime       : {self.leakoff_regime} "
            f"(confidence {self.leakoff_confidence:.2f})",
            f"Closure G            : {self.closure_G:.2f}",
            f"Closure pressure     : {self.closure_pressure_psi:,.0f} psi",
            f"Closure time         : "
            + (f"{self.closure_time_min:,.1f} min"
               if self.closure_time_min is not None else "n/a"),
            f"Net pressure         : {self.net_pressure_psi:,.0f} psi",
            f"After-closure regime : {self.after_closure_regime}",
            f"Radial flow supported: {self.radial_supported}",
            f"Pore pressure        : "
            + (f"{self.pore_pressure_psi:,.0f} psi"
               if self.pore_pressure_psi is not None else "not estimated"),
        ]
        return "\n".join(lines)


def analyze_dfit(
    time_min: np.ndarray,
    pressure_psi: np.ndarray,
    rate_bpm: np.ndarray,
    t_pump_min: float | None = None,
    alpha: float = ALPHA_HIGH_LEAKOFF,
    datum_depth_ft: float | None = None,
) -> DFITAnalysis:
    """
    Run the full DFIT interpretation pipeline.

    Parameters
    ----------
    time_min : array_like
        Time since start of injection (minutes).
    pressure_psi : array_like
        Pressure (psi).
    rate_bpm : array_like
        Injection rate (bbl/min). Used to detect shut-in if t_pump not given.
    t_pump_min : float, optional
        Pumping time. If omitted, inferred as the last time the rate is nonzero.
    alpha : float
        Leakoff exponent for the G-function.
    datum_depth_ft : float, optional
        Datum depth for a closure-gradient estimate.

    Returns
    -------
    DFITAnalysis
    """
    time_min = np.asarray(time_min, dtype=float)
    pressure_psi = np.asarray(pressure_psi, dtype=float)
    rate_bpm = np.asarray(rate_bpm, dtype=float)

    # Infer pump time from the rate schedule if not provided.
    if t_pump_min is None:
        nz = np.where(rate_bpm > 0)[0]
        if len(nz) == 0:
            raise ValueError("Cannot infer pump time: rate is zero everywhere.")
        t_pump_min = float(time_min[nz[-1]])

    # ISIP (log-time extrapolation, wellbore-decompression corrected).
    isip = isip_log_extrapolation(time_min, pressure_psi, t_pump_min)

    # G-function over the falloff.
    G = G_from_time(time_min, t_pump_min, alpha)
    fall = np.isfinite(G)
    Gf = G[fall]
    pf = pressure_psi[fall]
    tf = time_min[fall]

    order = np.argsort(Gf)
    Gf, pf, tf = Gf[order], pf[order], tf[order]

    # Closure pick (needed to bound the pre-closure window for leakoff).
    clo = pick_closure(Gf, pf, t=tf, datum_depth_ft=datum_depth_ft)

    # Leakoff classification over the pre-closure window only.
    leak = classify_leakoff(Gf, pf, closure_G=clo.closure_G)

    npress = net_pressure(isip.isip, clo.closure_pressure)

    # After-closure flow regimes.
    t_closure = clo.closure_time if clo.closure_time else t_pump_min * 2
    dt, dp, deriv = aca_derivative(time_min, pressure_psi, t_closure)
    flow = detect_flow_regimes(dt, dp, deriv)

    return DFITAnalysis(
        isip_psi=isip.isip,
        leakoff_regime=leak.regime,
        leakoff_confidence=leak.confidence,
        closure_G=clo.closure_G,
        closure_pressure_psi=clo.closure_pressure,
        closure_time_min=clo.closure_time,
        net_pressure_psi=npress,
        after_closure_regime=(
            "linear" if flow.linear_flow_window else
            ("radial" if flow.radial_supported else "undeveloped")
        ),
        radial_supported=flow.radial_supported,
        pore_pressure_psi=flow.pore_pressure,
        notes={
            "isip": isip.note,
            "leakoff": leak.note,
            "closure": clo.note,
            "after_closure": flow.note,
        },
    )
