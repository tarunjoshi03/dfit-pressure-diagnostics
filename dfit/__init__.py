"""
dfit-pressure-diagnostics
=========================

An open-source Python toolkit for Diagnostic Fracture Injection Test (DFIT)
interpretation: G-function before-closure analysis, automated fracture-closure
picking, leakoff-regime classification, after-closure flow-regime analysis, and
a synthetic DFIT generator for testing and teaching.

Developed as part of DFIT pressure-diagnostics research in the Harold Vance
Department of Petroleum Engineering at Texas A&M University.

Primary reference
-----------------
Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) "Diagnostic Fracture
Injection Tests: Common Mistakes, Misfires, and Misdiagnoses."
SPE Production & Operations 30 (2): 84-98. SPE-169539-PA.
"""

from .gfunction import (
    g_function,
    G_function,
    G_from_time,
    time_from_G,
    superposition_time,
    sqrt_shutin_time,
    ALPHA_HIGH_LEAKOFF,
    ALPHA_LOW_LEAKOFF,
)
from .derivatives import (
    bourdet_derivative,
    semilog_derivative,
    first_derivative,
)
from .isip import (
    isip_log_extrapolation,
    wellbore_decompression_pressure,
    ISIPResult,
)
from .closure import pick_closure, net_pressure, ClosureResult
from .leakoff import classify_leakoff, LeakoffResult, REGIMES
from .afterclosure import (
    aca_derivative,
    detect_flow_regimes,
    FlowRegimeResult,
)
from .synthetic import generate_dfit, to_dataframe, SyntheticDFIT
from .workflow import analyze_dfit, DFITAnalysis
from .fracdesign import fracture_design_from_dfit, design_summary, FractureDesignParams

__version__ = "1.0.0"

__all__ = [
    # gfunction
    "g_function", "G_function", "G_from_time", "time_from_G",
    "superposition_time", "sqrt_shutin_time",
    "ALPHA_HIGH_LEAKOFF", "ALPHA_LOW_LEAKOFF",
    # derivatives
    "bourdet_derivative", "semilog_derivative", "first_derivative",
    # isip
    "isip_log_extrapolation", "wellbore_decompression_pressure", "ISIPResult",
    # closure
    "pick_closure", "net_pressure", "ClosureResult",
    # leakoff
    "classify_leakoff", "LeakoffResult", "REGIMES",
    # afterclosure
    "aca_derivative", "detect_flow_regimes", "FlowRegimeResult",
    # synthetic
    "generate_dfit", "to_dataframe", "SyntheticDFIT",
    # workflow
    "analyze_dfit", "DFITAnalysis",
    # fracdesign
    "fracture_design_from_dfit", "design_summary", "FractureDesignParams",
    "__version__",
]
