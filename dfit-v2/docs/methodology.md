# Methodology

This document records the physics and the numerical choices behind each module,
so that results can be reproduced and defended.

## 1. The G-function

The G-function (Nolte, 1979) is a dimensionless time transform for the falloff
after a fracture injection. With pumping time `tp` and total time `t`, the
dimensionless shut-in time is

```
dtD = (t - tp) / tp
```

Nolte's g-function has two bounding forms set by the leakoff exponent alpha:

- **alpha = 1.0** (high-leakoff / low-efficiency bound):
  `g(dtD) = (1 + dtD) * arcsin((1 + dtD)^(-1/2)) + dtD^(1/2)`, with `g(0) = pi/2`.
- **alpha = 0.5** (low-leakoff / high-efficiency bound):
  `g(dtD) = (4/3) * ((1 + dtD)^(3/2) - dtD^(3/2))`, with `g(0) = 4/3`.

The G-function normalises this against its value at shut-in:

```
G(dtD) = (4 / pi) * (g(dtD) - g(0))
```

so `G(0) = 0` and `G` increases monotonically with shut-in time. Because it is
monotonic, it can be inverted; `time_from_G` builds a dense lookup of
`G(dtD)` and interpolates, which the synthetic generator uses to convert a
pressure-versus-G specification back into a time series.

## 2. Derivatives

DFIT interpretation is a derivative exercise, and field gauges are noisy.

- **Bourdet derivative** (`bourdet_derivative`): for each point, the routine
  reaches at least `L` units in `ln(x)` to each side, then takes a
  distance-weighted central difference. This is the standard pressure-transient
  derivative; `L` trades resolution against smoothness.
- **Semilog G-function derivative** (`semilog_derivative`): returns
  `-G * dP/dG`. The minus sign encodes the industry convention: pressure
  declines during a falloff, so `dP/dG < 0`, and the diagnostic is plotted as a
  positive quantity that rises on a straight line through the origin during
  normal leakoff.

## 3. ISIP

The instantaneous shut-in pressure is rarely the pressure at the instant the
pump stops, because near-well tortuosity and wellbore decompression keep
injecting fluid into the fracture for seconds to minutes afterwards.

- `isip_log_extrapolation` fits the stable pressure-versus-log(shut-in-time)
  trend and compares the earliest recorded falloff pressure against the
  extrapolated trend. If the early data sit above the trend, that excess is
  decompression overshoot and the lower extrapolated value is used; otherwise
  the earliest falloff pressure is the ISIP.
- `wellbore_decompression_pressure` forward-models the early decline from a
  square-root-of-rate near-well friction law, a wellbore stiffness (psi per
  barrel of volume change), and the resulting fluid expansion. With the
  worked-example inputs from SPE-169539-PA it reproduces the reported decline
  from 3,640 to ~3,562 psi in the first second after shut-in.

## 4. Leakoff classification

The classifier restricts attention to the pre-closure window (and stays inside
`0.9 * closure_G` to avoid the post-closure rollover), fits a straight reference
line, and computes two shape metrics:

- **normalized signed area** between `G*dP/dG` and a through-origin reference
  line - positive for a hump above the line (PDL), negative for a belly below it
  (height recession), and
- **intercept ratio** - the fitted line's intercept normalised by its mid-range
  value, which is clearly positive for fracture-tip extension.

Smooth sigmoid / bell discriminants convert the two metrics into per-regime
scores, and confidence is the margin between the top two scores. Thresholds were
calibrated against the synthetic archetypes and verified to recover all four
regimes across many noise realisations.

## 5. Closure

`pick_closure` fits a through-origin line to the early straight section of
`G*dP/dG`, then flags closure at the first G where the data fall persistently
below that line by more than a relative tolerance (with a persistence count to
reject single-point noise). If no clean departure is found it falls back to the
peak of `G*dP/dG`.

The pick carries a small systematic late bias (order 0.7-1.1 in G against the
synthetic truth) but very low variance. This is documented rather than tuned
away: real closure picks are ambiguous by a comparable margin, and a stable,
slightly-late pick that the analyst refines against the sqrt(t) and log-log
views is more honest than one over-fitted to synthetic data.

## 6. After-closure analysis

`aca_derivative` restricts to data beyond roughly three times the closure time
(the Barree rule of thumb that reservoir transients need about half a log cycle
past closure to stabilise) and computes the log-log Bourdet derivative of the
pressure change.

`detect_flow_regimes` reads the local slope on the log-log plot: about -1/2 for
pseudo-linear flow, about -1 for pseudo-radial flow. Crucially, a radial flag is
only reported as *supported* when it appears at late time and spans at least a
third of a log cycle. When it does not, pore pressure is estimated from the
linear-flow extrapolation, or left unestimated, rather than forced onto a Horner
straight line - directly implementing the central caution of SPE-169539-PA.

## 7. Synthetic generation

The generator specifies the pre-closure pressure-drop derivative `dD/dG` as a
positive regime-specific shape, scales it so the drop integrates exactly to
`ISIP - closure_pressure` at the chosen `closure_G` (inserting that node exactly
so the endpoint is exact), appends an after-closure reservoir decline, maps G
back to time via the inverse G-function, prepends an injection ramp, and adds
gauge noise. It reproduces the diagnostic signatures faithfully; it is not a
full fracture-mechanics simulator.
