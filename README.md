# dfit-pressure-diagnostics

**An open-source Python toolkit for Diagnostic Fracture Injection Test (DFIT) interpretation.**

G-function before-closure analysis, automated fracture-closure picking, leakoff-regime classification, after-closure flow-regime analysis, probabilistic uncertainty quantification, fracture design bridge, and a synthetic DFIT generator - in one documented, tested package.

Developed as part of DFIT pressure-diagnostics research in the Harold Vance Department of Petroleum Engineering at Texas A&M University.

---

## Why this exists

DFITs are small pump-in/falloff tests that yield closure stress, net pressure, leakoff mechanism, pore pressure, and permeability - the inputs that calibrate every hydraulic-fracture design. Yet, as Barree, Miskimins and Gilbert document in SPE-169539-PA, a large fraction of tests are misanalysed: ISIP is read off the wrong point, closure is picked inside a pressure-dependent-leakoff hump, and a pseudo-radial flow regime is forced out of data that never developed one.

The open-source tooling for DFITs is thin. A couple of G-function plotters exist, but there is no clean, well-documented Python package that brings together:

- **before-closure** G-function diagnostics with proper noise-tolerant derivatives,
- **leakoff-regime classification** into the four canonical Barree signatures,
- **after-closure** flow-regime analysis that refuses to over-call radial flow,
- a **probabilistic classifier** that outputs regime probabilities and closure stress confidence intervals, and
- a **synthetic data generator** so the whole workflow can be exercised without proprietary well data.

This toolkit fills that gap.

---

## Install

```bash
git clone https://github.com/tarunjoshi03/dfit-pressure-diagnostics.git
cd dfit-pressure-diagnostics
pip install -e .
```

Requires Python 3.9+, NumPy, SciPy, Matplotlib, and pandas.

---

## Quick start

```python
import dfit

# 1. Generate a synthetic DFIT with known ground truth
d = dfit.generate_dfit(
    regime="pressure_dependent",
    isip_psi=8000, closure_pressure_psi=6800, closure_G=6.0, seed=7,
)

# 2. Run the full interpretation pipeline
result = dfit.analyze_dfit(d.time_min, d.pressure_psi, d.rate_bpm)
print(result.summary())
```

```
DFIT INTERPRETATION SUMMARY
========================================
ISIP                 : 8,000 psi
Leakoff regime       : pressure_dependent (confidence 0.80)
Closure G            : 6.94
Closure pressure     : 6,083 psi
Closure time         : 62.2 min
Net pressure         : 1,917 psi
After-closure regime : undeveloped
Radial flow supported: False
Pore pressure        : not estimated
```

```python
# 3. Probabilistic classification with closure stress uncertainty
from dfit.probabilistic import probabilistic_classify
import numpy as np

fall = np.isfinite(d.G)
G = d.G[fall]; p = d.pressure_psi[fall]
o = np.argsort(G); G, p = G[o], p[o]

prob = probabilistic_classify(G, p, noise_psi=3.5, n_bootstrap=200)
print(prob.summary())
```
```
PROBABILISTIC DFIT INTERPRETATION
[ Regime probabilities ]
pressure_dependent     72.1%
normal                 18.4%
tip_extension           7.2%
height_recession        2.3%
Entropy: 0.821 nats (max = 1.386, certain = 0.000)
[ Closure pick distribution ]
Closure G  : 6.94 +/- 0.12  [P10=6.79, P90=7.08]
Closure P  : 6,083 +/- 21 psi  [P10=6,048, P90=6,118]
Bootstrap  : 198 realisations

```
---

## What's inside

| Module | Purpose |
|---|---|
| `dfit.gfunction` | Nolte g- and G-functions (both leakoff bounds), superposition and sqrt-time functions, inverse G-to-time map |
| `dfit.derivatives` | Bourdet windowed log-derivative and the semilog G·dP/dG derivative, with noise smoothing |
| `dfit.isip` | ISIP estimation by log-time extrapolation and a wellbore-storage / tortuosity decompression model |
| `dfit.leakoff` | Leakoff-regime classification into the four Barree signatures |
| `dfit.closure` | Automated fracture-closure picking and net-pressure calculation |
| `dfit.afterclosure` | After-closure flow-regime detection and pore-pressure extrapolation |
| `dfit.synthetic` | Forward-model synthetic DFITs for any regime, with exact ground truth |
| `dfit.fracdesign` | PKN fracture treatment design parameters from DFIT outputs: C_L, fluid efficiency, treating pressure, volume |
| `dfit.probabilistic` | Probabilistic leakoff-regime classifier with closure stress P10-P90 confidence intervals via bootstrap |
| `dfit.benchmark` | DFITBench: generate, save, and load the 1,008-record standardized benchmark dataset |
| `dfit.workflow` | One-call end-to-end interpretation pipeline |

---

## The four leakoff signatures

Classification keys on the shape of G·dP/dG relative to a straight reference line through the origin:

- **Normal** - straight line through the origin; closure is the departure below it.
- **Pressure-dependent leakoff (PDL)** - a hump *above* the line, from secondary fractures opening at high net pressure.
- **Fracture-height recession / transverse storage** - a belly *below* the line, from variable fracture compliance.
- **Fracture-tip extension** - the straight section has a positive intercept (does not pass through the origin).

Notebook `03_leakoff_regime_classification.ipynb` plots all four and walks through the discriminants.

---

## Notebooks

1. `01_before_closure_gfunction.ipynb` - the G-function from first principles to a closure pick.
2. `02_after_closure_analysis.ipynb` - log-log flow-regime diagnostics and the case against forcing radial flow.
3. `03_leakoff_regime_classification.ipynb` - the four signatures and the classifier.
4. `04_synthetic_data_generation.ipynb` - how the forward model works and how to control it.
5. `05_validation_barree_2015.ipynb` - quantitative checks against SPE-169539-PA; reproduces the paper's 3,640 to 3,562 psi first-second decline exactly.
6. `06_field_data_example.ipynb` - end-to-end interpretation of a realistic field-analog DFIT (Well 1, 518 s pump time).
7. `07_sensitivity_analysis.ipynb` - how gauge noise and sampling rate degrade closure pick accuracy; bootstrap confidence interval on closure pressure.
8. `08_fracture_design_bridge.ipynb` - from DFIT outputs to PKN fracture treatment design; DFIT-calibrated vs naive design comparison.
9. `09_probabilistic_classifier.ipynb` - probabilistic regime classification and closure stress P10-P90 uncertainty quantification.
10. `10_dfitbench_results.ipynb` - DFITBench evaluation: confusion matrices, accuracy by noise level, closure pressure bias analysis.

---

## DFITBench

DFITBench is a standardized synthetic benchmark for DFIT interpretation methods, introduced alongside this toolkit. It addresses the central limitation noted by Mohamed et al. (URTeC 2020): real DFIT data is proprietary and there is no public dataset with known ground truth for comparing methods.

The benchmark contains **1,008 synthetic DFITs** across:
- 4 leakoff regimes (normal, pressure-dependent, height-recession, tip-extension)
- 3 formation types (tight gas, shale, conventional)
- 3 noise levels (1.0, 3.5, 8.0 psi gauge noise)
- 28 seeds per cell

**Probabilistic classifier results on DFITBench:**

| Noise level | Gauge (psi) | Overall accuracy | PDL accuracy | Normal accuracy |
|---|---|---|---|---|
| Low | 1.0 | 100% | 100% | 100% |
| Medium | 3.5 | 98% | 98% | 100% |
| High | 8.0 | 89% | 86% | 94% |

Generate the benchmark locally:

```python
from dfit.benchmark import generate_benchmark, save_benchmark
records = generate_benchmark(n_per_cell=28)
save_benchmark(records, "data/benchmark")
```

---

## Validation

The synthetic generator and the interpreter form a closed loop: every test is built from known parameters, then fed back through the pipeline. Across the four leakoff regimes and many noise realisations, the classifier recovers the regime it was built from, ISIP is recovered to within ~1%, and the closure pick is stable with a small, documented late bias consistent with the real-world ambiguity the literature describes. The wellbore-decompression model reproduces the worked example in SPE-169539-PA to the psi.

Run the test suite with `pytest`:

```bash
pytest tests/ -q
```

---

## Scope and limitations

This is a research and teaching toolkit, not a commercial interpretation suite. In particular:

- The closure pick is deliberately conservative and carries a small systematic late bias; treat it as a starting point to refine against the sqrt(t) and log-log views, exactly as you would in practice.
- After-closure permeability is intentionally **not** reported when a flow regime is not cleanly developed, following the central argument of Barree et al. (2015) that forcing a radial solution overstates flow capacity.
- The synthetic generator is phenomenological - it reproduces the diagnostic *signatures* faithfully, but it is not a full fracture-mechanics simulator.
- The probabilistic classifier requires an estimate of gauge noise (psi). If unknown, 3.5 psi is a reasonable default for a typical field memory gauge.

---

## References

- Barree, R.D., Miskimins, J.L., Gilbert, J.V. (2015) "Diagnostic Fracture Injection Tests: Common Mistakes, Misfires, and Misdiagnoses." *SPE Production & Operations* 30 (2): 84-98. SPE-169539-PA.
- Barree, R.D., Barree, V.L., Craig, D.P. (2009) "Holistic Fracture Diagnostics: Consistent Interpretation of Prefrac Injection Tests Using Multiple Analysis Methods." SPE-107877-PA.
- Mohamed et al. (2020) "Advanced Machine Learning Methods for Prediction of Fracture Closure Pressure, Closure Time, Permeability and Time to Late Flow Regimes From DFIT." URTeC-2020-2762.
- Nolte, K.G. (1979) "Determination of Fracture Parameters from Fracturing Pressure Decline." SPE-8341-MS.
- Talley, G.R. et al. (1999) "Field Application of After-Closure Analysis of Fracture Calibration Tests." SPE-52220-MS.
- Bourdet, D., Ayoub, J.A., Pirard, Y.M. (1989) "Use of Pressure Derivative in Well Test Interpretation." SPE-12777-PA.

---

## License

MIT - see [LICENSE](LICENSE).

## Citation

If this toolkit supports your work, please cite it:
```
Joshi, T. (2026). dfit-pressure-diagnostics: An open-source Python toolkit for
Diagnostic Fracture Injection Test interpretation. GitHub repository.
https://github.com/tarunjoshi03/dfit-pressure-diagnostics

```