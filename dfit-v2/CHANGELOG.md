# Changelog

## v1.0.0

**Fracture design bridge and sensitivity analysis**

- Added `dfit.fracdesign` module: derives Carter leakoff coefficient, fluid
  efficiency, PKN fracture geometry, treatment pressures, and fluid volumes
  directly from DFIT interpretation outputs. Closes the loop from test to
  treatment design.
- Added notebook `08_fracture_design_bridge.ipynb`: full walkthrough from DFIT
  interpretation to hydraulic fracture treatment design, including a
  quantitative comparison of DFIT-calibrated vs naive (no-DFIT) design.
- Added notebook `07_sensitivity_analysis.ipynb`: systematic study of how gauge
  noise, sampling rate, and pump time degrade closure pick accuracy,
  leakoff classification, and ISIP recovery. Includes a bootstrap confidence
  interval on closure pressure (200 realisations at realistic noise levels).
- Added notebook `06_field_data_example.ipynb`: end-to-end interpretation of a
  realistic field-analog DFIT built from published Well 1 parameters (Siddiqui
  & Qureshi, MPL v2.0), pump time 518 s.
- Added `data/well1_field_analog.csv`: field-analog dataset with realistic
  gauge noise and non-round pump time.
- Bumped version to 1.0.0.

## v0.9.0

**Core toolkit**

- `dfit.gfunction`: Nolte g- and G-functions (both leakoff bounds), inverse
  G-to-time mapping, superposition time, sqrt-of-shut-in time.
- `dfit.derivatives`: Bourdet windowed log-derivative, semilog G*dP/dG
  derivative with industry sign convention, moving-average smoothing.
- `dfit.isip`: ISIP estimation by log-time extrapolation with wellbore-
  decompression correction; forward model of early pressure decay from
  tortuosity and wellbore stiffness.
- `dfit.leakoff`: automated classification into the four Barree leakoff
  regimes (normal, pressure-dependent, height-recession/transverse-storage,
  fracture-tip-extension) with confidence scores.
- `dfit.closure`: automated fracture-closure pick from G*dP/dG departure;
  net-pressure calculation.
- `dfit.afterclosure`: log-log after-closure flow-regime detection; refuses
  to report radial flow unless statistically supported (Barree 2015).
- `dfit.synthetic`: forward-model synthetic DFITs for all four leakoff
  regimes with exact ground truth.
- `dfit.workflow`: one-call end-to-end DFIT interpretation pipeline.
- 59 unit tests, 5 teaching notebooks (G-function through Barree validation),
  4 synthetic sample datasets, methodology documentation.
- Validation against SPE-169539-PA: wellbore-decompression model reproduces
  the paper's worked example to the psi; G-function bounds match Nolte (1979)
  exactly (4/3 and pi/2).
