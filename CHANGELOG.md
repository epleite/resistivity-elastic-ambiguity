# Changelog

## v1.0.0 — GMD reproducibility release

- Reorganized the public entry points around five geophysical questions while
  retaining historical filenames needed to trace the prospectively frozen
  experiments.
- Added a portable, dependency-light Python reproduction of the Oman GT3A
  blocked-validation control and strict checking of every frozen statistic in
  the default full run.
- Added a hash-checking Oman data-preparation script, provenance records, a
  data dictionary, and tests for both electrical predictors. The sample-level
  table is generated locally and is not redistributed.
- Added scripts for the seven publication figures, including corrected
  GJI-style multi-panel compositions with descriptive geophysical labels.
- Limited the public evidence boundary to the fixed quality-screen test and
  the independent Oman control; post-opening exploratory scorers are excluded.
- Added Python 3.10--3.12 continuous integration, environment files, computing
  guidance, citation metadata, Zenodo metadata, and the MIT License.

## v0.8.0 — Phase 1.3 prospective selective inversion

- Froze and hashed a deployable porosity policy before generating any new test
  observation: Phase-1.1 q90 reduced chi-square, porosity-bound mass and
  numerical-health checks only.
- Generated 64 new geological panels with three common-noise replicates across
  six declared stress families, for 1,152 independent test observations.
- Persisted the exact elastic/electrical observation bank, 10,368 candidate
  fits and 13,824 target recoveries.
- Added strict abstention with an elastic-only fallback, panel-cluster
  uncertainty, nested risk-coverage curves and a 5,000-draw equal-retention
  random-rejection comparator.
- Added invariance tests proving that truth, scenario labels, realised error,
  adapted weights and row order cannot alter acceptance.
- Confirmed safe joint transport for matched EMA, connectivity drift and patchy
  saturation, plus safe domain rejection of the combined stress.
- Found silent failures for hybrid conductivity and invasion mismatch: their
  harmful-false-confidence upper bounds exceed the predeclared 10% ceiling.
- Found no scenario with risk enrichment beyond random rejection at equal
  retention; neither q95 nor stricter historical chi-square thresholds repair
  the atomic misses.
- Final decision remains **STOP**. The next challenger must use structured
  elastic/deep-Rt/shallow-Rt residual diagnostics and be frozen before a new
  independent validation bank is generated.
- Added ten Phase-1.3 regression tests (45 tests total).

## v0.7.0 — Phase 1.2C independent off-library falsification

- Added an independent design with 24 geological panels, three repeated-noise
  realizations, eight depths and common random numbers across six scenarios.
- Kept the inverse library fixed at three electrical families crossed with
  rigid, partial and independent coupling; no off-library truth enters it.
- Added a matched EMA transport control and five predeclared stresses: a
  depth-varying latent-dual/network hybrid, depth-varying connectivity, a
  three-zone invasion mismatch, mean-preserving patchy saturation and their
  combined escalation.
- Added a deployable frozen rule: a family-balanced prior learned from Phase
  1.1 only and updated by predeclared local blocked-PRESS scores.
- Reconstructed the exact frozen estimator with true outer-panel exclusion;
  interval inflation is explicitly empirical, not formal cluster conformal.
- Added an adapted leave-one-panel-out OOD ceiling that is diagnostic only and
  cannot upgrade the confirmatory decision.
- Added cluster inference for coverage, RMSE/width ratios, false confidence and
  bias, plus convergence, surviving-prior and weighted critical-bound mass.
- Added a frozen reduced-chi-square OOD alarm and separates robust,
  detected-OOD/abstain, silent inferential and non-harmful gate failures.
- Persisted 5,184 recoveries, 3,888 candidate fits, training reconstruction,
  truth-library distances, figures, runtime versions and SHA-256 hashes.
- Scientific result: the matched control has strong porosity point/width gain
  but misses strict PASS because of overcoverage and critical-bound mass; zero
  atomic OOD cases PASS, one is CONDITIONAL and three FAIL. Final decision:
  STOP for field use.
- Added eight Phase-1.2C regression tests (35 tests total).

## v0.4.0 — Phase 1.1 latent carbonate coupling

- Added a distinct electrical-connectivity latent variable to the inverse.
- Centered the elastic latent before imposing exact correlations, preventing
  coupling weights from confounding correlation with a design-wide offset.
- Added rigid, partially pooled and independent elastic-electrical coupling
  hypotheses without redundant simultaneous estimation of correlation and
  latent offset.
- Added latent dual-porosity, GEM/EMA and dual-network candidate electrical
  families, including invaded-zone tool mixing for EMA/network candidates.
- Added linearized leave-one-depth-block-out predictive scoring and pseudo-BMA
  weights across nine constitutive/coupling candidates.
- Added Gaussian-mixture target intervals containing both within-model and
  between-model uncertainty.
- Added strategy-level coverage, RMSE, reduced-chi-square, false-confidence,
  effective-model and model-weight diagnostics.
- Added common-random-number pairing, panel-cluster bootstrap gates, bounded
  mixture intervals and a one-percent numerical tolerance at the relaxed width
  boundary.
- Persisted candidate estimates, local standard errors and every physical-bound
  flag; the report exposes raw and predictive-weighted boundary rates.
- Added atomic figure writes and SHA-256 hashes for every Phase-1.1 artifact and
  all governing Python sources, the Phase-1.1 test file and project metadata.
- Added a 500-evaluation convergence fallback for candidate and elastic fits;
  any candidate still nonconverged receives exactly zero predictive weight and
  the surviving candidates are renormalized per observation.
- Final screening result: mean model-average RMSE ratio 0.704, mean nominal-90%
  coverage 0.803, four CONDITIONAL and sixteen FAIL target-by-truth gates; field
  deployment remains STOP pending uncertainty calibration and off-library tests.
- Added seven Phase-1.1 regression tests (20 tests total).

## v0.3.0 — Phase 1 carbonate

- Added alternative General Effective Medium and dual-network percolation
  electrical truth generators for multimodal carbonates.
- Decoupled electrical connectivity from elastic aspect ratio through a latent
  correlation coefficient (0, 0.5 and 0.9 stress cases).
- Added finite deep/shallow tool-volume mixing and mud-filtrate resistivity
  mismatch to the synthetic truth.
- Added repeated-noise bounded nonlinear recovery, local 90% Wald intervals,
  empirical coverage, bias, RMSE, boundary-hit and convergence diagnostics.
- Added a paired false-confidence test: intervals that tighten by at least 20%
  while excluding truth are counted explicitly.
- Added PASS/CONDITIONAL/FAIL carbonate gates and regression tests.

## v0.2.0 — Phase 0.5

- Added physically distinct soft-sand, stiff-sand, contact-cement, multimodal
  DEM and single-pore DEM elastic families.
- Added Archie, simplified Waxman–Smits and connectivity-aware dual-porosity
  electrical families.
- Separated truth and inverse models for matched and deliberate mismatch tests.
- Added rigid, partial and independent elastic–electrical pore coupling.
- Replaced the single-depth decision experiment with hierarchical facies panels.
- Added the pointwise-free nuisance negative control.
- Added correlated deep/shallow log-Rt errors and invasion nuisance.
- Added eight independently randomized LHS batches and target-prior sensitivity.
- Added generalized directional information, practical weak-subspace rescue,
  absolute uncertainty and false-confidence diagnostics.
- Added unpenalized nonlinear hierarchical profiles and physical-bound checks.
- Added Phase-0.5 decision matrix, robustness maps, configuration comparison,
  mismatch plot, profile surfaces, manifest and scientific report.
- Expanded automated validation from three to nine tests.

Main scientific outcome: resistivity produces conditional improvement for
porosity/saturation and strong matched-model carbonate gains, but no target
passes every robust-GO gate under constitutive mismatch. Cement and coordination
remain negative controls.
# Phase 1.2A

- Added leave-one-panel-out conformal scaling of local-curvature uncertainty.
- Added a transferable global calibration and matched-case sensitivity analysis.
- Added calibrated joint-versus-elastic utility gates with panel-cluster bootstrap.
- Added coverage and calibration-factor diagnostics without rerunning Phase 1.1 fits.

# Phase 1.2B

- Added error-blind interior design-medoid sentinel selection.
- Added genuine whitened-data profile likelihoods with all nuisance parameters reoptimized.
- Added split parametric-bootstrap calibration and held-out coverage evaluation.
- Kept cross-fitted predictive weights fixed only for the operational model-average estimator.
- Added paired RMSE/width gates and an explicit secondary-porosity negative control.
