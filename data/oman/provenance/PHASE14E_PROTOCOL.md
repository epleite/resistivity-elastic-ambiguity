# Phase 1.4E protocol — public external mechanistic transfer

**Protocol ID:** `phase14e_oman_external_transfer_v1`  
**Status:** frozen before outcome modelling; source schema and availability only
were inspected before freezing  
**Date:** 2026-08-30 (UTC)  
**Parent evidence:** manuscript v0.6 through Phase 1.3D  
**External source:** Akamatsu et al. (2023), Oman Drilling Project Hole GT3A,
official Hiroshima University repository record 2000060

## Evidence boundary

The Phase-1.3 synthetic bank remains closed for confirmation. No Phase-1.4E
choice may modify its candidate library, q90-v2 release rule, intervals,
features, endpoints, or STOP decisions.

The Oman workbook was inspected only to verify provenance, sample count,
column definitions, units, missing-value representation, and the availability
of paired elastic and electrical measurements. No porosity model, association,
cross-validation result, or endpoint comparison was calculated before this
protocol was frozen.

Oman GT3A contains mafic and felsic crustal rocks rather than the synthetic
carbonate system used in the parent study. Phase 1.4E therefore tests only the
transferable mechanistic claim that electrical information can add porosity
information beyond wet elastic velocity and bulk density. It is not a direct
validation of the locked carbonate estimator and cannot validate its interval
coverage or release policy.

## Data and analysis unit

- Analysis unit: one physical core sample; directions x, y, and z are never
  treated as independent replicates.
- Available population: the 94 sample rows in Table S2.
- Outcome: laboratory porosity in percent from Table S2.
- Elastic predictors: wet geometric-mean P-wave velocity, wet geometric-mean
  S-wave velocity, and bulk density.
- Primary electrical predictor: `log10` of the measured 35 g/L NaCl
  geometric-mean resistivity from Table S2.
- Electrical sensitivity: `log10` of the geometric-mean formation factor from
  Table S3. This is derived from the multi-salinity electrical experiment and
  cannot replace the primary endpoint.
- Complete-case analysis only. No imputation is allowed. Missingness is
  reported before fitting.
- No orientation-level variables, crack-inversion outputs, surface
  conductivity, mineralogical labels, depth, sequence, or lithology enter the
  fitted predictor vector.

Thirty-five g/L is frozen as the primary salinity because every sample is
measured under the same near-seawater brine condition and the predictor remains
a direct observation. Formation factor is reserved for sensitivity because it
is a fitted electrical quantity, although it does not use porosity truth.

## Geological blocking

Samples are sorted by measured depth. Before complete-case filtering, they are
assigned to eight contiguous depth blocks using

```text
block = 1 + floor(8 * zero_based_depth_rank / 94).
```

All preprocessing, penalty selection, and fitting for a held-out block use
only the other seven blocks. This prevents adjacent samples from being split
randomly across training and test sets and makes the core sample, not an
orientation, the effective observation.

## Locked models

Two linear ridge models are compared:

```text
E: porosity_pct ~ wet_Vp_mean + wet_Vs_mean + bulk_density
J: porosity_pct ~ wet_Vp_mean + wet_Vs_mean + bulk_density + log10(R35_mean)
```

The intercept is unpenalized. Predictors are centered and scaled using outer-
training statistics only. Constant training predictors are assigned unit
scale. Porosity is modelled on its original percent scale; predictions are not
clipped.

For E and J separately, the ridge penalty is selected from
`{0.01, 0.1, 1, 10, 100}` by the lowest mean squared error across the seven
training depth blocks, with the largest penalty breaking exact ties. The
selected model is refit on all seven outer-training blocks and evaluated once
on the untouched block. Pooled metrics use only these outer held-out
predictions.

Random forests, boosting, splines, interactions, feature selection, alternative
salinities, outcome transformations, and post-hoc subgroup thresholds are
prohibited in the primary analysis.

## Primary estimand and inference

The primary estimand is

```text
RMSE ratio = RMSE_J / RMSE_E
```

on pooled outer-block predictions. MAE ratio and the paired sample-level
difference in squared error are secondary.

Uncertainty for the RMSE ratio uses 10,000 deterministic whole-depth-block
bootstrap draws with seed `14052026`. Every draw resamples the eight blocks
with replacement and carries all samples in a selected block together. The
reported interval is the percentile 95% interval.

Independent electrical contribution is tested with 999 deterministic
permutations using seed `14052027`. Within each named geological sequence, the
primary electrical values are permuted among samples. Every permutation reruns
the complete nested J pipeline, including training-only scaling and penalty
selection; E remains the identical frozen reference. With

```text
D = RMSE_E - RMSE_J,
p = (1 + number(D_perm >= D_observed)) / 1000.
```

the alternative is that genuine sample-matched resistivity improves held-out
porosity recovery more than a sequence-preserving arbitrary electrical
assignment.

## Frozen decision gates

`E_MECHANISTIC_TRANSFER_GO` requires all of the following on the primary raw
35 g/L resistivity analysis:

1. at least 80 complete physical samples and at least six nonempty outer
   blocks;
2. point RMSE ratio at most 0.90;
3. 95% whole-block bootstrap upper bound below 1.00;
4. permutation p-value at most 0.05;
5. point MAE ratio at most 1.00; and
6. no named sequence with at least 10 complete samples has a joint/elastic
   RMSE ratio above 1.25.

Failure of any gate yields `E_NO_TRANSFER_GO`. If the primary analysis fails
but the frozen formation-factor sensitivity passes gates 2--6, the only
allowed label is `E_DERIVED_ELECTRICAL_SIGNAL`; it is mechanistic evidence and
cannot replace the primary result.

## Required reporting

- provenance URLs, source filename, byte size, and SHA-256 hash;
- all 94 source sample identifiers and the complete-case mask;
- immutable depth-block assignments;
- outer predictions, selected penalties, and sample-level errors for E and J;
- bootstrap distribution summary and full-refit permutation result;
- descriptive metrics by sequence and by lithology only for groups with at
  least five complete samples;
- the primary decision label, including failed gates without reinterpretation;
- the formation-factor and velocity-only sensitivities clearly marked as
  non-primary.

## Role of the pending Sawayama request

Sawayama data are not required to execute or interpret Phase 1.4E. If supplied
later, they form a separately frozen external stage. They may strengthen
lithology relevance or test uncertainty calibration, but they cannot be used
to retune this protocol after the Oman outcome is opened.
