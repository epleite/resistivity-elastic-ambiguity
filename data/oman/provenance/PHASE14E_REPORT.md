# Phase 1.4E report — Oman GT3A external mechanistic transfer

**Protocol:** `phase14e_oman_external_transfer_v1`  
**Decision:** strong-transfer STOP (immutable protocol identifier:
`E_NO_TRANSFER_GO`)  
**Date:** 2026-08-30 (UTC)

## Result in one sentence

Measured resistivity supplied reproducible held-out porosity information beyond
wet Vp, wet Vs, and bulk density, but the improvement was smaller than the
prospectively required 10% RMSE reduction; the strong-transfer gate therefore
failed without erasing the independent electrical signal.

## Primary frozen analysis

The official Akamatsu et al. Oman GT3A workbook contains 94 physical samples.
Ninety-three samples were complete for porosity, wet mean Vp, wet mean Vs, bulk
density, and 35 g/L NaCl mean resistivity. Samples were assigned before
complete-case filtering to eight contiguous depth blocks. Ridge penalties were
selected inside each outer-training partition; all metrics below use only
outer held-out predictions.

Porosity in the complete set ranges from 0.09% to 11.27%, with sample standard
deviation 2.592 percentage points. An outer-fold mean baseline, calculated from
the seven training blocks for each held-out block, has RMSE 2.600 percentage
points. Relative to that leakage-free baseline, cross-validated R² is 0.774 for
the elastic model and 0.796 for the joint model. Both models therefore have
substantial absolute skill; the ratio below measures a modest increment between
two informative models, not two ineffective models.

| Metric | Elastic | Joint + log10 R35 | Ratio / inference |
|---|---:|---:|---:|
| RMSE (porosity percentage points) | 1.2345 | 1.1754 | 0.9521 |
| MAE (porosity percentage points) | 0.8155 | 0.7538 | 0.9243 |
| Whole-block bootstrap RMSE ratio | — | — | 95% CI 0.8681–0.9924 |
| Within-sequence full-refit permutation | — | — | p=0.001 |

The primary joint model reduced RMSE by 4.8% and MAE by 7.6%. Its whole-block
bootstrap upper 95% bound remained below one, and only zero of 999
sequence-preserving electrical permutations equalled or exceeded the observed
improvement after complete nested refitting. Thus the sample-matched electrical
measurement contains independent information under the frozen analysis.

The 999 restricted permutations operated within four named sequences: Upper
Dike (n=33), Lower Dike (n=28), Lower Gabbro (n=29), and Upper Gabbro (n=3).

## Gate accounting

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Complete samples | >=80 | 93 | PASS |
| Nonempty depth blocks | >=6 | 8 | PASS |
| RMSE ratio | <=0.90 | 0.9521 | **FAIL** |
| Bootstrap upper 95% bound | <1.00 | 0.9924 | PASS |
| Full-refit permutation | p<=0.05 | 0.001 | PASS |
| MAE ratio | <=1.00 | 0.9243 | PASS |
| Maximum sequence RMSE ratio, n>=10 | <=1.25 | 0.9836 | PASS |

Because every gate was conjunctive, the scientific decision is a
strong-transfer STOP. The immutable protocol stores the legacy identifier
`E_NO_TRANSFER_GO`; it is retained for audit continuity. The 0.90 threshold was
frozen as the minimum practically material improvement for a strong-transfer
claim: a 10% RMSE reduction is equivalent to a 19% MSE reduction.
Retrospectively weakening this gate is prohibited.

## Geological localization

The largest well-supported sequence benefit occurred in Lower Gabbro (n=29;
RMSE ratio 0.850). Lower Dike (n=28; 0.984) and Upper Dike (n=33; 0.976)
showed much smaller benefits. No sequence with at least ten samples exceeded
the frozen harm ratio of 1.25.

Lithology summaries are descriptive only. Gabbro (n=8) had RMSE ratio 0.690,
whereas diabase (n=46) was essentially unchanged at 0.991. Small lithologies
must not be interpreted individually.

## Frozen sensitivities

- Replacing raw 35 g/L resistivity with calculated formation factor produced
  RMSE ratio 0.9548, MAE ratio 0.9343, bootstrap upper bound 1.0006, and
  permutation p=0.002. It also failed the strong-transfer gates.
- Removing bulk density from the elastic baseline increased the apparent
  electrical benefit to an RMSE ratio of 0.9288. This is compatible with
  density already carrying substantial porosity information, but it is a
  sensitivity and cannot replace the primary result.

## Scientific interpretation

This external test supports a narrow but useful statement:

> Electrical measurements can contribute independent porosity information
> across an unrelated rock system, but the incremental gain beyond wet Vp,
> wet Vs, and density is modest and geology-dependent.

It does not validate the locked synthetic carbonate estimator, its release
rule, or its uncertainty intervals. Oman GT3A lacks deep/shallow log
resistivities, invasion, and log-scale spatial sampling, and its mafic/felsic
lithologies are outside the carbonate model family.

The primary predictors use geometric means across three orthogonal measurement
directions. This keeps the physical sample as the independent unit and avoids
pseudoreplication, but deliberately discards directional contrasts that may
encode crack orientation. A prospectively frozen directional analysis would
test a distinct anisotropic mechanism.

This result strengthens the manuscript because it agrees with the central
claim without overselling it: complementary physics can improve point recovery,
yet the magnitude and safety of that improvement cannot be assumed from data
fit alone.

## Sawayama and next evidence stage

The pending Sawayama response is not needed for this conclusion. If those data
arrive, they should enter a separately hashed protocol and may provide the
closer carbonate or saturation-path validation that Oman cannot provide.

Without waiting, the next stage is a targeted search for a public paired
carbonate dataset with sample-matched porosity, elastic velocity, and
resistivity. The ideal source also includes pressure or saturation trajectories,
allowing interval calibration or mechanism testing rather than another
cross-sectional association.

## Provenance

- Akamatsu et al. (2023), *Journal of Geophysical Research: Solid Earth*,
  doi:10.1029/2022JB026130.
- Official dataset: Hiroshima University repository record 2000060.
- Source workbook SHA-256:
  `dceaae2ee7e85d233349222447f3c17e50d726b580ee2c629bcc1aed9291ff8b`.
- Frozen protocol SHA-256:
  `68bae6eceb0f15ae9507413724bc09bfea7c3b814f20646559c83b4b33fd1e84`.
