# Scientific workflow and internal file map

The public manuscript is organized by scientific question. Historical source
filenames retain the internal identifiers used when each experiment was
frozen. This table is the authoritative crosswalk.

| Scientific question | Executable | Main frozen output |
|---|---|---|
| Local information after nuisance marginalization | `scripts/run_phase0.py`, `scripts/run_phase05.py` | `outputs/phase0_results.json`, `outputs/phase05/` |
| Nonlinear recovery under constitutive mismatch | `scripts/run_phase1_carbonate.py` | `outputs/phase1_carbonate/` |
| Rigid, partial, and independent coupling; predictive averaging | `scripts/run_phase11_carbonate.py` | `outputs/phase11_carbonate/` |
| Cross-fitted interval calibration | `scripts/run_phase12_calibration.py` | `outputs/phase12_calibration/` |
| Data-only profile and bootstrap diagnostics | `scripts/run_phase12b_sentinels.py` | `outputs/phase12b_sentinels/` |
| Independent model-error bank | `scripts/run_phase12c_falsification.py` | `outputs/phase12c_offlibrary/` |
| Fixed quality screen on a new bank | `scripts/run_phase13_selective.py` | `outputs/phase13_selective/` |
| External Oman GT3A control | `scripts/run_oman_external_control.py` | `results/oman/` |

The released workflow ends here. Exploratory diagnostics developed after the
fixed quality-screen bank was opened are not part of the manuscript evidence
and are deliberately absent from this public package.
