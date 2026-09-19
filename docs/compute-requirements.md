# Computing requirements

The repository supports two distinct reproducibility levels. Most readers can
audit the published results and regenerate every manuscript figure from the
included machine-readable tables. Rerunning the full synthetic inversion banks
is a substantially larger calculation and is not required to inspect the
paper's numerical evidence.

## Tested software

- Python 3.10--3.12 on 64-bit Linux
- NumPy 1.24 or newer, SciPy 1.10 or newer, pandas 2.0 or newer
- Matplotlib 3.7 or newer
- Optional report builders: ReportLab 4 and pypdf 4 or newer

`environment.yml` provides a conda environment. For pip, install
`requirements.txt` and then the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Continuous integration tests Python 3.10, 3.11, and 3.12. The release was
locally checked with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
and Matplotlib 3.10.8.

## Workload guide

| Task | Suggested resources | Expected scale |
|---|---:|---|
| Unit and frozen-result tests | 2 CPU cores, 4 GB RAM | minutes |
| Regenerate seven figures from included tables | 2 CPU cores, 4 GB RAM | minutes |
| Oman point estimates or reduced-draw smoke test | 1 CPU core, 2 GB RAM | minutes |
| Oman full 10,000-bootstrap/999-permutation run | 1 CPU core, 2 GB RAM | several minutes; pure Python |
| Full nonlinear synthetic banks | 8 or more CPU cores, 16 GB RAM | hours or longer, depending on panel counts and optimizer speed |

The last row comprises repeated nonlinear fits across panels, constitutive
families, coupling modes, and resamples. Runtime is strongly hardware- and
BLAS-dependent. The `--workers` options in the relevant scripts control
parallel execution; begin with a small panel count before allocating a long
run. Frozen outputs and manifests are included precisely so the principal
claims can be checked without repeating those expensive fits.

## Fast verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_frozen_outputs.py
python scripts/make_gmd_figures.py --output build/manuscript_figures
python scripts/run_oman_external_control.py \
  --bootstrap 200 --permutations 19 \
  --output results/oman/quickcheck
```

Reduced resampling counts exercise the Oman workflow but are not expected to
match the frozen confidence limits or permutation probabilities. The script
automatically performs strict reference verification only at the declared full
settings of 10,000 bootstrap draws and 999 permutations.

## Determinism and outputs

The experiment seeds and held-out block assignments are fixed in the scripts
and protocol files. Small floating-point differences may occur across BLAS
implementations; reference checks use a numerical tolerance. Write reruns to a
new directory where the runner exposes `--output`. For older runners with a
fixed destination, use a disposable clone. Retain the released `outputs/`
trees, policy locks, and manifests unchanged in the archival copy.
