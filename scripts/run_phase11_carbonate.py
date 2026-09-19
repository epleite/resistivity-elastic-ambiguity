#!/usr/bin/env python3
"""Run and report Phase 1.1 latent carbonate coupling/model averaging."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jointinv.phase11 import (
    CANDIDATE_FAMILIES, COUPLING_MODES, STRATEGIES, TARGETS, TRUTH_CASES,
    build_phase11_gates, phase11_design, realized_correlation, run_phase11,
    summarize_strategies, summarize_weights,
)


CASE_LABELS = {
    "dual_partial_truth": "Dual-porosity truth (35% connected)",
    "ema_rho0": "EMA truth ρ=0",
    "ema_rho05": "EMA truth ρ=.5",
    "ema_rho09": "EMA truth ρ=.9",
    "network_rho05": "Two-network truth ρ=.5",
}
STRATEGY_LABELS = {
    "elastic": "Elastic only", "rigid": "Rigid",
    "partial": "Partial", "independent": "Independent",
    "model_average": "Full model average",
}
FAMILY_LABELS = {
    "latent_dual": "Connected dual-porosity",
    "ema": "Effective-medium",
    "network": "Two-network",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#94a3b8", "font.size": 9.5,
        "axes.titleweight": "bold",
    })


def _save_figure(fig, path: Path):
    """Write figures atomically so interruption cannot leave a zero-byte PNG."""
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=180, bbox_inches="tight")
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Figure generation failed: {path}")
    temporary.replace(path)


def _worker(args_tuple):
    case, panels, replicates, panel_size, seed = args_tuple
    return run_phase11(
        n_panels=panels, replicates=replicates, panel_size=panel_size,
        seed=seed, truth_cases=(case,),
    )[:2]


def run_parallel(args):
    tasks = [
        (case, args.panels, args.replicates, args.panel_size, args.seed)
        for case in TRUTH_CASES
    ]
    raw_parts, component_parts = [], []
    if args.workers == 1:
        results = []
        for task in tasks:
            results.append(_worker(task))
            print(f"Completed truth case: {task[0].name}", flush=True)
    else:
        results_by_name = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = {pool.submit(_worker, task): task[0].name for task in tasks}
            for future in as_completed(futures):
                case_name = futures[future]
                results_by_name[case_name] = future.result()
                print(f"Completed truth case: {case_name}", flush=True)
        results = [results_by_name[case.name] for case in TRUTH_CASES]
    for raw, components in results:
        # Preserve a nullable numeric rho for the constant-connectivity dual
        # case without triggering all-NA concat inference warnings.
        raw = raw.copy()
        components = components.copy()
        raw["truth_rho"] = pd.array(raw["truth_rho"], dtype="Float64")
        components["truth_rho"] = pd.array(
            components["truth_rho"], dtype="Float64"
        )
        raw_parts.append(raw)
        component_parts.append(components)
    raw = pd.concat(raw_parts, ignore_index=True)
    components = pd.concat(component_parts, ignore_index=True)
    summary = summarize_strategies(raw)
    gates = build_phase11_gates(raw, summary)
    family, coupling = summarize_weights(components)
    return raw, components, summary, gates, family, coupling


def coverage_plot(summary: pd.DataFrame, path: Path):
    _style()
    order = [(case.name, strategy) for case in TRUTH_CASES for strategy in STRATEGIES[1:]]
    matrix = summary[summary["strategy"] != "elastic"].pivot(
        index=["case", "strategy"], columns="target", values="coverage90"
    ).reindex(index=pd.MultiIndex.from_tuples(order), columns=list(TARGETS))
    fig, ax = plt.subplots(figsize=(9.2, 10.2))
    image = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.50, vmax=1.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100*matrix.iloc[i, j]:.0f}%", ha="center", va="center", fontsize=8.5,
                    fontweight="bold")
    ax.set_xticks(range(len(TARGETS)), ["φ", "Sw", "aspect", "secondary φ"])
    ax.set_yticks(range(len(order)), [
        f"{CASE_LABELS[case]} — {STRATEGY_LABELS[strategy]}" for case, strategy in order
    ])
    ax.set_title("Phase 1.1: empirical coverage of bounded 90% intervals")
    fig.colorbar(image, ax=ax, pad=0.02, label="Coverage")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def weight_plot(family: pd.DataFrame, coupling: pd.DataFrame, path: Path):
    _style()
    cases = [case.name for case in TRUTH_CASES]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)
    left = np.zeros(len(cases))
    for name in CANDIDATE_FAMILIES:
        values = family[family["family"] == name].set_index("case").reindex(cases)["mean_weight"].to_numpy()
        axes[0].barh(range(len(cases)), values, left=left, label=FAMILY_LABELS[name])
        left += values
    left = np.zeros(len(cases))
    for name in COUPLING_MODES:
        values = coupling[coupling["coupling"] == name].set_index("case").reindex(cases)["mean_weight"].to_numpy()
        axes[1].barh(range(len(cases)), values, left=left, label=STRATEGY_LABELS[name])
        left += values
    axes[0].set_yticks(range(len(cases)), [CASE_LABELS[name] for name in cases])
    axes[0].invert_yaxis()
    axes[0].set_title("Constitutive-family weights")
    axes[1].set_title("Coupling-hypothesis weights")
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Cross-fitted predictive weight")
        ax.legend(
            frameon=False, fontsize=8, loc="upper center",
            bbox_to_anchor=(0.5, -0.14), ncol=3,
        )
    fig.suptitle("What the held-out panels support", fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _save_figure(fig, path)
    plt.close(fig)


def gate_plot(gates: pd.DataFrame, path: Path):
    _style()
    averaged = gates[gates["strategy"] == "model_average"].copy()
    matrix = averaged.pivot(index="case", columns="target", values="verdict").reindex(
        index=[case.name for case in TRUTH_CASES], columns=list(TARGETS)
    )
    values = matrix.apply(lambda column: column.map({"FAIL": 0, "CONDITIONAL": 1, "PASS": 2})).astype(float)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.imshow(values.values, cmap=ListedColormap(["#b91c1c", "#d97706", "#15803d"]),
              vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix.iloc[i, j], ha="center", va="center", color="white",
                    fontweight="bold", fontsize=8.5)
    ax.set_xticks(range(len(TARGETS)), ["φ", "Sw", "aspect", "secondary φ"])
    ax.set_yticks(range(len(matrix)), [CASE_LABELS[name] for name in matrix.index])
    ax.set_title("Full model-average decision matrix")
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _table(frame: pd.DataFrame) -> str:
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    columns = list(map(str, frame.columns))
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows += ["| " + " | ".join(fmt(value) for value in row) + " |"
             for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def write_observation_bank(args, path: Path):
    design = phase11_design(args.panels, args.seed)
    parameter_names = sorted(key for key in design[0] if isinstance(design[0][key], (int, float, np.floating)))
    parameters = np.array([[p[name] for name in parameter_names] for p in design])
    elastic_noise, rt_noise = [], []
    for panel in range(args.panels):
        e_panel, r_panel = [], []
        for replicate in range(args.replicates):
            rng = np.random.default_rng(args.seed + 10007 * panel + 101 * replicate)
            e_panel.append(rng.normal(size=(args.panel_size, 3)))
            r_panel.append(rng.normal(size=(args.panel_size, 2)))
        elastic_noise.append(e_panel); rt_noise.append(r_panel)
    np.savez_compressed(
        path, parameters=parameters, parameter_names=np.array(parameter_names),
        elastic_noise=np.asarray(elastic_noise), rt_noise=np.asarray(rt_noise),
    )


def write_report(summary, gates, family, coupling, components, args, out):
    averaged = gates[gates["strategy"] == "model_average"].copy()
    matrix = averaged.pivot(index="case", columns="target", values="verdict").reindex(
        index=[case.name for case in TRUTH_CASES], columns=list(TARGETS)
    ).reset_index()
    matrix["case"] = matrix["case"].map(CASE_LABELS)
    key = averaged[[
        "case", "target", "coverage90", "coverage90_cluster_low",
        "rmse_ratio_over_elastic", "rmse_ratio_cluster_upper",
        "false_confidence_cluster_upper", "standardized_bias",
        "median_width_ratio", "median_reduced_chi2", "verdict",
    ]].copy()
    key["case"] = key["case"].map(CASE_LABELS)
    family_table = family.pivot(index="case", columns="family", values="mean_weight").reindex(
        [case.name for case in TRUTH_CASES]
    ).reset_index()
    family_table["case"] = family_table["case"].map(CASE_LABELS)
    coupling_table = coupling.pivot(index="case", columns="coupling", values="mean_weight").reindex(
        [case.name for case in TRUTH_CASES]
    ).reset_index()
    coupling_table["case"] = coupling_table["case"].map(CASE_LABELS)
    strategy_counts = (
        gates.groupby(["strategy", "verdict"]).size().unstack(fill_value=0)
        .reindex(
            index=STRATEGIES[1:], columns=["PASS", "CONDITIONAL", "FAIL"],
            fill_value=0,
        )
    )
    strategy_metrics = gates.groupby("strategy").agg(
        mean_coverage90=("coverage90", "mean"),
        mean_rmse_ratio=("rmse_ratio_over_elastic", "mean"),
        mean_false_confidence=("false_confidence_fraction", "mean"),
        mean_width_ratio=("median_width_ratio", "mean"),
    ).reindex(STRATEGIES[1:])
    strategy_table = strategy_counts.join(strategy_metrics).reset_index()
    strategy_table["strategy"] = strategy_table["strategy"].map(STRATEGY_LABELS)
    target_counts = (
        averaged.groupby(["target", "verdict"]).size().unstack(fill_value=0)
        .reindex(
            index=TARGETS, columns=["PASS", "CONDITIONAL", "FAIL"],
            fill_value=0,
        )
    )
    target_metrics = averaged.groupby("target").agg(
        mean_coverage90=("coverage90", "mean"),
        mean_rmse_ratio=("rmse_ratio_over_elastic", "mean"),
        mean_false_confidence=("false_confidence_fraction", "mean"),
    ).reindex(TARGETS)
    target_table = target_counts.join(target_metrics).reset_index()
    design = phase11_design(args.panels, args.seed)
    correlations = pd.DataFrame([
        {"requested_rho": rho, "realized_rho": realized_correlation(design, rho)}
        for rho in (0.0, 0.5, 0.9, 1.0)
    ])
    counts = averaged["verdict"].value_counts()
    rigid_failures = int(
        ((gates["strategy"] == "rigid") & (gates["verdict"] == "FAIL")).sum()
    )
    partial_passes = int(
        ((gates["strategy"] == "partial") & (gates["verdict"] == "PASS")).sum()
    )
    mean_average_coverage = float(averaged["coverage90"].mean())
    mean_average_rmse_ratio = float(averaged["rmse_ratio_over_elastic"].mean())
    weight_total = max(float(components["full_weight"].sum()), 1e-12)
    weighted_any_bound = float(
        (components["full_weight"] * components["any_bound_hit"]).sum()
        / weight_total
    )
    weighted_critical_bound = float(
        (components["full_weight"] * components["critical_bound_hit"]).sum()
        / weight_total
    )
    weighted_nuisance_bound = float(
        (components["full_weight"] * components["nuisance_bound_hit"]).sum()
        / weight_total
    )
    elastic_diagnostics = components.drop_duplicates(
        ["case", "panel", "replicate"]
    )
    numerical_table = pd.DataFrame([
        {"diagnostic": "candidate fits", "value": len(components)},
        {"diagnostic": "successful fits", "value": components["success"].mean()},
        {"diagnostic": "adaptive restarts", "value": components["adaptive_restart"].mean()},
        {"diagnostic": "candidate convergence fallbacks", "value": components["convergence_fallback"].mean()},
        {"diagnostic": "nonconverged assigned weight", "value": components.loc[~components["success"], "full_weight"].sum()},
        {"diagnostic": "successful elastic fits", "value": elastic_diagnostics["elastic_success"].mean()},
        {"diagnostic": "elastic convergence fallbacks", "value": elastic_diagnostics["elastic_convergence_fallback"].mean()},
        {"diagnostic": "any final bound hit", "value": components["any_bound_hit"].mean()},
        {"diagnostic": "critical final bound hit", "value": components["critical_bound_hit"].mean()},
        {"diagnostic": "nuisance final bound hit", "value": components["nuisance_bound_hit"].mean()},
        {"diagnostic": "weighted any bound hit", "value": weighted_any_bound},
        {"diagnostic": "weighted critical bound hit", "value": weighted_critical_bound},
        {"diagnostic": "weighted nuisance bound hit", "value": weighted_nuisance_bound},
        {"diagnostic": "maximum nfev", "value": int(components["nfev"].max())},
        {"diagnostic": "maximum total nfev", "value": int(components["total_nfev"].max())},
    ])
    conclusion = (
        "Model averaging eliminated every robust failure." if counts.get("FAIL", 0) == 0
        else "Point recovery improves, but uncertainty calibration remains target-dependent."
    )
    text = f"""# Phase 1.1 carbonate: latent connectivity and predictive model averaging

## Decision

**{conclusion}** This stage compares rigid, partially pooled and independent
electrical–elastic coupling across latent-dual, EMA and dual-network electrical
families. The full average propagates within-model and between-model uncertainty
instead of selecting a single convenient constitutive law.

Design: {args.panels} independent carbonate panels × {args.replicates} common-noise
replicates × {args.panel_size} depths × five truth regimes. All methods see the same
noise realization. Secondary porosity is constrained by construction, not by a
nonsmooth penalty.

Across the 20 target-by-truth gates, rigid sharing fails {rigid_failures} cases.
Partial coupling produces {partial_passes} robust PASS result(s); full model
averaging produces {int(counts.get("CONDITIONAL", 0))} CONDITIONAL results and
{int(counts.get("FAIL", 0))} FAIL results. Averaged over all targets and truths,
the full average lowers RMSE to {100 * mean_average_rmse_ratio:.0f}% of
elastic-only RMSE, but its mean 90% coverage is only
{100 * mean_average_coverage:.0f}%. The method therefore improves point recovery
before it achieves calibrated uncertainty.

**Stage decision:** STOP for field deployment; GO only to an uncertainty-
calibration and off-library falsification stage. A low data misfit is not enough
to override undercoverage or boundary dependence.

## Strategy comparison

{_table(strategy_table)}

## Target-level diagnosis

{_table(target_table)}

## Full-average gate matrix

{_table(matrix)}

## Gate diagnostics

{_table(key)}

PASS requires cluster-bootstrap calibrated 90% coverage, RMSE improvement whose
upper confidence bound remains below one, false-confidence upper bound ≤10%,
small standardized bias, interval narrowing, optimizer success and reduced χ²
≤1.5. CONDITIONAL uses the documented relaxed limits, including a 1% numerical
tolerance at the no-wider-than-elastic boundary.

Porosity is the only target that repeatedly reaches the relaxed gate. Water
saturation achieves the largest point-RMSE reduction, but its local-curvature
intervals under-cover. Aspect ratio improves only when the electrical latent is
informative and still lacks stable panel-level calibration. Secondary porosity
is the weakest target: it can improve under dual/network truth but degrades under
EMA truth and frequently reaches its physical-fraction bound.

## Learned constitutive weights

{_table(family_table)}

## Learned coupling weights

{_table(coupling_table)}

Weights are panel-cluster cross-fitted: the held-out panel never chooses its own
weights. They are pseudo-BMA predictive weights derived from linearized blocked
PRESS scores, not posterior probabilities that a geological law is “true.”
Predictive combination is preferred in an M-open setting, following the logic of
[Yao et al. (2018)](https://doi.org/10.1214/17-BA1091). Evidence-based model
comparison has a long geophysical precedent
([Sambridge et al., 2006](https://doi.org/10.1111/j.1365-246X.2006.03155.x)),
while partially joint petrophysical inversion directly motivates relaxing an
invalid rigid law ([Söding et al., 2026](https://doi.org/10.1093/gji/ggaf531)).

Every synthetic truth family is present in the candidate library, so the nearly
perfect constitutive-family recovery is a matched-library ceiling. It validates
the weighting machinery; it does not yet demonstrate recognition of an
off-library or blended geological law.

## Realized latent correlations

{_table(correlations)}

Unlike Phase 1, the orthogonal design realizes the requested correlation to
machine precision. Its elastic latent is centered before correlation is imposed,
so coupling weights do not confound correlation with a cohort-wide offset. The
aspect coordinate remains finite at the physical bounds.

## Numerical diagnostics

{_table(numerical_table)}

Restart and fallback rates measure deliberately difficult candidate surfaces,
including families that are wrong for a given truth. A non-converged candidate
is continued from its best state; if it still fails, its weight is set exactly
to zero and the converged candidates are renormalized. Raw
bound rates include all nine
candidates; weighted rates show that boundary behavior remains material even
among predictively supported fits. Final bounds, estimates and local standard
errors are persisted by parameter in `fit_components.csv`; weighted success
alone must not hide failed or boundary-limited low-weight candidates.

## Statistical boundary

The intervals are bounded truncated-Gaussian mixtures based on local nonlinear
curvature. Cluster bootstrap treats the panel—not each noise replicate—as the
independent geological unit. This is substantially stronger than Phase 1 but
still a screening prototype. Any paper-level positive gate must subsequently be
confirmed by profile/parametric bootstrap or posterior sampling and true
leave-one-panel-out refitting.

## Next falsification test

Before field deployment, replace local-curvature intervals with profile or
parametric-bootstrap intervals and add off-library blended truths. The next
stage succeeds only if saturation and secondary-porosity coverage calibrate
without sacrificing the RMSE gains reported here.

## Reproduce

```bash
python scripts/run_phase11_carbonate.py --panels {args.panels} \\
  --replicates {args.replicates} --panel-size {args.panel_size} \\
  --seed {args.seed} --workers {args.workers}
```
"""
    (out / "PHASE11_CARBONATE_REPORT.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    out = ROOT / "outputs" / "phase11_carbonate"
    out.mkdir(parents=True, exist_ok=True)
    raw, components, summary, gates, family, coupling = run_parallel(args)
    raw.to_csv(out / "recovery_strategies.csv", index=False)
    components.to_csv(out / "fit_components.csv", index=False)
    summary.to_csv(out / "strategy_summary.csv", index=False)
    gates.to_csv(out / "decision_gates.csv", index=False)
    family.to_csv(out / "family_weights.csv", index=False)
    coupling.to_csv(out / "coupling_weights.csv", index=False)
    write_observation_bank(args, out / "observation_bank.npz")
    coverage_plot(summary, out / "coverage_matrix.png")
    weight_plot(family, coupling, out / "predictive_weights.png")
    gate_plot(gates, out / "model_average_gates.png")
    write_report(summary, gates, family, coupling, components, args, out)
    artifact_names = [
        "recovery_strategies.csv", "fit_components.csv", "strategy_summary.csv",
        "decision_gates.csv", "family_weights.csv", "coupling_weights.csv",
        "observation_bank.npz", "coverage_matrix.png", "predictive_weights.png",
        "model_average_gates.png", "PHASE11_CARBONATE_REPORT.md",
    ]
    for name in artifact_names:
        path = out / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output artifact: {path}")
    artifact_hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in artifact_names
    }
    source_paths = (
        sorted((ROOT / "src" / "jointinv").glob("*.py"))
        + [Path(__file__).resolve(), ROOT / "tests" / "test_phase11.py",
           ROOT / "pyproject.toml"]
    )
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    candidate_fits = len(components)
    adaptive_restarts = int(components["adaptive_restart"].sum())
    candidate_fallbacks = int(components["convergence_fallback"].sum())
    elastic_fits = len(TRUTH_CASES) * args.panels * args.replicates
    elastic_diagnostics = components.drop_duplicates(
        ["case", "panel", "replicate"]
    )
    elastic_fallbacks = int(
        elastic_diagnostics["elastic_convergence_fallback"].sum()
    )
    parameter_bound_columns = [
        name for name in components.columns
        if name.startswith("bound_hit_")
        and name not in ("bound_hit_phi", "bound_hit_sw", "bound_hit_aspect",
                         "bound_hit_secondary_fraction", "bound_hit_q_conn")
    ] + [
        "bound_hit_phi", "bound_hit_sw", "bound_hit_aspect",
        "bound_hit_secondary_fraction", "bound_hit_q_conn",
    ]
    parameter_bound_columns = list(dict.fromkeys(parameter_bound_columns))
    manifest = {
        "stage": "Phase 1.1 carbonate", "version": "0.4.0",
        "seed": args.seed, "panels": args.panels, "replicates": args.replicates,
        "panel_size": args.panel_size, "workers": args.workers,
        "truth_cases": [case.__dict__ for case in TRUTH_CASES],
        "candidate_families": list(CANDIDATE_FAMILIES),
        "coupling_modes": list(COUPLING_MODES),
        "joint_fits": candidate_fits,
        "joint_candidate_fits": candidate_fits,
        "adaptive_restarts": adaptive_restarts,
        "candidate_convergence_fallbacks": candidate_fallbacks,
        "joint_optimizer_calls": candidate_fits + adaptive_restarts + candidate_fallbacks,
        "elastic_fits": elastic_fits,
        "elastic_convergence_fallbacks": elastic_fallbacks,
        "elastic_optimizer_calls": elastic_fits + elastic_fallbacks,
        "total_optimizer_calls": (
            candidate_fits + adaptive_restarts + candidate_fallbacks
            + elastic_fits + elastic_fallbacks
        ),
        "successful_candidate_fits": int(components["success"].sum()),
        "nonconverged_candidate_fits": int((~components["success"]).sum()),
        "nonconverged_assigned_weight": float(
            components.loc[~components["success"], "full_weight"].sum()
        ),
        "successful_elastic_fits": int(elastic_diagnostics["elastic_success"].sum()),
        "final_any_bound_hits": int(components["any_bound_hit"].sum()),
        "final_critical_bound_hits": int(components["critical_bound_hit"].sum()),
        "final_nuisance_bound_hits": int(components["nuisance_bound_hit"].sum()),
        "final_bound_hits_by_parameter": {
            name.removeprefix("bound_hit_"): int(components[name].sum())
            for name in parameter_bound_columns
        },
        "weighted_bound_rates": {
            "any": float((components["full_weight"] * components["any_bound_hit"]).sum()
                         / max(components["full_weight"].sum(), 1e-12)),
            "critical": float((components["full_weight"] * components["critical_bound_hit"]).sum()
                              / max(components["full_weight"].sum(), 1e-12)),
            "nuisance": float((components["full_weight"] * components["nuisance_bound_hit"]).sum()
                              / max(components["full_weight"].sum(), 1e-12)),
        },
        "observation_bank_sha256": artifact_hashes["observation_bank.npz"],
        "artifact_sha256": artifact_hashes,
        "source_sha256": source_hashes,
        "realized_correlations": {
            str(rho): realized_correlation(phase11_design(args.panels, args.seed), rho)
            for rho in (0.0, 0.5, 0.9, 1.0)
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(gates[gates["strategy"] == "model_average"].groupby("verdict").size().to_string())
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
