#!/usr/bin/env python3
"""Run and persist the Phase-1.2C independent off-library falsification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jointinv.phase12c import OFF_LIBRARY_CASES, run_phase12c


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY = ROOT / "outputs/phase11_carbonate/recovery_strategies.csv"
DEFAULT_COMPONENTS = ROOT / "outputs/phase11_carbonate/fit_components.csv"
DEFAULT_OUTPUT = ROOT / "outputs/phase12c_offlibrary"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format=path.suffix.lstrip("."))
    temporary.replace(path)
    plt.close(fig)


def plot_phi_performance(gates: pd.DataFrame, output: Path) -> None:
    selected = gates[gates.target.eq("phi")].copy()
    order = [case.name for case in OFF_LIBRARY_CASES]
    labels = [
        "Represented\nphysics", "Hybrid\nlaw", "Connectivity\ndrift",
        "Invasion\nmismatch", "Patchy\nSw", "Combined\nstress",
    ]
    colors = {"frozen_press": "#2563eb", "adapted_average": "#f59e0b"}
    markers = {"frozen_press": "o", "adapted_average": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    metrics = [
        ("coverage90", "coverage90_cluster_low", "coverage90_cluster_high", "90% coverage", 0.90),
        ("rmse_ratio_over_elastic", None, "rmse_ratio_cluster_upper", "RMSE ratio", 1.0),
        ("median_width_ratio", None, "width_ratio_cluster_upper", "Width ratio", 1.0),
    ]
    x = np.arange(len(order))
    for axis, (value, low, high, ylabel, reference) in zip(axes, metrics):
        for offset, strategy in zip((-0.10, 0.10), colors):
            rows = selected[selected.strategy.eq(strategy)].set_index("case").loc[order]
            center = rows[value].to_numpy(float)
            lower = rows[low].to_numpy(float) if low else center
            upper = rows[high].to_numpy(float)
            axis.errorbar(
                x + offset, center, yerr=np.vstack([center - lower, upper - center]),
                fmt=markers[strategy], color=colors[strategy], capsize=3,
                label=(
                    {"frozen_press": "Pre-specified model average", "adapted_average": "Refitted comparison"}[strategy]
                    if axis is axes[0] else None
                ),
            )
        axis.axhline(reference, color="#64748b", linestyle="--", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.45, 1.02)
    axes[1].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Porosity when transport processes are omitted from the inversion", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_truth_distance(distance: pd.DataFrame, output: Path) -> None:
    best = distance.groupby(["case", "panel"], as_index=False)[
        "fixed_truth_standardized_rms"
    ].min()
    order = [case.name for case in OFF_LIBRARY_CASES]
    labels = ["Matched", "Hybrid", "Connectivity", "Invasion", "Patchy", "Combined"]
    data = [best.loc[best.case.eq(case), "fixed_truth_standardized_rms"] for case in order]
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    boxes = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, case in zip(boxes["boxes"], OFF_LIBRARY_CASES):
        patch.set_facecolor("#bfdbfe" if not case.is_ood else "#fecaca")
        patch.set_edgecolor("#475569")
    axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=1, label="1 noise SD RMS")
    axis.set_ylabel("Best fixed-truth standardized RMS")
    axis.set_title("Structural distance from the unchanged 3 x 3 candidate library", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output)


def plot_alarm_and_prior(
    diagnostics: pd.DataFrame, weights: pd.DataFrame, output: Path,
) -> None:
    order = [case.name for case in OFF_LIBRARY_CASES]
    labels = [
        "Represented", "Hybrid law", "Connectivity drift",
        "Invasion", "Patchy saturation", "Combined",
    ]
    alarms = diagnostics.groupby("case").ood_alarm.mean().reindex(order)
    matrix = weights.pivot(index="family", columns="coupling", values="frozen_weight").reindex(
        index=["latent_dual", "ema", "network"],
        columns=["rigid", "partial", "independent"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].bar(labels, alarms.to_numpy(), color="#dc2626")
    axes[0].axhline(0.5, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction flagged")
    axes[0].set_title("Pre-specified misfit criterion", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    image = axes[1].imshow(matrix.to_numpy(), cmap="Blues", vmin=0.0)
    axes[1].set_xticks(range(3), matrix.columns)
    axes[1].set_yticks(
        range(3), ["Connected dual-porosity", "Effective-medium", "Two-network"]
    )
    axes[1].set_title("Pre-specified candidate weights", fontweight="bold")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, output)


def summarize_bound_mass(components: pd.DataFrame) -> pd.DataFrame:
    """Decompose weighted critical-bound mass without changing the gate."""
    bound_columns = [
        "bound_hit_phi", "bound_hit_sw", "bound_hit_aspect",
        "bound_hit_secondary_fraction", "bound_hit_q_conn",
    ]
    rows = []
    for strategy, weight_column in (
        ("frozen_press", "frozen_weight"),
        ("adapted_average", "adapted_weight"),
    ):
        for case, group in components.groupby("case", sort=True):
            observation_keys = [group["panel"], group["replicate"]]
            for bound_column in bound_columns:
                per_observation = (
                    (group[weight_column] * group[bound_column].astype(float))
                    .groupby(observation_keys).sum()
                )
                rows.append({
                    "case": case, "strategy": strategy,
                    "bound_name": bound_column.removeprefix("bound_hit_"),
                    "mean_weighted_bound_mass": float(per_observation.mean()),
                })
    return pd.DataFrame(rows)


def summarize_control_verdicts(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, strategy), group in gates.groupby(["target", "strategy"], sort=True):
        counts = group["verdict"].value_counts()
        rows.append({
            "target": target, "strategy": strategy,
            "pass_count": int(counts.get("PASS", 0)),
            "conditional_count": int(counts.get("CONDITIONAL", 0)),
            "fail_count": int(counts.get("FAIL", 0)),
        })
    return pd.DataFrame(rows)


def summarize_alarm_selective_risk(raw: pd.DataFrame) -> pd.DataFrame:
    """Exploratory risk after applying the frozen per-panel OOD alarm."""
    selected = raw[
        raw["target"].eq("phi")
        & raw["strategy"].isin(["elastic", "frozen_press"])
    ]
    rows = []
    for case, group in selected.groupby("case", sort=True):
        paired = group.pivot(
            index=["panel", "replicate"], columns="strategy",
            values=["truth", "estimate", "interval_width90", "covered90", "ood_alarm"],
        )
        alarm = paired[("ood_alarm", "frozen_press")].astype(bool)
        covered = paired[("covered90", "frozen_press")].astype(bool)
        width_ratio = (
            paired[("interval_width90", "frozen_press")].astype(float)
            / paired[("interval_width90", "elastic")].astype(float).clip(lower=1e-12)
        )
        false_confidence = width_ratio.le(0.80) & ~covered
        no_alarm = ~alarm
        joint_error = (
            paired[("estimate", "frozen_press")].astype(float)
            - paired[("truth", "frozen_press")].astype(float)
        )
        elastic_error = (
            paired[("estimate", "elastic")].astype(float)
            - paired[("truth", "elastic")].astype(float)
        )
        no_alarm_ratio = (
            float(np.sqrt(np.mean(joint_error[no_alarm] ** 2))
                  / max(np.sqrt(np.mean(elastic_error[no_alarm] ** 2)), 1e-12))
            if no_alarm.any() else float("nan")
        )
        false_total = int(false_confidence.sum())
        false_alarmed = int((false_confidence & alarm).sum())
        rows.append({
            "case": case, "n": len(paired), "alarm_count": int(alarm.sum()),
            "alarm_fraction": float(alarm.mean()),
            "nonalarm_count": int(no_alarm.sum()),
            "nonalarm_coverage90": float(covered[no_alarm].mean()) if no_alarm.any() else np.nan,
            "nonalarm_rmse_ratio_over_elastic": no_alarm_ratio,
            "false_confidence_count": false_total,
            "false_confidence_alarmed_count": false_alarmed,
            "false_confidence_capture_fraction": (
                false_alarmed / false_total if false_total else np.nan
            ),
        })
    return pd.DataFrame(rows)


def report_text(
    gates: pd.DataFrame, overall: pd.DataFrame, calibration: pd.DataFrame,
    diagnostics: pd.DataFrame, distance: pd.DataFrame,
    bound_mass: pd.DataFrame, control_verdicts: pd.DataFrame,
    selective_risk: pd.DataFrame, n_panels: int,
    replicates: int, panel_size: int, seed: int,
) -> str:
    decision = overall.iloc[0]
    phi = gates[
        gates.target.eq("phi") & gates.strategy.eq("frozen_press")
    ].set_index("case")
    adaptive = gates[
        gates.target.eq("phi") & gates.strategy.eq("adapted_average")
    ].set_index("case")
    lines = [
        "# Phase 1.2C — Independent off-library falsification",
        "",
        f"**Primary decision: {decision.decision}.** {decision.rationale}",
        "",
        "The primary rule is frozen before the external test: a family-balanced candidate prior learned only from Phase 1.1 is updated by the current panel's predeclared blocked-PRESS scores. Its empirical interval inflation and the OOD alarm threshold are reconstructed on outer-held Phase-1.1 panels. No Phase-1.2C truth, error or scenario label trains the primary estimator.",
        "",
        "## Confirmatory porosity gates",
        "",
        "| Scenario | Frozen verdict | Coverage (low) | RMSE ratio (upper) | Width ratio (upper) | OOD alarm | Safety | Adapted ceiling |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for case in OFF_LIBRARY_CASES:
        row = phi.loc[case.name]
        adapted_row = adaptive.loc[case.name]
        lines.append(
            f"| {case.label} | **{row.verdict}** | "
            f"{row.coverage90:.3f} ({row.coverage90_cluster_low:.3f}) | "
            f"{row.rmse_ratio_over_elastic:.3f} ({row.rmse_ratio_cluster_upper:.3f}) | "
            f"{row.median_width_ratio:.3f} ({row.width_ratio_cluster_upper:.3f}) | "
            f"{row.ood_alarm_fraction:.3f} | {row.safety_class} | {adapted_row.verdict} |"
        )
    lines += [
        "",
        "The adapted average excludes the held panel but learns from other panels inside the same simulator-defined OOD scenario. It is an optimistic domain-adaptation ceiling and cannot upgrade the decision.",
        "",
        "## Why the primary rule did not receive GO",
        "",
    ]
    for case in OFF_LIBRARY_CASES:
        row = phi.loc[case.name]
        if row.verdict != "PASS":
            failed = str(row.failed_pass_criteria).replace(";", ", ").replace("_", " ")
            lines.append(f"- **{case.label} ({row.verdict}):** {failed}.")
    lines += [
        "",
        "## Design and frozen controls",
        "",
        f"- Independent seed: `{seed}`; {n_panels} geological panels; {replicates} repeated-noise realizations; {panel_size} depths per panel.",
        "- Common random numbers pair the six scenarios; inference resamples geological panels, never individual rows.",
        "- Candidate library is unchanged: latent-dual, EMA and network conductivity crossed with rigid, partial and independent coupling (nine candidates).",
        "- Four atomic OOD truths are predeclared; the combined case is an escalation test; an independent matched EMA case validates transport to the new design.",
        "- Water saturation, aspect ratio and secondary porosity remain stress/negative controls and are not promoted by this phase.",
        "",
        "## Bound-dependence sensitivity (decision unchanged)",
        "",
        "The predeclared gate uses mass on any critical bound. This is deliberately conservative, but secondary-porosity bounds dominate several porosity decisions. The table decomposes that aggregate; no threshold is relaxed after seeing the result.",
        "",
        "| Scenario | Aggregate critical | Phi bound | Secondary-fraction bound | q-connectivity bound |",
        "|---|---:|---:|---:|---:|",
    ]
    frozen_bounds = bound_mass[bound_mass.strategy.eq("frozen_press")].pivot(
        index="case", columns="bound_name", values="mean_weighted_bound_mass",
    )
    for case in OFF_LIBRARY_CASES:
        gate_row = phi.loc[case.name]
        bounds = frozen_bounds.loc[case.name]
        lines.append(
            f"| {case.label} | {gate_row.mean_critical_bound_mass:.3f} | "
            f"{bounds['phi']:.3f} | {bounds['secondary_fraction']:.3f} | "
            f"{bounds['q_conn']:.3f} |"
        )
    lines += [
        "",
        "The matched control, hybrid law and patchy case retain strong porosity accuracy despite failing/conditional aggregate-bound gates. Connectivity still has an RMSE-ratio upper bound above one; invasion still exceeds the false-confidence threshold; the combined stress fails decisively. Thus this sensitivity changes interpretation, not the STOP decision.",
        "",
        "## Stress and negative controls",
        "",
        "| Target | Strategy | PASS | CONDITIONAL | FAIL |",
        "|---|---|---:|---:|---:|",
    ]
    for row in control_verdicts.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.strategy} | {row.pass_count} | "
            f"{row.conditional_count} | {row.fail_count} |"
        )
    lines += [
        "",
        "Aspect ratio and secondary porosity do not pass any scenario. Saturation remains nonconfirmatory. The negative controls therefore are not spuriously promoted.",
        "",
        "## Exploratory alarm selectivity (not a decision gate)",
        "",
        "| Scenario | Alarmed | Non-alarm coverage | False confidence caught |",
        "|---|---:|---:|---:|",
    ]
    selective = selective_risk.set_index("case")
    for case in OFF_LIBRARY_CASES:
        row = selective.loc[case.name]
        capture = (
            "n/a" if pd.isna(row.false_confidence_capture_fraction)
            else f"{int(row.false_confidence_alarmed_count)}/{int(row.false_confidence_count)}"
        )
        nonalarm_coverage = (
            "n/a" if pd.isna(row.nonalarm_coverage90)
            else f"{row.nonalarm_coverage90:.3f} (n={int(row.nonalarm_count)})"
        )
        lines.append(
            f"| {case.label} | {int(row.alarm_count)}/{int(row.n)} | "
            f"{nonalarm_coverage} | {capture} |"
        )
    lines += [
        "",
        "This selective-risk table was added as a descriptive audit after the confirmatory run. It does not reclassify any gate. The frozen alarm detects the combined stress almost completely and enriches false-confidence risk in invasion/connectivity, but its weak atomic-case sensitivity is not sufficient for deployment.",
        "",
        "## Empirical interval inflation learned before Phase 1.2C",
        "",
        "| Strategy | Target | c90 | Training rows |",
        "|---|---|---:|---:|",
    ]
    for row in calibration.itertuples(index=False):
        lines.append(f"| {row.strategy} | {row.target} | {row.c90:.4f} | {row.n_training} |")
    alarm_threshold = float(diagnostics.alarm_threshold.iloc[0])
    best_distance = distance.groupby(["case", "panel"])[
        "fixed_truth_standardized_rms"
    ].min().groupby("case").median()
    lines += [
        "",
        "## Model checking",
        "",
        f"The mismatch alarm is frozen at best-candidate reduced chi-square > `{alarm_threshold:.4f}`, the 95th-percentile empirical order statistic of matched Phase-1.1 fits.",
        "",
        "Median noise-free distance to the closest unchanged candidate at the true parameters:",
        "",
    ]
    for case in OFF_LIBRARY_CASES:
        lines.append(f"- {case.label}: `{best_distance[case.name]:.3f}` standardized RMS.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is a synthetic falsification experiment, not field validation. A PASS would establish transfer across these specific predeclared constitutive discrepancies only. A harmful inferential failure with an alarm is a detected OOD/abstention result; harmful failure without an alarm is silent failure. Overcoverage or excessive critical-bound dependence is recorded separately as a nonrobust gate failure even when point recovery improves.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-recoveries", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--training-components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training_recoveries = pd.read_csv(args.training_recoveries)
    training_components = pd.read_csv(args.training_components)
    print(
        f"Phase 1.2C: {args.panels} panels × {args.replicates} replicates × "
        f"{len(OFF_LIBRARY_CASES)} cases; {args.workers} workers",
        flush=True,
    )
    outputs = run_phase12c(
        training_recoveries, training_components,
        n_panels=args.panels, replicates=args.replicates,
        panel_size=args.panel_size, seed=args.seed, workers=args.workers,
        n_bootstrap=args.bootstrap,
    )
    (
        raw, components, diagnostics, summary, gates, overall, calibration,
        weights, reconstruction, distance,
    ) = outputs
    frames = {
        "recovery_raw.csv": raw,
        "candidate_components.csv": components,
        "observation_diagnostics.csv": diagnostics,
        "strategy_summary.csv": summary,
        "decision_gates.csv": gates,
        "overall_decision.csv": overall,
        "frozen_calibration.csv": calibration,
        "frozen_prior_weights.csv": weights,
        "training_rule_reconstruction.csv": reconstruction,
        "truth_library_distance.csv": distance,
        "bound_mass_decomposition.csv": summarize_bound_mass(components),
        "control_verdict_summary.csv": summarize_control_verdicts(gates),
        "alarm_selective_risk.csv": summarize_alarm_selective_risk(raw),
    }
    for filename, frame in frames.items():
        frame.to_csv(args.output / filename, index=False)
    plot_phi_performance(gates, args.output / "phi_offlibrary_performance.png")
    plot_truth_distance(distance, args.output / "truth_library_distance.png")
    plot_alarm_and_prior(
        diagnostics, weights, args.output / "ood_alarm_and_frozen_prior.png",
    )
    report = report_text(
        gates, overall, calibration, diagnostics, distance,
        summarize_bound_mass(components), summarize_control_verdicts(gates),
        summarize_alarm_selective_risk(raw),
        args.panels, args.replicates, args.panel_size, args.seed,
    )
    (args.output / "PHASE12C_OFFLIBRARY_REPORT.md").write_text(report, encoding="utf-8")
    config = {
        "phase": "1.2C",
        "seed": args.seed,
        "n_panels": args.panels,
        "replicates": args.replicates,
        "panel_size": args.panel_size,
        "workers": args.workers,
        "cluster_bootstrap_draws": args.bootstrap,
        "candidate_order": [list(key) for key in zip(weights.family, weights.coupling)],
        "cases": [case.__dict__ for case in OFF_LIBRARY_CASES],
        "primary_strategy": "frozen_press",
        "adapted_strategy_role": "diagnostic_ceiling_only",
        "training_inputs": {
            str(args.training_recoveries): sha256(args.training_recoveries),
            str(args.training_components): sha256(args.training_components),
        },
        "runtime_versions": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8",
    )
    governed = [
        ROOT / "src/jointinv/analysis.py",
        ROOT / "src/jointinv/families.py",
        ROOT / "src/jointinv/phase05.py",
        ROOT / "src/jointinv/phase1.py",
        ROOT / "src/jointinv/phase12c.py",
        ROOT / "src/jointinv/phase11.py",
        ROOT / "src/jointinv/phase12.py",
        ROOT / "tests/test_phase12c.py",
        ROOT / "pyproject.toml",
        Path(__file__),
    ]
    artifacts = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "phase": "1.2C",
        "decision": overall.decision.iloc[0],
        "counts": {
            "recoveries": len(raw), "candidate_fits": len(components),
            "observations": len(diagnostics), "gates": len(gates),
        },
        "artifacts": {
            path.relative_to(args.output).as_posix(): sha256(path)
            for path in artifacts
        },
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in governed},
        "inputs": config["training_inputs"],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(overall.to_string(index=False), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
