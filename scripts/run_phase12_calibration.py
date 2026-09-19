#!/usr/bin/env python3
"""Run Phase 1.2A from an existing Phase 1.1 recovery bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jointinv.phase12 import build_phase12_gates, calibration_summary, crossfit_intervals


CASE_LABELS = {
    "dual_partial_truth": "Dual",
    "ema_rho0": "EMA ρ=0",
    "ema_rho05": "EMA ρ=.5",
    "ema_rho09": "EMA ρ=.9",
    "network_rho05": "Network ρ=.5",
}
TARGET_LABELS = {"phi": "Porosity", "sw": "Water saturation", "aspect": "Aspect ratio", "secondary": "Secondary porosity"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_coverage(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["strategy"].eq("model_average")].copy()
    cases = list(CASE_LABELS)
    targets = ["phi", "sw", "aspect", "secondary"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for axis, target in zip(axes.flat, targets):
        subset = data[data["target"].eq(target)].set_index("case").loc[cases]
        x = np.arange(len(cases))
        axis.plot(x, subset["original_coverage90"], "o--", label="Original")
        axis.plot(x, subset["calibrated_coverage90"], "o-", label="Cross-fitted calibrated")
        axis.fill_between(x, subset["coverage90_cluster_low"], subset["coverage90_cluster_high"], alpha=0.18)
        axis.axhspan(0.85, 0.95, color="#4daf4a", alpha=0.10)
        axis.axhline(0.90, color="black", lw=0.8)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xticks(x, [CASE_LABELS[c] for c in cases], rotation=25, ha="right")
        axis.set_ylim(0.45, 1.01)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Empirical 90% coverage")
    axes[1, 0].set_ylabel("Empirical 90% coverage")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Phase 1.2A — held-panel interval calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_factors(calibrated: pd.DataFrame, output: Path) -> None:
    data = calibrated.groupby(["target", "strategy"], as_index=False)["c90"].median()
    pivot = data.pivot(index="target", columns="strategy", values="c90").loc[["phi", "sw", "aspect", "secondary"]]
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot))
    width = 0.16
    for index, strategy in enumerate(["elastic", "rigid", "partial", "independent", "model_average"]):
        axis.bar(x + (index - 2) * width, pivot[strategy], width, label=strategy.replace("_", " "))
    axis.axhline(1.645, color="black", linestyle="--", lw=1, label="Gaussian 90%")
    axis.set_xticks(x, [TARGET_LABELS[t] for t in pivot.index])
    axis.set_ylabel("Median held-panel calibration factor c90")
    axis.set_title("How much local-curvature uncertainty was underestimated")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _report(global_summary: pd.DataFrame, gates: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    model = global_summary[global_summary["strategy"].eq("model_average")]
    target = model.groupby("target").agg(
        original_coverage=("original_coverage90", "mean"),
        calibrated_coverage=("calibrated_coverage90", "mean"),
        median_c90=("median_c90", "median"),
    ).reindex(["phi", "sw", "aspect", "secondary"])
    verdict_counts = gates["verdict"].value_counts()
    gate_target = gates.groupby("target")["verdict"].value_counts().unstack(fill_value=0).reindex(["phi", "sw", "aspect", "secondary"])
    sensitivity_model = sensitivity[sensitivity["strategy"].eq("model_average")]
    sensitivity_cov = sensitivity_model.groupby("target")["calibrated_coverage90"].mean().reindex(target.index)

    lines = [
        "# Phase 1.2A carbonate: held-panel uncertainty calibration",
        "",
        "## Decision",
        "",
        "The 4,800 Phase 1.1 recoveries were recalibrated without refitting. For every held-out geological panel, the 90% multiplier was learned only from the other 11 panels. The primary analysis pools constitutive truth regimes; a case-specific calibration is retained only as an optimistic matched-scenario sensitivity check.",
        "",
        f"The global calibration produces **{int(verdict_counts.get('PASS', 0))} PASS, {int(verdict_counts.get('CONDITIONAL', 0))} CONDITIONAL and {int(verdict_counts.get('FAIL', 0))} FAIL** target-by-truth gates for the full model average.",
        "",
        "This stage diagnoses the uncertainty approximation, not geological generalization. A PASS here authorizes profile/bootstrap confirmation; it does not authorize field deployment.",
        "",
        "## Target diagnosis",
        "",
        "| target | original coverage | calibrated coverage | median c90 | PASS | CONDITIONAL | FAIL | matched-case coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in target.iterrows():
        counts = gate_target.loc[name] if name in gate_target.index else {}
        lines.append(
            f"| {TARGET_LABELS[name]} | {row.original_coverage:.3f} | {row.calibrated_coverage:.3f} | {row.median_c90:.3f} | "
            f"{int(counts.get('PASS', 0))} | {int(counts.get('CONDITIONAL', 0))} | {int(counts.get('FAIL', 0))} | {sensitivity_cov.loc[name]:.3f} |"
        )
    lines += [
        "",
        "The Gaussian reference for a symmetric 90% interval is 1.645. Porosity needs only modest inflation. Aspect ratio and secondary porosity need materially larger factors. Saturation requires the largest correction, so its Phase 1.1 undercoverage is not a small numerical defect even when calibrated coverage is recovered.",
        "",
        "## Gate matrix — global transferable calibration",
        "",
        "| truth regime | porosity | water saturation | aspect ratio | secondary porosity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in CASE_LABELS:
        row = gates[gates["case"].eq(case)].set_index("target")
        lines.append(
            f"| {CASE_LABELS[case]} | {row.loc['phi', 'verdict']} | {row.loc['sw', 'verdict']} | {row.loc['aspect', 'verdict']} | {row.loc['secondary', 'verdict']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- The calibration factors are cross-fitted by panel, so the evaluated panel never calibrates itself.",
        "- Replicates within a panel are not treated as independent geology; all uncertainty gates use panel-cluster bootstrap.",
        "- The primary factor is shared across truth regimes. The case-specific result is an upper bound because a real field interval does not reveal its generating constitutive family.",
        "- Symmetric standard-deviation inflation cannot repair multimodality, bias, boundary pile-up or an off-library forward law. Those require Phase 1.2B.",
        "",
        "## Phase 1.2B authorization",
        "",
        "Run profile likelihood plus parametric bootstrap on sentinel panels for targets that are PASS or CONDITIONAL here. Water saturation must be treated as a high-priority stress test because its required c90 is structurally large. Secondary porosity remains the negative control. Then introduce hybrid/off-library truths before expanding to real logs.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/run_phase12_calibration.py --input outputs/phase11_carbonate/recovery_strategies.csv --output outputs/phase12_calibration",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    global_cal = crossfit_intervals(raw, "global")
    case_cal = crossfit_intervals(raw, "case")
    global_summary = calibration_summary(global_cal)
    case_summary = calibration_summary(case_cal)
    gates = build_phase12_gates(global_cal)

    files = {
        "calibrated_recoveries_global.csv": global_cal,
        "calibrated_recoveries_case_sensitivity.csv": case_cal,
        "calibration_summary_global.csv": global_summary,
        "calibration_summary_case_sensitivity.csv": case_summary,
        "decision_gates_global.csv": gates,
    }
    for name, frame in files.items():
        frame.to_csv(args.output / name, index=False)
    _plot_coverage(global_summary, args.output / "coverage_before_after.png")
    _plot_factors(global_cal, args.output / "calibration_factors.png")
    (args.output / "PHASE12_CALIBRATION_REPORT.md").write_text(
        _report(global_summary, gates, case_summary), encoding="utf-8"
    )
    manifest_files = sorted(path for path in args.output.iterdir() if path.name != "manifest.json")
    project_root = Path(__file__).resolve().parents[1]
    source_files = [
        project_root / "src/jointinv/phase12.py",
        project_root / "scripts/run_phase12_calibration.py",
        project_root / "tests/test_phase12.py",
    ]
    manifest = {
        "phase": "1.2A",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "rows": len(raw),
        "panels": int(raw["panel"].nunique()),
        "primary_scope": "global target-by-strategy, leave-one-panel-out",
        "nominal_coverage": 0.90,
        "files": {path.name: _sha256(path) for path in manifest_files},
        "sources": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in source_files
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(gates["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
