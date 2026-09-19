#!/usr/bin/env python3
"""Run and report the Phase-1 nonlinear carbonate stress test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jointinv.phase1 import CASES, TARGETS, run_carbonate_monte_carlo


LABELS = {
    "matched_dual": "Matched dual",
    "ema_rho0_dual": "EMA ρ=0 → dual",
    "ema_rho05_dual": "EMA ρ=.5 → dual",
    "ema_rho09_dual": "EMA ρ=.9 → dual",
    "network_rho05_dual": "Network ρ=.5 → dual",
    "network_rho05_archie": "Network ρ=.5 → Archie",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#94a3b8", "axes.grid": False,
        "font.size": 10, "axes.titleweight": "bold",
    })


def coverage_plot(summary: pd.DataFrame, path: Path):
    _style()
    joint = summary[summary["mode"] == "joint"]
    matrix = joint.pivot(index="case", columns="target", values="coverage90").reindex(
        index=[case.name for case in CASES], columns=list(TARGETS)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    image = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.50, vmax=1.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{100*value:.0f}%", ha="center", va="center", fontweight="bold")
    ax.set_xticks(range(len(matrix.columns)), [name.replace("secondary", "secondary φ") for name in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [LABELS[name] for name in matrix.index])
    ax.set_title("Empirical coverage of nominal 90% intervals")
    ax.set_xlabel("Recovered carbonate target")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def gate_plot(gates: pd.DataFrame, path: Path):
    _style()
    case_order = [case.name for case in CASES]
    colors = {"PASS": "#15803d", "CONDITIONAL": "#d97706", "FAIL": "#b91c1c"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    for y, case in enumerate(case_order):
        group = gates[gates["case"] == case].set_index("target").reindex(TARGETS)
        offsets = np.linspace(-0.24, 0.24, len(TARGETS))
        for offset, target in zip(offsets, TARGETS):
            row = group.loc[target]
            axes[0].scatter(row["rmse_ratio_joint_over_elastic"], y + offset, s=55,
                            color=colors[row["verdict"]], edgecolor="white", zorder=3)
            axes[1].scatter(row["false_confidence_fraction"], y + offset, s=55,
                            color=colors[row["verdict"]], edgecolor="white", zorder=3,
                            label=target if y == 0 else None)
    axes[0].axvline(1.0, color="#475569", linestyle="--", linewidth=1)
    axes[0].axvline(0.9, color="#15803d", linestyle=":", linewidth=1)
    axes[1].axvline(0.10, color="#15803d", linestyle=":", linewidth=1)
    axes[1].axvline(0.20, color="#b91c1c", linestyle="--", linewidth=1)
    axes[0].set_xlabel("RMSE joint / RMSE elastic")
    axes[1].set_xlabel("False-confidence fraction")
    axes[0].set_yticks(range(len(case_order)), [LABELS[name] for name in case_order])
    axes[0].invert_yaxis()
    axes[0].set_title("Recovery error")
    axes[1].set_title("Intervals narrower but wrong")
    fig.suptitle("Carbonate nonlinear stress-test gates", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def recovery_plot(raw: pd.DataFrame, path: Path):
    _style()
    selected = ["matched_dual", "ema_rho05_dual", "network_rho05_dual"]
    targets = ["aspect", "secondary"]
    fig, axes = plt.subplots(len(selected), len(targets), figsize=(9.4, 11.0))
    for i, case in enumerate(selected):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            g = raw[(raw["case"] == case) & (raw["target"] == target) & (raw["mode"] == "joint")]
            ax.scatter(g["truth"], g["estimate"], s=18, alpha=0.58, color="#2563eb", edgecolor="none")
            lo = min(g["truth"].min(), g["estimate"].min())
            hi = max(g["truth"].max(), g["estimate"].max())
            ax.plot([lo, hi], [lo, hi], color="#0f172a", linestyle="--", linewidth=1)
            ax.set_title(f"{LABELS[case]} — {target}")
            ax.set_xlabel("Truth")
            ax.set_ylabel("Joint estimate")
    fig.suptitle("Repeated-noise carbonate recovery", fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _table(frame: pd.DataFrame) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, gates: pd.DataFrame, args, out: Path):
    joint = summary[summary["mode"] == "joint"].copy()
    headline = gates.pivot(index="case", columns="target", values="verdict").reindex(
        index=[case.name for case in CASES], columns=list(TARGETS)
    ).reset_index()
    headline["case"] = headline["case"].map(LABELS)
    metrics = gates[[
        "case", "target", "joint_coverage90", "rmse_ratio_joint_over_elastic",
        "false_confidence_fraction", "joint_median_reduced_chi2", "verdict",
    ]].copy()
    metrics["case"] = metrics["case"].map(LABELS)
    worst = metrics.sort_values(["false_confidence_fraction", "joint_coverage90"], ascending=[False, True]).head(8)
    matched = metrics[metrics["case"] == LABELS["matched_dual"]]
    counts = gates["verdict"].value_counts()
    conclusion = (
        "The carbonate method is not robust to electrical constitutive mismatch."
        if counts.get("FAIL", 0) > 0 else
        "The tested carbonate method passed all current mismatch gates."
    )
    text = f"""# Phase 1 carbonate: nonlinear coverage stress test

## Decision

**{conclusion}** The matched case is a calibration ceiling; EMA and dual-network
truths determine whether apparent secondary-porosity/aspect-ratio resolution
survives when electrical connectivity is not the same object as elastic pore
shape.

Design: {args.panels} carbonate facies panels × {args.replicates} independent noise
realizations × {args.panel_size} depths, six truth/inversion cases. Each noisy
panel is fitted twice (elastic only and joint elastic–electrical). The nominal 90%
intervals are local bounded nonlinear Wald intervals.

## Gate matrix

{_table(headline)}

PASS requires: 90% coverage Wilson lower bound ≥80%, joint/elastic RMSE ≤0.90,
false confidence ≤10%, optimizer success ≥95%, and median reduced χ² ≤1.5.
CONDITIONAL relaxes empirical coverage to 75%, RMSE ratio to 1.05, false
confidence to 20%, and reduced χ² to 2.0.

## Matched-model ceiling

{_table(matched)}

## Largest risks

{_table(worst)}

False confidence means the joint 90% interval is at least 20% narrower than the
elastic-only interval while excluding the synthetic truth. It is therefore a
more stringent warning than data misfit alone.

## Physical construction

- **Matched dual:** the truth and inverse use the same partially connected
  dual-porosity law.
- **EMA truth:** conductivity follows a General Effective Medium percolation
  equation. The electrical percolation threshold is controlled by a latent
  variable whose correlation with elastic aspect ratio is ρ=0, 0.5, or 0.9.
- **Network truth:** primary and secondary pore networks conduct through
  separate percolating branches and a stochastic bottleneck. The inverse sees
  either the simpler dual-porosity law or total-porosity Archie.
- Deep and shallow curves mix virgin and invaded zones and use an unmodeled
  filtrate/brine resistivity ratio.

This construction follows the spirit of symmetric effective-medium joint
modeling in [Kazatchenko et al. (2004)](https://doi.org/10.1029/2003JB002443),
double-porosity carbonate inversion in
[Kazatchenko et al. (2006)](https://doi.org/10.1016/j.jappgeo.2005.07.004), and
dual-network electrical transport in bimodal carbonates in
[Tang et al. (2015)](https://doi.org/10.1093/gji/ggv096). It is a synthetic
falsification benchmark, not an implementation of any one paper verbatim.

## Interpretation boundary

The experiment intentionally holds clay volume, cement and coordination fixed
at their panel truths, so it isolates the electrical-connectivity question. A
PASS here is necessary but not sufficient for field deployment. A FAIL is
already sufficient to reject rigid microstructure sharing for that target and
truth class.

## Reproduce

```bash
python scripts/run_phase1_carbonate.py --panels {args.panels} \\
  --replicates {args.replicates} --panel-size {args.panel_size} --seed {args.seed}
```
"""
    (out / "PHASE1_CARBONATE_REPORT.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    out = ROOT / "outputs" / "phase1_carbonate"
    out.mkdir(parents=True, exist_ok=True)
    raw, summary, gates = run_carbonate_monte_carlo(
        n_panels=args.panels, replicates=args.replicates,
        panel_size=args.panel_size, seed=args.seed,
    )
    raw.to_csv(out / "recovery_raw.csv", index=False)
    summary.to_csv(out / "recovery_summary.csv", index=False)
    gates.to_csv(out / "decision_gates.csv", index=False)
    coverage_plot(summary, out / "coverage_matrix.png")
    gate_plot(gates, out / "recovery_and_false_confidence.png")
    recovery_plot(raw, out / "aspect_secondary_recovery.png")
    manifest = {
        "stage": "Phase 1 carbonate", "version": "0.3.0",
        "seed": args.seed, "panels": args.panels,
        "replicates": args.replicates, "panel_size": args.panel_size,
        "cases": [case.__dict__ for case in CASES],
        "targets": list(TARGETS), "fits": int(2 * args.panels * args.replicates * len(CASES)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(summary, gates, args, out)
    print(gates.groupby(["verdict"]).size().to_string())
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
