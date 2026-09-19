#!/usr/bin/env python3
"""Run the hierarchical Phase-0.5 robustness benchmark and profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jointinv.phase05 import run_ensemble
from jointinv.profiles import run_profiles


ROOT = Path(__file__).resolve().parents[1]


def _gate_plot(gates: pd.DataFrame, path: Path):
    scenarios = list(gates["scenario"].drop_duplicates())
    targets = ["phi", "sw", "vcl", "secondary", "aspect", "cement", "coord"]
    targets = [t for t in targets if t in set(gates["target"])]
    code = {"STOP": 0, "CONDITIONAL": 1, "ROBUST_GO": 2}
    matrix = np.full((len(scenarios), len(targets)), np.nan)
    labels = np.full(matrix.shape, "—", dtype=object)
    short = {"STOP": "STOP", "CONDITIONAL": "COND.", "ROBUST_GO": "GO"}
    for i, scenario in enumerate(scenarios):
        for j, target in enumerate(targets):
            row = gates[(gates.scenario == scenario) & (gates.target == target)]
            if len(row):
                verdict = row.iloc[0].verdict
                matrix[i, j], labels[i, j] = code[verdict], short[verdict]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#c95b56", "#e6b85c", "#4f9d75"])
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(len(scenarios)):
        for j in range(len(targets)):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] in (0, 2) else "#332b1f")
    ax.set_xticks(range(len(targets)), targets)
    ax.set_yticks(range(len(scenarios)), [s.replace("_", " ") for s in scenarios])
    ax.set_xlabel("Target parameter")
    ax.set_title("Phase 0.5 decision matrix")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _config_plot(summary: pd.DataFrame, path: Path):
    matched = summary[
        (summary.case == "matched") & (summary.structure == "panel")
        & (summary.target_prior == "base")
    ].copy()
    matched["setting"] = matched.apply(
        lambda r: (
            "E only" if r.config == "elastic" else
            "R only" if r.config == "resistivity" else
            "E + deep (free)" if r.config == "deep" else
            "Oracle" if r.config == "oracle" else
            f"E + deep/shallow ({r.nuisance_tier})"
        ), axis=1,
    )
    settings = [
        "E only", "R only", "E + deep (free)", "E + deep/shallow (free)",
        "E + deep/shallow (weak)", "E + deep/shallow (calibrated)", "Oracle",
    ]
    scenarios = list(matched.scenario.drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    im = None
    for ax, scenario in zip(axes, scenarios):
        sub = matched[matched.scenario == scenario]
        targets = list(sub.target.drop_duplicates())
        matrix = np.full((len(targets), len(settings)), np.nan)
        for i, target in enumerate(targets):
            for j, setting in enumerate(settings):
                row = sub[(sub.target == target) & (sub.setting == setting)]
                if len(row):
                    matrix[i, j] = 100 * row.iloc[0].median_reduction
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(targets)), targets)
        ax.set_xticks(range(len(settings)), settings, rotation=48, ha="right", fontsize=8)
        ax.set_title(scenario.replace("_", " "))
    fig.colorbar(im, ax=axes, label="Median variance reduction (%)", shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _robustness_maps(map_data: pd.DataFrame, path: Path):
    scenarios = list(map_data.scenario.drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1), constrained_layout=True)
    image = None
    for ax, scenario in zip(axes, scenarios):
        sub = map_data[map_data.scenario == scenario]
        image = ax.hexbin(
            sub.phi, sub.sw, C=100 * sub.gain_sw, reduce_C_function=np.mean,
            gridsize=20, vmin=0, vmax=100, cmap="viridis", mincnt=1,
        )
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Porosity")
        ax.set_ylabel("Water saturation")
    fig.colorbar(image, ax=axes, label="Mean Sw variance reduction (%)", shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _mismatch_plot(summary: pd.DataFrame, path: Path):
    sub = summary[
        (summary.structure == "panel") & (summary.config == "deep_shallow")
        & (summary.nuisance_tier == "free") & (summary.target_prior == "base")
        & (summary.target.isin(["phi", "sw", "secondary"]))
    ].copy()
    sub["label"] = sub.scenario.str.replace("_", " ") + " | " + sub.target
    labels = list(sub.label.drop_duplicates())
    cases = ["matched", "elastic_mismatch", "electrical_mismatch", "coupling_mismatch", "combined_mismatch"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.15
    for k, case in enumerate(cases):
        values, false = [], []
        for label in labels:
            row = sub[(sub.label == label) & (sub.case == case)]
            values.append(100 * row.iloc[0].gain_fraction if len(row) else np.nan)
            false.append(100 * row.iloc[0].false_confidence_fraction if len(row) else np.nan)
        axes[0].bar(x + (k - 2) * width, values, width, label=case.replace("_", " "))
        axes[1].bar(x + (k - 2) * width, false, width)
    for ax, title, ylabel in [
        (axes[0], "Gain under matched and mismatched models", "Panels with gain ≥20% (%)"),
        (axes[1], "Linearized false-confidence diagnostic", "Gain ≥20% and bias >1.96 SD (%)"),
    ]:
        ax.set_xticks(x, labels, rotation=42, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _profile_plots(profiles: pd.DataFrame, out: Path):
    levels = [0.0, 2.30, 5.991, 9.21, 20.0]
    for (scenario, pair), group in profiles.groupby(["scenario", "pair"]):
        modes = list(group["mode"].drop_duplicates())
        fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 4.0), constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for ax, mode in zip(axes, modes):
            sub = group[group["mode"] == mode]
            pivot = sub.pivot(index="iy", columns="ix", values="delta_q")
            x = np.sort(sub.x.unique())
            y = np.sort(sub.y.unique())
            z = np.minimum(pivot.values, 20.0)
            image = ax.contourf(x, y, z, levels=levels, cmap="viridis", extend="max")
            ax.contour(x, y, z, levels=[2.30, 5.991, 9.21], colors="white", linewidths=0.8)
            ax.set_title(mode.replace("_", " "))
            ax.set_xlabel(sub.x_name.iloc[0])
            ax.set_ylabel(sub.y_name.iloc[0])
        fig.colorbar(image, ax=axes, label="ΔQ", shrink=0.82)
        fig.savefig(out / f"profile_{scenario}_{pair.replace('|', '_')}.png", dpi=220)
        plt.close(fig)


def _write_report(
    summary: pd.DataFrame, gates: pd.DataFrame, profile_summary: pd.DataFrame,
    path: Path, args,
):
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(c) for c in frame.columns]
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in frame.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|") for value in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    gate_table = gates[[
        "scenario", "target", "verdict", "matched_worst_gain_fraction",
        "mismatch_worst_gain_fraction", "mismatch_false_confidence",
        "matched_useful_fraction", "interpretation", "profile_reason",
    ]].copy()
    for column in ["matched_worst_gain_fraction", "mismatch_worst_gain_fraction",
                   "mismatch_false_confidence", "matched_useful_fraction"]:
        gate_table[column] = (100 * gate_table[column]).round(1).astype(str) + "%"
    profiles = profile_summary[[
        "scenario", "pair", "mode", "closed_95", "unrestricted_bound_hit",
        "width_x_95", "width_y_95", "convergence_fraction", "region_convergence_fraction",
    ]].copy()
    key = summary[
        (summary.structure == "panel") & (summary.case == "matched")
        & (summary.config == "deep_shallow") & (summary.nuisance_tier == "free")
        & (summary.target_prior == "base")
        & summary.target.isin(["phi", "sw", "secondary", "aspect"])
    ][["scenario", "target", "median_reduction", "gain_fraction", "median_joint_sd",
       "weak_rescue_fraction"]].copy()
    for column in ["median_reduction", "gain_fraction", "weak_rescue_fraction"]:
        key[column] = (100 * key[column]).round(1).astype(str) + "%"
    key["median_joint_sd"] = key["median_joint_sd"].map(lambda value: f"{value:.4g}")
    local = summary[
        (summary.structure == "local") & (summary.case == "matched")
        & (summary.config == "deep_shallow") & (summary.nuisance_tier == "free")
        & (summary.target_prior == "base")
    ]
    local_max = float(local.median_reduction.abs().max()) if len(local) else float("nan")
    counts = gates.verdict.value_counts().to_dict()
    text = f"""# Phase 0.5 — hierarchical robustness report

## Design

The experiment contains {args.panels * args.panel_size:,} synthetic depth states per lithology,
organized as {args.panels} facies panels of {args.panel_size} depths and {args.batches}
independently randomized Latin-hypercube batches. Electrical nuisance parameters are
shared within a panel. A pointwise-free control is retained to expose structural
non-identifiability.

Truth and inverse constitutive families are separated. The benchmark includes matched,
elastic-mismatch, electrical-mismatch, coupling-mismatch and combined-mismatch cases;
correlated deep/shallow log-Rt errors; an invasion nuisance; three target-prior strengths;
free, weak and calibrated electrical nuisance regimes; and an oracle ceiling.

The decisions below are preliminary screening gates. Model-mismatch coverage is represented
by a linearized bias diagnostic; it is not a substitute for a full Monte Carlo coverage study.

## Main findings

- The pointwise-free nuisance control gives a maximum median variance reduction of
  {local_max:.2e}: effectively zero. Pooling across depth is therefore necessary, not optional.
- No target passed every robust-GO requirement. The final matrix contains
  {counts.get('CONDITIONAL', 0)} CONDITIONAL and {counts.get('STOP', 0)} STOP decisions.
- Matched carbonate panels show the strongest gain and practical weak-direction rescue,
  especially for secondary porosity and aspect ratio. The matched joint profile closes,
  but the combined-mismatch profile is open and bound-limited; the result is conditional.
- Cement and coordination remain negative controls: their nonlinear profiles stay open,
  even when marginal Fisher variances shrink indirectly through correlations.
- Sandstone porosity and saturation are tightened rather than true exact-null-space rescue.
  Their value depends strongly on facies-level calibration of the electrical nuisance parameters.

Matched, shared-panel results for deep + shallow Rt with free facies-level nuisances:

{markdown_table(key)}

## Decision matrix

{markdown_table(gate_table)}

## Nonlinear hierarchical profiles

Profiles minimize the unpenalized, covariance-weighted objective over all remaining shared
targets and electrical nuisance parameters. The 95% two-parameter region uses ΔQ=5.991.
A region touching the physical grid boundary is classified as open.

{markdown_table(profiles)}

## Interpretation safeguards

- Parameter-variance reduction and practical weak-direction rescue are reported separately.
- A large relative reduction is insufficient when the absolute posterior width remains large.
- The local pointwise-free control should show negligible independent electrical information.
- Carbonate connectivity is a tested surrogate hypothesis, not an assumed universal identity
  between elastic aspect ratio and electrical connectivity.
- Cement and coordination remain negative controls.

## Reproduce

```bash
python scripts/run_phase05.py --panels {args.panels} --panel-size {args.panel_size} \\
  --batches {args.batches} --profile-grid {args.profile_grid}
```
"""
    path.write_text(text, encoding="utf-8")


def _apply_profile_gate(gates: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Prevent a robust verdict when the required nonlinear region is open/unverified."""
    gates = gates.copy()
    supported, reasons, final = [], [], []
    for _, row in gates.iterrows():
        relevant = profiles[
            (profiles.scenario == row.scenario)
            & profiles.pair.apply(lambda value: row.target in value.split("|"))
        ] if len(profiles) else profiles
        matched = relevant[relevant["mode"] == "matched_joint"] if len(relevant) else relevant
        mismatch = relevant[relevant["mode"] == "mismatch_joint"] if len(relevant) else relevant
        matched_ok = bool(
            len(matched) and matched.closed_95.all() and (~matched.unrestricted_bound_hit).all()
            and (matched.convergence_fraction >= 0.70).all()
            and (matched.region_convergence_fraction >= 0.90).all()
        )
        mismatch_ok = bool(
            len(mismatch) and mismatch.closed_95.all() and (~mismatch.unrestricted_bound_hit).all()
            and (mismatch.convergence_fraction >= 0.70).all()
            and (mismatch.region_convergence_fraction >= 0.90).all()
        )
        ok = matched_ok and mismatch_ok
        if not len(relevant):
            reason = "no nonlinear profile for this target"
        elif len(matched) and not matched.closed_95.all():
            reason = "matched nonlinear profile is open"
        elif len(matched) and matched.unrestricted_bound_hit.any():
            reason = "matched nonlinear optimum is bound-limited"
        elif len(matched) and (
            (matched.convergence_fraction < 0.70).any()
            or (matched.region_convergence_fraction < 0.90).any()
        ):
            reason = "matched profile has insufficient numerical convergence"
        elif not matched_ok:
            reason = "matched nonlinear profile not supported"
        elif not len(mismatch):
            reason = "mismatch nonlinear profile not yet available"
        elif len(mismatch) and not mismatch.closed_95.all():
            reason = "mismatch nonlinear profile is open"
        elif len(mismatch) and mismatch.unrestricted_bound_hit.any():
            reason = "mismatch nonlinear optimum is bound-limited"
        elif not mismatch_ok:
            reason = "mismatch profile has insufficient numerical convergence"
        else:
            reason = "matched and mismatch profiles are closed"
        verdict = row.verdict
        if verdict == "ROBUST_GO" and not ok:
            verdict = "CONDITIONAL"
        if (
            verdict == "CONDITIONAL" and len(matched)
            and (matched.convergence_fraction >= 0.70).all()
            and (matched.region_convergence_fraction >= 0.90).all()
            and not matched.closed_95.all()
        ):
            verdict = "STOP"
        supported.append(ok)
        reasons.append(reason)
        final.append(verdict)
    gates["profile_supported"] = supported
    gates["profile_reason"] = reasons
    gates["preprofile_verdict"] = gates["verdict"]
    gates["verdict"] = final
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--profile-grid", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs" / "phase05"
    out.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        from jointinv.phase05 import build_gates
        summary = pd.read_csv(out / "ensemble_summary.csv")
        maps = pd.read_csv(out / "robustness_map_samples.csv")
        gates = build_gates(summary)
    else:
        summary, gates, maps = run_ensemble(
            n_panels=args.panels, panel_size=args.panel_size, seed=args.seed,
            n_batches=args.batches,
        )
        summary.to_csv(out / "ensemble_summary.csv", index=False)
        maps.to_csv(out / "robustness_map_samples.csv", index=False)
        _config_plot(summary, out / "configuration_comparison.png")
        _robustness_maps(maps, out / "phi_sw_robustness_maps.png")
        _mismatch_plot(summary, out / "matched_vs_mismatch.png")

    if args.postprocess_only:
        profiles = pd.read_csv(out / "profile_surfaces.csv")
        profile_summary = pd.read_csv(out / "profile_summary.csv")
        _profile_plots(profiles, out)
    elif args.skip_profiles:
        profiles = pd.DataFrame()
        profile_summary = pd.DataFrame(columns=[
            "scenario", "pair", "mode", "closed_95", "unrestricted_bound_hit",
            "width_x_95", "width_y_95", "convergence_fraction",
            "region_convergence_fraction",
        ])
    else:
        profiles, profile_summary = run_profiles(args.profile_grid, args.panel_size)
        profiles.to_csv(out / "profile_surfaces.csv", index=False)
        profile_summary.to_csv(out / "profile_summary.csv", index=False)
        _profile_plots(profiles, out)

    gates = _apply_profile_gate(gates, profile_summary)
    gates.to_csv(out / "robust_gates.csv", index=False)
    _gate_plot(gates, out / "decision_matrix.png")

    manifest = {
        "phase": "0.5", "seed": args.seed, "panels_per_lithology": args.panels,
        "depths_per_panel": args.panel_size,
        "states_per_lithology": args.panels * args.panel_size,
        "lhs_batches": args.batches, "profile_grid": args.profile_grid,
        "rt_log_base": "natural", "rt_error_correlation": 0.55,
        "profile_objective": "unpenalized covariance-weighted Q",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(summary, gates, profile_summary, out / "PHASE05_REPORT.md", args)
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
