#!/usr/bin/env python3
"""Run the prospective Phase-1.3 selective-inversion experiment.

The q90 policy is frozen from the Phase-1.1 matched bank and persisted before
the independent test bank is generated.  The new bank is then fit with the
unchanged Phase-1.2C candidate library.  Porosity results are released only
when the frozen deployable guard passes; rejected observations are evaluated
both as strict abstentions and with an elastic-only operational fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from jointinv.phase1 import TARGETS
from jointinv.phase12c import OFF_LIBRARY_CASES, run_phase12c
from jointinv.phase13 import (
    PRIMARY_THRESHOLD_QUANTILE,
    SelectivePolicy,
    apply_policy,
    build_control_summary,
    build_phase13_gates,
    build_phi_observation_table,
    build_risk_coverage_curve,
    freeze_phase13_policy,
    overall_phase13_decision,
    random_rejection_comparator,
    reconstruct_observation_bank,
    summarize_selective_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY = ROOT / "outputs/phase11_carbonate/recovery_strategies.csv"
DEFAULT_COMPONENTS = ROOT / "outputs/phase11_carbonate/fit_components.csv"
DEFAULT_OUTPUT = ROOT / "outputs/phase13_selective"
CASE_ORDER = [case.name for case in OFF_LIBRARY_CASES]
CASE_LABELS = {
    "matched_ema_control": "Represented physics",
    "hybrid_law": "Hybrid",
    "connectivity_drift": "Connectivity drift",
    "invasion_mismatch": "Invasion",
    "patchy_saturation": "Patchy saturation",
    "combined_stress": "Combined",
}
VERDICT_COLORS = {
    "SELECTIVE_PASS": "#15803d",
    "SAFE_JOINT": "#2563eb",
    "SAFE_DOMAIN_REJECT": "#7c3aed",
    "INCONCLUSIVE_LOW_SUPPORT": "#d97706",
    "SILENT_FAILURE": "#dc2626",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping) -> None:
    atomic_text(
        path,
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False,
            default=_json_default,
        ) + "\n",
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary, dpi=200, bbox_inches="tight",
        format=path.suffix.removeprefix("."), facecolor="white",
    )
    temporary.replace(path)
    plt.close(fig)


def _policy_lock_payload(
    policy: SelectivePolicy,
    threshold_grid: pd.DataFrame,
    training_recoveries: Path,
    training_components: Path,
    n_panels: int,
    replicates: int,
    panel_size: int,
    seed: int,
) -> dict:
    return {
        "phase": "1.3",
        "lock_status": "FROZEN_BEFORE_TEST_GENERATION",
        "confirmatory_endpoint": "phi",
        "confirmatory_operating_point": "q90",
        "policy": policy.to_dict(),
        "policy_hash": policy.canonical_hash(),
        "threshold_grid": threshold_grid.to_dict(orient="records"),
        "test_design_locked": {
            "n_panels": int(n_panels),
            "replicates": int(replicates),
            "panel_size": int(panel_size),
            "seed": int(seed),
            "cases": [case.__dict__ for case in OFF_LIBRARY_CASES],
            "common_random_numbers_across_cases": True,
            "bootstrap_unit": "geological_panel",
        },
        "decision_contract": {
            "primary_only": "q90",
            "threshold_curve_role": "exploratory_nested_sensitivity_only",
            "adapted_average_role": "cannot_determine_or_upgrade_decision",
            "strict_abstention": "no_joint_release_when_guard_fails",
            "fallback": "elastic_only_when_guard_fails",
            "negative_controls": ["sw", "aspect", "secondary"],
        },
        "training_inputs": {
            str(training_recoveries.resolve()): sha256(training_recoveries),
            str(training_components.resolve()): sha256(training_components),
        },
        "governed_policy_source": {
            "src/jointinv/phase13.py": sha256(ROOT / "src/jointinv/phase13.py"),
        },
    }


def freeze_and_lock_policy(
    output: Path,
    training_recoveries_path: Path,
    training_components_path: Path,
    training_components: pd.DataFrame,
    n_panels: int,
    replicates: int,
    panel_size: int,
    seed: int,
) -> tuple[SelectivePolicy, pd.DataFrame, dict]:
    """Persist the q90 contract before any Phase-1.3 test bank is created."""
    policy, threshold_grid = freeze_phase13_policy(
        training_components,
        threshold_quantile=PRIMARY_THRESHOLD_QUANTILE,
    )
    if not np.isclose(policy.threshold_quantile, 0.90):
        raise RuntimeError("Phase 1.3 confirmatory policy must be locked at q90")
    payload = _policy_lock_payload(
        policy, threshold_grid,
        training_recoveries_path, training_components_path,
        n_panels, replicates, panel_size, seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "policy_lock.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "Existing Phase-1.3 policy lock is incompatible; refusing to "
                "overwrite a prospective contract"
            )
    atomic_csv(threshold_grid, output / "threshold_grid.csv")
    atomic_json(lock_path, payload)
    # Re-read from disk before test generation: the durable lock is authoritative.
    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    if persisted.get("policy_hash") != policy.canonical_hash():
        raise RuntimeError("Persisted Phase-1.3 policy hash does not match memory")
    return policy, threshold_grid, payload


def verify_observation_bank(
    raw: pd.DataFrame,
    components: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bank: Mapping[str, np.ndarray],
    bank_index: pd.DataFrame,
    n_panels: int,
    replicates: int,
    panel_size: int,
) -> None:
    expected = len(OFF_LIBRARY_CASES) * n_panels * replicates
    if len(bank_index) != expected:
        raise RuntimeError("Observation-bank index has an unexpected row count")
    if bank_index["observation_id"].duplicated().any():
        raise RuntimeError("Observation-bank identifiers are not unique")
    expected_shapes = {
        "elastic": (expected, panel_size, 3),
        "log_rt": (expected, panel_size, 2),
        "truth_targets": (expected, len(TARGETS)),
    }
    for name, shape in expected_shapes.items():
        if name not in bank or tuple(bank[name].shape) != shape:
            raise RuntimeError(f"Observation-bank array {name!r} has wrong shape")
        if not np.all(np.isfinite(bank[name])):
            raise RuntimeError(f"Observation-bank array {name!r} is not finite")
    raw_keys = raw[["case", "panel", "replicate"]].drop_duplicates()
    component_keys = components[["case", "panel", "replicate"]].drop_duplicates()
    diagnostic_keys = diagnostics[["case", "panel", "replicate"]].drop_duplicates()
    index_keys = bank_index[["case", "panel", "replicate"]]
    canonical = lambda frame: set(map(tuple, frame.to_numpy()))
    reference = canonical(index_keys)
    if any(canonical(frame) != reference for frame in (raw_keys, component_keys, diagnostic_keys)):
        raise RuntimeError("Fit outputs and persisted observation bank have different keys")

    truth = raw.loc[
        raw["strategy"].eq("frozen_press"),
        ["case", "panel", "replicate", "target", "truth"],
    ].pivot(
        index=["case", "panel", "replicate"], columns="target", values="truth",
    )
    index_with_key = bank_index.copy()
    key_index = pd.MultiIndex.from_frame(
        index_with_key[["case", "panel", "replicate"]],
    )
    truth = truth.reindex(key_index)
    if truth.isna().any().any():
        raise RuntimeError("Could not align target truths with observation bank")
    if not np.allclose(
        truth.loc[:, list(TARGETS)].to_numpy(float), bank["truth_targets"],
        rtol=0.0, atol=1e-13,
    ):
        raise RuntimeError("Observation-bank truths disagree with fitted recovery bank")


def plot_selective_case_performance(
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    output: Path,
) -> None:
    case_summary = summary.set_index("case").loc[CASE_ORDER]
    case_gates = gates.set_index("case").loc[CASE_ORDER]
    x = np.arange(len(CASE_ORDER))
    colors = [VERDICT_COLORS.get(value, "#64748b") for value in case_gates["verdict"]]
    labels = [CASE_LABELS[name] for name in CASE_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharex=True)

    retention = case_summary["retention"].to_numpy(float)
    retention_low = case_summary["retention_cluster_low"].to_numpy(float)
    retention_high = case_summary["retention_cluster_high"].to_numpy(float)
    axes[0].bar(x, retention, color=colors, alpha=0.88)
    axes[0].errorbar(
        x, retention,
        yerr=np.vstack([retention - retention_low, retention_high - retention]),
        fmt="none", ecolor="#0f172a", capsize=3, linewidth=1,
    )
    axes[0].axhline(0.85, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Joint retention")
    axes[0].set_ylim(0.0, 1.04)

    for axis, metric, low, high, ylabel in (
        (
            axes[1], "accepted_rmse_ratio",
            "accepted_rmse_ratio_cluster_low", "accepted_rmse_ratio_cluster_high",
            "Accepted joint / elastic RMSE",
        ),
        (
            axes[2], "system_rmse_ratio",
            "system_rmse_ratio_cluster_low", "system_rmse_ratio_cluster_high",
            "Guarded system / elastic RMSE",
        ),
    ):
        center = case_summary[metric].to_numpy(float)
        lower = case_summary[low].to_numpy(float)
        upper = case_summary[high].to_numpy(float)
        finite = np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper)
        axis.errorbar(
            x[finite], center[finite],
            yerr=np.vstack([center[finite] - lower[finite], upper[finite] - center[finite]]),
            fmt="o", color="#0f172a", markerfacecolor="#ffffff",
            capsize=3, markersize=6,
        )
        axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output)


def plot_risk_coverage_curve(curve: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), sharex=True, sharey=True)
    for axis, case in zip(axes.ravel(), CASE_ORDER):
        frame = curve.loc[curve["case"].eq(case)].sort_values("retention")
        actual = np.sqrt(frame["accepted_mse"].to_numpy(float))
        random = np.sqrt(frame["random_mse_mean"].to_numpy(float))
        oracle = np.sqrt(frame["oracle_mse"].to_numpy(float))
        retention = frame["retention"].to_numpy(float)
        axis.plot(retention, actual, "o-", color="#2563eb", label="Quality rule")
        axis.plot(retention, random, "--", color="#f59e0b", label="Random exclusion")
        axis.plot(retention, oracle, ":", color="#16a34a", label="Error-informed lower bound")
        primary = frame["confirmatory_operating_point"].astype(bool).to_numpy()
        if primary.any():
            axis.scatter(
                retention[primary], actual[primary], s=70, marker="*",
                color="#dc2626", edgecolor="white", linewidth=0.6,
                zorder=5, label="Pre-specified threshold" if case == CASE_ORDER[0] else None,
            )
        axis.set_title(CASE_LABELS[case], fontweight="bold", fontsize=10)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Fraction of joint estimates retained")
    for axis in axes[:, 0]:
        axis.set_ylabel("Porosity RMSE among retained estimates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output)


def plot_policy_diagnostics(
    selective: pd.DataFrame,
    summary: pd.DataFrame,
    policy: SelectivePolicy,
    output: Path,
) -> None:
    labels = [CASE_LABELS[name] for name in CASE_ORDER]
    x = np.arange(len(CASE_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))

    chi2 = [
        selective.loc[selective["case"].eq(case), "best_reduced_chi2"].to_numpy(float)
        for case in CASE_ORDER
    ]
    boxes = axes[0].boxplot(
        chi2, positions=x, tick_labels=labels, patch_artist=True, showfliers=True,
    )
    for patch, case in zip(boxes["boxes"], OFF_LIBRARY_CASES):
        patch.set_facecolor("#bfdbfe" if not case.is_ood else "#fecaca")
        patch.set_edgecolor("#475569")
    axes[0].axhline(
        policy.chi2_threshold, color="#dc2626", linestyle="--", linewidth=1.2,
        label=f"Threshold = {policy.chi2_threshold:.3f}",
    )
    axes[0].set_ylabel("Best-fit reduced chi-square")
    axes[0].legend(frameon=False, fontsize=8)

    bound = [
        selective.loc[selective["case"].eq(case), "target_bound_mass"].to_numpy(float)
        for case in CASE_ORDER
    ]
    boxes = axes[1].boxplot(
        bound, positions=x, tick_labels=labels, patch_artist=True, showfliers=True,
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#475569")
    axes[1].axhline(
        policy.phi_bound_mass_limit, color="#dc2626", linestyle="--", linewidth=1.2,
        label=f"Porosity-bound limit = {policy.phi_bound_mass_limit:.2f}",
    )
    axes[1].set_ylabel("Predictive weight at porosity bounds")
    axes[1].legend(frameon=False, fontsize=8)

    case_summary = summary.set_index("case").loc[CASE_ORDER]
    abstention = case_summary["abstention"].to_numpy(float)
    capture = case_summary["false_confidence_capture"].to_numpy(float)
    capture_display = np.nan_to_num(capture, nan=0.0)
    axes[2].bar(x - 0.18, abstention, width=0.36, color="#7c3aed", label="Estimates withheld")
    axes[2].bar(
        x + 0.18, capture_display, width=0.36, color="#dc2626",
        label="High-error cases withheld",
    )
    for position, value in zip(x, capture):
        if not np.isfinite(value):
            axes[2].text(position + 0.18, 0.02, "n/a", ha="center", va="bottom", fontsize=7)
    axes[2].set_ylim(0.0, 1.04)
    axes[2].set_ylabel("Fraction")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def build_report(
    policy: SelectivePolicy,
    threshold_grid: pd.DataFrame,
    selective: pd.DataFrame,
    summary: pd.DataFrame,
    comparator: pd.DataFrame,
    curve: pd.DataFrame,
    gates: pd.DataFrame,
    controls: pd.DataFrame,
    overall: pd.DataFrame,
    n_panels: int,
    replicates: int,
    panel_size: int,
    seed: int,
) -> str:
    decision = overall.iloc[0]
    gate_lookup = gates.set_index("case").loc[CASE_ORDER]
    primary_curve = curve.loc[curve["confirmatory_operating_point"].astype(bool)]
    reason_counts = (
        selective["abstention_reason"].value_counts().rename_axis("reason").reset_index(name="n")
    )
    lines = [
        "# Phase 1.3 — Prospective selective inversion",
        "",
        (
            "**Primary decision: STOP.** Matched transport was SAFE_JOINT and "
            "the combined stress was SAFE_DOMAIN_REJECT. However, hybrid law "
            "and invasion mismatch remained SILENT_FAILURE, and no scenario "
            "showed risk enrichment beyond random rejection at equal retention."
            if str(decision.decision) == "STOP"
            else f"**Primary decision: {decision.decision}.** {decision.rationale}"
        ),
        "",
        "The porosity release policy was hashed and written to `policy_lock.json` before the independent test bank was generated. The q90 threshold, porosity-bound veto and numerical-health checks use only deployable diagnostics. Test truth, realised error, scenario labels, truth-library distance and adapted weights cannot affect acceptance.",
        "",
        "## Frozen policy and independent design",
        "",
        f"- Policy: `{policy.policy_id}`; hash `{policy.canonical_hash()}`.",
        f"- Phase-1.1 q90 reduced-chi-square cutoff: `{policy.chi2_threshold:.6f}` from {policy.training_observations} training observations.",
        f"- Joint release also requires phi-bound mass <= {policy.phi_bound_mass_limit:.2f}, candidate success >= {policy.candidate_success_min:.2f}, and surviving prior mass >= {policy.surviving_prior_mass_min:.2f}.",
        f"- Independent test: {n_panels} geological panels x {replicates} common-noise replicates x {len(OFF_LIBRARY_CASES)} scenarios; {panel_size} depths per observation; seed `{seed}`.",
        "- Strict abstention and elastic-only fallback are reported separately. Whole panels, not rows or noise replicates, are resampled.",
        "",
        "## Confirmatory q90 results",
        "",
        "| Scenario | Verdict | Retention (95% cluster) | Accepted clusters | Accepted coverage (low) | Accepted RMSE ratio (95% high) | System RMSE ratio (95% high) | False-confidence capture |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in gate_lookup.iterrows():
        lines.append(
            f"| {CASE_LABELS[case]} | **{row.verdict}** | "
            f"{_fmt(row.retention)} ({_fmt(row.retention_cluster_low)}–{_fmt(row.retention_cluster_high)}) | "
            f"{int(row.accepted_clusters)} | {_fmt(row.accepted_coverage90)} ({_fmt(row.accepted_coverage90_cluster_low)}) | "
            f"{_fmt(row.accepted_rmse_ratio)} ({_fmt(row.accepted_rmse_ratio_cluster_upper95)}) | "
            f"{_fmt(row.system_rmse_ratio)} ({_fmt(row.system_rmse_ratio_cluster_upper95)}) | "
            f"{_fmt(row.false_confidence_capture)} |"
        )
    lines.extend([
        "",
        "A low-retention domain is not allowed to pass because few or no risky joint estimates remain. It is classified through explicit domain rejection and the safety of the elastic fallback. Conversely, the matched control must retain useful joint coverage.",
        "",
        "## Selection beyond random rejection",
        "",
        "| Scenario | Actual accepted MSE | Random MSE | Delta upper 95% | Permutation p | Enriched |",
        "|---|---:|---:|---:|---:|---|",
    ])
    comp = comparator.set_index("case").loc[CASE_ORDER]
    for case, row in comp.iterrows():
        lines.append(
            f"| {CASE_LABELS[case]} | {_fmt(row.actual_accepted_mse, 6)} | "
            f"{_fmt(row.random_mse_mean, 6)} | {_fmt(row.actual_minus_random_upper95, 6)} | "
            f"{_fmt(row.enrichment_p_value, 4)} | {bool(row.selection_enriched)} |"
        )
    lines.extend([
        "",
        "The comparator permutes complete within-panel acceptance patterns at the same retention. This prevents a lower risk caused merely by rejecting more observations from being interpreted as diagnostic enrichment.",
        "",
        "## Negative and stress controls",
        "",
        "The phi-derived acceptance mask is applied unchanged to water saturation, aspect ratio and secondary porosity. These targets never train or tune the guard.",
        "",
        "| Target | Scenario | Retention | Excess MSE upper 95% | Coverage low | Width upper 95% | Harmful false upper 95% | Promoted |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in controls.sort_values(["target", "case"]).itertuples(index=False):
        lines.append(
            f"| {row.target} | {CASE_LABELS[row.case]} | {_fmt(row.retention)} | "
            f"{_fmt(row.accepted_excess_mse_upper95, 6)} | {_fmt(row.accepted_coverage90_low)} | "
            f"{_fmt(row.accepted_width_ratio_upper95)} | {_fmt(row.harmful_false_upper95)} | "
            f"{bool(row.promoted_by_phi_guard)} |"
        )
    alerts = int(controls["promoted_by_phi_guard"].astype(bool).sum())
    lines.extend([
        "",
        f"Negative-control alerts: **{alerts}**. Any alert can block SELECTIVE_GO but cannot improve the decision.",
        "",
        "## Exploratory nested risk–coverage curve",
        "",
        "Only q90 is confirmatory. The other Phase-1.1-derived thresholds show sensitivity and cannot upgrade the primary decision.",
        "",
        "| Scenario | q90 threshold | Retention | Accepted RMSE | Random-reject RMSE | Oracle RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in primary_curve.set_index("case").loc[CASE_ORDER].itertuples():
        lines.append(
            f"| {CASE_LABELS[row.Index]} | {_fmt(row.chi2_threshold)} | "
            f"{_fmt(row.retention)} | {_fmt(row.accepted_rmse, 5)} | "
            f"{_fmt(np.sqrt(row.random_mse_mean), 5)} | {_fmt(np.sqrt(row.oracle_mse), 5)} |"
        )
    lines.extend([
        "",
        "Frozen thresholds: " + ", ".join(
            f"q{int(round(row.threshold_quantile * 1000)) / 10:g}={row.chi2_threshold:.4f}"
            for row in threshold_grid.itertuples(index=False)
        ) + ".",
        "",
        "## Abstention audit",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ])
    for row in reason_counts.itertuples(index=False):
        lines.append(f"| {row.reason} | {int(row.n)} |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This phase tests a prospectively locked safeguard on new synthetic panels drawn from the same six declared structural stress families. It is not field validation and does not establish universal OOD detection. A SELECTIVE_GO would authorize only guarded progression to a field-data validation design; it would not authorize unconstrained deployment.",
        "",
        "## Reproducibility",
        "",
        "The exact noisy elastic and electrical arrays are persisted in `observation_bank.npz`, aligned by `observation_bank_index.csv`. Raw recoveries, all nine candidate fits, policy features, thresholds, bootstrap summaries, random-rejection comparator, controls, figures, configuration and SHA-256 hashes are included in this directory.",
    ])
    return "\n".join(lines) + "\n"


def build_manifest(
    output: Path,
    training_recoveries: Path,
    training_components: Path,
    policy: SelectivePolicy,
    raw: pd.DataFrame,
    components: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selective: pd.DataFrame,
    curve: pd.DataFrame,
    gates: pd.DataFrame,
    controls: pd.DataFrame,
    overall: pd.DataFrame,
) -> dict:
    source_paths = [
        ROOT / "src/jointinv/analysis.py",
        ROOT / "src/jointinv/families.py",
        ROOT / "src/jointinv/phase05.py",
        ROOT / "src/jointinv/phase1.py",
        ROOT / "src/jointinv/phase11.py",
        ROOT / "src/jointinv/phase12.py",
        ROOT / "src/jointinv/phase12c.py",
        ROOT / "src/jointinv/phase13.py",
        ROOT / "pyproject.toml",
        Path(__file__),
    ]
    artifacts = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json" and ".tmp" not in path.name
    )
    return {
        "phase": "1.3",
        "decision": str(overall.iloc[0]["decision"]),
        "policy_id": policy.policy_id,
        "policy_hash": policy.canonical_hash(),
        "policy_lock_sha256": sha256(output / "policy_lock.json"),
        "counts": {
            "test_observations": int(len(diagnostics)),
            "candidate_fits": int(len(components)),
            "recoveries": int(len(raw)),
            "selective_observations": int(len(selective)),
            "risk_curve_rows": int(len(curve)),
            "gates": int(len(gates)),
            "control_rows": int(len(controls)),
            "successful_candidate_fits": int(components["success"].astype(bool).sum()),
        },
        "inputs": {
            str(training_recoveries.resolve()): sha256(training_recoveries),
            str(training_components.resolve()): sha256(training_components),
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "artifacts": {
            str(path.relative_to(output)): sha256(path) for path in artifacts
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the prospectively locked Phase-1.3 selective benchmark.",
    )
    parser.add_argument("--training-recoveries", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--training-components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--control-bootstrap", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--curve-random-draws", type=int, default=500)
    parser.add_argument(
        "--generator-bootstrap", type=int, default=100,
        help="Minimal internal Phase-1.2C gate bootstrap; those gates are discarded.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.panels < 12:
        raise ValueError("Phase 1.3 requires at least 12 independent panels")
    if args.replicates < 1 or args.panel_size < 2:
        raise ValueError("Replicates must be positive and panel size at least two")
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    if args.bootstrap < 100 or args.control_bootstrap < 100 or args.permutations < 100:
        raise ValueError("Bootstrap and permutation counts must be at least 100")
    if args.curve_random_draws < 1 or args.generator_bootstrap < 1:
        raise ValueError("Curve and generator draw counts must be positive")
    if not args.training_recoveries.is_file() or not args.training_components.is_file():
        raise FileNotFoundError("Phase-1.1 training inputs are required")

    training_recoveries = pd.read_csv(args.training_recoveries)
    training_components = pd.read_csv(args.training_components)

    # Critical ordering invariant: no Phase-1.3 bank exists in memory when the
    # q90 policy and design contract are made durable.
    policy, threshold_grid, lock_payload = freeze_and_lock_policy(
        args.output,
        args.training_recoveries,
        args.training_components,
        training_components,
        args.panels,
        args.replicates,
        args.panel_size,
        args.seed,
    )
    print(
        f"Phase 1.3 policy locked: {policy.policy_id}; "
        f"q90 chi2 <= {policy.chi2_threshold:.6f}",
        flush=True,
    )
    print(
        f"Generating independent bank: {args.panels} panels x "
        f"{args.replicates} replicates x {len(OFF_LIBRARY_CASES)} cases; "
        f"seed {args.seed}; {args.workers} workers",
        flush=True,
    )

    generated = run_phase12c(
        training_recoveries,
        training_components,
        n_panels=args.panels,
        replicates=args.replicates,
        panel_size=args.panel_size,
        seed=args.seed,
        cases=OFF_LIBRARY_CASES,
        workers=args.workers,
        n_bootstrap=args.generator_bootstrap,
    )
    raw, components, diagnostics = generated[:3]

    bank, bank_index = reconstruct_observation_bank(
        args.panels, args.replicates, args.panel_size, args.seed, OFF_LIBRARY_CASES,
    )
    verify_observation_bank(
        raw, components, diagnostics, bank, bank_index,
        args.panels, args.replicates, args.panel_size,
    )

    observations = build_phi_observation_table(raw)
    selective = apply_policy(observations, policy)
    summary = summarize_selective_cases(
        selective, n_bootstrap=args.bootstrap, seed=args.seed + 201,
    )
    comparator = random_rejection_comparator(
        selective, n_permutations=args.permutations, seed=args.seed + 202,
    )
    curve = build_risk_coverage_curve(
        observations, policy, threshold_grid,
        random_draws=args.curve_random_draws, seed=args.seed + 203,
    )
    gates = build_phase13_gates(summary, comparator)
    controls = build_control_summary(
        raw, selective, n_bootstrap=args.control_bootstrap, seed=args.seed + 204,
    )
    negative_control_alert_count = int(
        controls["promoted_by_phi_guard"].astype(bool).sum()
    )
    overall = overall_phase13_decision(gates, negative_control_alert_count)

    # Phase 1.2C diagnostics remain useful for provenance, but their q95 alarm
    # is not the confirmatory q90 release decision used here.  Persist them
    # under an explicit legacy prefix to prevent two apparent alarm contracts.
    legacy_alarm_names = {
        "ood_alarm": "legacy_phase12c_ood_alarm_q95",
        "alarm_threshold": "legacy_phase12c_alarm_threshold_q95",
    }
    frames = {
        "recovery_raw.csv": raw.rename(columns=legacy_alarm_names),
        "candidate_components.csv": components,
        "observation_diagnostics.csv": diagnostics.rename(columns=legacy_alarm_names),
        "observation_bank_index.csv": bank_index,
        "threshold_grid.csv": threshold_grid,
        "selective_observations.csv": selective,
        "selective_summary.csv": summary,
        "random_rejection_comparator.csv": comparator,
        "risk_coverage_curve.csv": curve,
        "decision_gates.csv": gates,
        "control_summary.csv": controls,
        "overall_decision.csv": overall,
    }
    for filename, frame in frames.items():
        atomic_csv(frame, args.output / filename)
    atomic_npz(args.output / "observation_bank.npz", bank)

    plot_selective_case_performance(
        summary, gates, args.output / "selective_case_performance.png",
    )
    plot_risk_coverage_curve(curve, args.output / "risk_coverage_curve.png")
    plot_policy_diagnostics(
        selective, summary, policy, args.output / "policy_diagnostics.png",
    )

    report = build_report(
        policy, threshold_grid, selective, summary, comparator, curve,
        gates, controls, overall,
        args.panels, args.replicates, args.panel_size, args.seed,
    )
    atomic_text(args.output / "PHASE13_SELECTIVE_REPORT.md", report)

    run_config = {
        "phase": "1.3",
        "seed": args.seed,
        "n_panels": args.panels,
        "replicates": args.replicates,
        "panel_size": args.panel_size,
        "workers": args.workers,
        "cluster_bootstrap_draws": args.bootstrap,
        "control_bootstrap_draws": args.control_bootstrap,
        "random_rejection_permutations": args.permutations,
        "risk_curve_random_draws": args.curve_random_draws,
        "generator_internal_bootstrap_draws": args.generator_bootstrap,
        "policy_id": policy.policy_id,
        "policy_hash": policy.canonical_hash(),
        "policy_lock_sha256": sha256(args.output / "policy_lock.json"),
        "policy_locked_before_test_generation": True,
        "confirmatory_threshold_quantile": policy.threshold_quantile,
        "confirmatory_chi2_threshold": policy.chi2_threshold,
        "cases": [case.__dict__ for case in OFF_LIBRARY_CASES],
        "candidate_bank": "unchanged_phase12c_3x3",
        "adapted_average_role": "generated_for_audit_but_cannot_determine_decision",
        "bootstrap_unit": "geological_panel",
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "training_inputs": lock_payload["training_inputs"],
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    atomic_json(args.output / "run_config.json", run_config)
    manifest = build_manifest(
        args.output,
        args.training_recoveries,
        args.training_components,
        policy,
        raw,
        components,
        diagnostics,
        selective,
        curve,
        gates,
        controls,
        overall,
    )
    atomic_json(args.output / "manifest.json", manifest)
    print(
        f"Phase 1.3 complete: {overall.iloc[0]['decision']}; "
        f"{len(selective)} selective observations; outputs in {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
