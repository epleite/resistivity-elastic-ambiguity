#!/usr/bin/env python3
"""Run Phase 1.2B profile and parametric-bootstrap sentinel validation."""

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

from jointinv.phase1 import CARBONATE, FIT_NAMES as PHASE1_FIT_NAMES, TARGETS, fit_panel
from jointinv.phase11 import (
    TRUTH_CASES, _range, converged_weights, fit_candidate, mixture_recovery,
    phase11_design, simulate_phase11_observations,
)
from jointinv.phase12 import conformal_order_statistic
from jointinv.phase12b import (
    SENTINEL_SPECS, active_component_weights, connected_profile_interval,
    fit_profile_point, fit_unrestricted_candidate, paired_rmse_ratio_interval,
    select_sentinels, wilson_interval,
)


CASE_LABELS = {
    "dual_partial_truth": "Connected dual-porosity",
    "ema_rho0": "Effective-medium ρ=0",
    "ema_rho05": "Effective-medium ρ=.5",
    "ema_rho09": "Effective-medium ρ=.9",
    "network_rho05": "Two-network ρ=.5",
}
TARGET_LABELS = {
    "phi": "Porosity", "sw": "Water saturation",
    "aspect": "Aspect ratio", "secondary": "Secondary porosity",
}
TRUTH_LOOKUP = {case.name: case for case in TRUTH_CASES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_state(truth: dict) -> dict:
    fixed = dict(truth)
    for name in PHASE1_FIT_NAMES:
        fixed.pop(name, None)
    fixed.update({
        "vcl": truth["vcl"], "cement": truth["cement"],
        "coord": truth["coord"], "phi_e": truth["phi_e"],
    })
    return fixed


def _operational_fits(observed_e, observed_r, fixed, panel_size, elastic, active):
    fits, weights = [], []
    weight_lookup = {
        (row["family"], row["coupling"]): float(row["full_weight"])
        for row in active
    }
    # Preserve the original Phase-1.1 warm-start sequence.  Even a zero-weight
    # rigid candidate seeds the subsequent partial fit and can alter which
    # nonlinear basin is reached.
    for family in sorted({row["family"] for row in active}):
        prior = None
        for coupling in ("rigid", "partial", "independent"):
            fitted = fit_candidate(
                observed_e, observed_r, fixed, family, coupling, panel_size,
                elastic["estimate"], prior,
            )
            prior = fitted["estimate"]
            if (family, coupling) in weight_lookup:
                fits.append(fitted)
                weights.append(weight_lookup[(family, coupling)])
    weights = converged_weights(fits, np.asarray(weights, dtype=float))
    return fits, weights


def _primary_family(active):
    totals = {}
    for row in active:
        totals[row["family"]] = totals.get(row["family"], 0.0) + float(row["full_weight"])
    return max(sorted(totals), key=lambda name: totals[name])


def _reconstruct_original(task):
    seed, panel_size = task["seed"], task["panel_size"]
    sentinel, active = task["sentinel"], task["active"]
    truth = phase11_design(task["panels"], seed)[int(sentinel["panel"])]
    truth_case = TRUTH_LOOKUP[sentinel["case"]]
    replicate = int(sentinel["replicate"])
    rng = np.random.default_rng(seed + 10007 * int(sentinel["panel"]) + 101 * replicate)
    observed_e, observed_r = simulate_phase11_observations(
        truth, truth_case, panel_size,
        rng.normal(size=(panel_size, 3)), rng.normal(size=(panel_size, 2)),
    )
    fixed = _fixed_state(truth)
    fit_rng = np.random.default_rng(seed + 50021 * int(sentinel["panel"]) + 307 * replicate)
    elastic = fit_panel(
        observed_e, observed_r, fixed, "dual_porosity", False,
        panel_size, fit_rng, n_starts=1,
    )
    fits, weights = _operational_fits(
        observed_e, observed_r, fixed, panel_size, elastic, active
    )
    return truth, observed_e, observed_r, fixed, elastic, fits, weights


def _profile_worker(task):
    sentinel, active = task["sentinel"], task["active"]
    truth, observed_e, observed_r, fixed, elastic, fits, weights = _reconstruct_original(task)
    target = sentinel["target"]
    operational = mixture_recovery(fits, weights, target)
    family = _primary_family(active)
    initial = None
    for fit in fits:
        if fit["family"] == family and fit["coupling"] == "independent":
            initial = fit["estimate"]
            break
    if initial is None:
        initial = next(fit["estimate"] for fit in fits if fit["family"] == family)
        initial = dict(initial)
        initial.setdefault("q_conn", 0.0)
    unrestricted = fit_unrestricted_candidate(
        observed_e, observed_r, fixed, family, "independent", task["panel_size"],
        initial, objective_mode="data",
    )
    lower, upper = ((0.0, 0.12) if target == "secondary" else _range(target))
    grid = np.linspace(lower, upper, task["profile_points"])
    grid = np.unique(np.r_[grid, unrestricted["state"][target], truth[target]])
    rows = []
    for value in grid:
        fitted = fit_profile_point(
            observed_e, observed_r, fixed, family, "independent",
            task["panel_size"], target, float(value), unrestricted["state"],
            objective_mode="data",
        )
        rows.append({
            "sentinel": sentinel["sentinel"], "role": sentinel["role"],
            "case": sentinel["case"], "target": target,
            "panel": int(sentinel["panel"]), "replicate": int(sentinel["replicate"]),
            "truth": float(truth[target]), "profile_family": family,
            "profile_coupling": "free_q", "profile_value": float(value),
            "objective": fitted["objective"], "data_objective": fitted["data_objective"],
            "success": fitted["success"], "feasibility_residual": fitted["feasibility_residual"],
            "nfev": fitted["nfev"], "total_nfev": fitted["total_nfev"],
        })
    return rows, {
        "sentinel": sentinel["sentinel"], "profile_family": family,
        "truth": float(truth[target]), "unrestricted_estimate": float(unrestricted["state"][target]),
        "unrestricted_objective": float(unrestricted["objective"]),
        "unrestricted_success": bool(unrestricted["success"]),
        "operational_estimate": float(operational["estimate"]),
        "phase12a_estimate": float(sentinel["phase12a_estimate"]),
        "reproduction_error": float(operational["estimate"] - sentinel["phase12a_estimate"]),
    }


def _bootstrap_worker(task):
    sentinel, active = task["sentinel"], task["active"]
    seed, panel_size = task["seed"], task["panel_size"]
    bootstrap_index = task["bootstrap_index"]
    truth = phase11_design(task["panels"], seed)[int(sentinel["panel"])]
    truth_case = TRUTH_LOOKUP[sentinel["case"]]
    noise_seed = seed + 1_000_003 + task["sentinel_index"] * 100_003 + bootstrap_index * 997
    noise_rng = np.random.default_rng(noise_seed)
    observed_e, observed_r = simulate_phase11_observations(
        truth, truth_case, panel_size,
        noise_rng.normal(size=(panel_size, 3)), noise_rng.normal(size=(panel_size, 2)),
    )
    fixed = _fixed_state(truth)
    elastic_rng = np.random.default_rng(noise_seed + 41)
    elastic = fit_panel(
        observed_e, observed_r, fixed, "dual_porosity", False,
        panel_size, elastic_rng, n_starts=1,
    )
    fits, weights = _operational_fits(
        observed_e, observed_r, fixed, panel_size, elastic, active
    )
    target = sentinel["target"]
    operational = mixture_recovery(fits, weights, target)
    elastic_component = {
        "estimate": elastic["estimate"], "sd": elastic["sd"],
        "reduced_chi2": elastic["cost"] / max(3 * panel_size - len(TARGETS), 1),
        "success": elastic["success"],
    }
    elastic_result = mixture_recovery([elastic_component], np.array([1.0]), target)

    family = _primary_family(active)
    initial = None
    for fit in fits:
        if fit["family"] == family and fit["coupling"] == "independent":
            initial = fit["estimate"]
            break
    if initial is None:
        initial = dict(next(fit["estimate"] for fit in fits if fit["family"] == family))
        initial.setdefault("q_conn", 0.0)
    unrestricted = fit_unrestricted_candidate(
        observed_e, observed_r, fixed, family, "independent", panel_size,
        initial, objective_mode="data",
    )
    conditioned = fit_profile_point(
        observed_e, observed_r, fixed, family, "independent", panel_size,
        target, float(truth[target]), unrestricted["state"], objective_mode="data",
    )
    if conditioned["objective"] < unrestricted["objective"] - 1e-4:
        retried = fit_unrestricted_candidate(
            observed_e, observed_r, fixed, family, "independent", panel_size,
            conditioned["state"], objective_mode="data",
        )
        if retried["objective"] < unrestricted["objective"]:
            unrestricted = retried
    lr_raw = float(conditioned["objective"] - unrestricted["objective"])
    lr = max(lr_raw, 0.0)

    physical_lower, physical_upper = _range(target)
    lower_cal = max(physical_lower, operational["estimate"] - sentinel["phase12a_c90"] * operational["sd"])
    upper_cal = min(physical_upper, operational["estimate"] + sentinel["phase12a_c90"] * operational["sd"])
    lower_elastic = max(physical_lower, elastic_result["estimate"] - sentinel["elastic_c90"] * elastic_result["sd"])
    upper_elastic = min(physical_upper, elastic_result["estimate"] + sentinel["elastic_c90"] * elastic_result["sd"])
    calibrated_covered = lower_cal <= truth[target] <= upper_cal
    elastic_covered = lower_elastic <= truth[target] <= upper_elastic
    width = upper_cal - lower_cal
    elastic_width = upper_elastic - lower_elastic
    false_confidence = width <= 0.8 * elastic_width and not calibrated_covered
    return {
        "sentinel": sentinel["sentinel"], "role": sentinel["role"],
        "case": sentinel["case"], "target": target,
        "panel": int(sentinel["panel"]), "bootstrap_index": bootstrap_index,
        "split": "calibration" if bootstrap_index < task["calibration_count"] else "validation",
        "truth": float(truth[target]), "estimate": operational["estimate"],
        "sd": operational["sd"], "original_lower90": operational["lower90"],
        "original_upper90": operational["upper90"],
        "original_covered90": operational["lower90"] <= truth[target] <= operational["upper90"],
        "calibrated_lower90": lower_cal, "calibrated_upper90": upper_cal,
        "calibrated_covered90": calibrated_covered, "calibrated_width90": width,
        "elastic_estimate": elastic_result["estimate"], "elastic_sd": elastic_result["sd"],
        "elastic_calibrated_lower90": lower_elastic, "elastic_calibrated_upper90": upper_elastic,
        "elastic_calibrated_covered90": elastic_covered, "elastic_calibrated_width90": elastic_width,
        "width_ratio": width / max(elastic_width, 1e-12),
        "false_confidence": false_confidence,
        "harmful_false_confidence": false_confidence and elastic_covered,
        "standardized_absolute_error": abs(operational["estimate"] - truth[target]) / max(operational["sd"], 1e-12),
        "profile_lr_truth": lr, "profile_lr_raw": lr_raw,
        "profile_unrestricted_success": unrestricted["success"],
        "profile_conditioned_success": conditioned["success"],
        "operational_success": all(fit["success"] for fit in fits),
        "elastic_success": elastic["success"],
        "any_critical_bound": any(
            fit["bound_hit"].get(name, False)
            for fit in fits for name in ("phi", "sw", "aspect", "secondary_fraction", "q_conn")
        ),
        "total_nfev": int(
            sum(fit["total_nfev"] for fit in fits) + elastic["total_nfev"]
            + unrestricted["total_nfev"] + conditioned["total_nfev"]
        ),
    }


def _paired_interval(values, seed, statistic=np.median, n_bootstrap=5000):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    draws = np.apply_along_axis(statistic, 1, array[indices])
    return float(statistic(array)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(profile_raw, bootstrap_raw, sentinels):
    summaries = []
    for _, sentinel in sentinels.iterrows():
        name = sentinel["sentinel"]
        calibration = bootstrap_raw[(bootstrap_raw.sentinel == name) & (bootstrap_raw.split == "calibration")]
        validation = bootstrap_raw[(bootstrap_raw.sentinel == name) & (bootstrap_raw.split == "validation")]
        all_replicates = bootstrap_raw[bootstrap_raw.sentinel == name]
        q90_lr = conformal_order_statistic(calibration.profile_lr_truth, 0.90)
        q90_c = conformal_order_statistic(calibration.standardized_absolute_error, 0.90)
        wilks_threshold = 2.705543454095404
        profile = profile_raw[profile_raw.sentinel == name].sort_values("profile_value").copy()
        profile["delta_chi2"] = profile.objective - profile.objective.min()
        wilks_interval = connected_profile_interval(profile.profile_value, profile.delta_chi2, threshold=wilks_threshold)
        bootstrap_interval = connected_profile_interval(profile.profile_value, profile.delta_chi2, threshold=q90_lr)
        truth = float(sentinel["truth"])
        wilks_original_covers = wilks_interval["lower"] <= truth <= wilks_interval["upper"]
        bootstrap_original_covers = bootstrap_interval["lower"] <= truth <= bootstrap_interval["upper"]
        wald_success = int(validation.calibrated_covered90.sum())
        profile_success = int((validation.profile_lr_truth <= q90_lr).sum())
        wilks_success = int((validation.profile_lr_truth <= wilks_threshold).sum())
        original_success = int(validation.original_covered90.sum())
        wald_low, wald_high = wilson_interval(wald_success, len(validation), z=1.96)
        profile_low, profile_high = wilson_interval(profile_success, len(validation), z=1.96)
        wilks_low, wilks_high = wilson_interval(wilks_success, len(validation), z=1.96)
        original_low, original_high = wilson_interval(original_success, len(validation), z=1.96)
        joint_error = validation.estimate - validation.truth
        elastic_error = validation.elastic_estimate - validation.truth
        rmse_ratio, rmse_low, rmse_high = paired_rmse_ratio_interval(
            joint_error, elastic_error, seed=12001 + int(sentinel["panel"]), n_bootstrap=5000
        )
        width_ratio, width_low, width_high = _paired_interval(
            validation.width_ratio, seed=13001 + int(sentinel["panel"])
        )
        false_fraction = float(validation.false_confidence.mean())
        false_low, false_high = wilson_interval(int(validation.false_confidence.sum()), len(validation), z=1.96)
        fit_valid = all_replicates[[
            "profile_unrestricted_success", "profile_conditioned_success",
            "operational_success", "elastic_success",
        ]].all(axis=1)
        fit_success = float(fit_valid.mean())
        numerical_failure_count = int((~fit_valid).sum())
        negative_lr = float((validation.profile_lr_raw < -1e-4).mean())
        error_calibration = calibration.estimate - calibration.truth
        basic_lower = max(_range(sentinel["target"])[0], sentinel["phase12a_estimate"] - np.quantile(error_calibration, 0.95))
        basic_upper = min(_range(sentinel["target"])[1], sentinel["phase12a_estimate"] - np.quantile(error_calibration, 0.05))
        basic_covers = basic_lower <= truth <= basic_upper

        strict = (
            wald_low >= 0.85 and profile_low >= 0.85
            and rmse_high < 1.0 and width_high < 1.0
            and false_high <= 0.10 and fit_success >= 0.99
            and wilks_original_covers and bootstrap_original_covers and basic_covers
            and not bootstrap_interval["lower_bound_limited"]
            and not bootstrap_interval["upper_bound_limited"]
            and not bootstrap_interval["disconnected"]
            and negative_lr == 0.0
        )
        relaxed = (
            validation.calibrated_covered90.mean() >= 0.85
            and (validation.profile_lr_truth <= q90_lr).mean() >= 0.85
            and rmse_ratio < 1.0 and width_ratio <= 1.10
            and bootstrap_original_covers and fit_success >= 0.95
            and negative_lr <= 0.01
        )
        verdict = "PASS" if strict else ("CONDITIONAL" if relaxed else "FAIL")
        if sentinel["role"] == "negative_control":
            control_status = {
                "PASS": "UNEXPECTED_PASS", "CONDITIONAL": "CONTROL_AMBIGUOUS",
                "FAIL": "NEGATIVE_CONTROL_CONFIRMED",
            }[verdict]
        else:
            control_status = "not_applicable"
        summaries.append({
            **sentinel.to_dict(), "bootstrap_calibration_n": len(calibration),
            "bootstrap_validation_n": len(validation), "profile_family": profile.profile_family.iloc[0],
            "bootstrap_profile_threshold90": q90_lr,
            "phase12a_c90": sentinel["phase12a_c90"], "bootstrap_required_c90": q90_c,
            "original_coverage90": float(validation.original_covered90.mean()),
            "original_coverage_low": original_low, "original_coverage_high": original_high,
            "calibrated_wald_coverage90": float(validation.calibrated_covered90.mean()),
            "calibrated_wald_coverage_low": wald_low, "calibrated_wald_coverage_high": wald_high,
            "wilks_profile_coverage90": float((validation.profile_lr_truth <= wilks_threshold).mean()),
            "wilks_profile_coverage_low": wilks_low, "wilks_profile_coverage_high": wilks_high,
            "bootstrap_profile_coverage90": float((validation.profile_lr_truth <= q90_lr).mean()),
            "bootstrap_profile_coverage_low": profile_low, "bootstrap_profile_coverage_high": profile_high,
            "rmse_ratio_joint_over_elastic": rmse_ratio, "rmse_ratio_low": rmse_low,
            "rmse_ratio_high": rmse_high, "median_width_ratio": width_ratio,
            "width_ratio_low": width_low, "width_ratio_high": width_high,
            "false_confidence_fraction": false_fraction, "false_confidence_low": false_low,
            "false_confidence_high": false_high, "fit_success_fraction": fit_success,
            "numerical_failure_count": numerical_failure_count,
            "critical_bound_fraction": float(validation.any_critical_bound.mean()),
            "negative_lr_fraction": negative_lr,
            "wilks_interval_lower": wilks_interval["lower"], "wilks_interval_upper": wilks_interval["upper"],
            "wilks_interval_covers_truth": wilks_original_covers,
            "bootstrap_profile_lower": bootstrap_interval["lower"],
            "bootstrap_profile_upper": bootstrap_interval["upper"],
            "bootstrap_profile_covers_truth": bootstrap_original_covers,
            "profile_bound_limited": bool(
                bootstrap_interval["lower_bound_limited"] or bootstrap_interval["upper_bound_limited"]
            ),
            "profile_disconnected": bool(bootstrap_interval["disconnected"]),
            "bootstrap_basic_lower": basic_lower, "bootstrap_basic_upper": basic_upper,
            "bootstrap_basic_covers_truth": basic_covers,
            "verdict": verdict, "control_status": control_status,
        })
    return pd.DataFrame(summaries)


def plot_profiles(profile_raw, summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for axis, sentinel in zip(axes.flat, SENTINEL_SPECS):
        data = profile_raw[profile_raw.sentinel == sentinel.name].sort_values("profile_value").copy()
        data["delta"] = data.objective - data.objective.min()
        row = summary[summary.sentinel == sentinel.name].iloc[0]
        axis.plot(data.profile_value, data.delta, color="#1f77b4", lw=2)
        axis.axhline(2.70554, color="black", ls="--", label="Wilks 90% threshold")
        axis.axhline(
            row.bootstrap_profile_threshold90,
            color="#d95f02",
            ls=":",
            lw=2,
            label="Bootstrap 90% threshold",
        )
        axis.axvline(row.truth, color="#15803d", lw=1.5, label="Truth")
        axis.axvspan(
            row.bootstrap_profile_lower,
            row.bootstrap_profile_upper,
            color="#d95f02",
            alpha=0.12,
            label="Bootstrap 90% profile interval",
        )
        axis.set_title(f"{TARGET_LABELS[sentinel.target]} — {CASE_LABELS[sentinel.case]}")
        axis.set_xlabel(TARGET_LABELS[sentinel.target])
        axis.set_ylabel("Profile Δχ² (data only)")
        axis.set_ylim(0, max(12.0, 3.0 * row.bootstrap_profile_threshold90))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Data-only likelihood profiles and bootstrap thresholds")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coverages(summary, output):
    labels = [TARGET_LABELS[target] for target in summary.target]
    metrics = [
        ("original_coverage90", "Original local", "#6b7280"),
        ("calibrated_wald_coverage90", "Held-panel calibrated", "#1f77b4"),
        ("wilks_profile_coverage90", "Wilks profile", "#9467bd"),
        ("bootstrap_profile_coverage90", "Bootstrap-profile", "#d95f02"),
    ]
    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(summary)); width = 0.19
    for index, (column, label, color) in enumerate(metrics):
        axis.bar(x + (index - 1.5) * width, summary[column], width, label=label, color=color)
    axis.axhspan(0.85, 0.95, color="#15803d", alpha=0.08)
    axis.axhline(0.90, color="black", lw=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.45, 1.01)
    axis.set_ylabel("Conditional repeated-noise coverage")
    axis.set_title("Sentinel coverage on the held validation bootstrap split")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(summary, output):
    counts = summary.verdict.value_counts()
    lines = [
        "# Phase 1.2B carbonate: profile and parametric-bootstrap sentinels", "",
        "## Decision", "",
        f"The predeclared sentinel screen yields **{int(counts.get('PASS', 0))} PASS, {int(counts.get('CONDITIONAL', 0))} CONDITIONAL and {int(counts.get('FAIL', 0))} FAIL** results.", "",
        "Sentinels were selected as interior medoids of the petrophysical design, with replicate zero fixed before fitting. Realized error, coverage and convergence were not used for selection. Porosity is the sole primary endpoint; saturation and aspect ratio are stress endpoints; secondary porosity is the negative control.", "",
        "The primary profile uses only whitened elastic and electrical data residuals. It excludes the weak nuisance centering and the partial/independent coupling penalties. Cross-fitted predictive weights remain fixed only for the operational model-average estimator and are not interpreted as model probabilities.", "",
        "## Sentinel results", "",
        "| target | role | profile family | Phase-1.2A c90 | bootstrap c90 | Wald coverage | bootstrap-profile coverage | RMSE ratio (95% upper) | width ratio (95% upper) | bound/disconnected | verdict |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.role} | {row.profile_family}/free-q | {row.phase12a_c90:.3f} | {row.bootstrap_required_c90:.3f} | {row.calibrated_wald_coverage90:.3f} | {row.bootstrap_profile_coverage90:.3f} | {row.rmse_ratio_joint_over_elastic:.3f} ({row.rmse_ratio_high:.3f}) | {row.median_width_ratio:.3f} ({row.width_ratio_high:.3f}) | {'yes' if (row.profile_bound_limited or row.profile_disconnected) else 'no'} | {row.verdict} |"
        )
    lines += ["", "## Non-Gaussian interval diagnostics", "",
              "| target | truth | Wilks interval | bootstrap-profile interval | bootstrap basic interval | negative-LR rate | fit success |", "| --- | ---: | --- | --- | --- | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {TARGET_LABELS[row.target]} | {row.truth:.4f} | [{row.wilks_interval_lower:.4f}, {row.wilks_interval_upper:.4f}] | [{row.bootstrap_profile_lower:.4f}, {row.bootstrap_profile_upper:.4f}] | [{row.bootstrap_basic_lower:.4f}, {row.bootstrap_basic_upper:.4f}] | {row.negative_lr_fraction:.3f} | {row.fit_success_fraction:.3f} |"
        )
    lines += ["", "## Interpretation boundary", "",
              "- The bootstrap is conditional on one representative geological panel per endpoint. It tests repeated noise, nonlinearity and bounds; it does not establish across-panel geological generalization.",
              "- The bootstrap split is frozen: the first half calibrates the profile threshold and the second half evaluates coverage. Phase-1.2A multipliers are never relearned from these simulations.",
              "- A bounded or disconnected profile blocks GO even if point RMSE improves.",
              "- A negative-control FAIL is the scientifically expected result; a sudden PASS for secondary porosity would indicate model locking.",
              "- All truths remain in the candidate library. Off-library hybrid laws are the next falsification stage before field logs.",
              "- With 199 replicates per sentinel this is a gate-level screen. The aspect-ratio endpoint remains too close to the RMSE/width boundary for a positive paper-level claim; expansion is required only if that endpoint is pursued.",
              "", "## Reproduce", "", "```bash",
              "python scripts/run_phase12b_sentinels.py --bootstrap 199 --workers 5 --profile-points 41", "```", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase12a", type=Path, default=ROOT / "outputs/phase12_calibration/calibrated_recoveries_global.csv")
    parser.add_argument("--components", type=Path, default=ROOT / "outputs/phase11_carbonate/fit_components.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase12b_sentinels")
    parser.add_argument("--bootstrap", type=int, default=199)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--profile-points", type=int, default=41)
    parser.add_argument("--panels", type=int, default=12)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 39:
        raise ValueError("At least 39 bootstrap replicates are required")
    args.output.mkdir(parents=True, exist_ok=True)
    calibrated = pd.read_csv(args.phase12a)
    components = pd.read_csv(args.components)
    design = pd.DataFrame(phase11_design(args.panels, args.seed))
    design.insert(0, "panel", np.arange(len(design)))
    sentinels = select_sentinels(calibrated, design)
    for index, row in sentinels.iterrows():
        elastic = calibrated[
            calibrated.case.eq(row.case) & calibrated.target.eq(row.target)
            & calibrated.strategy.eq("elastic") & calibrated.panel.eq(row.panel)
            & calibrated.replicate.eq(row.replicate)
        ].iloc[0]
        sentinels.loc[index, "elastic_c90"] = float(elastic.c90)
    sentinels.to_csv(args.output / "sentinel_selection.csv", index=False)

    base_tasks = []
    for sentinel_index, sentinel in sentinels.iterrows():
        active = active_component_weights(components, sentinel).to_dict("records")
        base_tasks.append({
            "sentinel": sentinel.to_dict(), "sentinel_index": int(sentinel_index),
            "active": active, "seed": args.seed, "panels": args.panels,
            "panel_size": args.panel_size, "profile_points": args.profile_points,
            "calibration_count": args.bootstrap // 2,
        })

    if args.reuse_existing:
        profile_raw = pd.read_csv(args.output / "profile_raw.csv")
        bootstrap_raw = pd.read_csv(args.output / "bootstrap_raw.csv")
        expected = len(sentinels) * args.bootstrap
        if len(bootstrap_raw) != expected:
            raise ValueError(f"Existing bootstrap has {len(bootstrap_raw)} rows; expected {expected}")
    else:
        profile_rows, profile_base = [], []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(base_tasks))) as pool:
            futures = {pool.submit(_profile_worker, task): task["sentinel"]["sentinel"] for task in base_tasks}
            for future in as_completed(futures):
                rows, base = future.result()
                profile_rows.extend(rows); profile_base.append(base)
                print(f"Completed profile: {futures[future]}", flush=True)
        profile_raw = pd.DataFrame(profile_rows).sort_values(["sentinel", "profile_value"])
        profile_raw.to_csv(args.output / "profile_raw.csv", index=False)
        pd.DataFrame(profile_base).sort_values("sentinel").to_csv(args.output / "profile_base_fits.csv", index=False)

        bootstrap_tasks = []
        for task in base_tasks:
            for bootstrap_index in range(args.bootstrap):
                current = dict(task); current["bootstrap_index"] = bootstrap_index
                bootstrap_tasks.append(current)
        bootstrap_rows = []
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_bootstrap_worker, task): (task["sentinel"]["sentinel"], task["bootstrap_index"]) for task in bootstrap_tasks}
            for future in as_completed(futures):
                bootstrap_rows.append(future.result()); completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    checkpoint = pd.DataFrame(bootstrap_rows).sort_values(["sentinel", "bootstrap_index"])
                    checkpoint.to_csv(args.output / "bootstrap_checkpoint.csv", index=False)
                    print(f"Completed bootstrap fits: {completed}/{len(futures)}", flush=True)
        bootstrap_raw = pd.DataFrame(bootstrap_rows).sort_values(["sentinel", "bootstrap_index"])
        bootstrap_raw.to_csv(args.output / "bootstrap_raw.csv", index=False)
    summary = summarize(profile_raw, bootstrap_raw, sentinels)
    summary.to_csv(args.output / "sentinel_summary.csv", index=False)
    plot_profiles(profile_raw, summary, args.output / "profile_likelihoods.png")
    plot_coverages(summary, args.output / "sentinel_coverages.png")
    write_report(summary, args.output / "PHASE12B_SENTINEL_REPORT.md")

    artifact_paths = sorted(path for path in args.output.iterdir() if path.name != "manifest.json")
    source_paths = [
        ROOT / "src/jointinv/phase12b.py", ROOT / "scripts/run_phase12b_sentinels.py",
        ROOT / "tests/test_phase12b.py",
    ]
    manifest = {
        "phase": "1.2B", "bootstrap_replicates_per_sentinel": args.bootstrap,
        "bootstrap_calibration_count": args.bootstrap // 2,
        "bootstrap_validation_count": args.bootstrap - args.bootstrap // 2,
        "profile_points_nominal": args.profile_points, "seed": args.seed,
        "sentinels": sentinels[["sentinel", "case", "target", "panel", "replicate"]].to_dict("records"),
        "input_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in [args.phase12a, args.components]},
        "artifacts": {path.name: _sha256(path) for path in artifact_paths},
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary[["sentinel", "verdict", "control_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
