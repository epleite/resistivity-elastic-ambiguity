"""Phase 1.2A: panel-cross-fitted calibration of Phase 1.1 intervals.

This module deliberately reuses the Phase 1.1 recovery bank.  It asks whether
the local-curvature standard deviations can be repaired by a multiplicative
calibration learned without access to the held-out geological panel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .phase11 import _range, _stable_seed


CALIBRATION_SCOPES = {
    "global": ("target", "strategy"),
    "case": ("case", "target", "strategy"),
}


def conformal_order_statistic(scores: Sequence[float], coverage: float = 0.90) -> float:
    """Return the finite-sample split-conformal order statistic.

    The ``ceil((n + 1) * coverage)`` rank is used and clipped to the largest
    observed score when the nominal rank is beyond the calibration sample.
    """
    values = np.sort(np.asarray(scores, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Calibration scores are empty")
    rank = min(values.size, int(np.ceil((values.size + 1) * coverage)))
    return float(values[rank - 1])


def crossfit_intervals(
    raw: pd.DataFrame, scope: str = "global", coverage: float = 0.90,
) -> pd.DataFrame:
    """Calibrate symmetric intervals while holding out the complete panel.

    ``global`` pools truth regimes and is the transferable primary analysis.
    ``case`` learns within the known synthetic regime and is retained only as
    an optimistic matched-scenario sensitivity analysis.
    """
    if scope not in CALIBRATION_SCOPES:
        raise ValueError(f"Unknown calibration scope: {scope}")
    required = {"case", "panel", "strategy", "target", "truth", "estimate", "sd"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = raw.copy()
    for name in ("lower90", "upper90", "interval_width90", "covered90"):
        if name in result:
            result[f"original_{name}"] = result[name]
    result["standardized_absolute_error"] = (
        (result["estimate"] - result["truth"]).abs()
        / result["sd"].clip(lower=1e-12)
    )
    keys = CALIBRATION_SCOPES[scope]
    factors = np.empty(len(result), dtype=float)
    for panel in sorted(result["panel"].unique()):
        held = result["panel"].eq(panel)
        training = result.loc[~held]
        factor_lookup = {
            group_key: conformal_order_statistic(group["standardized_absolute_error"], coverage)
            for group_key, group in training.groupby(list(keys), sort=False)
        }
        held_positions = np.flatnonzero(held.to_numpy())
        for position in held_positions:
            row = result.iloc[position]
            group_key = tuple(row[key] for key in keys)
            if len(keys) == 1:
                group_key = group_key[0]
            factors[position] = factor_lookup[group_key]

    result["calibration_scope"] = scope
    result["c90"] = factors
    lower, upper = [], []
    for row in result.itertuples(index=False):
        physical_lower, physical_upper = _range(row.target)
        lower.append(max(physical_lower, row.estimate - row.c90 * row.sd))
        upper.append(min(physical_upper, row.estimate + row.c90 * row.sd))
    result["lower90"] = lower
    result["upper90"] = upper
    result["interval_width90"] = result["upper90"] - result["lower90"]
    result["covered90"] = (
        (result["truth"] >= result["lower90"])
        & (result["truth"] <= result["upper90"])
    )
    return result


def _cluster_interval(
    values: pd.Series, panels: pd.Series, seed_parts: tuple[str, ...],
    n_bootstrap: int = 4000,
) -> tuple[float, float]:
    panel_ids = np.array(sorted(panels.unique()))
    means = np.array([
        float(values.loc[panels.eq(panel)].mean()) for panel in panel_ids
    ])
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    draws = rng.integers(0, len(means), size=(n_bootstrap, len(means)))
    sampled = means[draws].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def calibration_summary(calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target, strategy), group in calibrated.groupby(
        ["case", "target", "strategy"], sort=True
    ):
        low, high = _cluster_interval(
            group["covered90"].astype(float), group["panel"],
            (case, target, strategy, "calibrated_coverage"),
        )
        rows.append({
            "case": case,
            "target": target,
            "strategy": strategy,
            "n": len(group),
            "original_coverage90": float(group["original_covered90"].mean()),
            "calibrated_coverage90": float(group["covered90"].mean()),
            "coverage90_cluster_low": low,
            "coverage90_cluster_high": high,
            "median_c90": float(group["c90"].median()),
            "median_interval_width": float(group["interval_width90"].median()),
            "rmse": float(np.sqrt(np.mean((group["estimate"] - group["truth"]) ** 2))),
        })
    return pd.DataFrame(rows)


def build_phase12_gates(
    calibrated: pd.DataFrame, strategy: str = "model_average",
    n_bootstrap: int = 4000,
) -> pd.DataFrame:
    """Apply utility gates to calibrated joint intervals versus calibrated elastic.

    PASS requires honest uncertainty *and* point-recovery gain without intervals
    wider than the equally calibrated elastic baseline.
    """
    rows = []
    selected = calibrated[calibrated["strategy"].isin(["elastic", strategy])]
    for (case, target), group in selected.groupby(["case", "target"], sort=True):
        paired = group.pivot(
            index=["panel", "replicate"], columns="strategy",
            values=["truth", "estimate", "interval_width90", "covered90", "weighted_reduced_chi2", "success_weight"],
        )
        joint_error = paired[("estimate", strategy)] - paired[("truth", strategy)]
        elastic_error = paired[("estimate", "elastic")] - paired[("truth", "elastic")]
        rmse_ratio = float(
            np.sqrt(np.mean(joint_error**2))
            / max(np.sqrt(np.mean(elastic_error**2)), 1e-12)
        )
        width_ratio_values = (
            paired[("interval_width90", strategy)]
            / paired[("interval_width90", "elastic")]
        )
        width_ratio = float(np.median(width_ratio_values))
        joint_covered = paired[("covered90", strategy)].astype(bool)
        elastic_covered = paired[("covered90", "elastic")].astype(bool)
        narrower = width_ratio_values <= 0.80
        false_confidence = narrower & ~joint_covered
        harmful_false = false_confidence & elastic_covered

        panels = np.array(sorted(paired.index.get_level_values("panel").unique()))
        rng = np.random.default_rng(_stable_seed(case, target, strategy, "phase12"))
        ratio_draws, false_draws, harmful_draws, coverage_draws = [], [], [], []
        panel_index = paired.index.get_level_values("panel").to_numpy()
        for _ in range(n_bootstrap):
            sampled_panels = rng.choice(panels, size=len(panels), replace=True)
            indices = np.concatenate([
                np.flatnonzero(panel_index == panel) for panel in sampled_panels
            ])
            ratio_draws.append(
                np.sqrt(np.mean(joint_error.to_numpy()[indices] ** 2))
                / max(np.sqrt(np.mean(elastic_error.to_numpy()[indices] ** 2)), 1e-12)
            )
            false_draws.append(float(false_confidence.to_numpy()[indices].mean()))
            harmful_draws.append(float(harmful_false.to_numpy()[indices].mean()))
            coverage_draws.append(float(joint_covered.to_numpy()[indices].mean()))

        coverage = float(joint_covered.mean())
        coverage_low = float(np.quantile(coverage_draws, 0.025))
        coverage_high = float(np.quantile(coverage_draws, 0.975))
        ratio_upper = float(np.quantile(ratio_draws, 0.975))
        false_upper = float(np.quantile(false_draws, 0.975))
        harmful_upper = float(np.quantile(harmful_draws, 0.975))
        joint_rows = group[group["strategy"].eq(strategy)]
        truth_by_panel = joint_rows.groupby("panel")["truth"].first()
        standardized_bias = abs(float(joint_error.mean())) / max(
            float(truth_by_panel.std(ddof=0)), 1e-12
        )
        reduced_chi2 = float(joint_rows["weighted_reduced_chi2"].median())
        success = float(joint_rows["success_weight"].mean())

        if (
            0.85 <= coverage <= 0.97 and coverage_low >= 0.80
            and rmse_ratio <= 0.90 and ratio_upper < 1.0
            and false_upper <= 0.10 and harmful_upper <= 0.10
            and width_ratio <= 1.00 and standardized_bias <= 0.10
            and reduced_chi2 <= 1.50 and success >= 0.95
        ):
            verdict = "PASS"
        elif (
            coverage >= 0.80 and coverage_low >= 0.70
            and rmse_ratio <= 1.05 and false_upper <= 0.20
            and width_ratio <= 1.10 and standardized_bias <= 0.20
            and reduced_chi2 <= 2.00
        ):
            verdict = "CONDITIONAL"
        else:
            verdict = "FAIL"
        rows.append({
            "case": case,
            "target": target,
            "strategy": strategy,
            "verdict": verdict,
            "coverage90": coverage,
            "coverage90_cluster_low": coverage_low,
            "coverage90_cluster_high": coverage_high,
            "rmse_ratio_over_elastic": rmse_ratio,
            "rmse_ratio_cluster_upper": ratio_upper,
            "false_confidence_fraction": float(false_confidence.mean()),
            "false_confidence_cluster_upper": false_upper,
            "harmful_false_confidence_fraction": float(harmful_false.mean()),
            "harmful_false_confidence_cluster_upper": harmful_upper,
            "standardized_bias": standardized_bias,
            "median_width_ratio": width_ratio,
            "median_reduced_chi2": reduced_chi2,
            "success_fraction": success,
        })
    return pd.DataFrame(rows)

