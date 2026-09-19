"""Phase 1.3: prospective selective inversion with fail-closed abstention.

The confirmatory policy is deliberately simple.  Its reduced-chi-square
threshold is the 90% conformal order statistic of the Phase-1.1 matched-library
bank.  A joint porosity result is released only when that diagnostic, the
porosity-specific boundary mass and numerical-health checks all pass.  No
Phase-1.2C or Phase-1.3 truth, scenario label or realised error enters the
policy.

Phase 1.3 evaluates both strict abstention and an operational elastic-only
fallback on a new independent panel bank.  All uncertainty summaries resample
whole geological panels, keeping repeated-noise realisations paired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis import DATA_SIGMA
from .phase1 import TARGETS, _rt_cholesky
from .phase11 import _stable_seed, phase11_design
from .phase12 import conformal_order_statistic
from .phase12c import (
    CANDIDATE_KEYS,
    OFF_LIBRARY_CASES,
    OffLibraryCase,
    _case_lookup,
    simulate_offlibrary_observations,
)


DEPLOYABLE_FEATURES = (
    "best_reduced_chi2",
    "target_bound_mass",
    "raw_candidate_success_fraction",
    "surviving_prior_mass",
)
FORBIDDEN_POLICY_FIELDS = (
    "truth", "error", "covered90", "false_confidence", "harmful_false",
    "case", "case_label", "is_ood", "atomic", "truth_library_distance",
    "adapted_average", "q_clip_fraction", "invasion_clip_fraction",
    "rmf_clip_fraction", "mean_patch_delta",
)
THRESHOLD_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
PRIMARY_THRESHOLD_QUANTILE = 0.90


@dataclass(frozen=True)
class SelectivePolicy:
    """Immutable, deployable Phase-1.3 porosity release rule."""

    policy_id: str
    endpoint: str
    threshold_source: str
    threshold_quantile: float
    chi2_threshold: float
    phi_bound_mass_limit: float
    candidate_success_min: float
    surviving_prior_mass_min: float
    accept_when: str
    deployable_features: tuple[str, ...]
    training_observations: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["deployable_features"] = list(self.deployable_features)
        return payload

    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _validate_phase11_components(components: pd.DataFrame) -> None:
    required = {
        "case", "panel", "replicate", "family", "coupling", "success",
        "reduced_chi2",
    }
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(f"Missing Phase-1.1 component columns: {sorted(missing)}")
    keys = ["case", "panel", "replicate", "family", "coupling"]
    if components.duplicated(keys).any():
        raise ValueError("Phase-1.1 component keys must be unique")
    expected = set(CANDIDATE_KEYS)
    for _, group in components.groupby(["case", "panel", "replicate"], sort=False):
        present = set(zip(group["family"], group["coupling"]))
        if present != expected:
            raise ValueError("Every training observation must contain the nine candidates")
        active = group.loc[group["success"].astype(bool), "reduced_chi2"]
        if active.empty or not np.all(np.isfinite(active.astype(float))):
            raise ValueError("Every training observation needs a finite successful candidate")


def freeze_threshold_grid(
    training_components: pd.DataFrame,
    quantiles: Sequence[float] = THRESHOLD_QUANTILES,
) -> pd.DataFrame:
    """Freeze nested chi-square cutoffs from the Phase-1.1 bank only."""
    _validate_phase11_components(training_components)
    best = (
        training_components.loc[training_components["success"].astype(bool)]
        .groupby(["case", "panel", "replicate"], sort=True)["reduced_chi2"]
        .min()
        .astype(float)
        .to_numpy()
    )
    values = []
    last = -np.inf
    for quantile in quantiles:
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("Threshold quantiles must lie strictly between zero and one")
        cutoff = float(conformal_order_statistic(best, quantile))
        if cutoff < last:
            raise RuntimeError("Frozen chi-square thresholds are not nested")
        values.append({
            "threshold_quantile": quantile,
            "chi2_threshold": cutoff,
            "training_observations": int(len(best)),
            "threshold_source": "phase11_matched_candidate_bank",
        })
        last = cutoff
    return pd.DataFrame(values)


def freeze_phase13_policy(
    training_components: pd.DataFrame,
    threshold_quantile: float = PRIMARY_THRESHOLD_QUANTILE,
    phi_bound_mass_limit: float = 0.10,
    candidate_success_min: float = 0.99,
    surviving_prior_mass_min: float = 0.99,
) -> tuple[SelectivePolicy, pd.DataFrame]:
    """Create the confirmatory policy without accepting any test-bank input."""
    grid_quantiles = tuple(sorted(set(THRESHOLD_QUANTILES + (threshold_quantile,))))
    grid = freeze_threshold_grid(training_components, grid_quantiles)
    match = grid.loc[np.isclose(grid["threshold_quantile"], threshold_quantile)]
    if len(match) != 1:
        raise RuntimeError("Primary threshold quantile was not resolved uniquely")
    cutoff = float(match.iloc[0]["chi2_threshold"])
    contract = (
        "finite diagnostics AND best_reduced_chi2 <= chi2_threshold AND "
        "target_bound_mass <= phi_bound_mass_limit AND "
        "raw_candidate_success_fraction >= candidate_success_min AND "
        "surviving_prior_mass >= surviving_prior_mass_min"
    )
    policy = SelectivePolicy(
        policy_id="phase13_phi_q90_guard_v1",
        endpoint="phi",
        threshold_source="phase11_matched_candidate_bank",
        threshold_quantile=float(threshold_quantile),
        chi2_threshold=cutoff,
        phi_bound_mass_limit=float(phi_bound_mass_limit),
        candidate_success_min=float(candidate_success_min),
        surviving_prior_mass_min=float(surviving_prior_mass_min),
        accept_when=contract,
        deployable_features=DEPLOYABLE_FEATURES,
        training_observations=int(match.iloc[0]["training_observations"]),
    )
    return policy, grid


def _flatten_pivot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "__".join(str(part) for part in column) if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def build_phi_observation_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Pair elastic and frozen porosity results once per test observation."""
    required = {
        "case", "panel", "replicate", "strategy", "target", "truth",
        "estimate", "sd", "c90", "lower90", "upper90", "interval_width90",
        "covered90", "best_reduced_chi2", "target_bound_mass",
        "critical_bound_mass", "raw_candidate_success_fraction",
        "surviving_prior_mass", "weighted_reduced_chi2",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing recovery columns: {sorted(missing)}")
    selected = raw.loc[
        raw["target"].eq("phi")
        & raw["strategy"].isin(["elastic", "frozen_press"])
    ].copy()
    keys = ["case", "panel", "replicate", "strategy"]
    if selected.duplicated(keys).any():
        raise ValueError("Porosity strategy rows must be unique")
    counts = selected.groupby(["case", "panel", "replicate"])["strategy"].nunique()
    if not counts.eq(2).all():
        raise ValueError("Every observation needs one elastic and one frozen porosity row")
    values = [
        "truth", "estimate", "sd", "c90", "lower90", "upper90",
        "interval_width90", "covered90", "best_reduced_chi2",
        "target_bound_mass", "critical_bound_mass",
        "raw_candidate_success_fraction", "surviving_prior_mass",
        "weighted_reduced_chi2",
    ]
    paired = selected.pivot(
        index=["case", "panel", "replicate"], columns="strategy", values=values,
    )
    paired = _flatten_pivot_columns(paired).reset_index()
    for value in values:
        left = paired[f"{value}__elastic"]
        right = paired[f"{value}__frozen_press"]
        if value == "truth" and not np.allclose(
            left.astype(float), right.astype(float), rtol=0.0, atol=1e-14,
        ):
            raise ValueError("Elastic and frozen strategies disagree on truth")
    result = pd.DataFrame({
        "case": paired["case"],
        "panel": paired["panel"].astype(int),
        "replicate": paired["replicate"].astype(int),
        "truth": paired["truth__frozen_press"].astype(float),
        "joint_estimate": paired["estimate__frozen_press"].astype(float),
        "elastic_estimate": paired["estimate__elastic"].astype(float),
        "joint_sd": paired["sd__frozen_press"].astype(float),
        "elastic_sd": paired["sd__elastic"].astype(float),
        "joint_c90": paired["c90__frozen_press"].astype(float),
        "elastic_c90": paired["c90__elastic"].astype(float),
        "joint_lower90": paired["lower90__frozen_press"].astype(float),
        "joint_upper90": paired["upper90__frozen_press"].astype(float),
        "elastic_lower90": paired["lower90__elastic"].astype(float),
        "elastic_upper90": paired["upper90__elastic"].astype(float),
        "joint_width90": paired["interval_width90__frozen_press"].astype(float),
        "elastic_width90": paired["interval_width90__elastic"].astype(float),
        "joint_covered90": paired["covered90__frozen_press"].astype(bool),
        "elastic_covered90": paired["covered90__elastic"].astype(bool),
        "best_reduced_chi2": paired["best_reduced_chi2__frozen_press"].astype(float),
        "target_bound_mass": paired["target_bound_mass__frozen_press"].astype(float),
        "critical_bound_mass": paired["critical_bound_mass__frozen_press"].astype(float),
        "raw_candidate_success_fraction": paired[
            "raw_candidate_success_fraction__frozen_press"
        ].astype(float),
        "surviving_prior_mass": paired[
            "surviving_prior_mass__frozen_press"
        ].astype(float),
        "joint_weighted_reduced_chi2": paired[
            "weighted_reduced_chi2__frozen_press"
        ].astype(float),
    })
    result["joint_error"] = result["joint_estimate"] - result["truth"]
    result["elastic_error"] = result["elastic_estimate"] - result["truth"]
    result["joint_squared_error"] = result["joint_error"] ** 2
    result["elastic_squared_error"] = result["elastic_error"] ** 2
    result["excess_squared_error"] = (
        result["joint_squared_error"] - result["elastic_squared_error"]
    )
    result["width_ratio"] = result["joint_width90"] / result[
        "elastic_width90"
    ].clip(lower=1e-12)
    result["false_confidence"] = (
        result["width_ratio"].le(0.80) & ~result["joint_covered90"]
    )
    result["harmful_false_confidence"] = (
        result["false_confidence"] & result["elastic_covered90"]
    )
    result["observation_id"] = result.apply(
        lambda row: f"{row['case']}:p{int(row['panel']):03d}:r{int(row['replicate']):02d}",
        axis=1,
    )
    if result["observation_id"].duplicated().any():
        raise ValueError("Observation identifiers must be unique")
    if not np.all(np.isfinite(result.select_dtypes(include=[np.number]))):
        raise ValueError("Selective observation table contains non-finite values")
    return result.sort_values(
        ["case", "panel", "replicate"], ignore_index=True,
    )


def apply_policy(
    observations: pd.DataFrame,
    policy: SelectivePolicy,
    chi2_threshold: float | None = None,
) -> pd.DataFrame:
    """Apply a frozen policy; truth-dependent columns never affect acceptance."""
    missing = set(policy.deployable_features).difference(observations.columns)
    if missing:
        raise ValueError(f"Missing deployable policy features: {sorted(missing)}")
    threshold = policy.chi2_threshold if chi2_threshold is None else float(chi2_threshold)
    feature_values = observations.loc[:, policy.deployable_features].astype(float)
    finite = np.isfinite(feature_values).all(axis=1)
    pass_chi2 = observations["best_reduced_chi2"].astype(float).le(threshold)
    pass_bound = observations["target_bound_mass"].astype(float).le(
        policy.phi_bound_mass_limit
    )
    pass_success = observations["raw_candidate_success_fraction"].astype(float).ge(
        policy.candidate_success_min
    )
    pass_mass = observations["surviving_prior_mass"].astype(float).ge(
        policy.surviving_prior_mass_min
    )
    accepted = finite & pass_chi2 & pass_bound & pass_success & pass_mass
    reasons = []
    for index in range(len(observations)):
        failed = []
        if not bool(finite.iloc[index]):
            failed.append("nonfinite_diagnostic")
        if not bool(pass_chi2.iloc[index]):
            failed.append("chi2")
        if not bool(pass_bound.iloc[index]):
            failed.append("phi_bound")
        if not bool(pass_success.iloc[index]):
            failed.append("candidate_success")
        if not bool(pass_mass.iloc[index]):
            failed.append("surviving_prior_mass")
        reasons.append("accepted" if not failed else ";".join(failed))
    result = observations.copy()
    result["policy_id"] = policy.policy_id
    result["policy_hash"] = policy.canonical_hash()
    result["chi2_threshold"] = threshold
    result["accepted"] = accepted.to_numpy(bool)
    result["abstained"] = ~result["accepted"]
    result["abstention_reason"] = reasons
    result["system_estimate"] = np.where(
        result["accepted"], result["joint_estimate"], result["elastic_estimate"],
    )
    result["system_lower90"] = np.where(
        result["accepted"], result["joint_lower90"], result["elastic_lower90"],
    )
    result["system_upper90"] = np.where(
        result["accepted"], result["joint_upper90"], result["elastic_upper90"],
    )
    result["system_width90"] = np.where(
        result["accepted"], result["joint_width90"], result["elastic_width90"],
    )
    result["system_error"] = result["system_estimate"] - result["truth"]
    result["system_squared_error"] = result["system_error"] ** 2
    result["system_excess_squared_error"] = (
        result["system_squared_error"] - result["elastic_squared_error"]
    )
    result["system_covered90"] = (
        result["truth"].ge(result["system_lower90"])
        & result["truth"].le(result["system_upper90"])
    )
    result["system_width_ratio"] = result["system_width90"] / result[
        "elastic_width90"
    ].clip(lower=1e-12)
    result["system_false_confidence"] = (
        result["system_width_ratio"].le(0.80) & ~result["system_covered90"]
    )
    result["system_harmful_false_confidence"] = (
        result["system_false_confidence"] & result["elastic_covered90"]
    )
    return result


def _metric_snapshot(frame: pd.DataFrame) -> dict[str, float]:
    accepted = frame["accepted"].astype(bool).to_numpy()
    n_total = len(frame)
    n_accepted = int(accepted.sum())
    rejected = ~accepted
    result: dict[str, float] = {
        "n": float(n_total),
        "accepted_n": float(n_accepted),
        "retention": float(n_accepted / n_total) if n_total else np.nan,
        "abstention": float(rejected.mean()) if n_total else np.nan,
        "accepted_clusters": float(frame.loc[accepted, "panel"].nunique()),
    }
    if n_accepted:
        joint_sq = frame.loc[accepted, "joint_squared_error"].to_numpy(float)
        elastic_sq = frame.loc[accepted, "elastic_squared_error"].to_numpy(float)
        result.update({
            "accepted_coverage90": float(frame.loc[accepted, "joint_covered90"].mean()),
            "accepted_mse": float(joint_sq.mean()),
            "accepted_elastic_mse": float(elastic_sq.mean()),
            "accepted_excess_mse": float((joint_sq - elastic_sq).mean()),
            "accepted_rmse_ratio": float(
                np.sqrt(joint_sq.mean()) / max(np.sqrt(elastic_sq.mean()), 1e-12)
            ),
            "accepted_width_ratio": float(frame.loc[accepted, "width_ratio"].median()),
            "accepted_false_confidence": float(
                frame.loc[accepted, "false_confidence"].mean()
            ),
            "accepted_harmful_false_confidence": float(
                frame.loc[accepted, "harmful_false_confidence"].mean()
            ),
            "accepted_phi_bound_mass": float(
                frame.loc[accepted, "target_bound_mass"].mean()
            ),
        })
    else:
        for name in (
            "accepted_coverage90", "accepted_mse", "accepted_elastic_mse",
            "accepted_excess_mse", "accepted_rmse_ratio", "accepted_width_ratio",
            "accepted_false_confidence", "accepted_harmful_false_confidence",
            "accepted_phi_bound_mass",
        ):
            result[name] = np.nan
    result.update({
        "system_coverage90": float(frame["system_covered90"].mean()),
        "system_mse": float(frame["system_squared_error"].mean()),
        "system_excess_mse": float(frame["system_excess_squared_error"].mean()),
        "system_rmse_ratio": float(
            np.sqrt(frame["system_squared_error"].mean())
            / max(np.sqrt(frame["elastic_squared_error"].mean()), 1e-12)
        ),
        "system_width_ratio": float(frame["system_width_ratio"].median()),
        "system_false_confidence": float(frame["system_false_confidence"].mean()),
        "system_harmful_false_confidence": float(
            frame["system_harmful_false_confidence"].mean()
        ),
        "accepted_and_harmful_fraction": float(
            (frame["accepted"].astype(bool)
             & frame["harmful_false_confidence"].astype(bool)).mean()
        ),
    })
    reason = frame["abstention_reason"].astype(str)
    ood_abstain = frame["abstained"].astype(bool) & reason.str.contains(
        r"(?:^|;)chi2(?:$|;)", regex=True,
    )
    technical_abstain = frame["abstained"].astype(bool) & ~ood_abstain
    result["ood_abstention"] = float(ood_abstain.mean())
    result["technical_abstention"] = float(technical_abstain.mean())
    result["ood_share_of_abstentions"] = (
        float(ood_abstain.sum() / max(frame["abstained"].sum(), 1))
    )
    failure = frame["false_confidence"].astype(bool).to_numpy()
    harmful = frame["harmful_false_confidence"].astype(bool).to_numpy()
    uncovered = ~frame["joint_covered90"].astype(bool).to_numpy()
    for label, event in (
        ("false_confidence", failure),
        ("harmful_false", harmful),
        ("uncovered", uncovered),
    ):
        count = int(event.sum())
        captured = int((event & rejected).sum())
        fraction = captured / count if count else np.nan
        result[f"{label}_count"] = float(count)
        result[f"{label}_captured_n"] = float(captured)
        result[f"{label}_capture"] = float(fraction) if np.isfinite(fraction) else np.nan
        result[f"{label}_capture_lift"] = (
            float(fraction / max(rejected.mean(), 1e-12))
            if count and rejected.any() else np.nan
        )
    return result


def _cluster_draw_indices(
    frame: pd.DataFrame, rng: np.random.Generator,
) -> np.ndarray:
    panels = np.array(sorted(frame["panel"].unique()))
    selected = rng.choice(panels, size=len(panels), replace=True)
    panel_values = frame["panel"].to_numpy()
    return np.concatenate([np.flatnonzero(panel_values == panel) for panel in selected])


def summarize_selective_cases(
    observations: pd.DataFrame,
    n_bootstrap: int = 5000,
    seed: int = 20261301,
) -> pd.DataFrame:
    """Casewise and pooled metrics with whole-panel uncertainty intervals."""
    if n_bootstrap < 100:
        raise ValueError("At least 100 cluster bootstrap draws are required")
    scopes: list[tuple[str, pd.DataFrame]] = [
        (str(case), group.copy())
        for case, group in observations.groupby("case", sort=True)
    ]
    atomic_names = {case.name for case in OFF_LIBRARY_CASES if case.atomic}
    scopes.append(("__atomic_macro__", observations[observations["case"].isin(atomic_names)].copy()))
    scopes.append(("__all__", observations.copy()))
    rows = []
    interval_metrics = (
        "retention", "abstention", "accepted_coverage90", "accepted_excess_mse",
        "accepted_rmse_ratio", "accepted_width_ratio",
        "accepted_false_confidence", "accepted_harmful_false_confidence",
        "system_coverage90", "system_excess_mse", "system_rmse_ratio",
        "system_width_ratio", "system_false_confidence",
        "system_harmful_false_confidence", "accepted_and_harmful_fraction",
        "ood_abstention", "technical_abstention", "ood_share_of_abstentions",
    )
    for scope, frame in scopes:
        frame = frame.reset_index(drop=True)
        point = _metric_snapshot(frame)
        draws = {name: [] for name in interval_metrics}
        rng = np.random.default_rng(_stable_seed(
            scope, str(seed), "phase13_cluster",
        ))
        for _ in range(n_bootstrap):
            sampled = frame.iloc[_cluster_draw_indices(frame, rng)].reset_index(drop=True)
            snapshot = _metric_snapshot(sampled)
            for name in interval_metrics:
                value = snapshot[name]
                if np.isfinite(value):
                    draws[name].append(float(value))
        row: dict[str, object] = {"case": scope, **point}
        for name in interval_metrics:
            values = np.asarray(draws[name], dtype=float)
            row[f"{name}_cluster_low"] = (
                float(np.quantile(values, 0.025)) if values.size else np.nan
            )
            row[f"{name}_cluster_high"] = (
                float(np.quantile(values, 0.975)) if values.size else np.nan
            )
            row[f"{name}_cluster_upper95"] = (
                float(np.quantile(values, 0.95)) if values.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def random_rejection_comparator(
    observations: pd.DataFrame,
    n_permutations: int = 5000,
    seed: int = 20261302,
) -> pd.DataFrame:
    """Compare policy risk with panel-pattern permutation at equal retention."""
    if n_permutations < 100:
        raise ValueError("At least 100 random-rejection permutations are required")
    rows = []
    for case, frame in observations.groupby("case", sort=True):
        frame = frame.sort_values(["panel", "replicate"]).reset_index(drop=True)
        panels = np.array(sorted(frame["panel"].unique()))
        groups = {panel: np.flatnonzero(frame["panel"].to_numpy() == panel) for panel in panels}
        patterns = {panel: frame.loc[indices, "accepted"].to_numpy(bool) for panel, indices in groups.items()}
        lengths = {len(value) for value in patterns.values()}
        if len(lengths) != 1:
            raise ValueError("Every panel must have the same number of replicates")
        actual = frame.loc[frame["accepted"], "joint_squared_error"].to_numpy(float)
        if not len(actual) or len(actual) == len(frame):
            rows.append({
                "case": case, "accepted_n": len(actual), "n": len(frame),
                "retention": len(actual) / len(frame),
                "actual_accepted_mse": float(actual.mean()) if len(actual) else np.nan,
                "random_mse_mean": np.nan, "random_mse_low": np.nan,
                "random_mse_high": np.nan, "actual_minus_random_mean": np.nan,
                "actual_minus_random_upper95": np.nan,
                "enrichment_p_value": np.nan, "selection_enriched": False,
            })
            continue
        rng = np.random.default_rng(_stable_seed(
            str(case), str(seed), "phase13_random",
        ))
        random_mse = []
        for _ in range(n_permutations):
            permuted = rng.permutation(panels)
            mask = np.zeros(len(frame), dtype=bool)
            for outcome_panel, pattern_panel in zip(panels, permuted):
                mask[groups[outcome_panel]] = patterns[pattern_panel]
            random_mse.append(float(frame.loc[mask, "joint_squared_error"].mean()))
        random_mse = np.asarray(random_mse)
        actual_mse = float(actual.mean())
        delta = actual_mse - random_mse
        upper = float(np.quantile(delta, 0.95))
        rows.append({
            "case": case, "accepted_n": len(actual), "n": len(frame),
            "retention": len(actual) / len(frame),
            "actual_accepted_mse": actual_mse,
            "random_mse_mean": float(random_mse.mean()),
            "random_mse_low": float(np.quantile(random_mse, 0.025)),
            "random_mse_high": float(np.quantile(random_mse, 0.975)),
            "actual_minus_random_mean": float(delta.mean()),
            "actual_minus_random_upper95": upper,
            "enrichment_p_value": float(
                (1 + np.count_nonzero(random_mse <= actual_mse))
                / (n_permutations + 1)
            ),
            "selection_enriched": bool(upper < 0.0),
        })
    return pd.DataFrame(rows)


def build_risk_coverage_curve(
    observations: pd.DataFrame,
    policy: SelectivePolicy,
    threshold_grid: pd.DataFrame,
    random_draws: int = 400,
    seed: int = 20261303,
) -> pd.DataFrame:
    """Exploratory nested curve; only the q90 point is confirmatory."""
    rows = []
    for threshold_row in threshold_grid.itertuples(index=False):
        applied = apply_policy(observations, policy, threshold_row.chi2_threshold)
        for case, frame in applied.groupby("case", sort=True):
            actual = _metric_snapshot(frame)
            k = int(actual["accepted_n"])
            joint_sq = frame["joint_squared_error"].to_numpy(float)
            oracle_mse = float(np.mean(np.sort(joint_sq)[:k])) if k else np.nan
            rng = np.random.default_rng(_stable_seed(
                str(case), str(threshold_row.threshold_quantile), str(seed),
                "curve_random",
            ))
            random_values = []
            if 0 < k < len(frame):
                for _ in range(random_draws):
                    indices = rng.choice(len(frame), size=k, replace=False)
                    random_values.append(float(joint_sq[indices].mean()))
            elif k == len(frame):
                random_values = [float(joint_sq.mean())]
            rows.append({
                "case": case,
                "threshold_quantile": float(threshold_row.threshold_quantile),
                "chi2_threshold": float(threshold_row.chi2_threshold),
                "confirmatory_operating_point": bool(np.isclose(
                    threshold_row.threshold_quantile, policy.threshold_quantile,
                )),
                "retention": actual["retention"],
                "accepted_mse": actual["accepted_mse"],
                "accepted_rmse": (
                    float(np.sqrt(actual["accepted_mse"]))
                    if np.isfinite(actual["accepted_mse"]) else np.nan
                ),
                "accepted_excess_mse": actual["accepted_excess_mse"],
                "accepted_coverage90": actual["accepted_coverage90"],
                "accepted_false_confidence": actual["accepted_false_confidence"],
                "random_mse_mean": (
                    float(np.mean(random_values)) if random_values else np.nan
                ),
                "random_mse_low": (
                    float(np.quantile(random_values, 0.025)) if random_values else np.nan
                ),
                "random_mse_high": (
                    float(np.quantile(random_values, 0.975)) if random_values else np.nan
                ),
                "oracle_mse": oracle_mse,
            })
    return pd.DataFrame(rows)


def build_phase13_gates(
    summary: pd.DataFrame,
    random_comparator: pd.DataFrame,
) -> pd.DataFrame:
    """Predeclared safety and utility classification at the frozen q90 point."""
    merged = summary.loc[~summary["case"].str.startswith("__")].merge(
        random_comparator, on="case", suffixes=("", "_random"), validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        spec = _case_lookup(row.case)
        min_retention = 0.90 if row.case == "matched_ema_control" else 0.60
        min_retention_low = 0.80 if row.case == "matched_ema_control" else 0.50
        support_ok = row.accepted_clusters >= 12 and row.retention >= min_retention
        joint_checks = {
            "accepted_support": bool(support_ok),
            "retention_low": bool(row.retention_cluster_low >= min_retention_low),
            "accepted_coverage_ge_0.85": bool(row.accepted_coverage90 >= 0.85),
            "coverage_cluster_low_ge_0.80": bool(
                row.accepted_coverage90_cluster_low >= 0.80
            ),
            "accepted_excess_mse_upper_lt_0": bool(
                row.accepted_excess_mse_cluster_upper95 < 0.0
            ),
            "accepted_width_ratio_le_0.80": bool(row.accepted_width_ratio <= 0.80),
            "width_ratio_cluster_upper_lt_1": bool(
                row.accepted_width_ratio_cluster_upper95 < 1.0
            ),
            "harmful_false_upper_le_0.10": bool(
                row.accepted_harmful_false_confidence_cluster_upper95 <= 0.10
            ),
            "phi_bound_mass_le_0.10": bool(row.accepted_phi_bound_mass <= 0.10),
        }
        fallback_checks = {
            "system_coverage_ge_0.85": bool(row.system_coverage90 >= 0.85),
            "system_coverage_low_ge_0.80": bool(
                row.system_coverage90_cluster_low >= 0.80
            ),
            "system_excess_mse_upper_le_0": bool(
                row.system_excess_mse_cluster_upper95 <= 0.0
            ),
            "system_harmful_false_upper_le_0.10": bool(
                row.system_harmful_false_confidence_cluster_upper95 <= 0.10
            ),
        }
        joint_pass = all(joint_checks.values())
        fallback_pass = all(fallback_checks.values())
        enriched = bool(row.selection_enriched)
        safe_reject_checks = {
            "abstention_ge_0.90": bool(row.abstention >= 0.90),
            "abstention_cluster_low_ge_0.80": bool(
                row.abstention_cluster_low >= 0.80
            ),
            "accepted_harmful_upper_le_0.10": bool(
                row.accepted_and_harmful_fraction_cluster_upper95 <= 0.10
            ),
            "ood_share_of_abstentions_ge_0.90": bool(
                row.ood_share_of_abstentions >= 0.90
            ),
            "fallback_coverage_safe": bool(
                row.system_coverage90 >= 0.85
                and row.system_coverage90_cluster_low >= 0.80
                and row.system_harmful_false_confidence_cluster_upper95 <= 0.10
            ),
        }
        safe_reject = all(safe_reject_checks.values())
        if joint_pass and enriched and row.retention <= 0.95:
            verdict = "SELECTIVE_PASS"
        elif joint_pass:
            verdict = "SAFE_JOINT"
        elif spec.is_ood and safe_reject:
            verdict = "SAFE_DOMAIN_REJECT"
        elif row.retention < min_retention:
            verdict = "INCONCLUSIVE_LOW_SUPPORT"
        else:
            verdict = "SILENT_FAILURE"
        rows.append({
            "case": row.case, "case_label": spec.label,
            "is_ood": spec.is_ood, "atomic": spec.atomic,
            "verdict": verdict,
            "joint_pass": joint_pass, "fallback_pass": fallback_pass,
            "selection_enriched": enriched, "safe_domain_reject": safe_reject,
            "failed_joint_criteria": ";".join(
                key for key, value in joint_checks.items() if not value
            ),
            "failed_fallback_criteria": ";".join(
                key for key, value in fallback_checks.items() if not value
            ),
            "failed_safe_reject_criteria": ";".join(
                key for key, value in safe_reject_checks.items() if not value
            ),
            **{
                key: getattr(row, key) for key in (
                    "n", "accepted_n", "retention", "retention_cluster_low",
                    "retention_cluster_high", "abstention",
                    "abstention_cluster_low", "accepted_clusters",
                    "accepted_coverage90", "accepted_coverage90_cluster_low",
                    "accepted_excess_mse", "accepted_excess_mse_cluster_upper95",
                    "accepted_rmse_ratio", "accepted_rmse_ratio_cluster_upper95",
                    "accepted_width_ratio", "accepted_width_ratio_cluster_upper95",
                    "accepted_false_confidence",
                    "accepted_false_confidence_cluster_upper95",
                    "accepted_harmful_false_confidence",
                    "accepted_harmful_false_confidence_cluster_upper95",
                    "accepted_phi_bound_mass", "false_confidence_count",
                    "false_confidence_capture", "false_confidence_capture_lift",
                    "accepted_and_harmful_fraction",
                    "accepted_and_harmful_fraction_cluster_upper95",
                    "ood_abstention", "technical_abstention",
                    "ood_share_of_abstentions",
                    "system_coverage90", "system_coverage90_cluster_low",
                    "system_excess_mse", "system_excess_mse_cluster_upper95",
                    "system_rmse_ratio", "system_rmse_ratio_cluster_upper95",
                    "system_harmful_false_confidence",
                    "system_harmful_false_confidence_cluster_upper95",
                    "actual_minus_random_mean", "actual_minus_random_upper95",
                    "enrichment_p_value",
                )
            },
        })
    return pd.DataFrame(rows)


def overall_phase13_decision(
    gates: pd.DataFrame,
    negative_control_alert_count: int = 0,
) -> pd.DataFrame:
    """Only the frozen q90 policy can determine the Phase-1.3 decision."""
    expected = {case.name for case in OFF_LIBRARY_CASES}
    if set(gates["case"]) != expected:
        raise ValueError("Overall decision requires all six Phase-1.3 cases")
    lookup = gates.set_index("case")
    matched = str(lookup.loc["matched_ema_control", "verdict"])
    combined = str(lookup.loc["combined_stress", "verdict"])
    atomic = gates.loc[gates["atomic"].astype(bool)]
    unsafe = {"SILENT_FAILURE", "INCONCLUSIVE_LOW_SUPPORT"}
    safe_matched = matched in {"SAFE_JOINT", "SELECTIVE_PASS"}
    safe_combined = combined in {"SAFE_DOMAIN_REJECT", "SAFE_JOINT", "SELECTIVE_PASS"}
    any_unsafe_atomic = atomic["verdict"].isin(unsafe).any()
    enriched_atomic = int(atomic["verdict"].eq("SELECTIVE_PASS").sum())
    if (
        safe_matched and safe_combined and not any_unsafe_atomic
        and enriched_atomic >= 2 and negative_control_alert_count == 0
    ):
        decision = "SELECTIVE_GO"
        rationale = (
            "Matched transport and every OOD mechanism were handled safely; "
            "at least two atomic stresses showed selection beyond random rejection."
        )
    elif safe_matched and safe_combined and not any_unsafe_atomic:
        decision = "CONDITIONAL"
        rationale = (
            "The frozen guard avoided silent failure, but selective enrichment "
            "or negative-control separation was not strong enough for GO."
        )
    else:
        decision = "STOP"
        rationale = (
            "The frozen guard failed matched transport, left an unsafe atomic "
            "stress, or did not safely reject the combined domain shift."
        )
    return pd.DataFrame([{
        "endpoint": "phi", "primary_policy": "phase13_phi_q90_guard_v1",
        "decision": decision, "rationale": rationale,
        "matched_verdict": matched, "combined_verdict": combined,
        "atomic_selective_pass_count": enriched_atomic,
        "atomic_safe_joint_count": int(atomic["verdict"].eq("SAFE_JOINT").sum()),
        "atomic_safe_reject_count": int(
            atomic["verdict"].eq("SAFE_DOMAIN_REJECT").sum()
        ),
        "atomic_unsafe_count": int(atomic["verdict"].isin(unsafe).sum()),
        "negative_control_alert_count": int(negative_control_alert_count),
    }])


def build_control_summary(
    raw: pd.DataFrame,
    primary_phi_observations: pd.DataFrame,
    n_bootstrap: int = 3000,
    seed: int = 20261304,
) -> pd.DataFrame:
    """Apply the phi acceptance mask to controls without tuning on them."""
    acceptance = primary_phi_observations.set_index(
        ["case", "panel", "replicate"]
    )["accepted"]
    rows = []
    for target in ("sw", "aspect", "secondary"):
        subset = raw.loc[
            raw["target"].eq(target)
            & raw["strategy"].isin(["elastic", "frozen_press"])
        ]
        pivot = subset.pivot(
            index=["case", "panel", "replicate"], columns="strategy",
            values=["truth", "estimate", "interval_width90", "covered90"],
        )
        if len(pivot) != len(acceptance) or not pivot.index.equals(acceptance.index):
            pivot = pivot.reindex(acceptance.index)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete control pairing for {target}")
        for case in sorted(subset["case"].unique()):
            index = pivot.index.get_level_values("case") == case
            local = pivot.loc[index]
            accepted = acceptance.loc[local.index].to_numpy(bool)
            panels = local.index.get_level_values("panel").to_numpy(int)
            truth = local[("truth", "frozen_press")].to_numpy(float)
            joint_error = local[("estimate", "frozen_press")].to_numpy(float) - truth
            elastic_error = local[("estimate", "elastic")].to_numpy(float) - truth
            width_ratio = (
                local[("interval_width90", "frozen_press")].to_numpy(float)
                / np.clip(local[("interval_width90", "elastic")].to_numpy(float), 1e-12, None)
            )
            covered = local[("covered90", "frozen_press")].to_numpy(bool)
            false = width_ratio <= 0.80
            harmful = false & ~covered & local[("covered90", "elastic")].to_numpy(bool)
            if not accepted.any():
                rows.append({
                    "target": target, "case": case, "n": len(local),
                    "accepted_n": 0, "retention": 0.0,
                    "accepted_excess_mse": np.nan,
                    "accepted_excess_mse_upper95": np.nan,
                    "accepted_coverage90": np.nan,
                    "accepted_coverage90_low": np.nan,
                    "accepted_width_ratio": np.nan,
                    "accepted_width_ratio_upper95": np.nan,
                    "harmful_false_upper95": np.nan,
                    "promoted_by_phi_guard": False,
                })
                continue
            point_delta = float(np.mean(joint_error[accepted] ** 2 - elastic_error[accepted] ** 2))
            point_width = float(np.median(width_ratio[accepted]))
            point_coverage = float(np.mean(covered[accepted]))
            rng = np.random.default_rng(_stable_seed(
                str(target), str(case), str(seed), "control",
            ))
            unique_panels = np.array(sorted(np.unique(panels)))
            delta_draws, width_draws, coverage_draws, harmful_draws = [], [], [], []
            for _ in range(n_bootstrap):
                selected_panels = rng.choice(
                    unique_panels, size=len(unique_panels), replace=True,
                )
                indices = np.concatenate([
                    np.flatnonzero(panels == panel) for panel in selected_panels
                ])
                keep = accepted[indices]
                if not keep.any():
                    continue
                chosen = indices[keep]
                delta_draws.append(float(np.mean(
                    joint_error[chosen] ** 2 - elastic_error[chosen] ** 2
                )))
                width_draws.append(float(np.median(width_ratio[chosen])))
                coverage_draws.append(float(np.mean(covered[chosen])))
                harmful_draws.append(float(np.mean(harmful[chosen])))
            promoted = bool(
                accepted.mean() >= 0.50
                and np.quantile(delta_draws, 0.95) < 0.0
                and np.quantile(coverage_draws, 0.025) >= 0.80
                and np.quantile(width_draws, 0.95) < 1.0
                and np.quantile(harmful_draws, 0.95) <= 0.10
            )
            rows.append({
                "target": target, "case": case, "n": len(local),
                "accepted_n": int(accepted.sum()), "retention": float(accepted.mean()),
                "accepted_excess_mse": point_delta,
                "accepted_excess_mse_upper95": float(np.quantile(delta_draws, 0.95)),
                "accepted_coverage90": point_coverage,
                "accepted_coverage90_low": float(np.quantile(coverage_draws, 0.025)),
                "accepted_width_ratio": point_width,
                "accepted_width_ratio_upper95": float(np.quantile(width_draws, 0.95)),
                "harmful_false_upper95": float(np.quantile(harmful_draws, 0.95)),
                "promoted_by_phi_guard": promoted,
            })
    return pd.DataFrame(rows)


def reconstruct_observation_bank(
    n_panels: int,
    replicates: int,
    panel_size: int,
    seed: int,
    cases: Sequence[OffLibraryCase] = OFF_LIBRARY_CASES,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Recreate and persist the exact noisy observations used by the fit bank."""
    design = phase11_design(n_panels, seed)
    arrays: dict[str, list[np.ndarray]] = {
        "elastic": [], "log_rt": [], "truth_targets": [],
    }
    rows = []
    for case in cases:
        spec = _case_lookup(case)
        for panel, truth in enumerate(design):
            for replicate in range(replicates):
                rng = np.random.default_rng(seed + 17011 * panel + 223 * replicate)
                elastic_noise = rng.normal(size=(panel_size, 3))
                rt_noise = rng.normal(size=(panel_size, 2))
                elastic, log_rt, _ = simulate_offlibrary_observations(
                    truth, spec, panel_size, elastic_noise, rt_noise,
                )
                arrays["elastic"].append(elastic)
                arrays["log_rt"].append(log_rt)
                arrays["truth_targets"].append(np.array([
                    float(truth[target]) for target in TARGETS
                ]))
                rows.append({
                    "observation_id": f"{spec.name}:p{panel:03d}:r{replicate:02d}",
                    "noise_id": f"seed{seed}:p{panel:03d}:r{replicate:02d}",
                    "case": spec.name, "panel": panel, "replicate": replicate,
                })
    bank = {name: np.stack(values) for name, values in arrays.items()}
    bank["target_names"] = np.array(TARGETS, dtype="U32")
    bank["elastic_sigma"] = np.array([
        DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"],
    ])
    bank["rt_cholesky"] = _rt_cholesky()
    index = pd.DataFrame(rows)
    return bank, index
