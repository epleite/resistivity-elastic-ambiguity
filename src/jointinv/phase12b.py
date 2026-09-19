"""Phase 1.2B: non-Gaussian confirmation on predeclared sentinel cases.

The stage combines a weighted profile-likelihood diagnostic with a conditional
repeated-noise parametric bootstrap.  It is intentionally a sentinel test: it
checks interval geometry at representative observations and does not replace
the panel-level calibration or the future off-library falsification stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

from .analysis import DATA_SIGMA
from .phase05 import _panel_states
from .phase1 import CARBONATE, NUISANCE, _rt_cholesky
from .phase11 import (
    PARTIAL_TAU,
    TARGETS,
    _bounds,
    _candidate_names,
    _candidate_vectors,
    _range,
    _unpack,
    elastic_latent,
)


@dataclass(frozen=True)
class SentinelSpec:
    name: str
    case: str
    target: str
    role: str


SENTINEL_SPECS = (
    SentinelSpec("phi_positive", "network_rho05", "phi", "positive_candidate"),
    SentinelSpec("sw_stress", "ema_rho05", "sw", "calibration_stress"),
    SentinelSpec("aspect_ambiguous", "ema_rho05", "aspect", "ambiguity_stress"),
    SentinelSpec("secondary_negative", "ema_rho05", "secondary", "negative_control"),
)


def select_sentinels(
    calibrated: pd.DataFrame,
    design: pd.DataFrame,
    specs: Sequence[SentinelSpec] = SENTINEL_SPECS,
) -> pd.DataFrame:
    """Select an interior design medoid without looking at realized error."""
    if "panel" not in design:
        raise ValueError("Design table must contain panel identifiers")
    feature_names = [
        name for name in (
            "phi", "sw", "aspect", "secondary", "vcl", "cement", "coord",
            "log_rw", "archie_m", "archie_n", "surface_cond", "invasion",
            "phi_e", "q_elastic_design", "q_perp",
        ) if name in design
    ]
    standardized = design[feature_names].astype(float).copy()
    scales = standardized.std(ddof=0).replace(0.0, 1.0)
    standardized = (standardized - standardized.mean()) / scales
    design_distance = np.sqrt(np.mean(standardized**2, axis=1))
    rows = []
    for spec in specs:
        lower, upper = _range(spec.target)
        target_position = (
            (design[spec.target].astype(float) - lower) / (upper - lower)
        )
        eligible = target_position.between(0.15, 0.85)
        if not eligible.any():
            eligible = pd.Series(True, index=design.index)
        candidates = design.loc[eligible, ["panel"]].copy()
        candidates["design_distance"] = design_distance[eligible].to_numpy()
        chosen = candidates.sort_values(
            ["design_distance", "panel"], kind="stable"
        ).iloc[0]
        panel = int(chosen["panel"])
        replicate = 0
        group = calibrated[
            calibrated["case"].eq(spec.case)
            & calibrated["target"].eq(spec.target)
            & calibrated["strategy"].eq("model_average")
            & calibrated["panel"].eq(panel)
            & calibrated["replicate"].eq(replicate)
        ].copy()
        if len(group) != 1:
            raise ValueError(f"No rows for sentinel {spec.name}")
        row = group.iloc[0]
        rows.append({
            "sentinel": spec.name,
            "role": spec.role,
            "case": spec.case,
            "target": spec.target,
            "panel": panel,
            "replicate": replicate,
            "truth": float(row["truth"]),
            "phase12a_estimate": float(row["estimate"]),
            "phase12a_sd": float(row["sd"]),
            "phase12a_c90": float(row["c90"]),
            "design_distance": float(chosen["design_distance"]),
            "target_boundary_fraction": float(
                min(target_position.loc[design["panel"].eq(panel)].iloc[0],
                    1.0 - target_position.loc[design["panel"].eq(panel)].iloc[0])
            ),
            "selection_rule": "interior standardized design medoid; replicate fixed at zero",
        })
    return pd.DataFrame(rows)


def active_component_weights(
    components: pd.DataFrame, sentinel: Mapping, threshold: float = 1e-6,
) -> pd.DataFrame:
    """Return predictively supported candidates for a sentinel observation."""
    selected = components[
        components["case"].eq(sentinel["case"])
        & components["panel"].eq(int(sentinel["panel"]))
        & components["replicate"].eq(int(sentinel["replicate"]))
        & components["success"].astype(bool)
        & components["full_weight"].gt(threshold)
    ][["family", "coupling", "full_weight"]].copy()
    if selected.empty:
        raise ValueError(f"No supported candidates for {sentinel['sentinel']}")
    selected["full_weight"] /= selected["full_weight"].sum()
    return selected.sort_values(["family", "coupling"]).reset_index(drop=True)


def _profile_state(
    z: np.ndarray,
    free_names: Sequence[str],
    fixed: Mapping[str, float],
    target: str,
    value: float,
) -> tuple[dict[str, float], float]:
    lo, hi = _bounds(free_names)
    p = dict(fixed)
    p.update({
        name: float(parameter)
        for name, parameter in zip(free_names, lo + np.asarray(z) * (hi - lo))
    })
    feasibility = 0.0
    if target == "secondary":
        cap = min(0.12, 0.72 * float(p["phi"]))
        p["secondary"] = float(value)
        p["secondary_fraction"] = float(np.clip(value / max(cap, 1e-12), 0.0, 1.0))
        feasibility = max(value - cap, 0.0) / 0.002
    else:
        p[target] = float(value)
        if "secondary_fraction" in p:
            p["secondary"] = float(
                p["secondary_fraction"] * min(0.12, 0.72 * p["phi"])
            )
    return p, feasibility


def candidate_profile_residual(
    z: np.ndarray,
    free_names: Sequence[str],
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    fixed: Mapping[str, float],
    family: str,
    coupling: str,
    panel_size: int,
    target: str,
    value: float,
    objective_mode: str = "data",
) -> np.ndarray:
    """Candidate residual with one physical target held fixed.

    ``data`` is the genuine whitened-data likelihood. ``structural`` adds the
    coupling law. ``phase11`` additionally includes the weak numerical nuisance
    centering used by the operational Phase-1.1 estimator.
    """
    if objective_mode not in {"data", "structural", "phase11"}:
        raise ValueError(f"Unknown objective mode: {objective_mode}")
    p, feasibility = _profile_state(z, free_names, fixed, target, value)
    pred_e, pred_r = _candidate_vectors(p, family, coupling, panel_size)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    rt_whitener = np.linalg.inv(_rt_cholesky())
    pieces = [((pred_e - observed_elastic) / sigma_e).ravel()]
    pieces.append(np.hstack([
        rt_whitener @ row for row in pred_r - observed_log_rt
    ]))
    candidate_names = _candidate_names(family, coupling)
    if objective_mode == "phase11":
        nuisance_values = []
        for name in NUISANCE:
            if name in candidate_names:
                lo, hi = _range(name)
                normalized = (p[name] - lo) / (hi - lo)
                nuisance_values.append(0.10 * (normalized - 0.5))
        if nuisance_values:
            pieces.append(np.asarray(nuisance_values))
    if objective_mode in {"structural", "phase11"}:
        if coupling == "partial":
            pieces.append(np.array([
                (p["q_conn"] - elastic_latent(p["aspect"])) / PARTIAL_TAU
            ]))
        elif coupling == "independent":
            pieces.append(np.array([p["q_conn"]]))
    if target == "secondary":
        pieces.append(np.array([feasibility]))
    return np.concatenate(pieces)


def candidate_penalized_objective(
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    state: Mapping[str, float],
    family: str,
    coupling: str,
    panel_size: int,
    objective_mode: str = "phase11",
) -> float:
    """Evaluate an unrestricted candidate objective at a physical state."""
    if objective_mode not in {"data", "structural", "phase11"}:
        raise ValueError(f"Unknown objective mode: {objective_mode}")
    pred_e, pred_r = _candidate_vectors(state, family, coupling, panel_size)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    rt_whitener = np.linalg.inv(_rt_cholesky())
    pieces = [((pred_e - observed_elastic) / sigma_e).ravel()]
    pieces.append(np.hstack([
        rt_whitener @ row for row in pred_r - observed_log_rt
    ]))
    if objective_mode == "phase11":
        nuisance_values = []
        for name in NUISANCE:
            lo, hi = _range(name)
            normalized = (state[name] - lo) / (hi - lo)
            nuisance_values.append(0.10 * (normalized - 0.5))
        pieces.append(np.asarray(nuisance_values))
    if objective_mode in {"structural", "phase11"}:
        if coupling == "partial":
            pieces.append(np.array([
                (state["q_conn"] - elastic_latent(state["aspect"])) / PARTIAL_TAU
            ]))
        elif coupling == "independent":
            pieces.append(np.array([state["q_conn"]]))
    residual = np.concatenate(pieces)
    return float(np.dot(residual, residual))


def fit_profile_point(
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    fixed: Mapping[str, float],
    family: str,
    coupling: str,
    panel_size: int,
    target: str,
    value: float,
    initial_state: Mapping[str, float],
    objective_mode: str = "data",
) -> dict:
    """Minimize the penalized objective conditional on one target value."""
    candidate_names = list(_candidate_names(family, coupling))
    held_name = "secondary_fraction" if target == "secondary" else target
    if held_name not in candidate_names:
        raise ValueError(f"Target {target} is unavailable for {family}/{coupling}")
    free_names = tuple(name for name in candidate_names if name != held_name)
    lo, hi = _bounds(free_names)
    start_values = np.array([
        initial_state.get(name, 0.5 * (lower + upper))
        for name, lower, upper in zip(free_names, lo, hi)
    ])
    start = np.clip((start_values - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)

    def residual(z: np.ndarray) -> np.ndarray:
        return candidate_profile_residual(
            z, free_names, observed_elastic, observed_log_rt, fixed,
            family, coupling, panel_size, target, value, objective_mode,
        )

    attempts = []
    starts = [start, np.where(np.arange(len(free_names)) % 2, 0.65, 0.35)]
    for candidate_start in starts:
        attempts.append(least_squares(
            residual, candidate_start,
            bounds=(np.zeros(len(free_names)), np.ones(len(free_names))),
            method="trf", x_scale="jac", max_nfev=120,
            ftol=2e-7, xtol=2e-7, gtol=2e-7,
        ))
    fit = min(attempts, key=lambda result: float(np.dot(result.fun, result.fun)))
    if not fit.success:
        fallback = least_squares(
            residual, fit.x,
            bounds=(np.zeros(len(free_names)), np.ones(len(free_names))),
            method="trf", x_scale="jac", max_nfev=3000,
            ftol=2e-7, xtol=2e-7, gtol=2e-7,
        )
        attempts.append(fallback)
        if fallback.success or np.dot(fallback.fun, fallback.fun) < np.dot(fit.fun, fit.fun):
            fit = fallback
    state, feasibility = _profile_state(fit.x, free_names, fixed, target, value)
    data_count = 5 * panel_size
    return {
        "state": state,
        "success": bool(fit.success),
        "objective": float(np.dot(fit.fun, fit.fun)),
        "data_objective": float(np.dot(fit.fun[:data_count], fit.fun[:data_count])),
        "feasibility_residual": float(feasibility),
        "nfev": int(fit.nfev),
        "total_nfev": int(sum(attempt.nfev for attempt in attempts)),
    }


def fit_unrestricted_candidate(
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    fixed: Mapping[str, float],
    family: str,
    coupling: str,
    panel_size: int,
    initial_state: Mapping[str, float],
    objective_mode: str = "data",
) -> dict:
    """Fit one candidate without priors when used for the primary profile."""
    names = tuple(_candidate_names(family, coupling))
    lo, hi = _bounds(names)
    start_values = np.array([
        initial_state.get(name, 0.5 * (lower + upper))
        for name, lower, upper in zip(names, lo, hi)
    ])
    start = np.clip((start_values - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)

    def residual(z: np.ndarray) -> np.ndarray:
        state = _unpack(z, names, fixed)
        state["secondary"] = float(
            state["secondary_fraction"] * min(0.12, 0.72 * state["phi"])
        )
        pred_e, pred_r = _candidate_vectors(state, family, coupling, panel_size)
        sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
        rt_whitener = np.linalg.inv(_rt_cholesky())
        pieces = [((pred_e - observed_elastic) / sigma_e).ravel()]
        pieces.append(np.hstack([
            rt_whitener @ row for row in pred_r - observed_log_rt
        ]))
        if objective_mode == "phase11":
            nuisance_values = []
            for name in NUISANCE:
                lower, upper = _range(name)
                nuisance_values.append(
                    0.10 * ((state[name] - lower) / (upper - lower) - 0.5)
                )
            pieces.append(np.asarray(nuisance_values))
        if objective_mode in {"structural", "phase11"}:
            if coupling == "partial":
                pieces.append(np.array([
                    (state["q_conn"] - elastic_latent(state["aspect"])) / PARTIAL_TAU
                ]))
            elif coupling == "independent":
                pieces.append(np.array([state["q_conn"]]))
        return np.concatenate(pieces)

    if objective_mode not in {"data", "structural", "phase11"}:
        raise ValueError(f"Unknown objective mode: {objective_mode}")
    attempts = []
    starts = [start, np.where(np.arange(len(names)) % 2, 0.65, 0.35)]
    for candidate_start in starts:
        attempts.append(least_squares(
            residual, candidate_start,
            bounds=(np.zeros(len(names)), np.ones(len(names))),
            method="trf", x_scale="jac", max_nfev=160,
            ftol=2e-7, xtol=2e-7, gtol=2e-7,
        ))
    fit = min(attempts, key=lambda result: float(np.dot(result.fun, result.fun)))
    if not fit.success:
        fallback = least_squares(
            residual, fit.x, bounds=(np.zeros(len(names)), np.ones(len(names))),
            method="trf", x_scale="jac", max_nfev=3000,
            ftol=2e-7, xtol=2e-7, gtol=2e-7,
        )
        attempts.append(fallback)
        if fallback.success or np.dot(fallback.fun, fallback.fun) < np.dot(fit.fun, fit.fun):
            fit = fallback
    state = _unpack(fit.x, names, fixed)
    state["secondary"] = float(
        state["secondary_fraction"] * min(0.12, 0.72 * state["phi"])
    )
    return {
        "state": state,
        "success": bool(fit.success),
        "objective": float(np.dot(fit.fun, fit.fun)),
        "data_objective": float(np.dot(fit.fun[:5 * panel_size], fit.fun[:5 * panel_size])),
        "nfev": int(fit.nfev),
        "total_nfev": int(sum(attempt.nfev for attempt in attempts)),
        "bound_hit": {
            name: bool(value) for name, value in zip(
                names, (fit.x <= 1e-4) | (fit.x >= 1.0 - 1e-4)
            )
        },
    }


def connected_profile_interval(
    grid: Sequence[float], delta_chi2: Sequence[float],
    level: float = 0.90, threshold: float | None = None,
) -> dict[str, float | bool]:
    """Threshold the connected likelihood region containing the optimum."""
    x = np.asarray(grid, dtype=float)
    delta = np.asarray(delta_chi2, dtype=float)
    if x.ndim != 1 or delta.shape != x.shape or len(x) < 3:
        raise ValueError("Profile grid and delta must be aligned one-dimensional arrays")
    order = np.argsort(x)
    x, delta = x[order], delta[order]
    threshold = float(chi2.ppf(level, 1) if threshold is None else threshold)
    minimum = int(np.nanargmin(delta))
    inside = np.isfinite(delta) & (delta <= threshold)
    runs = int(np.sum(inside & np.r_[True, ~inside[:-1]]))
    if not inside[minimum]:
        raise ValueError("Profile minimum is outside its own confidence set")
    left = minimum
    while left > 0 and inside[left - 1]:
        left -= 1
    right = minimum
    while right < len(x) - 1 and inside[right + 1]:
        right += 1

    def crossing(outside_index: int, inside_index: int) -> float:
        x0, x1 = x[outside_index], x[inside_index]
        y0, y1 = delta[outside_index], delta[inside_index]
        if not np.isfinite(y0) or abs(y1 - y0) < 1e-14:
            return float(x1)
        return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    lower_limited = left == 0
    upper_limited = right == len(x) - 1
    lower = float(x[0]) if lower_limited else crossing(left - 1, left)
    upper = float(x[-1]) if upper_limited else crossing(right + 1, right)
    return {
        "lower": lower,
        "upper": upper,
        "threshold": threshold,
        "minimum": float(x[minimum]),
        "lower_bound_limited": lower_limited,
        "upper_bound_limited": upper_limited,
        "disconnected": runs > 1,
    }


def weighted_profile(profile_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Combine candidate relative likelihoods using frozen predictive weights."""
    data = profile_rows.copy()
    data["candidate_minimum"] = data.groupby(
        ["family", "coupling"]
    )["objective"].transform("min")
    data["relative_likelihood"] = np.exp(
        -0.5 * np.clip(data["objective"] - data["candidate_minimum"], 0.0, 1400.0)
    )
    combined = (
        data.assign(weighted=lambda frame: frame["weight"] * frame["relative_likelihood"])
        .groupby("profile_value", as_index=False)
        .agg(weighted_likelihood=("weighted", "sum"), all_success=("success", "all"))
        .sort_values("profile_value")
    )
    combined["weighted_likelihood"] /= combined["weighted_likelihood"].max()
    combined["delta_chi2"] = -2.0 * np.log(
        combined["weighted_likelihood"].clip(lower=1e-300)
    )
    interval = connected_profile_interval(
        combined["profile_value"], combined["delta_chi2"], 0.90
    )
    return combined, interval


def weighted_profile_statistic(
    unrestricted_objectives: Sequence[float],
    conditioned_objectives: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Likelihood-ratio statistic for a fixed target under a model mixture."""
    unrestricted = np.asarray(unrestricted_objectives, dtype=float)
    conditioned = np.asarray(conditioned_objectives, dtype=float)
    model_weights = np.asarray(weights, dtype=float)
    if not (unrestricted.shape == conditioned.shape == model_weights.shape):
        raise ValueError("Objectives and weights must have identical shapes")
    model_weights = model_weights / model_weights.sum()
    delta = np.maximum(conditioned - unrestricted, 0.0)
    ratio = float(np.dot(model_weights, np.exp(-0.5 * np.clip(delta, 0.0, 1400.0))))
    return float(-2.0 * np.log(max(ratio, 1e-300)))


def wilson_interval(successes: int, total: int, z: float = 1.644854) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z / denominator * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)
    )
    return float(center - half), float(center + half)


def paired_rmse_ratio_interval(
    joint_error: Sequence[float], elastic_error: Sequence[float],
    seed: int, n_bootstrap: int = 5000,
) -> tuple[float, float, float]:
    joint = np.asarray(joint_error, dtype=float)
    elastic = np.asarray(elastic_error, dtype=float)
    if joint.shape != elastic.shape or joint.size == 0:
        raise ValueError("Paired nonempty error arrays are required")
    ratio = float(np.sqrt(np.mean(joint**2)) / max(np.sqrt(np.mean(elastic**2)), 1e-12))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(joint), size=(n_bootstrap, len(joint)))
    joint_rmse = np.sqrt(np.mean(joint[indices] ** 2, axis=1))
    elastic_rmse = np.sqrt(np.mean(elastic[indices] ** 2, axis=1))
    draws = joint_rmse / np.maximum(elastic_rmse, 1e-12)
    return ratio, float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))
