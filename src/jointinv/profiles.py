"""Hierarchical nonlinear profile-likelihood experiments for Phase 0.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .analysis import DATA_SIGMA, PARAM_SCALES
from .phase05 import (
    BASE_ELECTRICAL_NUISANCE,
    RANGES,
    RT_CORRELATION,
    _electrical_matrix,
    _elastic_vector,
    _nuisance_names,
    cases_for,
    targets_for,
)
from .models import SCENARIOS, Scenario


@dataclass(frozen=True)
class ProfileSpec:
    scenario: str
    x: str
    y: str
    label: str


PROFILE_SPECS = (
    ProfileSpec("clean_sandstone", "phi", "sw", "porosity–saturation"),
    ProfileSpec("clayey_sandstone", "phi", "sw", "porosity–saturation"),
    ProfileSpec("multimodal_carbonate", "secondary", "aspect", "secondary porosity–aspect ratio"),
    ProfileSpec("multimodal_carbonate", "cement", "coord", "cement–coordination negative control"),
)


def reference_state(scenario: Scenario) -> dict[str, float]:
    p = dict(scenario.truth)
    ranges = RANGES[scenario.name]
    for name, (lower, upper) in ranges.items():
        p.setdefault(name, 0.5 * (lower + upper))
        p[name] = float(np.clip(p[name], lower + 1e-8, upper - 1e-8))
    p["secondary"] = min(p["secondary"], 0.72 * p["phi"])
    p["phi_e"] = p["phi"] - 0.65 * p["secondary"]
    return p


def _rt_whiten(residual: np.ndarray) -> np.ndarray:
    sigma = DATA_SIGMA["rt_log"]
    covariance = sigma**2 * np.array([[1.0, RT_CORRELATION], [RT_CORRELATION, 1.0]])
    w = np.linalg.inv(np.linalg.cholesky(covariance))
    return np.hstack([w @ row for row in residual])


def _observed_data(p: Mapping[str, float], scenario: Scenario, case, panel_size: int):
    elastic = _elastic_vector(p, scenario, case.truth_elastic, panel_size)
    electrical = np.log(_electrical_matrix(
        p, scenario, case.truth_electrical, case.truth_coupling, panel_size
    ))
    return elastic, electrical


def _residual(
    p: Mapping[str, float], scenario: Scenario, case, panel_size: int,
    elastic_data: np.ndarray, electrical_data: np.ndarray, include_electrical: bool,
):
    pred_e = _elastic_vector(p, scenario, case.inverse_elastic, panel_size)
    sigma_e = np.tile([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]], panel_size)
    residuals = [(elastic_data - pred_e) / sigma_e]
    if include_electrical:
        pred_r = np.log(_electrical_matrix(
            p, scenario, case.inverse_electrical, case.inverse_coupling, panel_size
        ))
        residuals.append(_rt_whiten(electrical_data - pred_r))
    return np.concatenate(residuals)


def _scaled_bounds(names: Sequence[str], scenario: Scenario):
    ranges = RANGES[scenario.name]
    lower = np.array([ranges[name][0] / PARAM_SCALES[name] for name in names])
    upper = np.array([ranges[name][1] / PARAM_SCALES[name] for name in names])
    return lower, upper


def _fit(
    start_p: Mapping[str, float], names: Sequence[str], fixed: Mapping[str, float],
    scenario: Scenario, case, panel_size: int, elastic_data: np.ndarray,
    electrical_data: np.ndarray, include_electrical: bool, starts: int = 2,
):
    lower, upper = _scaled_bounds(names, scenario)

    def unpack(z):
        p = dict(start_p)
        p.update(fixed)
        for name, value in zip(names, z):
            p[name] = float(value * PARAM_SCALES[name])
        p["secondary"] = min(p.get("secondary", 0.0), 0.72 * p["phi"])
        return p

    def fun(z):
        return _residual(
            unpack(z), scenario, case, panel_size, elastic_data, electrical_data,
            include_electrical,
        )

    z_truth = np.array([start_p[name] / PARAM_SCALES[name] for name in names])
    z_mid = 0.5 * (lower + upper)
    candidates = [np.clip(z_truth, lower, upper), z_mid][:max(starts, 1)]
    best = None
    for z0 in candidates:
        fit = least_squares(
            fun, z0, bounds=(lower, upper), max_nfev=180,
            x_scale="jac", ftol=1e-7, xtol=1e-7, gtol=1e-7,
        )
        cost = float(np.dot(fit.fun, fit.fun))
        if best is None or cost < best[0]:
            best = (cost, fit, unpack(fit.x))
    return best


def compute_profile(
    spec: ProfileSpec, mode: str, grid_size: int = 15, panel_size: int = 16,
) -> tuple[pd.DataFrame, dict]:
    """Compute an unpenalized hierarchical profile on a bounded physical grid."""
    scenario = SCENARIOS[spec.scenario]
    matched, *_, combined = cases_for(scenario)
    if mode == "matched_elastic":
        case, include_electrical = matched, False
    elif mode == "matched_joint":
        case, include_electrical = matched, True
    elif mode == "mismatch_joint":
        case, include_electrical = combined, True
    else:
        raise ValueError(f"Unknown profile mode: {mode}")

    truth = reference_state(scenario)
    elastic_data, electrical_data = _observed_data(truth, scenario, case, panel_size)
    names = targets_for(scenario)
    if include_electrical:
        names = names + _nuisance_names(case.inverse_coupling, "deep_shallow")
    names = list(dict.fromkeys(names))

    unrestricted = _fit(
        truth, names, {}, scenario, case, panel_size, elastic_data, electrical_data,
        include_electrical, starts=2,
    )
    qmin, fit0, optimum = unrestricted
    free = [name for name in names if name not in (spec.x, spec.y)]
    xr, yr = RANGES[scenario.name][spec.x], RANGES[scenario.name][spec.y]
    xgrid = np.linspace(*xr, grid_size)
    ygrid = np.linspace(*yr, grid_size)
    rows = []
    warm = optimum
    for iy, yvalue in enumerate(ygrid):
        for ix, xvalue in enumerate(xgrid):
            fixed = {spec.x: float(xvalue), spec.y: float(yvalue)}
            cost, fit, fitted_p = _fit(
                warm, free, fixed, scenario, case, panel_size,
                elastic_data, electrical_data, include_electrical, starts=1,
            )
            warm = fitted_p
            rows.append({
                "scenario": scenario.name, "pair": f"{spec.x}|{spec.y}",
                "pair_label": spec.label, "mode": mode,
                "x_name": spec.x, "y_name": spec.y,
                "ix": ix, "iy": iy, "x": xvalue, "y": yvalue,
                "q": cost, "delta_q": max(cost - qmin, 0.0),
                "converged": bool(fit.success), "nfev": int(fit.nfev),
            })
    frame = pd.DataFrame(rows)
    region = frame[frame["delta_q"] <= 5.991]
    touches = bool(
        len(region) == 0
        or (region["ix"].isin([0, grid_size - 1])).any()
        or (region["iy"].isin([0, grid_size - 1])).any()
    )
    lower, upper = _scaled_bounds(names, scenario)
    tol = 1e-4 * np.maximum(upper - lower, 1.0)
    bound_hit = bool(np.any((fit0.x - lower) < tol) or np.any((upper - fit0.x) < tol))
    summary = {
        "scenario": scenario.name, "pair": f"{spec.x}|{spec.y}",
        "pair_label": spec.label, "mode": mode, "qmin": qmin,
        "closed_95": not touches, "unrestricted_bound_hit": bound_hit,
        "convergence_fraction": float(frame["converged"].mean()),
        "region_convergence_fraction": float(region["converged"].mean()) if len(region) else 0.0,
        "region_points": int(len(region)),
        "width_x_95": float(region["x"].max() - region["x"].min()) if len(region) else np.nan,
        "width_y_95": float(region["y"].max() - region["y"].min()) if len(region) else np.nan,
        "grid_size": grid_size, "panel_size": panel_size,
    }
    return frame, summary


def run_profiles(grid_size: int = 9, panel_size: int = 16):
    frames, summaries = [], []
    for spec in PROFILE_SPECS:
        modes = ["matched_elastic", "matched_joint"]
        if spec.scenario == "multimodal_carbonate" and spec.x == "secondary":
            modes.append("mismatch_joint")
        for mode in modes:
            frame, summary = compute_profile(spec, mode, grid_size, panel_size)
            frames.append(frame)
            summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)
