"""Jacobian, Fisher/Schur and weak-direction analysis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .models import SCENARIOS, Scenario, forward_elastic, forward_electrical

TARGETS = ["phi", "sw", "cement", "coord", "aspect", "secondary"]
ELECTRICAL_NUISANCE = ["log_rw", "archie_m", "archie_n", "surface_cond"]

# Scaling makes columns dimensionless and gives the weak direction a physical
# interpretation in plausible parameter perturbations.
PARAM_SCALES = {
    "phi": 0.04, "sw": 0.15, "vcl": 0.10, "cement": 0.02,
    "coord": 1.5, "aspect": 0.04, "secondary": 0.025,
    "log_rw": 0.45, "archie_m": 0.25, "archie_n": 0.30,
    "surface_cond": 0.015,
    "phi_e": 0.04, "invasion": 0.12,
}
DATA_SIGMA = {"vp": 0.06, "vs": 0.04, "rho": 0.025, "rt_log": 0.12}


def finite_difference_jacobian(
    func: Callable[[Mapping[str, float]], np.ndarray],
    p: Mapping[str, float], names: Sequence[str], log_output: bool = False,
) -> np.ndarray:
    """Central finite-difference Jacobian with dimensionless parameter columns."""
    base = dict(p)
    columns = []
    for name in names:
        h = 1e-4 * PARAM_SCALES[name]
        plus, minus = dict(base), dict(base)
        plus[name], minus[name] = base[name] + h, base[name] - h
        yp, ym = func(plus), func(minus)
        if log_output:
            yp, ym = np.log(yp), np.log(ym)
        columns.append((yp - ym) / (2.0 * h) * PARAM_SCALES[name])
    return np.column_stack(columns)


def _effective_target_information(
    j_target: np.ndarray, j_nuis: np.ndarray, prior_nuis: np.ndarray,
) -> np.ndarray:
    """Schur complement after marginalizing nuisance parameters."""
    att = j_target.T @ j_target
    if j_nuis.size == 0:
        return att
    atn = j_target.T @ j_nuis
    ann = j_nuis.T @ j_nuis + prior_nuis
    return att - atn @ np.linalg.pinv(ann, rcond=1e-11) @ atn.T


def _posterior_covariance(info: np.ndarray) -> np.ndarray:
    # Weak, common target prior: five scaled standard deviations. Its only role
    # is to make parameter-wise variances finite in an underdetermined problem.
    return np.linalg.inv(info + 0.04 * np.eye(info.shape[0]))


def analyze_scenario(scenario: Scenario) -> dict:
    p = dict(scenario.truth)
    targets = TARGETS if scenario.name == "multimodal_carbonate" else TARGETS[:-1]

    je = finite_difference_jacobian(lambda x: forward_elastic(x, scenario), p, targets)
    je = je / np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])[:, None]
    jr_t = finite_difference_jacobian(
        lambda x: forward_electrical(x, scenario)[:1], p, targets, log_output=True
    ) / DATA_SIGMA["rt_log"]
    jr_n = finite_difference_jacobian(
        lambda x: forward_electrical(x, scenario)[:1], p, ELECTRICAL_NUISANCE,
        log_output=True,
    ) / DATA_SIGMA["rt_log"]
    jr2_t = finite_difference_jacobian(
        lambda x: forward_electrical(x, scenario), p, targets, log_output=True
    ) / DATA_SIGMA["rt_log"]
    jr2_n = finite_difference_jacobian(
        lambda x: forward_electrical(x, scenario), p, ELECTRICAL_NUISANCE,
        log_output=True,
    ) / DATA_SIGMA["rt_log"]

    info_e = je.T @ je
    _, singular_values, vt = np.linalg.svd(je, full_matrices=True)
    rank = int(np.linalg.matrix_rank(je, tol=1e-9))
    null_basis = vt[rank:].T
    cov_e = _posterior_covariance(info_e)

    # Prior precision in scaled coordinates. Broad represents realistic weak
    # knowledge; tight approximates local calibration of electrical parameters.
    priors = {
        "broad": np.diag([0.15, 0.10, 0.10, 0.05]),
        "tight": np.diag([9.0, 4.0, 4.0, 2.0]),
    }
    cases = {}
    for prior_name, prior in priors.items():
        for curve_name, jt, jn in [
            ("deep", jr_t, jr_n), ("deep+shallow", jr2_t, jr2_n)
        ]:
            increment = _effective_target_information(jt, jn, prior)
            info_joint = info_e + increment
            cov_joint = _posterior_covariance(info_joint)
            reductions = 1.0 - np.diag(cov_joint) / np.diag(cov_e)
            if null_basis.shape[1]:
                projected = null_basis.T @ increment @ null_basis
                null_evals, null_evecs = np.linalg.eigh(projected)
                best_null = null_basis @ null_evecs[:, -1]
                max_null_gamma = float(max(null_evals[-1], 0.0))
                null_trace = float(max(np.trace(projected), 0.0))
            else:
                best_null = np.zeros(len(targets))
                max_null_gamma = null_trace = 0.0
            cases[f"{curve_name}|{prior_name}"] = {
                "max_null_gamma": max_null_gamma,
                "null_information_trace": null_trace,
                "parameter_variance_reduction": {
                    name: float(value) for name, value in zip(targets, reductions)
                },
                "most_rescued_null_direction": {
                    name: float(value) for name, value in zip(targets, best_null)
                },
            }

    broad_gains = cases["deep+shallow|broad"]["parameter_variance_reduction"]
    tight_gains = cases["deep+shallow|tight"]["parameter_variance_reduction"]
    target_verdicts = {}
    for name in targets:
        if broad_gains[name] >= 0.20:
            target_verdicts[name] = "GO"
        elif tight_gains[name] >= 0.20:
            target_verdicts[name] = "CONDITIONAL"
        else:
            target_verdicts[name] = "STOP"
    verdict = "GO" if "GO" in target_verdicts.values() else (
        "CONDITIONAL" if "CONDITIONAL" in target_verdicts.values() else "STOP"
    )
    return {
        "scenario": scenario.name,
        "label": scenario.label,
        "targets": targets,
        "elastic_rank": rank,
        "elastic_nullity": len(targets) - rank,
        "elastic_singular_values": singular_values.tolist(),
        "cases": cases,
        "target_verdicts": target_verdicts,
        "verdict": verdict,
    }


def run_benchmark() -> tuple[list[dict], pd.DataFrame]:
    results = [analyze_scenario(s) for s in SCENARIOS.values()]
    rows = []
    for result in results:
        for case, metrics in result["cases"].items():
            curves, prior = case.split("|")
            for target, reduction in metrics["parameter_variance_reduction"].items():
                rows.append({
                    "scenario": result["scenario"], "label": result["label"],
                    "target": target, "curves": curves, "nuisance_prior": prior,
                    "max_null_gamma": metrics["max_null_gamma"],
                    "null_information_trace": metrics["null_information_trace"],
                    "variance_reduction": reduction,
                    "target_verdict": result["target_verdicts"][target],
                    "scenario_verdict": result["verdict"],
                })
    return results, pd.DataFrame(rows)
