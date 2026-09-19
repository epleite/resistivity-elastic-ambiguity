"""Ensemble robustness and model-mismatch experiments for Phase 0.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .analysis import (
    DATA_SIGMA,
    PARAM_SCALES,
    TARGETS,
    finite_difference_jacobian,
)
from .families import forward_elastic_family, forward_electrical_family
from .models import SCENARIOS, Scenario

TARGET_PRIORS = {"weak": 0.01, "base": 0.04, "moderate": 0.10}
NUISANCE_TIERS = ("free", "weak", "calibrated")
BASE_ELECTRICAL_NUISANCE = [
    "log_rw", "archie_m", "archie_n", "surface_cond", "invasion"
]
DATA_CONFIGS = ("elastic", "resistivity", "deep", "deep_shallow", "oracle")
RT_CORRELATION = 0.55
USEFUL_SD = {
    "phi": 0.03, "sw": 0.10, "vcl": 0.07, "cement": 0.015,
    "coord": 1.0, "aspect": 0.03, "secondary": 0.025,
}

_ELASTIC_JAC_CACHE: dict[tuple, np.ndarray] = {}
_ELECTRICAL_JAC_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _parameter_signature(p: Mapping[str, float]) -> tuple:
    return tuple(sorted((name, float(value)) for name, value in p.items()))


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    truth_elastic: str
    inverse_elastic: str
    truth_electrical: str
    inverse_electrical: str
    truth_coupling: str
    inverse_coupling: str


RANGES = {
    "clean_sandstone": {
        "phi": (0.12, 0.32), "sw": (0.25, 0.95), "vcl": (0.00, 0.06),
        "cement": (0.005, 0.080), "coord": (6.0, 11.0),
        "aspect": (0.05, 0.25), "secondary": (0.000, 0.020),
        "log_rw": (np.log(0.03), np.log(0.25)), "archie_m": (1.6, 2.3),
        "archie_n": (1.7, 2.5), "surface_cond": (0.000, 0.003),
        "invasion": (0.05, 0.45), "phi_e": (0.08, 0.34),
    },
    "clayey_sandstone": {
        "phi": (0.12, 0.32), "sw": (0.25, 0.95), "vcl": (0.08, 0.35),
        "cement": (0.005, 0.080), "coord": (5.5, 10.0),
        "aspect": (0.04, 0.20), "secondary": (0.000, 0.030),
        "log_rw": (np.log(0.03), np.log(0.30)), "archie_m": (1.8, 2.7),
        "archie_n": (1.7, 2.8), "surface_cond": (0.005, 0.080),
        "invasion": (0.05, 0.45), "phi_e": (0.07, 0.34),
    },
    "multimodal_carbonate": {
        "phi": (0.06, 0.28), "sw": (0.20, 0.98), "vcl": (0.00, 0.08),
        "cement": (0.010, 0.120), "coord": (5.0, 10.0),
        "aspect": (0.015, 0.15), "secondary": (0.005, 0.120),
        "log_rw": (np.log(0.025), np.log(0.30)), "archie_m": (1.8, 3.2),
        "archie_n": (1.6, 3.0), "surface_cond": (0.000, 0.012),
        "invasion": (0.05, 0.50), "phi_e": (0.03, 0.30),
    },
}


def targets_for(scenario: Scenario) -> list[str]:
    core = ["phi", "sw", "vcl", "cement", "coord", "aspect"]
    return core + (["secondary"] if scenario.name == "multimodal_carbonate" else [])


def cases_for(scenario: Scenario) -> list[ExperimentCase]:
    """Matched and one-factor/combined mismatch cases."""
    if scenario.name == "clean_sandstone":
        base_e, alt_e = "soft_sand", "stiff_sand"
        base_r, alt_r = "archie", "dual_porosity"
        base_c = "rigid"
    elif scenario.name == "clayey_sandstone":
        base_e, alt_e = "soft_sand", "contact_cement"
        base_r, alt_r = "waxman_smits", "archie"
        base_c = "partial"
    else:
        base_e, alt_e = "dem", "dem_single"
        base_r, alt_r = "dual_porosity", "archie"
        base_c = "partial"
    wrong_c = "rigid" if base_c == "partial" else "partial"
    return [
        ExperimentCase("matched", base_e, base_e, base_r, base_r, base_c, base_c),
        ExperimentCase("elastic_mismatch", alt_e, base_e, base_r, base_r, base_c, base_c),
        ExperimentCase("electrical_mismatch", base_e, base_e, base_r, alt_r, base_c, base_c),
        ExperimentCase("coupling_mismatch", base_e, base_e, base_r, base_r, base_c, wrong_c),
        ExperimentCase("combined_mismatch", alt_e, base_e, base_r, alt_r, base_c, wrong_c),
    ]


def latin_hypercube(scenario: Scenario, n: int, seed: int) -> list[dict[str, float]]:
    ranges = RANGES[scenario.name]
    names = list(ranges)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n)
    lower = np.array([ranges[name][0] for name in names])
    upper = np.array([ranges[name][1] for name in names])
    values = qmc.scale(unit, lower, upper)
    samples = []
    for row in values:
        p = {name: float(value) for name, value in zip(names, row)}
        p["secondary"] = min(p["secondary"], 0.72 * p["phi"])
        # Independent electrical porosity is centered on, but not equal to, the
        # partially connected pore volume.
        connected = p["phi"] - 0.65 * p["secondary"]
        p["phi_e"] = float(np.clip(0.65 * p["phi_e"] + 0.35 * connected, 0.02, 0.45))
        samples.append(p)
    return samples


def _nuisance_names(coupling: str, config: str) -> list[str]:
    if config in ("elastic", "oracle"):
        return []
    names = BASE_ELECTRICAL_NUISANCE.copy()
    if config == "deep":
        names.remove("invasion")
    if coupling == "independent":
        names.append("phi_e")
    return names


def _nuisance_precision(names: Sequence[str], tier: str) -> np.ndarray:
    if tier == "free":
        return np.zeros((len(names), len(names)))
    weak = {
        "log_rw": 0.15, "archie_m": 0.10, "archie_n": 0.10,
        "surface_cond": 0.05, "invasion": 0.18, "phi_e": 0.10,
    }
    factor = 1.0 if tier == "weak" else 25.0
    return np.diag([factor * weak[name] for name in names])


def _safe_inverse(a: np.ndarray) -> np.ndarray:
    a = 0.5 * (a + a.T)
    evals, evecs = np.linalg.eigh(a)
    floor = max(1e-10, 1e-12 * max(float(evals[-1]), 1.0))
    return (evecs * (1.0 / np.maximum(evals, floor))) @ evecs.T


def _electrical_whitener(config: str) -> np.ndarray:
    sigma = DATA_SIGMA["rt_log"]
    if config == "deep":
        return np.array([[1.0 / sigma]])
    covariance = sigma**2 * np.array([[1.0, RT_CORRELATION], [RT_CORRELATION, 1.0]])
    return np.linalg.inv(np.linalg.cholesky(covariance))


def _effective_information(jt: np.ndarray, jn: np.ndarray, pn: np.ndarray) -> np.ndarray:
    if not jn.size:
        return jt.T @ jt
    ann = jn.T @ jn + pn
    if not np.any(pn):
        ann_inv = np.linalg.pinv(ann, rcond=1e-10)
    else:
        ann_inv = _safe_inverse(ann)
    result = jt.T @ jt - jt.T @ jn @ ann_inv @ jn.T @ jt
    return 0.5 * (result + result.T)


def _panel_states(p: Mapping[str, float], size: int) -> list[dict[str, float]]:
    """Deterministic synthetic facies panel with shared constitutive parameters."""
    if size == 1:
        return [dict(p)]
    phase = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    states = []
    for angle in phase:
        q = dict(p)
        q["phi"] = float(np.clip(p["phi"] + 0.028 * np.sin(angle) + 0.010 * np.sin(3 * angle), 0.045, 0.36))
        q["sw"] = float(np.clip(p["sw"] + 0.14 * np.cos(2 * angle + 0.35), 0.08, 0.995))
        q["vcl"] = float(np.clip(p["vcl"] + 0.035 * np.sin(angle + 1.2), 0.0, 0.42))
        sec_factor = 0.70 + 0.30 * (1.0 + np.sin(2 * angle - 0.4))
        q["secondary"] = float(min(p["secondary"] * sec_factor, 0.72 * q["phi"]))
        connected = q["phi"] - 0.65 * q["secondary"]
        q["phi_e"] = float(np.clip(p["phi_e"] + 0.45 * (connected - (p["phi"] - 0.65 * p["secondary"])), 0.02, 0.45))
        states.append(q)
    return states


def _elastic_vector(p: Mapping[str, float], scenario: Scenario, family: str, panel_size: int):
    return np.vstack([
        forward_elastic_family(q, scenario, family) for q in _panel_states(p, panel_size)
    ]).ravel()


def _electrical_matrix(
    p: Mapping[str, float], scenario: Scenario, family: str, coupling: str,
    panel_size: int,
):
    return np.vstack([
        forward_electrical_family(q, scenario, family, coupling)
        for q in _panel_states(p, panel_size)
    ])


def _elastic_jacobian(
    p: Mapping[str, float], scenario: Scenario, family: str, targets, panel_size: int,
):
    key = (_parameter_signature(p), scenario.name, family, tuple(targets), panel_size)
    cached = _ELASTIC_JAC_CACHE.get(key)
    if cached is not None:
        return cached
    j = finite_difference_jacobian(
        lambda x: _elastic_vector(x, scenario, family, panel_size), p, targets
    )
    sigma = np.tile([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]], panel_size)
    result = j / sigma[:, None]
    _ELASTIC_JAC_CACHE[key] = result
    return result


def _electrical_jacobians(
    p: Mapping[str, float], scenario: Scenario, family: str, coupling: str,
    config: str, targets: Sequence[str], nuisance: Sequence[str],
    panel_size: int,
):
    key = (
        _parameter_signature(p), scenario.name, family, coupling, config,
        tuple(targets), tuple(nuisance), panel_size,
    )
    cached = _ELECTRICAL_JAC_CACHE.get(key)
    if cached is not None:
        return cached
    indices = slice(0, 1) if config == "deep" else slice(None)
    whitener = _electrical_whitener(config)
    def raw(x):
        return _electrical_matrix(x, scenario, family, coupling, panel_size)[:, indices].ravel()
    jt_raw = finite_difference_jacobian(raw, p, targets, log_output=True)
    ncurves = 1 if config == "deep" else 2
    jt = np.vstack([whitener @ jt_raw[i:i+ncurves] for i in range(0, len(jt_raw), ncurves)])
    if nuisance:
        jn_raw = finite_difference_jacobian(raw, p, nuisance, log_output=True)
        jn = np.vstack([whitener @ jn_raw[i:i+ncurves] for i in range(0, len(jn_raw), ncurves)])
    else:
        jn = np.empty((jt.shape[0], 0))
    result = (jt, jn)
    _ELECTRICAL_JAC_CACHE[key] = result
    return result


def _normalized_model_residual(
    p: Mapping[str, float], scenario: Scenario, case: ExperimentCase, config: str,
    panel_size: int,
):
    if config != "resistivity":
        elastic_truth = _elastic_vector(p, scenario, case.truth_elastic, panel_size)
        elastic_inverse = _elastic_vector(p, scenario, case.inverse_elastic, panel_size)
        sigma = np.tile([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]], panel_size)
        re = (elastic_truth - elastic_inverse) / sigma
    else:
        re = np.empty(0)
    if config == "elastic":
        return re
    econfig = "deep_shallow" if config in ("resistivity", "oracle") else config
    indices = slice(0, 1) if econfig == "deep" else slice(None)
    rt_truth = np.log(_electrical_matrix(
        p, scenario, case.truth_electrical, case.truth_coupling, panel_size
    )[:, indices])
    rt_inverse = np.log(_electrical_matrix(
        p, scenario, case.inverse_electrical, case.inverse_coupling, panel_size
    )[:, indices])
    whitener = _electrical_whitener(econfig)
    rr = np.hstack([whitener @ row for row in (rt_truth - rt_inverse)])
    return np.concatenate([re, rr])


def _generalized_update(info_e: np.ndarray, increment: np.ndarray, precision: float):
    h = info_e + precision * np.eye(info_e.shape[0])
    evals, evecs = np.linalg.eigh(h)
    hinvhalf = (evecs * (1.0 / np.sqrt(np.maximum(evals, 1e-12)))) @ evecs.T
    normalized = 0.5 * (hinvhalf @ increment @ hinvhalf)
    normalized = normalized + normalized.T
    lam = np.maximum(np.linalg.eigvalsh(normalized), 0.0)
    gains = lam / (1.0 + lam)
    info_gain = 0.5 * float(np.sum(np.log1p(lam)))
    return gains, info_gain


def _weak_subspace_gain(info_e: np.ndarray, increment: np.ndarray, precision: float) -> float:
    """Largest update in elastic directions with <=1 Fisher unit of information."""
    evals, evecs = np.linalg.eigh(0.5 * (info_e + info_e.T))
    weak = evecs[:, evals <= 1.0]
    if weak.shape[1] == 0:
        return 0.0
    h = weak.T @ (info_e + precision * np.eye(info_e.shape[0])) @ weak
    a = weak.T @ increment @ weak
    he, hv = np.linalg.eigh(0.5 * (h + h.T))
    hinvhalf = (hv * (1.0 / np.sqrt(np.maximum(he, 1e-12)))) @ hv.T
    norm = hinvhalf @ (0.5 * (a + a.T)) @ hinvhalf
    lam = max(float(np.linalg.eigvalsh(norm)[-1]), 0.0)
    return lam / (1.0 + lam)


def _one_state(
    p: Mapping[str, float], scenario: Scenario, case: ExperimentCase,
    config: str, nuisance_tier: str, panel_size: int,
) -> dict:
    targets = targets_for(scenario)
    je = _elastic_jacobian(p, scenario, case.inverse_elastic, targets, panel_size)
    if config == "resistivity":
        je = np.empty((0, len(targets)))
    info_e = je.T @ je
    _, _, vt = np.linalg.svd(je, full_matrices=True)
    rank = int(np.linalg.matrix_rank(je, tol=1e-8))
    null_basis = vt[rank:].T

    nuisance = _nuisance_names(case.inverse_coupling, config)
    if config == "elastic":
        jt = np.empty((0, len(targets)))
        jn = np.empty((0, 0))
        pn = np.empty((0, 0))
        increment = np.zeros_like(info_e)
    else:
        econfig = "deep_shallow" if config in ("resistivity", "oracle") else config
        jt, jn = _electrical_jacobians(
            p, scenario, case.inverse_electrical, case.inverse_coupling,
            econfig, targets, nuisance, panel_size,
        )
        pn = _nuisance_precision(nuisance, nuisance_tier)
        increment = _effective_information(jt, jn, pn)

    if null_basis.shape[1]:
        projected = 0.5 * (null_basis.T @ increment @ null_basis)
        projected = projected + projected.T
        null_evals = np.linalg.eigvalsh(projected)
        max_null_gamma = float(max(null_evals[-1], 0.0))
        null_rank_added = int(np.count_nonzero(null_evals > 1.0))
    else:
        max_null_gamma, null_rank_added = 0.0, 0

    reductions, joint_sd, useful, directional = {}, {}, {}, {}
    for prior_name, target_precision in TARGET_PRIORS.items():
        cov_e = _safe_inverse(info_e + target_precision * np.eye(len(targets)))
        cov_j = _safe_inverse(info_e + increment + target_precision * np.eye(len(targets)))
        reductions[prior_name] = 1.0 - np.diag(cov_j) / np.diag(cov_e)
        joint_sd[prior_name] = np.sqrt(np.diag(cov_j)) * np.array([PARAM_SCALES[n] for n in targets])
        useful[prior_name] = joint_sd[prior_name] <= np.array([USEFUL_SD[n] for n in targets])
        dgain, information_gain = _generalized_update(info_e, increment, target_precision)
        directional[prior_name] = {
            "leading_gain": float(dgain[-1]),
            "second_gain": float(dgain[-2]) if len(dgain) > 1 else 0.0,
            "effective_dimensions": float(np.sum(dgain)),
            "information_gain": information_gain,
            "weak_subspace_gain": _weak_subspace_gain(info_e, increment, target_precision),
        }

    # Linearized MAP bias under deliberate constitutive mismatch.  Bias and
    # posterior standard deviations are both expressed in scaled coordinates.
    if case.name == "matched":
        bias_z = np.zeros(len(targets))
    elif config == "elastic":
        jfull = je
        prior_full = TARGET_PRIORS["base"] * np.eye(len(targets))
    elif config == "resistivity":
        jfull = np.block([jt, jn])
        prior_full = np.zeros((len(targets) + len(nuisance),) * 2)
        prior_full[:len(targets), :len(targets)] = TARGET_PRIORS["base"] * np.eye(len(targets))
        if nuisance:
            prior_full[len(targets):, len(targets):] = pn
    else:
        zeros = np.zeros((je.shape[0], len(nuisance)))
        jfull = np.block([[je, zeros], [jt, jn]])
        prior_full = np.zeros((len(targets) + len(nuisance),) * 2)
        prior_full[:len(targets), :len(targets)] = TARGET_PRIORS["base"] * np.eye(len(targets))
        if nuisance:
            prior_full[len(targets):, len(targets):] = pn
    if case.name != "matched":
        cov_full = _safe_inverse(jfull.T @ jfull + prior_full)
        residual = _normalized_model_residual(p, scenario, case, config, panel_size)
        delta = cov_full @ jfull.T @ residual
        bias_z = np.abs(delta[:len(targets)]) / np.sqrt(np.diag(cov_full)[:len(targets)])

    return {
        "reductions": reductions,
        "joint_sd": joint_sd,
        "useful": useful,
        "directional": directional,
        "bias_z": bias_z,
        "max_null_gamma": max_null_gamma,
        "null_rank_added": null_rank_added,
    }


def run_ensemble(
    n_panels: int = 512, panel_size: int = 16, seed: int = 20260829,
    n_batches: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 512x16=8192 depth states per scenario by default."""
    if n_panels % n_batches:
        raise ValueError("n_panels must be divisible by n_batches")
    _ELASTIC_JAC_CACHE.clear()
    _ELECTRICAL_JAC_CACHE.clear()
    summaries: list[dict] = []
    map_rows: list[dict] = []
    for scenario_index, scenario in enumerate(SCENARIOS.values()):
        per_batch = n_panels // n_batches
        samples = []
        batch_ids = []
        for batch in range(n_batches):
            samples.extend(latin_hypercube(
                scenario, per_batch, seed + 1009 * scenario_index + 7919 * batch
            ))
            batch_ids.extend([batch] * per_batch)
        batch_ids = np.asarray(batch_ids)
        targets = targets_for(scenario)
        for case in cases_for(scenario):
            settings = (
                [
                    ("panel", "elastic", "free"),
                    ("panel", "resistivity", "free"),
                    ("panel", "deep", "free"),
                    ("panel", "deep_shallow", "free"),
                    ("panel", "deep_shallow", "weak"),
                    ("panel", "deep_shallow", "calibrated"),
                    ("panel", "oracle", "calibrated"),
                    ("local", "deep_shallow", "free"),
                ] if case.name == "matched" else [
                    ("panel", "deep_shallow", "free"),
                ]
            )
            for structure, config, nuisance_tier in settings:
                active_panel = panel_size if structure == "panel" else 1
                state_results = [
                    _one_state(p, scenario, case, config, nuisance_tier, active_panel)
                    for p in samples
                ]
                null_gamma = np.array([r["max_null_gamma"] for r in state_results])
                null_added = np.array([r["null_rank_added"] for r in state_results])
                bias = np.vstack([r["bias_z"] for r in state_results])
                for prior_name in TARGET_PRIORS:
                    reductions = np.vstack([r["reductions"][prior_name] for r in state_results])
                    sds = np.vstack([r["joint_sd"][prior_name] for r in state_results])
                    useful = np.vstack([r["useful"][prior_name] for r in state_results])
                    leading = np.array([r["directional"][prior_name]["leading_gain"] for r in state_results])
                    second = np.array([r["directional"][prior_name]["second_gain"] for r in state_results])
                    edims = np.array([r["directional"][prior_name]["effective_dimensions"] for r in state_results])
                    igain = np.array([r["directional"][prior_name]["information_gain"] for r in state_results])
                    weak_gain = np.array([r["directional"][prior_name]["weak_subspace_gain"] for r in state_results])
                    for target_index, target in enumerate(targets):
                        gain = reductions[:, target_index]
                        bz = bias[:, target_index]
                        batch_gain = [
                            float(np.mean(gain[batch_ids == batch] >= 0.20))
                            for batch in range(n_batches)
                        ]
                        summaries.append({
                            "scenario": scenario.name,
                            "case": case.name, "structure": structure,
                            "config": config,
                            "nuisance_tier": nuisance_tier,
                            "target_prior": prior_name,
                            "target": target,
                            "n": n_panels,
                            "median_reduction": float(np.median(gain)),
                            "p10_reduction": float(np.quantile(gain, 0.10)),
                            "p90_reduction": float(np.quantile(gain, 0.90)),
                            "gain_fraction": float(np.mean(gain >= 0.20)),
                            "gain_fraction_wilson_lower": _wilson_lower(int(np.count_nonzero(gain >= 0.20)), n_panels),
                            "gain_fraction_batch_sd": float(np.std(batch_gain, ddof=1)) if n_batches > 1 else 0.0,
                            "median_joint_sd": float(np.median(sds[:, target_index])),
                            "useful_fraction": float(np.mean(useful[:, target_index])),
                            "median_bias_z": float(np.median(bz)),
                            "p90_bias_z": float(np.quantile(bz, 0.90)),
                            "coverage_risk": float(np.mean(bz > 1.96)),
                            "false_confidence_fraction": float(np.mean((gain >= 0.20) & (bz > 1.96))),
                            "median_max_null_gamma": float(np.median(null_gamma)),
                            "null_rescue_fraction": float(np.mean(null_gamma >= 1.0)),
                            "rank_added_fraction": float(np.mean(null_added >= 1)),
                            "leading_directional_gain_median": float(np.median(leading)),
                            "second_directional_gain_median": float(np.median(second)),
                            "effective_dimensions_median": float(np.median(edims)),
                            "information_gain_median": float(np.median(igain)),
                            "weak_subspace_gain_median": float(np.median(weak_gain)),
                            "weak_rescue_fraction": float(np.mean(weak_gain >= 0.20)),
                            "weak_rescue_wilson_lower": _wilson_lower(int(np.count_nonzero(weak_gain >= 0.20)), n_panels),
                        })
                if case.name == "matched" and structure == "panel" and config == "deep_shallow" and nuisance_tier == "free":
                    base_red = np.vstack([r["reductions"]["base"] for r in state_results])
                    for idx, (p, result) in enumerate(zip(samples, state_results)):
                        row = {
                            "scenario": scenario.name, "sample": idx,
                            "phi": p["phi"], "sw": p["sw"], "vcl": p["vcl"],
                            "secondary": p["secondary"], "log_rw": p["log_rw"],
                            "max_null_gamma": result["max_null_gamma"],
                        }
                        for j, target in enumerate(targets):
                            row[f"gain_{target}"] = float(base_red[idx, j])
                        map_rows.append(row)

    summary = pd.DataFrame(summaries)
    gates = build_gates(summary)
    return summary, gates, pd.DataFrame(map_rows)


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    p = successes / total
    den = 1.0 + z**2 / total
    center = p + z**2 / (2.0 * total)
    spread = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2))
    return float((center - spread) / den)


def build_gates(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the explicit robust/conditional/stop decision rules."""
    rows = []
    main = summary[
        (summary["structure"] == "panel")
        & (summary["config"] == "deep_shallow")
        & (summary["nuisance_tier"] == "free")
    ]
    oracle = summary[
        (summary["structure"] == "panel") & (summary["config"] == "oracle")
        & (summary["target_prior"] == "base")
    ]
    for (scenario, target), group in main.groupby(["scenario", "target"]):
        matched = group[group["case"] == "matched"]
        mismatch = group[group["case"] != "matched"]
        # Requiring success for all target-prior strengths prevents a gate from
        # being determined by the arbitrary ridge used to regularize null spaces.
        matched_worst_gain = float(matched.groupby("target_prior")["gain_fraction_wilson_lower"].first().min())
        mismatch_worst_gain = float(mismatch.groupby("target_prior")["gain_fraction_wilson_lower"].mean().min())
        mismatch_false = float(mismatch[mismatch["target_prior"] == "base"]["false_confidence_fraction"].mean())
        oracle_row = oracle[
            (oracle["scenario"] == scenario) & (oracle["target"] == target)
            & (oracle["case"] == "matched")
        ]
        oracle_gain = float(oracle_row["gain_fraction"].iloc[0])
        matched_null = float(matched[matched["target_prior"] == "base"]["null_rescue_fraction"].iloc[0])
        matched_weak = float(matched[matched["target_prior"] == "base"]["weak_rescue_wilson_lower"].iloc[0])
        matched_useful = float(matched[matched["target_prior"] == "base"]["useful_fraction"].iloc[0])
        matched_bias = float(mismatch[mismatch["target_prior"] == "base"]["median_bias_z"].median())
        if (
            matched_worst_gain >= 0.70 and mismatch_worst_gain >= 0.70
            and mismatch_false <= 0.10 and matched_useful >= 0.70 and matched_bias <= 0.50
            and matched_weak >= 0.70
        ):
            verdict = "ROBUST_GO"
        elif oracle_gain >= 0.70 or matched_worst_gain >= 0.30:
            verdict = "CONDITIONAL"
        else:
            verdict = "STOP"
        rows.append({
            "scenario": scenario, "target": target, "verdict": verdict,
            "matched_worst_gain_fraction": matched_worst_gain,
            "mismatch_worst_gain_fraction": mismatch_worst_gain,
            "mismatch_false_confidence": mismatch_false,
            "oracle_gain_fraction": oracle_gain,
            "matched_null_rescue_fraction": matched_null,
            "matched_weak_rescue_wilson_lower": matched_weak,
            "matched_useful_fraction": matched_useful,
            "mismatch_median_bias_z": matched_bias,
            "interpretation": (
                "practical weak-direction rescue" if matched_weak >= 0.70
                else "tightening without robust weak-direction rescue"
            ),
        })
    return pd.DataFrame(rows)
