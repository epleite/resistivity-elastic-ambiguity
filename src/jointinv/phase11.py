"""Phase 1.1: learn carbonate electrical-elastic coupling uncertainty.

Three coupling hypotheses are compared for each of three electrical
constitutive families:

``rigid``
    The electrical connectivity latent equals the elastic pore-shape latent.
``partial``
    Electrical connectivity is free but shrunk toward elastic pore shape.
``independent``
    Electrical connectivity is free with an independent standard-normal prior.

Candidate predictions are combined with panel-cluster cross-fitted weights
derived from linearized leave-one-depth-block-out scores. Target intervals use
the complete Gaussian-mixture
variance and quantiles, retaining both within-model and between-model
uncertainty.  This is a computational screening approximation to predictive
stacking, not a substitute for posterior sampling on field data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import expit
from scipy.stats import norm

from .analysis import DATA_SIGMA
from .families import forward_elastic_family
from .phase05 import RANGES, _panel_states
from .phase1 import (
    CARBONATE, FIT_NAMES as PHASE1_FIT_NAMES,
    NUISANCE, TARGETS, _rt_cholesky, carbonate_design,
    gem_conductivity,
)


CANDIDATE_FAMILIES = ("latent_dual", "ema", "network")
COUPLING_MODES = ("rigid", "partial", "independent")
STRATEGIES = ("elastic", "rigid", "partial", "independent", "model_average")
Q_RANGE = (-3.0, 3.0)
RMF_RANGE = (0.25, 1.75)
PARTIAL_TAU = 0.75


@dataclass(frozen=True)
class Phase11TruthCase:
    name: str
    truth_electrical: str
    coupling_rho: float | None


TRUTH_CASES = (
    # The dual case fixes 35% connected secondary porosity; it does not define
    # a panelwise correlation coefficient and must not be labelled rho=1.
    Phase11TruthCase("dual_partial_truth", "dual", None),
    Phase11TruthCase("ema_rho0", "ema", 0.0),
    Phase11TruthCase("ema_rho05", "ema", 0.5),
    Phase11TruthCase("ema_rho09", "ema", 0.9),
    Phase11TruthCase("network_rho05", "network", 0.5),
)


def elastic_latent(aspect: float) -> float:
    """Finite log-aspect coordinate with the physical bounds at about ±2."""
    lo, hi = RANGES[CARBONATE.name]["aspect"]
    center = 0.5 * (np.log(lo) + np.log(hi))
    scale = np.log(hi / lo) / 4.0
    return float((np.log(np.clip(aspect, lo, hi)) - center) / scale)


def connectivity_latent(p: Mapping[str, float], coupling: str) -> float:
    if coupling == "rigid":
        return elastic_latent(float(p["aspect"]))
    if coupling not in ("partial", "independent"):
        raise ValueError(f"Unknown coupling mode: {coupling}")
    return float(np.clip(p["q_conn"], *Q_RANGE))


def _candidate_names(family: str, coupling: str) -> tuple[str, ...]:
    names = ["phi", "sw", "aspect", "secondary_fraction"] + list(NUISANCE)
    # A common observation operator prevents q_conn from absorbing mud-
    # filtrate and finite-volume tool effects in only some model families.
    names.append("rmf_ratio")
    if coupling != "rigid":
        names.append("q_conn")
    return tuple(names)


def _range(name: str) -> tuple[float, float]:
    if name == "q_conn":
        return Q_RANGE
    if name == "rmf_ratio":
        return RMF_RANGE
    if name == "secondary_fraction":
        return (0.0, 1.0)
    return RANGES[CARBONATE.name][name]


def _bounds(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([_range(name)[0] for name in names], dtype=float)
    hi = np.array([_range(name)[1] for name in names], dtype=float)
    return lo, hi


def _pack(p: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    lo, hi = _bounds(names)
    values = []
    for name in names:
        if name == "secondary_fraction":
            if "secondary_fraction" in p:
                values.append(float(p["secondary_fraction"]))
            else:
                cap = min(0.12, 0.72 * float(p["phi"]))
                values.append(float(p["secondary"]) / max(cap, 1e-8))
        else:
            values.append(p[name])
    values = np.asarray(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)


def _unpack(z: np.ndarray, names: Sequence[str], fixed: Mapping[str, float]) -> dict[str, float]:
    lo, hi = _bounds(names)
    p = dict(fixed)
    p.update({name: float(value) for name, value in zip(names, lo + z * (hi - lo))})
    if "secondary_fraction" in p:
        p["secondary"] = float(
            p["secondary_fraction"] * min(0.12, 0.72 * p["phi"])
        )
    return p


def _latent_dual_conductivity(
    p: Mapping[str, float], saturation: float, water_resistivity: float,
) -> float:
    phi = float(p["phi"])
    secondary = float(np.clip(p["secondary"], 0.0, 0.72 * phi))
    primary = max(phi - secondary, 0.005)
    connected_fraction = 0.05 + 0.90 * expit(float(p["q_effective"]))
    secondary_connected = connected_fraction * secondary
    phi_connected = max(primary + secondary_connected, 0.01)
    m, n = float(p["archie_m"]), float(p["archie_n"])
    surface = max(float(p["surface_cond"]), 0.0)
    vcl = float(p["vcl"])
    macro = primary**m * saturation**n / water_resistivity
    micro_m = float(np.clip(m - 0.35, 1.2, 3.2))
    micro_n = float(np.clip(n + 0.20, 1.3, 3.5))
    micro = 0.65 * secondary_connected**micro_m * saturation**micro_n / water_resistivity
    clay = 0.55 * surface * vcl * (0.35 + saturation) / max(phi_connected, 0.03)
    sigma = macro + micro + clay
    return max(float(sigma), 1e-10)


def forward_candidate_state(
    p: Mapping[str, float], family: str, coupling: str,
) -> np.ndarray:
    """Deep/shallow Rt for one candidate family and coupling hypothesis."""
    if family not in CANDIDATE_FAMILIES:
        raise ValueError(f"Unknown candidate family: {family}")
    q = connectivity_latent(p, coupling)
    qstate = dict(p)
    qstate["q_effective"] = q
    phi = float(p["phi"])
    secondary = float(np.clip(p["secondary"], 0.0, 0.72 * phi))
    primary = max(phi - secondary, 1e-5)
    sw = float(np.clip(p["sw"], 0.01, 0.999))
    rw = float(np.exp(p["log_rw"]))
    n = float(p["archie_n"])
    surface = max(float(p["surface_cond"]), 0.0)
    vcl = float(p["vcl"])
    invasion = float(np.clip(p["invasion"], 0.0, 0.65))

    def intrinsic(saturation: float, water_resistivity: float) -> float:
        if family == "latent_dual":
            return _latent_dual_conductivity(qstate, saturation, water_resistivity)
        sigma_fluid = saturation**n / max(water_resistivity, 1e-8)
        if family == "ema":
            pc = 0.025 + 0.19 * expit(-q)
            exponent = 1.55 + 0.85 * expit(q)
            sigma = gem_conductivity(phi, sigma_fluid, 2e-7, pc, exponent)
        else:
            link = expit(q)
            pc_primary = 0.015 + 0.055 * expit(-q)
            pc_secondary = 0.004 + 0.045 * expit(q)
            def smooth_positive(value: float, eps: float = 0.002) -> float:
                return 0.5 * (value + np.sqrt(value * value + eps * eps))
            xp = smooth_positive((primary - pc_primary) / (1.0 - pc_primary))
            xs = smooth_positive((secondary - pc_secondary) / (1.0 - pc_secondary))
            sig_primary = sigma_fluid * xp ** (1.75 + 0.30 * (1.0 - link))
            sig_secondary = sigma_fluid * xs ** (1.35 + 0.65 * link)
            sigma = sig_primary + link * sig_secondary
            sigma += 0.35 * link * np.sqrt(max(sig_primary * sig_secondary, 0.0))
        clay = surface * vcl * (0.30 + saturation) / max(phi, 0.03)
        return max(float(sigma + clay), 1e-10)

    sw_invaded = float(np.clip(sw + invasion * (1.0 - sw), 0.01, 0.999))
    rmf = rw * float(np.clip(p["rmf_ratio"], *RMF_RANGE))
    virgin = intrinsic(sw, rw)
    invaded = intrinsic(sw_invaded, rmf)
    deep = 0.86 * virgin + 0.14 * invaded
    shallow = 0.22 * virgin + 0.78 * invaded
    return 1.0 / np.array([deep, shallow], dtype=float)


def _candidate_vectors(
    p: Mapping[str, float], family: str, coupling: str, panel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = _panel_states(p, panel_size)
    elastic = np.vstack([forward_elastic_family(q, CARBONATE, "dem") for q in states])
    log_rt = np.log(np.vstack([forward_candidate_state(q, family, coupling) for q in states]))
    return elastic, log_rt


def phase11_design(n_panels: int, seed: int) -> list[dict[str, float]]:
    """Centered carbonate design with an orthogonal electrical latent direction."""
    if n_panels < 3:
        raise ValueError("Phase 1.1 requires at least three panels")
    design = carbonate_design(n_panels, seed)
    q_elastic = np.array([elastic_latent(p["aspect"]) for p in design])
    q_elastic -= q_elastic.mean()
    # Preserve the relative aspect ordering while keeping the centered design
    # inside the finite physical aspect coordinate. Centering prevents coupling
    # weights from confusing correlation with a cohort-wide latent offset.
    q_elastic *= min(1.0, 1.90 / max(float(np.max(np.abs(q_elastic))), 1e-12))
    aspect_lo, aspect_hi = RANGES[CARBONATE.name]["aspect"]
    log_center = 0.5 * (np.log(aspect_lo) + np.log(aspect_hi))
    log_scale = np.log(aspect_hi / aspect_lo) / 4.0
    for p, qe in zip(design, q_elastic):
        p["aspect"] = float(np.exp(log_center + qe * log_scale))
    centered = q_elastic
    raw = np.array([p["connectivity_latent"] for p in design], dtype=float)
    raw -= raw.mean()
    raw -= centered * np.dot(raw, centered) / max(np.dot(centered, centered), 1e-12)
    if np.linalg.norm(raw) < 1e-8:
        raw = np.sin(np.linspace(0.0, 2.0 * np.pi, n_panels, endpoint=False))
        raw -= raw.mean()
        raw -= centered * np.dot(raw, centered) / max(np.dot(centered, centered), 1e-12)
    raw *= np.std(centered, ddof=0) / max(np.std(raw, ddof=0), 1e-12)
    for p, qe, qp in zip(design, q_elastic, raw):
        p["q_elastic_design"] = float(qe)
        p["q_perp"] = float(qp)
    return design


def realized_correlation(design: Sequence[Mapping[str, float]], rho: float) -> float:
    qe = np.array([p["q_elastic_design"] for p in design])
    qp = np.array([p["q_perp"] for p in design])
    qr = rho * qe + np.sqrt(max(1.0 - rho * rho, 0.0)) * qp
    return float(np.corrcoef(qe, qr)[0, 1])


def _truth_q(p: Mapping[str, float], truth_case: Phase11TruthCase) -> float:
    if truth_case.truth_electrical == "dual":
        # 0.35 connected secondary porosity, matching the original Phase-1
        # partial-dual truth but now under the common observation operator.
        return float(np.log((0.35 - 0.05) / (0.95 - 0.35)))
    rho = float(truth_case.coupling_rho)
    return float(
        rho * p["q_elastic_design"]
        + np.sqrt(max(1.0 - rho**2, 0.0)) * p["q_perp"]
    )


def simulate_phase11_observations(
    p: Mapping[str, float], truth_case: Phase11TruthCase, panel_size: int,
    elastic_noise: np.ndarray, rt_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one paired observation using externally supplied common noise."""
    truth = dict(p)
    truth["q_conn"] = _truth_q(p, truth_case)
    states = _panel_states(truth, panel_size)
    elastic = np.vstack([forward_elastic_family(q, CARBONATE, "dem") for q in states])
    family = "latent_dual" if truth_case.truth_electrical == "dual" else truth_case.truth_electrical
    log_rt = np.log(np.vstack([
        forward_candidate_state(q, family, "independent") for q in states
    ]))
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    return elastic + elastic_noise * sigma_e, log_rt + rt_noise @ _rt_cholesky().T


def _initial_state(
    fixed: Mapping[str, float], names: Sequence[str],
    elastic_estimate: Mapping[str, float], prior_fit: Mapping[str, float] | None = None,
) -> dict[str, float]:
    state = dict(fixed)
    for name in names:
        lo, hi = _range(name)
        state[name] = 0.5 * (lo + hi)
    state.update({name: elastic_estimate[name] for name in ("phi", "sw", "aspect")})
    cap = min(0.12, 0.72 * state["phi"])
    state["secondary"] = elastic_estimate["secondary"]
    state["secondary_fraction"] = np.clip(
        elastic_estimate["secondary"] / max(cap, 1e-8), 1e-4, 1.0 - 1e-4
    )
    state["rmf_ratio"] = 0.75
    if "q_conn" in names:
        state["q_conn"] = np.clip(elastic_latent(state["aspect"]), *Q_RANGE)
    if prior_fit is not None:
        for name in names:
            if name in prior_fit:
                state[name] = prior_fit[name]
    return state


def _block_press_score(
    residual_data: np.ndarray, jacobian_data: np.ndarray,
    covariance_z: np.ndarray, panel_size: int,
) -> float:
    """Linearized leave-one-depth-block-out predictive chi-square."""
    elastic_count = 3 * panel_size
    score = 0.0
    for depth in range(panel_size):
        indices = np.array([
            3 * depth, 3 * depth + 1, 3 * depth + 2,
            elastic_count + 2 * depth, elastic_count + 2 * depth + 1,
        ])
        jg = jacobian_data[indices]
        rg = residual_data[indices]
        leverage = 0.5 * (jg @ covariance_z @ jg.T)
        leverage = leverage + leverage.T
        # Local nonlinear fits can put a block leverage arbitrarily close to
        # one, making raw PRESS numerically explosive.  Regularized influence
        # clipping is explicit and common to every candidate.
        evals, evecs = np.linalg.eigh(leverage)
        leverage = (evecs * np.clip(evals, 0.0, 0.85)) @ evecs.T
        deleted = np.linalg.pinv(np.eye(len(indices)) - leverage, rcond=1e-8) @ rg
        score += float(np.dot(deleted, deleted))
    return min(score, 1e8)


def fit_candidate(
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    fixed: Mapping[str, float],
    family: str,
    coupling: str,
    panel_size: int,
    elastic_estimate: Mapping[str, float],
    prior_fit: Mapping[str, float] | None = None,
) -> dict:
    """Fit one constitutive-family/coupling candidate."""
    names = _candidate_names(family, coupling)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    rt_whitener = np.linalg.inv(_rt_cholesky())
    n_data = 5 * panel_size

    def residual(z: np.ndarray) -> np.ndarray:
        p = _unpack(z, names, fixed)
        pred_e, pred_r = _candidate_vectors(p, family, coupling, panel_size)
        data = [((pred_e - observed_elastic) / sigma_e).ravel()]
        data.append(np.hstack([rt_whitener @ row for row in pred_r - observed_log_rt]))
        pieces = data
        common_nuisance = [name for name in NUISANCE if name in names]
        if common_nuisance:
            lo, hi = _bounds(names)
            positions = [names.index(name) for name in common_nuisance]
            pieces.append(0.10 * (z[positions] - 0.5))
        if coupling == "partial":
            pieces.append(np.array([
                (p["q_conn"] - elastic_latent(p["aspect"])) / PARTIAL_TAU
            ]))
        elif coupling == "independent":
            pieces.append(np.array([p["q_conn"]]))
        return np.concatenate(pieces)

    initial = _initial_state(fixed, names, elastic_estimate, prior_fit)
    attempts = []

    def solve(start, max_nfev=75):
        return least_squares(
            residual, start,
            bounds=(np.zeros(len(names)), np.ones(len(names))), method="trf",
            x_scale="jac", max_nfev=max_nfev,
            ftol=3e-7, xtol=3e-7, gtol=3e-7,
        )

    fit = solve(_pack(initial, names))
    attempts.append(fit)
    first_data_cost = float(np.dot(fit.fun[:n_data], fit.fun[:n_data]))
    first_dof = max(n_data - len(names), 1)
    critical_positions = [
        names.index(name) for name in
        ("phi", "sw", "aspect", "secondary_fraction", "q_conn")
        if name in names
    ]
    critical_bound = np.any(
        (fit.x[critical_positions] <= 1e-4)
        | (fit.x[critical_positions] >= 1.0 - 1e-4)
    )
    if not fit.success or first_data_cost / first_dof > 2.5 or critical_bound:
        alternate = np.where(np.arange(len(names)) % 2, 0.68, 0.32)
        second = solve(alternate)
        attempts.append(second)
        if float(np.dot(second.fun, second.fun)) < float(np.dot(fit.fun, fit.fun)):
            fit = second
    convergence_fallback = False
    if not fit.success:
        # A non-converged candidate must never receive material predictive
        # weight. Continue from the best current state before scoring it.
        final = solve(fit.x, max_nfev=500)
        attempts.append(final)
        convergence_fallback = True
        if final.success or float(np.dot(final.fun, final.fun)) <= float(
            np.dot(fit.fun, fit.fun)
        ):
            fit = final
    p_hat = _unpack(fit.x, names, fixed)
    jtj = fit.jac.T @ fit.jac
    evals, evecs = np.linalg.eigh(0.5 * (jtj + jtj.T))
    floor = max(1e-8, 1e-10 * max(float(evals[-1]), 1.0))
    covariance_z = (evecs * (1.0 / np.maximum(evals, floor))) @ evecs.T
    lo, hi = _bounds(names)
    physical_covariance = covariance_z * (hi - lo)[:, None] * (hi - lo)[None, :]
    sd = {name: float(value) for name, value in zip(
        names, np.sqrt(np.maximum(np.diag(physical_covariance), 0.0))
    )}
    # Delta-method secondary-porosity uncertainty for the constrained
    # reparameterization secondary = fraction * min(0.12, 0.72 phi).
    gradient = np.zeros(len(names))
    phi_position = names.index("phi")
    fraction_position = names.index("secondary_fraction")
    cap = min(0.12, 0.72 * p_hat["phi"])
    gradient[fraction_position] = cap
    if 0.72 * p_hat["phi"] < 0.12:
        gradient[phi_position] = 0.72 * p_hat["secondary_fraction"]
    secondary_variance = float(gradient @ physical_covariance @ gradient)
    sd["secondary"] = float(np.sqrt(max(secondary_variance, 0.0)))
    data_residual = fit.fun[:n_data]
    data_jacobian = fit.jac[:n_data]
    data_cost = float(np.dot(data_residual, data_residual))
    dof = max(n_data - len(names), 1)
    predictive_score = _block_press_score(
        data_residual, data_jacobian, covariance_z, panel_size
    )
    return {
        "family": family,
        "coupling": coupling,
        "names": names,
        "estimate": p_hat,
        "sd": sd,
        "success": bool(fit.success),
        "data_cost": data_cost,
        "reduced_chi2": data_cost / dof,
        "predictive_score": predictive_score,
        "nfev": int(fit.nfev),
        "total_nfev": int(sum(attempt.nfev for attempt in attempts)),
        "adaptive_restart": len(attempts) >= 2,
        "convergence_fallback": convergence_fallback,
        "bound_hit": {
            name: bool(value) for name, value in zip(
                names, (fit.x <= 1e-4) | (fit.x >= 1.0 - 1e-4)
            )
        },
    }


def predictive_weights(fits: Sequence[Mapping], selector=None) -> np.ndarray:
    """Pseudo-BMA weights from blocked predictive scores."""
    active = [i for i, fit in enumerate(fits) if selector is None or selector(fit)]
    if not active:
        raise ValueError("No candidate models selected")
    scores = np.array([fits[i]["predictive_score"] for i in active], dtype=float)
    delta = np.clip(scores - np.min(scores), 0.0, 1400.0)
    raw = np.exp(-0.5 * delta)
    raw /= raw.sum()
    weights = np.zeros(len(fits), dtype=float)
    weights[active] = raw
    return weights


def converged_weights(
    fits: Sequence[Mapping], weights: np.ndarray,
) -> np.ndarray:
    """Zero non-converged candidates and renormalize for one observation."""
    adjusted = np.asarray(weights, dtype=float).copy()
    adjusted *= np.array([bool(fit["success"]) for fit in fits], dtype=float)
    total = float(adjusted.sum())
    if total <= 0.0:
        raise RuntimeError("No converged candidate remains for this observation")
    return adjusted / total


def _truncated_component_moments(
    means: np.ndarray, sds: np.ndarray, lower: float, upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    sds = np.maximum(sds, 1e-10)
    alpha = (lower - means) / sds
    beta = (upper - means) / sds
    z = np.maximum(norm.cdf(beta) - norm.cdf(alpha), 1e-14)
    delta = (norm.pdf(alpha) - norm.pdf(beta)) / z
    truncated_mean = means + sds * delta
    truncated_variance = sds**2 * (
        1.0 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / z - delta**2
    )
    return truncated_mean, np.maximum(truncated_variance, 1e-14)


def _mixture_quantile(
    means: np.ndarray, sds: np.ndarray, weights: np.ndarray, q: float,
    lower: float, upper: float,
) -> float:
    sds = np.maximum(sds, 1e-10)
    alpha = (lower - means) / sds
    beta = (upper - means) / sds
    normalization = np.maximum(norm.cdf(beta) - norm.cdf(alpha), 1e-14)

    def equation(value: float) -> float:
        component = (
            norm.cdf((value - means) / sds) - norm.cdf(alpha)
        ) / normalization
        return float(np.sum(weights * component) - q)

    return float(brentq(equation, lower, upper, xtol=1e-10))


def mixture_recovery(
    fits: Sequence[Mapping], weights: np.ndarray, target: str,
) -> dict[str, float]:
    means = np.array([fit["estimate"][target] for fit in fits], dtype=float)
    sds = np.array([fit["sd"][target] for fit in fits], dtype=float)
    physical_lower, physical_upper = _range(target)
    truncated_means, truncated_variances = _truncated_component_moments(
        means, sds, physical_lower, physical_upper
    )
    mean = float(np.dot(weights, truncated_means))
    variance = float(np.dot(
        weights, truncated_variances + truncated_means**2
    ) - mean**2)
    variance = max(variance, 1e-12)
    lower = _mixture_quantile(
        means, sds, weights, 0.05, physical_lower, physical_upper
    )
    upper = _mixture_quantile(
        means, sds, weights, 0.95, physical_lower, physical_upper
    )
    return {
        "estimate": mean,
        "sd": float(np.sqrt(variance)),
        "lower90": lower,
        "upper90": upper,
        "interval_width90": upper - lower,
        "weighted_reduced_chi2": float(np.dot(
            weights, [fit["reduced_chi2"] for fit in fits]
        )),
        "success_weight": float(np.dot(weights, [fit["success"] for fit in fits])),
        "effective_models": float(1.0 / np.sum(weights**2)),
        "max_weight": float(np.max(weights)),
    }


def _stable_seed(*parts: str) -> int:
    text = "|".join(parts)
    return 9109 + sum((index + 1) * ord(char) for index, char in enumerate(text))


def _cluster_mean_interval(
    group: pd.DataFrame, column: str, transform=lambda x: x,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    panels = sorted(group["panel"].unique())
    by_panel = np.array([
        float(np.mean(transform(group.loc[group["panel"] == panel, column].to_numpy())))
        for panel in panels
    ])
    rng = np.random.default_rng(_stable_seed(str(group["case"].iloc[0]), column))
    draws = rng.integers(0, len(panels), size=(n_bootstrap, len(panels)))
    bootstrap = by_panel[draws].mean(axis=1)
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def summarize_strategies(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target, strategy), group in raw.groupby(["case", "target", "strategy"]):
        error = group["estimate"] - group["truth"]
        covered = group["covered90"].astype(bool)
        lo, hi = _cluster_mean_interval(group, "covered90")
        rows.append({
            "case": case, "target": target, "strategy": strategy, "n": len(group),
            "bias": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "coverage90": float(covered.mean()),
            "coverage90_cluster_low": lo,
            "coverage90_cluster_high": hi,
            "median_interval_width": float(group["interval_width90"].median()),
            "median_reduced_chi2": float(group["weighted_reduced_chi2"].median()),
            "mean_effective_models": float(group["effective_models"].mean()),
            "mean_max_weight": float(group["max_weight"].mean()),
            "success_fraction": float(group["success_weight"].mean()),
        })
    return pd.DataFrame(rows)


def build_phase11_gates(raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target), group in summary.groupby(["case", "target"]):
        elastic = group[group["strategy"] == "elastic"].iloc[0]
        for strategy in STRATEGIES[1:]:
            current = group[group["strategy"] == strategy].iloc[0]
            subset = raw[
                (raw["case"] == case) & (raw["target"] == target)
                & (raw["strategy"].isin(["elastic", strategy]))
            ]
            paired = subset.pivot(index=["panel", "replicate"], columns="strategy",
                    values=["interval_width90", "covered90", "estimate", "truth"])
            narrower = paired[("interval_width90", strategy)] <= 0.8 * paired[("interval_width90", "elastic")]
            false_conf = narrower & ~paired[("covered90", strategy)].astype(bool)
            false_fraction = float(false_conf.mean())
            harmful_false = false_conf & paired[("covered90", "elastic")].astype(bool)
            rmse_ratio = float(current["rmse"] / max(elastic["rmse"], 1e-12))
            adequate = float(current["median_reduced_chi2"])
            panels = sorted(subset["panel"].unique())
            rng = np.random.default_rng(_stable_seed(case, target, strategy))
            ratio_boot, false_boot, harmful_boot = [], [], []
            for _ in range(1000):
                selected = rng.choice(panels, size=len(panels), replace=True)
                indices = np.concatenate([
                    np.flatnonzero(paired.index.get_level_values("panel") == panel)
                    for panel in selected
                ])
                err_joint = (
                    paired[("estimate", strategy)].to_numpy()[indices]
                    - paired[("truth", strategy)].to_numpy()[indices]
                )
                err_elastic = (
                    paired[("estimate", "elastic")].to_numpy()[indices]
                    - paired[("truth", "elastic")].to_numpy()[indices]
                )
                ratio_boot.append(np.sqrt(np.mean(err_joint**2)) / max(np.sqrt(np.mean(err_elastic**2)), 1e-12))
                false_boot.append(float(false_conf.to_numpy()[indices].mean()))
                harmful_boot.append(float(harmful_false.to_numpy()[indices].mean()))
            ratio_upper = float(np.quantile(ratio_boot, 0.975))
            false_upper = float(np.quantile(false_boot, 0.975))
            harmful_upper = float(np.quantile(harmful_boot, 0.975))
            truth_by_panel = subset[subset["strategy"] == strategy].groupby("panel")["truth"].first()
            design_sd = max(float(truth_by_panel.std(ddof=0)), 1e-12)
            standardized_bias = abs(float(current["bias"])) / design_sd
            width_ratio = float(
                np.median(
                    paired[("interval_width90", strategy)]
                    / paired[("interval_width90", "elastic")]
                )
            )
            if (
                0.85 <= current["coverage90"] <= 0.97
                and current["coverage90_cluster_low"] >= 0.80
                and rmse_ratio <= 0.90 and ratio_upper < 1.0
                and false_upper <= 0.10 and current["success_fraction"] >= 0.95
                and adequate <= 1.50 and standardized_bias <= 0.10
                and width_ratio <= 0.80
            ):
                verdict = "PASS"
            elif (
                current["coverage90"] >= 0.80
                and current["coverage90_cluster_low"] >= 0.70
                and rmse_ratio <= 1.05 and false_upper <= 0.20
                and adequate <= 2.00 and standardized_bias <= 0.20
                # A one-percent tolerance prevents floating-point equality at
                # the elastic-width boundary from creating a categorical FAIL.
                and width_ratio <= 1.01
            ):
                verdict = "CONDITIONAL"
            else:
                verdict = "FAIL"
            rows.append({
                "case": case, "target": target, "strategy": strategy,
                "verdict": verdict,
                "coverage90": float(current["coverage90"]),
                "coverage90_cluster_low": float(current["coverage90_cluster_low"]),
                "rmse_ratio_over_elastic": rmse_ratio,
                "rmse_ratio_cluster_upper": ratio_upper,
                "false_confidence_fraction": false_fraction,
                "false_confidence_cluster_upper": false_upper,
                "harmful_false_confidence_fraction": float(harmful_false.mean()),
                "harmful_false_confidence_cluster_upper": harmful_upper,
                "standardized_bias": standardized_bias,
                "median_width_ratio": width_ratio,
                "median_reduced_chi2": adequate,
                "success_fraction": float(current["success_fraction"]),
                "mean_effective_models": float(current["mean_effective_models"]),
            })
    return pd.DataFrame(rows)


def summarize_weights(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = (
        components.groupby(["case", "panel", "replicate", "family"], as_index=False)["full_weight"].sum()
        .groupby(["case", "family"], as_index=False)["full_weight"].mean()
        .rename(columns={"full_weight": "mean_weight"})
    )
    coupling = (
        components.groupby(["case", "panel", "replicate", "coupling"], as_index=False)["full_weight"].sum()
        .groupby(["case", "coupling"], as_index=False)["full_weight"].mean()
        .rename(columns={"full_weight": "mean_weight"})
    )
    return family, coupling


def _crossfit_weights(
    fit_bank: Mapping[tuple[int, int], Sequence[Mapping]],
    heldout_panel: int,
    selector=None,
) -> np.ndarray:
    """Panel-cluster cross-fitted weights; held-out panel never sets its weights."""
    training = [fits for (panel, _), fits in fit_bank.items() if panel != heldout_panel]
    if not training:
        raise ValueError("Cross-fitted weights require at least two panels")
    n_candidates = len(training[0])
    templates = []
    for index in range(n_candidates):
        template = dict(training[0][index])
        template["predictive_score"] = float(np.mean([
            fits[index]["predictive_score"] for fits in training
        ]))
        templates.append(template)
    return predictive_weights(templates, selector=selector)


def run_phase11(
    n_panels: int = 12,
    replicates: int = 4,
    panel_size: int = 8,
    seed: int = 20260829,
    truth_cases: Sequence[Phase11TruthCase] = TRUTH_CASES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase-1.1 repeated-noise coupling/model-averaging benchmark."""
    design = phase11_design(n_panels, seed)
    # Common random numbers make every truth/coupling/model comparison paired.
    noise_bank = {}
    for panel_index in range(n_panels):
        for replicate in range(replicates):
            noise_rng = np.random.default_rng(seed + 10007 * panel_index + 101 * replicate)
            noise_bank[(panel_index, replicate)] = (
                noise_rng.normal(size=(panel_size, 3)),
                noise_rng.normal(size=(panel_size, 2)),
            )
    recovery_rows: list[dict] = []
    component_rows: list[dict] = []
    for truth_case in truth_cases:
        fit_bank: dict[tuple[int, int], list[dict]] = {}
        elastic_bank: dict[tuple[int, int], dict] = {}
        for panel_index, truth in enumerate(design):
            fixed = dict(truth)
            for name in PHASE1_FIT_NAMES:
                fixed.pop(name, None)
            fixed.update({
                "vcl": truth["vcl"], "cement": truth["cement"],
                "coord": truth["coord"], "phi_e": truth["phi_e"],
            })
            for replicate in range(replicates):
                observed_e, observed_r = simulate_phase11_observations(
                    truth, truth_case, panel_size, *noise_bank[(panel_index, replicate)]
                )
                from .phase1 import fit_panel  # avoids a module-level circular API surface
                fit_rng = np.random.default_rng(seed + 50021 * panel_index + 307 * replicate)
                elastic = fit_panel(
                    observed_e, observed_r, fixed, "dual_porosity", False,
                    panel_size, fit_rng, n_starts=1,
                )
                fits = []
                for family in CANDIDATE_FAMILIES:
                    prior = None
                    for coupling in COUPLING_MODES:
                        fitted = fit_candidate(
                            observed_e, observed_r, fixed, family, coupling,
                            panel_size, elastic["estimate"], prior,
                        )
                        fits.append(fitted)
                        prior = fitted["estimate"]
                fit_bank[(panel_index, replicate)] = fits
                elastic_bank[(panel_index, replicate)] = elastic

        panel_weights = {}
        for panel_index in range(n_panels):
            panel_weights[(panel_index, "model_average")] = _crossfit_weights(
                fit_bank, panel_index
            )
            for coupling in COUPLING_MODES:
                panel_weights[(panel_index, coupling)] = _crossfit_weights(
                    fit_bank, panel_index,
                    selector=lambda fit, mode=coupling: fit["coupling"] == mode,
                )

        for panel_index, truth in enumerate(design):
            for replicate in range(replicates):
                fits = fit_bank[(panel_index, replicate)]
                elastic = elastic_bank[(panel_index, replicate)]
                full_weights = converged_weights(
                    fits, panel_weights[(panel_index, "model_average")]
                )
                for fit, weight in zip(fits, full_weights):
                    bound_hit = fit["bound_hit"]
                    critical_names = (
                        "phi", "sw", "aspect", "secondary_fraction", "q_conn"
                    )
                    recorded_names = (
                        "phi", "sw", "aspect", "secondary", "secondary_fraction",
                        *NUISANCE, "rmf_ratio", "q_conn",
                    )
                    component_rows.append({
                        "case": truth_case.name, "truth_family": truth_case.truth_electrical,
                        "truth_rho": truth_case.coupling_rho, "panel": panel_index,
                        "replicate": replicate, "family": fit["family"],
                        "coupling": fit["coupling"], "predictive_score": fit["predictive_score"],
                        "full_weight": float(weight), "reduced_chi2": fit["reduced_chi2"],
                        "success": fit["success"], "nfev": fit["nfev"],
                        "total_nfev": fit["total_nfev"],
                        "adaptive_restart": fit["adaptive_restart"],
                        "convergence_fallback": fit["convergence_fallback"],
                        "elastic_success": elastic["success"],
                        "elastic_nfev": elastic["nfev"],
                        "elastic_total_nfev": elastic["total_nfev"],
                        "elastic_convergence_fallback": elastic[
                            "convergence_fallback"
                        ],
                        "any_bound_hit": any(bound_hit.values()),
                        "critical_bound_hit": any(
                            bound_hit.get(name, False) for name in critical_names
                        ),
                        "bound_hit_phi": bound_hit.get("phi", False),
                        "bound_hit_sw": bound_hit.get("sw", False),
                        "bound_hit_aspect": bound_hit.get("aspect", False),
                        "bound_hit_secondary_fraction": bound_hit.get(
                            "secondary_fraction", False
                        ),
                        "bound_hit_q_conn": bound_hit.get("q_conn", False),
                        "nuisance_bound_hit": any(
                            value for name, value in bound_hit.items()
                            if name not in critical_names
                        ),
                        **{
                            f"bound_hit_{name}": bound_hit.get(name, False)
                            for name in (*NUISANCE, "rmf_ratio")
                        },
                        **{
                            f"estimate_{name}": fit["estimate"].get(name, np.nan)
                            for name in recorded_names
                        },
                        **{
                            f"sd_{name}": fit["sd"].get(name, np.nan)
                            for name in recorded_names
                        },
                        "q_hat": fit["estimate"].get(
                            "q_conn", elastic_latent(fit["estimate"]["aspect"])
                        ),
                        "q_truth": _truth_q(truth, truth_case),
                    })

                strategy_weights = {
                    coupling: converged_weights(
                        fits, panel_weights[(panel_index, coupling)]
                    )
                    for coupling in COUPLING_MODES
                }
                strategy_weights["model_average"] = full_weights

                for target in TARGETS:
                    truth_value = truth[target]
                    elastic_component = {
                        "estimate": elastic["estimate"], "sd": elastic["sd"],
                        "reduced_chi2": elastic["cost"] / max(3 * panel_size - len(TARGETS), 1),
                        "success": elastic["success"],
                    }
                    elastic_result = mixture_recovery(
                        [elastic_component], np.array([1.0]), target
                    )
                    recovery_rows.append({
                        "case": truth_case.name, "truth_family": truth_case.truth_electrical,
                        "truth_rho": truth_case.coupling_rho, "panel": panel_index,
                        "replicate": replicate, "strategy": "elastic", "target": target,
                        "truth": truth_value, "estimate": elastic_result["estimate"],
                        "sd": elastic_result["sd"], "lower90": elastic_result["lower90"],
                        "upper90": elastic_result["upper90"],
                        "interval_width90": elastic_result["interval_width90"],
                        "covered90": elastic_result["lower90"] <= truth_value <= elastic_result["upper90"],
                        "weighted_reduced_chi2": elastic_result["weighted_reduced_chi2"],
                        "success_weight": float(elastic["success"]),
                        "effective_models": 1.0, "max_weight": 1.0,
                    })
                    for strategy, weights in strategy_weights.items():
                        result = mixture_recovery(fits, weights, target)
                        recovery_rows.append({
                            "case": truth_case.name, "truth_family": truth_case.truth_electrical,
                            "truth_rho": truth_case.coupling_rho, "panel": panel_index,
                            "replicate": replicate, "strategy": strategy, "target": target,
                            "truth": truth_value, "estimate": result["estimate"],
                            "sd": result["sd"], "lower90": result["lower90"],
                            "upper90": result["upper90"],
                            "interval_width90": result["interval_width90"],
                            "covered90": result["lower90"] <= truth_value <= result["upper90"],
                            "weighted_reduced_chi2": result["weighted_reduced_chi2"],
                            "success_weight": result["success_weight"],
                            "effective_models": result["effective_models"],
                            "max_weight": result["max_weight"],
                        })

    raw = pd.DataFrame(recovery_rows)
    components = pd.DataFrame(component_rows)
    summary = summarize_strategies(raw)
    gates = build_phase11_gates(raw, summary)
    family_weights, coupling_weights = summarize_weights(components)
    return raw, components, summary, gates, family_weights, coupling_weights
