"""Phase 1: nonlinear carbonate stress test with alternative electrical truths.

This module deliberately separates the elastic pore-shape latent variable from
the electrical connectivity latent variable.  The latter can be correlated
with the former by a user-controlled coefficient, but equality is never
assumed.  Repeated-noise nonlinear inversions quantify empirical coverage,
bias, RMSE, boundary hits, and false confidence.

The models are transparent research surrogates, not reservoir-calibrated laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import norm, qmc

from .analysis import DATA_SIGMA
from .families import forward_elastic_family, forward_electrical_family
from .models import SCENARIOS, Scenario
from .phase05 import RANGES, RT_CORRELATION, _panel_states


CARBONATE = SCENARIOS["multimodal_carbonate"]
TARGETS = ("phi", "sw", "aspect", "secondary")
NUISANCE = ("log_rw", "archie_m", "archie_n", "surface_cond", "invasion")
FIT_NAMES = TARGETS + NUISANCE


@dataclass(frozen=True)
class CarbonateCase:
    name: str
    truth_electrical: str
    inverse_electrical: str
    coupling_rho: float


CASES = (
    CarbonateCase("matched_dual", "dual", "dual_porosity", 1.0),
    CarbonateCase("ema_rho0_dual", "ema", "dual_porosity", 0.0),
    CarbonateCase("ema_rho05_dual", "ema", "dual_porosity", 0.5),
    CarbonateCase("ema_rho09_dual", "ema", "dual_porosity", 0.9),
    CarbonateCase("network_rho05_dual", "network", "dual_porosity", 0.5),
    CarbonateCase("network_rho05_archie", "network", "archie", 0.5),
)


def _electrical_latent(p: Mapping[str, float], rho: float) -> float:
    """Correlated-standard-normal electrical connectivity latent variable."""
    lo, hi = RANGES[CARBONATE.name]["aspect"]
    u = (np.log(p["aspect"]) - np.log(lo)) / (np.log(hi) - np.log(lo))
    q_elastic = norm.ppf(np.clip(u, 1e-4, 1.0 - 1e-4))
    q_independent = float(p.get("connectivity_latent", 0.0))
    rho = float(np.clip(rho, 0.0, 1.0))
    return rho * q_elastic + np.sqrt(max(1.0 - rho * rho, 0.0)) * q_independent


def gem_conductivity(
    conductive_fraction: float,
    sigma_fluid: float,
    sigma_solid: float,
    percolation_threshold: float,
    exponent: float,
) -> float:
    """Two-phase General Effective Medium conductivity.

    The root is solved in ``sigma**(1/t)``.  ``A=(1-pc)/pc`` places the
    conductor-insulator transition at the requested percolation threshold.
    """
    f = float(np.clip(conductive_fraction, 1e-7, 1.0 - 1e-7))
    sf = max(float(sigma_fluid), 1e-12) ** (1.0 / exponent)
    ss = max(float(sigma_solid), 1e-12) ** (1.0 / exponent)
    a = (1.0 - percolation_threshold) / percolation_threshold

    # Expanding the two-phase GEM equation gives
    # A*x² - B*x - ss*sf = 0.  The positive root is exact and much faster than
    # repeated scalar root finding inside nonlinear inversion.
    b = (1.0 - f) * (a * ss - sf) + f * (a * sf - ss)
    x = (b + np.sqrt(b * b + 4.0 * a * ss * sf)) / (2.0 * a)
    return float(x**exponent)


def _percolating_branch(
    pore_fraction: float, threshold: float, exponent: float, sigma_fluid: float,
) -> float:
    excess = max((pore_fraction - threshold) / (1.0 - threshold), 0.0)
    return float(sigma_fluid * excess**exponent)


def forward_electrical_truth(
    p: Mapping[str, float], truth_family: str, rho: float,
) -> np.ndarray:
    """Alternative deep/shallow carbonate electrical truth generators."""
    if truth_family == "dual":
        return forward_electrical_family(p, CARBONATE, "dual_porosity", "partial")
    if truth_family not in ("ema", "network"):
        raise ValueError(f"Unknown truth family: {truth_family}")

    phi = float(p["phi"])
    secondary = float(np.clip(p["secondary"], 0.0, 0.72 * phi))
    primary = max(phi - secondary, 1e-5)
    sw = float(np.clip(p["sw"], 0.01, 0.999))
    rw = float(np.exp(p["log_rw"]))
    n = float(p["archie_n"])
    vcl = float(p["vcl"])
    surface = max(float(p["surface_cond"]), 0.0)
    invasion = float(np.clip(p["invasion"], 0.0, 0.65))
    q_conn = _electrical_latent(p, rho)

    def intrinsic(saturation: float, water_resistivity: float) -> float:
        sigma_fluid = saturation**n / max(water_resistivity, 1e-8)
        if truth_family == "ema":
            # Connectivity is not a deterministic transform of secondary
            # porosity or elastic aspect ratio.  It enters through a latent
            # percolation threshold and transport exponent.
            pc = 0.025 + 0.19 * expit(-q_conn)
            exponent = 1.55 + 0.85 * expit(q_conn)
            sigma = gem_conductivity(phi, sigma_fluid, 2e-7, pc, exponent)
        else:
            # Two subnetworks connected in parallel through a stochastic
            # bottleneck.  The macropore branch may remain isolated even when
            # its elastic signature is strong.
            link = expit(q_conn)
            pc_primary = 0.015 + 0.055 * expit(-q_conn)
            pc_secondary = 0.004 + 0.045 * expit(q_conn)
            sig_primary = _percolating_branch(
                primary, pc_primary, 1.75 + 0.30 * (1.0 - link), sigma_fluid
            )
            sig_secondary = _percolating_branch(
                secondary, pc_secondary, 1.35 + 0.65 * link, sigma_fluid
            )
            cross = 0.35 * link * np.sqrt(max(sig_primary * sig_secondary, 0.0))
            sigma = sig_primary + link * sig_secondary + cross
        clay_surface = surface * vcl * (0.30 + saturation) / max(phi, 0.03)
        return max(sigma + clay_surface, 1e-10)

    sw_invaded = float(np.clip(sw + invasion * (1.0 - sw), 0.01, 0.999))
    rmf_ratio = float(np.clip(p.get("rmf_ratio", 0.65), 0.25, 1.75))
    virgin = intrinsic(sw, rw)
    invaded = intrinsic(sw_invaded, rw * rmf_ratio)
    # Finite tool volumes: neither measurement is a pure zone response.
    deep = 0.86 * virgin + 0.14 * invaded
    shallow = 0.22 * virgin + 0.78 * invaded
    return 1.0 / np.array([deep, shallow], dtype=float)


def carbonate_design(n_panels: int, seed: int) -> list[dict[str, float]]:
    """Stratified carbonate panel design with independent connectivity latent."""
    ranges = RANGES[CARBONATE.name]
    names = list(ranges)
    sampler = qmc.LatinHypercube(d=len(names) + 2, seed=seed)
    unit = sampler.random(n_panels)
    lower = np.array([ranges[name][0] for name in names])
    upper = np.array([ranges[name][1] for name in names])
    values = qmc.scale(unit[:, :len(names)], lower, upper)
    design = []
    for i, row in enumerate(values):
        p = {name: float(value) for name, value in zip(names, row)}
        p["secondary"] = min(p["secondary"], 0.72 * p["phi"])
        p["connectivity_latent"] = float(norm.ppf(np.clip(unit[i, -2], 1e-4, 1 - 1e-4)))
        p["rmf_ratio"] = float(0.45 + 0.65 * unit[i, -1])
        design.append(p)
    return design


def _truth_vectors(
    p: Mapping[str, float], case: CarbonateCase, panel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = _panel_states(p, panel_size)
    elastic = np.vstack([forward_elastic_family(q, CARBONATE, "dem") for q in states])
    electrical = np.vstack([
        forward_electrical_truth(q, case.truth_electrical, case.coupling_rho)
        for q in states
    ])
    return elastic, np.log(electrical)


def _inverse_vectors(
    p: Mapping[str, float], inverse_family: str, panel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = _panel_states(p, panel_size)
    elastic = np.vstack([forward_elastic_family(q, CARBONATE, "dem") for q in states])
    coupling = "partial" if inverse_family == "dual_porosity" else "rigid"
    electrical = np.vstack([
        forward_electrical_family(q, CARBONATE, inverse_family, coupling)
        for q in states
    ])
    return elastic, np.log(electrical)


def _rt_cholesky() -> np.ndarray:
    covariance = DATA_SIGMA["rt_log"] ** 2 * np.array(
        [[1.0, RT_CORRELATION], [RT_CORRELATION, 1.0]]
    )
    return np.linalg.cholesky(covariance)


def simulate_observations(
    p: Mapping[str, float], case: CarbonateCase, panel_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    elastic, log_rt = _truth_vectors(p, case, panel_size)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    noisy_elastic = elastic + rng.normal(size=elastic.shape) * sigma_e
    noisy_log_rt = log_rt + rng.normal(size=log_rt.shape) @ _rt_cholesky().T
    return noisy_elastic, noisy_log_rt


def _bounds(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    ranges = RANGES[CARBONATE.name]
    lo = np.array([ranges[name][0] for name in names], dtype=float)
    hi = np.array([ranges[name][1] for name in names], dtype=float)
    return lo, hi


def _unpack(z: np.ndarray, names: Sequence[str], fixed: Mapping[str, float]) -> dict[str, float]:
    lo, hi = _bounds(names)
    p = dict(fixed)
    p.update({name: float(value) for name, value in zip(names, lo + z * (hi - lo))})
    return p


def fit_panel(
    observed_elastic: np.ndarray,
    observed_log_rt: np.ndarray,
    fixed: Mapping[str, float],
    inverse_family: str,
    joint: bool,
    panel_size: int,
    rng: np.random.Generator,
    n_starts: int = 1,
) -> dict:
    """Bounded nonlinear fit and local-Wald covariance for one noisy panel."""
    names = FIT_NAMES if joint else TARGETS
    lo, hi = _bounds(names)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    rt_whitener = np.linalg.inv(_rt_cholesky())

    def residual(z: np.ndarray) -> np.ndarray:
        p = _unpack(z, names, fixed)
        if joint:
            pred_e, pred_r = _inverse_vectors(p, inverse_family, panel_size)
        else:
            pred_e = np.vstack([
                forward_elastic_family(q, CARBONATE, "dem")
                for q in _panel_states(p, panel_size)
            ])
        pieces = [((pred_e - observed_elastic) / sigma_e).ravel()]
        if joint:
            pieces.append(np.hstack([rt_whitener @ row for row in pred_r - observed_log_rt]))
            # Extremely broad centering prior stabilizes numerical flatness but
            # contributes less than one Fisher unit across each full range.
            pieces.append(0.10 * (z[len(TARGETS):] - 0.5))
        # Smooth-enough physical feasibility penalty for secondary <= 0.72 phi.
        pieces.append(np.array([max(p["secondary"] - 0.72 * p["phi"], 0.0) / 0.006]))
        return np.concatenate(pieces)

    starts = [np.full(len(names), 0.5)]
    starts.extend(rng.uniform(0.15, 0.85, size=len(names)) for _ in range(n_starts - 1))
    fits = [
        least_squares(
            residual, start, bounds=(np.zeros(len(names)), np.ones(len(names))),
            method="trf", x_scale="jac", max_nfev=90, ftol=2e-7, xtol=2e-7,
            gtol=2e-7,
        )
        for start in starts
    ]
    fit = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    convergence_fallback = False
    if not fit.success:
        final = least_squares(
            residual, fit.x,
            bounds=(np.zeros(len(names)), np.ones(len(names))),
            method="trf", x_scale="jac", max_nfev=500,
            ftol=2e-7, xtol=2e-7, gtol=2e-7,
        )
        fits.append(final)
        convergence_fallback = True
        if final.success or float(np.dot(final.fun, final.fun)) <= float(
            np.dot(fit.fun, fit.fun)
        ):
            fit = final
    p_hat = _unpack(fit.x, names, fixed)

    jtj = fit.jac.T @ fit.jac
    evals, evecs = np.linalg.eigh(0.5 * (jtj + jtj.T))
    # A nearly null direction must yield a wide interval, not zero variance as
    # it would under a Moore-Penrose inverse.
    floor = max(1e-8, 1e-10 * max(float(evals[-1]), 1.0))
    cov_z = (evecs * (1.0 / np.maximum(evals, floor))) @ evecs.T
    scale = hi - lo
    covariance = cov_z * scale[:, None] * scale[None, :]
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    active = (fit.x <= 1e-4) | (fit.x >= 1.0 - 1e-4)
    return {
        "estimate": p_hat,
        "sd": {name: float(value) for name, value in zip(names, sd)},
        "success": bool(fit.success),
        "cost": float(2.0 * fit.cost),
        "bound_hit": {name: bool(value) for name, value in zip(names, active)},
        "nfev": int(fit.nfev),
        "total_nfev": int(sum(result.nfev for result in fits)),
        "convergence_fallback": convergence_fallback,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    phat = successes / total
    den = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / den
    half = z / den * np.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total**2))
    return float(center - half), float(center + half)


def summarize_recovery(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target, mode), group in raw.groupby(["case", "target", "mode"]):
        errors = group["estimate"] - group["truth"]
        covered = group["covered90"].astype(bool)
        lo_cov, hi_cov = _wilson_interval(int(covered.sum()), len(group))
        rows.append({
            "case": case,
            "target": target,
            "mode": mode,
            "n": len(group),
            "bias": float(errors.mean()),
            "median_bias": float(errors.median()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "coverage90": float(covered.mean()),
            "coverage90_wilson_low": lo_cov,
            "coverage90_wilson_high": hi_cov,
            "median_interval_width": float(group["interval_width90"].median()),
            "bound_hit_fraction": float(group["bound_hit"].mean()),
            "success_fraction": float(group["success"].mean()),
            "median_nfev": float(group["nfev"].median()),
            "median_cost": float(group["cost"].median()),
            "median_reduced_chi2": float(
                (group["cost"] / group["degrees_freedom"]).median()
            ),
        })
    return pd.DataFrame(rows)


def build_phase1_gates(raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target), group in summary.groupby(["case", "target"]):
        elastic = group[group["mode"] == "elastic"].iloc[0]
        joint = group[group["mode"] == "joint"].iloc[0]
        paired = raw[(raw["case"] == case) & (raw["target"] == target)].pivot(
            index=["panel", "replicate"], columns="mode",
            values=["interval_width90", "covered90"],
        )
        narrower = paired[("interval_width90", "joint")] <= 0.8 * paired[("interval_width90", "elastic")]
        false_conf = narrower & ~paired[("covered90", "joint")].astype(bool)
        false_fraction = float(false_conf.mean())
        rmse_ratio = float(joint["rmse"] / max(elastic["rmse"], 1e-12))
        adequate = float(joint["median_reduced_chi2"])
        if (
            joint["coverage90_wilson_low"] >= 0.80
            and rmse_ratio <= 0.90
            and false_fraction <= 0.10
            and joint["success_fraction"] >= 0.95
            and adequate <= 1.50
        ):
            verdict = "PASS"
        elif (
            joint["coverage90"] >= 0.75
            and rmse_ratio <= 1.05
            and false_fraction <= 0.20
            and adequate <= 2.00
        ):
            verdict = "CONDITIONAL"
        else:
            verdict = "FAIL"
        rows.append({
            "case": case,
            "target": target,
            "verdict": verdict,
            "joint_coverage90": float(joint["coverage90"]),
            "joint_coverage_wilson_low": float(joint["coverage90_wilson_low"]),
            "rmse_ratio_joint_over_elastic": rmse_ratio,
            "false_confidence_fraction": false_fraction,
            "joint_bound_hit_fraction": float(joint["bound_hit_fraction"]),
            "joint_success_fraction": float(joint["success_fraction"]),
            "joint_median_reduced_chi2": adequate,
        })
    return pd.DataFrame(rows)


def run_carbonate_monte_carlo(
    n_panels: int = 8,
    replicates: int = 10,
    panel_size: int = 8,
    seed: int = 20260829,
    cases: Sequence[CarbonateCase] = CASES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated-noise nonlinear carbonate recovery experiments."""
    design = carbonate_design(n_panels, seed)
    rng = np.random.default_rng(seed + 17)
    rows = []
    for case in cases:
        for panel_index, truth in enumerate(design):
            fixed = dict(truth)
            for name in FIT_NAMES:
                fixed.pop(name, None)
            # Elastic constitutive parameters not targeted in Phase 1 remain
            # known, isolating the electrical-coupling question.
            fixed["vcl"] = truth["vcl"]
            fixed["cement"] = truth["cement"]
            fixed["coord"] = truth["coord"]
            fixed["phi_e"] = truth["phi_e"]
            for replicate in range(replicates):
                observed_e, observed_r = simulate_observations(
                    truth, case, panel_size, rng
                )
                fits = {
                    "elastic": fit_panel(
                        observed_e, observed_r, fixed, case.inverse_electrical,
                        False, panel_size, rng,
                    ),
                    "joint": fit_panel(
                        observed_e, observed_r, fixed, case.inverse_electrical,
                        True, panel_size, rng,
                    ),
                }
                for mode, fit in fits.items():
                    for target in TARGETS:
                        estimate = fit["estimate"][target]
                        sd = fit["sd"][target]
                        half_width = 1.644854 * sd
                        rows.append({
                            "case": case.name,
                            "truth_electrical": case.truth_electrical,
                            "inverse_electrical": case.inverse_electrical,
                            "coupling_rho": case.coupling_rho,
                            "panel": panel_index,
                            "replicate": replicate,
                            "mode": mode,
                            "target": target,
                            "truth": truth[target],
                            "estimate": estimate,
                            "sd": sd,
                            "interval_width90": 2.0 * half_width,
                            "covered90": estimate - half_width <= truth[target] <= estimate + half_width,
                            "bound_hit": fit["bound_hit"][target],
                            "success": fit["success"],
                            "cost": fit["cost"],
                            "nfev": fit["nfev"],
                            "panel_size": panel_size,
                            "degrees_freedom": (
                                5 * panel_size - len(FIT_NAMES)
                                if mode == "joint"
                                else 3 * panel_size - len(TARGETS)
                            ),
                        })
    raw = pd.DataFrame(rows)
    summary = summarize_recovery(raw)
    gates = build_phase1_gates(raw, summary)
    return raw, summary, gates
