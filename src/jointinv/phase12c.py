"""Phase 1.2C: independent off-library falsification.

The primary estimator in this module is a *frozen deployment rule*.  Its
candidate weights, interval inflation factors and out-of-distribution alarm
threshold are learned only from the Phase-1.1 matched-library bank.  The new
Phase-1.2C panels are used once, as an external falsification set.

Five deliberately misspecified truths are evaluated without adding them to the
inverse library: a depth-varying hybrid conductivity, depth-varying electrical
connectivity, a three-zone invasion operator, patchy saturation, and a combined
stress.  An independent matched EMA control checks transport to the new design.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis import DATA_SIGMA
from .families import forward_elastic_family
from .phase05 import _panel_states
from .phase1 import (
    CARBONATE, FIT_NAMES as PHASE1_FIT_NAMES, TARGETS, _rt_cholesky, fit_panel,
)
from .phase11 import (
    CANDIDATE_FAMILIES, COUPLING_MODES, _candidate_vectors, _crossfit_weights,
    _range, _stable_seed, converged_weights, fit_candidate,
    forward_candidate_state, mixture_recovery, phase11_design,
)
from .phase12 import conformal_order_statistic


CANDIDATE_KEYS = tuple(
    (family, coupling)
    for family in CANDIDATE_FAMILIES
    for coupling in COUPLING_MODES
)
STRATEGIES = ("elastic", "frozen_press", "adapted_average")


@dataclass(frozen=True)
class OffLibraryCase:
    name: str
    label: str
    kind: str
    is_ood: bool
    atomic: bool


OFF_LIBRARY_CASES = (
    OffLibraryCase(
        "matched_ema_control", "Matched EMA control", "matched", False, False,
    ),
    OffLibraryCase(
        "hybrid_law", "Depth-varying hybrid law", "hybrid", True, True,
    ),
    OffLibraryCase(
        "connectivity_drift", "Depth-varying connectivity", "q_drift", True, True,
    ),
    OffLibraryCase(
        "invasion_mismatch", "Three-zone invasion mismatch", "invasion", True, True,
    ),
    OffLibraryCase(
        "patchy_saturation", "Patchy saturation", "patchy", True, True,
    ),
    OffLibraryCase(
        "combined_stress", "Combined off-library stress", "combined", True, False,
    ),
)


def _case_lookup(case: str | OffLibraryCase) -> OffLibraryCase:
    if isinstance(case, OffLibraryCase):
        return case
    matches = [item for item in OFF_LIBRARY_CASES if item.name == case]
    if not matches:
        raise ValueError(f"Unknown Phase-1.2C case: {case}")
    return matches[0]


def _base_connectivity(p: Mapping[str, float]) -> float:
    """The same panel-level rho=.5 latent used by the Phase-1.1 truth design."""
    return float(np.clip(
        0.5 * float(p["q_elastic_design"])
        + np.sqrt(0.75) * float(p["q_perp"]),
        -3.0, 3.0,
    ))


def patch_saturations(
    saturation: float, cap: float = 0.25, scale: float = 0.90,
) -> tuple[float, float]:
    """Symmetric sub-resolution saturations preserving the nominal mean."""
    saturation = float(np.clip(saturation, 0.01, 0.999))
    room = min(saturation - 0.01, 0.999 - saturation)
    delta = max(0.0, min(float(cap), float(scale) * room))
    return saturation - delta, saturation + delta


def _intrinsic_conductivity(
    p: Mapping[str, float], family: str, q_conn: float,
    saturation: float, water_resistivity: float,
) -> float:
    """Reuse a candidate intrinsic law without reusing its tool operator."""
    state = dict(p)
    state.update({
        "sw": float(np.clip(saturation, 0.01, 0.999)),
        "log_rw": float(np.log(max(water_resistivity, 1e-8))),
        "invasion": 0.0,
        "rmf_ratio": 1.0,
        "q_conn": float(np.clip(q_conn, -3.0, 3.0)),
    })
    resistivity = forward_candidate_state(state, family, "independent")
    conductivity = 1.0 / resistivity
    # With zero invasion and rmf_ratio=1 the two tool responses are identical.
    return max(float(np.mean(conductivity)), 1e-10)


def _hybrid_conductivity(
    p: Mapping[str, float], q_conn: float, saturation: float,
    water_resistivity: float, latent_fraction: float,
) -> float:
    latent = _intrinsic_conductivity(
        p, "latent_dual", q_conn, saturation, water_resistivity,
    )
    network = _intrinsic_conductivity(
        p, "network", q_conn, saturation, water_resistivity,
    )
    weight = float(np.clip(latent_fraction, 0.02, 0.98))
    return float(np.exp(weight * np.log(latent) + (1.0 - weight) * np.log(network)))


def _hill_average_elastic(
    first: np.ndarray, second: np.ndarray,
) -> np.ndarray:
    """Hill-average two elastic patches expressed as Vp, Vs and density."""
    arrays = [np.asarray(first, dtype=float), np.asarray(second, dtype=float)]
    moduli = []
    for vector in arrays:
        vp, vs, rho = vector
        shear = rho * vs * vs
        bulk = rho * vp * vp - 4.0 * shear / 3.0
        moduli.append((max(bulk, 1e-8), max(shear, 1e-8), rho))
    k_values = np.array([value[0] for value in moduli])
    g_values = np.array([value[1] for value in moduli])
    rho = float(np.mean([value[2] for value in moduli]))
    k_voigt, g_voigt = float(k_values.mean()), float(g_values.mean())
    k_reuss = float(1.0 / np.mean(1.0 / k_values))
    g_reuss = float(1.0 / np.mean(1.0 / g_values))
    bulk = 0.5 * (k_voigt + k_reuss)
    shear = 0.5 * (g_voigt + g_reuss)
    return np.array([
        np.sqrt((bulk + 4.0 * shear / 3.0) / rho),
        np.sqrt(shear / rho),
        rho,
    ])


def _patch_conductivity(
    p: Mapping[str, float], family: str, q_conn: float,
    saturation: float, water_resistivity: float,
    cap: float, scale: float, hybrid_fraction: float | None = None,
) -> float:
    low, high = patch_saturations(saturation, cap=cap, scale=scale)
    values = []
    for sub_saturation in (low, high):
        if hybrid_fraction is None:
            value = _intrinsic_conductivity(
                p, family, q_conn, sub_saturation, water_resistivity,
            )
        else:
            value = _hybrid_conductivity(
                p, q_conn, sub_saturation, water_resistivity, hybrid_fraction,
            )
        values.append(value)
    # Parallel sub-resolution water pathways are deliberately different from
    # the homogeneous saturation assumed by every inverse candidate.
    return float(np.mean(values))


def _standard_two_zone(
    p: Mapping[str, float], family: str, q_conn: float,
    hybrid_fraction: float | None = None,
    patch: tuple[float, float] | None = None,
) -> np.ndarray:
    sw = float(np.clip(p["sw"], 0.01, 0.999))
    rw = float(np.exp(p["log_rw"]))
    invasion = float(np.clip(p["invasion"], 0.0, 0.65))
    rmf_ratio = float(np.clip(p.get("rmf_ratio", 0.65), 0.25, 1.75))
    invaded_sw = float(np.clip(sw + invasion * (1.0 - sw), 0.01, 0.999))

    def sigma(saturation: float, water_resistivity: float) -> float:
        if patch is not None:
            return _patch_conductivity(
                p, family, q_conn, saturation, water_resistivity,
                cap=patch[0], scale=patch[1],
                hybrid_fraction=hybrid_fraction,
            )
        if hybrid_fraction is not None:
            return _hybrid_conductivity(
                p, q_conn, saturation, water_resistivity, hybrid_fraction,
            )
        return _intrinsic_conductivity(
            p, family, q_conn, saturation, water_resistivity,
        )

    virgin = sigma(sw, rw)
    invaded = sigma(invaded_sw, rw * rmf_ratio)
    return np.array([
        0.86 * virgin + 0.14 * invaded,
        0.22 * virgin + 0.78 * invaded,
    ])


def _three_zone(
    p: Mapping[str, float], q_conn: float, depth_x: float, phase: float,
    combined: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    sw = float(np.clip(p["sw"], 0.01, 0.999))
    rw = float(np.exp(p["log_rw"]))
    base_invasion = float(p["invasion"])
    base_ratio = float(p.get("rmf_ratio", 0.65))
    if combined:
        invasion = base_invasion + 0.16 * np.sin(phase + 0.5) + 0.08 * depth_x
        ratio = base_ratio * np.exp(0.45 * np.cos(2.0 * phase - 0.3))
        deep_weights = np.array([0.70, 0.20, 0.10])
        shallow_weights = np.array([0.04, 0.22, 0.74])
        cap, scale = 0.30, 0.95
        hybrid_fraction = 0.50 + 0.30 * np.sin(phase + 0.4)
    else:
        invasion = base_invasion + 0.10 * np.sin(phase + 0.5) + 0.06 * depth_x
        ratio = base_ratio * np.exp(0.30 * np.cos(2.0 * phase - 0.3))
        deep_weights = np.array([0.78, 0.15, 0.07])
        shallow_weights = np.array([0.10, 0.25, 0.65])
        cap, scale = 0.0, 0.0
        hybrid_fraction = None
    invasion_clipped = float(np.clip(invasion, 0.01, 0.65))
    ratio_clipped = float(np.clip(ratio, 0.20, 2.00))
    saturations = (
        sw,
        float(np.clip(sw + 0.45 * invasion_clipped * (1.0 - sw), 0.01, 0.999)),
        float(np.clip(sw + invasion_clipped * (1.0 - sw), 0.01, 0.999)),
    )
    resistivities = (rw, rw * np.sqrt(ratio_clipped), rw * ratio_clipped)
    zone_sigma = []
    for saturation, water_resistivity in zip(saturations, resistivities):
        if combined:
            zone_sigma.append(_patch_conductivity(
                p, "network", q_conn, saturation, water_resistivity,
                cap=cap, scale=scale, hybrid_fraction=hybrid_fraction,
            ))
        else:
            zone_sigma.append(_intrinsic_conductivity(
                p, "ema", q_conn, saturation, water_resistivity,
            ))
    zone_sigma = np.asarray(zone_sigma)
    return np.array([
        float(deep_weights @ zone_sigma),
        float(shallow_weights @ zone_sigma),
    ]), {
        "invasion_clipped": float(invasion != invasion_clipped),
        "rmf_clipped": float(ratio != ratio_clipped),
    }


def offlibrary_truth_vectors(
    p: Mapping[str, float], case: str | OffLibraryCase, panel_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return noiseless elastic and log-Rt vectors for one external truth."""
    spec = _case_lookup(case)
    states = _panel_states(p, panel_size)
    phases = np.linspace(0.0, 2.0 * np.pi, panel_size, endpoint=False)
    depth_x = np.zeros(panel_size) if panel_size == 1 else np.linspace(-1.0, 1.0, panel_size)
    q0 = _base_connectivity(p)
    elastic_rows, conductivity_rows = [], []
    q_clips = invasion_clips = rmf_clips = 0.0
    patch_deltas = []

    for state, phase, xvalue in zip(states, phases, depth_x):
        q = q0
        if spec.kind == "q_drift":
            raw_q = q0 + 0.75 * xvalue + 0.30 * np.sin(2.0 * phase + 0.7)
            q = float(np.clip(raw_q, -3.0, 3.0))
            q_clips += float(q != raw_q)
        elif spec.kind == "combined":
            raw_q = q0 + 1.15 * xvalue + 0.45 * np.sin(2.0 * phase + 0.7)
            q = float(np.clip(raw_q, -3.0, 3.0))
            q_clips += float(q != raw_q)

        if spec.kind in ("patchy", "combined"):
            cap, scale = ((0.25, 0.90) if spec.kind == "patchy" else (0.30, 0.95))
            sw_low, sw_high = patch_saturations(state["sw"], cap=cap, scale=scale)
            patch_deltas.append(0.5 * (sw_high - sw_low))
            first_state, second_state = dict(state), dict(state)
            first_state["sw"], second_state["sw"] = sw_low, sw_high
            elastic_rows.append(_hill_average_elastic(
                forward_elastic_family(first_state, CARBONATE, "dem"),
                forward_elastic_family(second_state, CARBONATE, "dem"),
            ))
        else:
            elastic_rows.append(forward_elastic_family(state, CARBONATE, "dem"))

        if spec.kind == "matched":
            conductivity = _standard_two_zone(state, "ema", q)
        elif spec.kind == "hybrid":
            latent_fraction = 0.50 + 0.15 * np.sin(phase + 0.4)
            conductivity = _standard_two_zone(
                state, "network", q, hybrid_fraction=latent_fraction,
            )
        elif spec.kind == "q_drift":
            conductivity = _standard_two_zone(state, "network", q)
        elif spec.kind == "invasion":
            conductivity, flags = _three_zone(state, q, xvalue, phase, False)
            invasion_clips += flags["invasion_clipped"]
            rmf_clips += flags["rmf_clipped"]
        elif spec.kind == "patchy":
            conductivity = _standard_two_zone(
                state, "ema", q, patch=(0.25, 0.90),
            )
        else:
            conductivity, flags = _three_zone(state, q, xvalue, phase, True)
            invasion_clips += flags["invasion_clipped"]
            rmf_clips += flags["rmf_clipped"]
        conductivity_rows.append(conductivity)

    diagnostics = {
        "q_clip_fraction": q_clips / panel_size,
        "invasion_clip_fraction": invasion_clips / panel_size,
        "rmf_clip_fraction": rmf_clips / panel_size,
        "mean_patch_delta": float(np.mean(patch_deltas)) if patch_deltas else 0.0,
    }
    elastic = np.vstack(elastic_rows)
    log_rt = -np.log(np.vstack(conductivity_rows))
    if not (np.all(np.isfinite(elastic)) and np.all(np.isfinite(log_rt))):
        raise FloatingPointError(f"Non-finite off-library truth for {spec.name}")
    return elastic, log_rt, diagnostics


def simulate_offlibrary_observations(
    p: Mapping[str, float], case: str | OffLibraryCase, panel_size: int,
    elastic_noise: np.ndarray, rt_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    elastic, log_rt, diagnostics = offlibrary_truth_vectors(p, case, panel_size)
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    return (
        elastic + np.asarray(elastic_noise) * sigma_e,
        log_rt + np.asarray(rt_noise) @ _rt_cholesky().T,
        diagnostics,
    )


def learn_frozen_weights(
    training_components: pd.DataFrame, excluded_panel: int | None = None,
) -> np.ndarray:
    """Learn a universal PRESS prior with dual/EMA/network truths equally weighted.

    The prior is reconstructed directly from per-observation predictive scores,
    not from Phase-1.1 ``full_weight`` values.  Those historical weights were
    cross-fitted inside each truth case and would indirectly retain an outer
    held panel through the weights assigned to the other panels.
    """
    required = {
        "case", "truth_family", "panel", "replicate", "family", "coupling",
        "predictive_score", "success",
    }
    missing = required.difference(training_components.columns)
    if missing:
        raise ValueError(f"Missing training component columns: {sorted(missing)}")
    data = training_components.copy()
    if excluded_panel is not None:
        data = data.loc[~data["panel"].eq(excluded_panel)]
    if data.empty:
        raise ValueError("No matched-library rows remain for frozen weights")
    local_rows = []
    for keys, group in data.groupby(
        ["truth_family", "case", "panel", "replicate"], sort=True,
    ):
        indexed = group.set_index(["family", "coupling"], drop=False)
        if indexed.index.has_duplicates or set(indexed.index) != set(CANDIDATE_KEYS):
            raise ValueError("Every matched observation must contain the nine candidates")
        scores = np.array([
            float(indexed.loc[key, "predictive_score"]) for key in CANDIDATE_KEYS
        ])
        success = np.array([
            bool(indexed.loc[key, "success"]) for key in CANDIDATE_KEYS
        ])
        active = success & np.isfinite(scores)
        if not np.any(active):
            raise RuntimeError("Matched observation has no valid candidate score")
        weights = np.zeros(len(CANDIDATE_KEYS), dtype=float)
        delta = np.clip(scores[active] - np.min(scores[active]), 0.0, 1400.0)
        weights[active] = np.exp(-0.5 * delta)
        weights /= weights.sum()
        truth_family, case, panel, replicate = keys
        for (family, coupling), weight in zip(CANDIDATE_KEYS, weights):
            local_rows.append({
                "truth_family": truth_family, "case": case, "panel": panel,
                "replicate": replicate, "family": family, "coupling": coupling,
                "local_press_weight": float(weight),
            })
    local = pd.DataFrame(local_rows)
    case_means = local.groupby(
        ["truth_family", "case", "family", "coupling"], as_index=False,
    )["local_press_weight"].mean()
    truth_balanced = case_means.groupby(
        ["truth_family", "family", "coupling"], as_index=False,
    )["local_press_weight"].mean()
    universal = truth_balanced.groupby(
        ["family", "coupling"], as_index=False,
    )["local_press_weight"].mean()
    lookup = {
        (row.family, row.coupling): float(row.local_press_weight)
        for row in universal.itertuples(index=False)
    }
    if set(lookup) != set(CANDIDATE_KEYS):
        raise ValueError("Matched-library bank does not contain all nine candidates")
    weights = np.array([lookup[key] for key in CANDIDATE_KEYS], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Frozen candidate weights must be finite and nonnegative")
    if weights.sum() <= 0.0:
        raise ValueError("Frozen candidate weights have zero total mass")
    return weights / weights.sum()


def frozen_press_weights(
    fits: Sequence[Mapping], frozen_prior: np.ndarray,
) -> np.ndarray:
    """Apply a predeclared local blocked-PRESS update to a frozen prior.

    This is a deployable rule: it uses the current panel's observations only
    through the candidates' depth-block predictive scores, never through the
    synthetic truth or realized target error.  The prior and temperature are
    fixed before Phase 1.2C.
    """
    prior = np.asarray(frozen_prior, dtype=float)
    if prior.shape != (len(fits),):
        raise ValueError("Frozen prior and candidate bank have different sizes")
    success = np.array([bool(fit["success"]) for fit in fits])
    scores = np.array([float(fit["predictive_score"]) for fit in fits])
    active = success & np.isfinite(scores) & (prior > 0.0)
    if not np.any(active):
        raise RuntimeError("No active candidate remains for the frozen PRESS rule")
    log_weight = np.full(len(fits), -np.inf)
    delta = np.clip(scores[active] - np.min(scores[active]), 0.0, 1400.0)
    log_weight[active] = np.log(prior[active]) - 0.5 * delta
    finite_max = float(np.max(log_weight[active]))
    weights = np.exp(log_weight - finite_max)
    return weights / weights.sum()


def _fits_from_component_group(group: pd.DataFrame) -> list[dict]:
    indexed = group.set_index(["family", "coupling"], drop=False)
    if indexed.index.has_duplicates or set(indexed.index) != set(CANDIDATE_KEYS):
        raise ValueError("Training observation must contain each candidate exactly once")
    fits = []
    for family, coupling in CANDIDATE_KEYS:
        row = indexed.loc[(family, coupling)]
        fits.append({
            "family": family,
            "coupling": coupling,
            "estimate": {target: float(row[f"estimate_{target}"]) for target in TARGETS},
            "sd": {target: float(row[f"sd_{target}"]) for target in TARGETS},
            "reduced_chi2": float(row["reduced_chi2"]),
            "predictive_score": float(row["predictive_score"]),
            "success": bool(row["success"]),
        })
    return fits


def prepare_frozen_training_rule(
    training_recoveries: pd.DataFrame, training_components: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, float]:
    """Outer-panel reconstruction of the exact frozen deployment estimator.

    The resulting factors are empirical inflation factors.  They are not
    advertised as finite-sample cluster-conformal guarantees because repeated
    case/noise observations remain nested inside only twelve training panels.
    """
    final_weights = learn_frozen_weights(training_components)
    truth_rows = training_recoveries.loc[
        training_recoveries["strategy"].eq("elastic"),
        ["case", "panel", "replicate", "target", "truth"],
    ].copy()
    if truth_rows.duplicated(["case", "panel", "replicate", "target"]).any():
        raise ValueError("Training recovery truth keys are duplicated")
    truth_lookup = truth_rows.set_index(
        ["case", "panel", "replicate", "target"],
    )["truth"]

    reconstructed = []
    group_keys = ["case", "panel", "replicate"]
    outer_prior_by_panel = {
        int(panel): learn_frozen_weights(
            training_components, excluded_panel=int(panel),
        )
        for panel in sorted(training_components["panel"].unique())
    }
    for keys, group in training_components.groupby(group_keys, sort=True):
        case, panel, replicate = keys
        outer_weights = outer_prior_by_panel[int(panel)]
        fits = _fits_from_component_group(group)
        usable = frozen_press_weights(fits, outer_weights)
        for target in TARGETS:
            result = mixture_recovery(fits, usable, target)
            truth = float(truth_lookup.loc[(case, panel, replicate, target)])
            reconstructed.append({
                "case": case, "panel": int(panel), "replicate": int(replicate),
                "strategy": "frozen_press", "target": target, "truth": truth,
                "estimate": result["estimate"], "sd": result["sd"],
                "standardized_absolute_error": abs(result["estimate"] - truth)
                / max(result["sd"], 1e-12),
            })
    reconstruction = pd.DataFrame(reconstructed)

    factors = []
    for strategy, source_strategy in (
        ("elastic", "elastic"),
        ("adapted_average", "model_average"),
    ):
        source = training_recoveries.loc[
            training_recoveries["strategy"].eq(source_strategy),
        ].copy()
        source["standardized_absolute_error"] = (
            (source["estimate"] - source["truth"]).abs()
            / source["sd"].clip(lower=1e-12)
        )
        for target, group in source.groupby("target"):
            factors.append({
                "strategy": strategy, "target": target,
                "c90": conformal_order_statistic(
                    group["standardized_absolute_error"], 0.90,
                ),
                "n_training": len(group),
                "calibration_source": f"phase11_{source_strategy}_empirical_bank",
            })
    for target, group in reconstruction.groupby("target"):
        factors.append({
            "strategy": "frozen_press", "target": target,
            "c90": conformal_order_statistic(
                group["standardized_absolute_error"], 0.90,
            ),
            "n_training": len(group),
            "calibration_source": "phase11_outer_panel_frozen_reconstruction_empirical",
        })
    calibration = pd.DataFrame(factors).sort_values(
        ["strategy", "target"], ignore_index=True,
    )

    best_training = (
        training_components.loc[training_components["success"].astype(bool)]
        .groupby(["case", "panel", "replicate"])["reduced_chi2"].min()
    )
    alarm_threshold = conformal_order_statistic(best_training, 0.95)
    return final_weights, calibration, reconstruction, alarm_threshold


def _fixed_for_inverse(truth: Mapping[str, float]) -> dict[str, float]:
    fixed = dict(truth)
    for name in PHASE1_FIT_NAMES:
        fixed.pop(name, None)
    fixed.update({
        "vcl": truth["vcl"], "cement": truth["cement"],
        "coord": truth["coord"], "phi_e": truth["phi_e"],
    })
    return fixed


def _fit_phase12c_observation(task: tuple) -> dict:
    (
        spec, panel_index, replicate, truth, panel_size, elastic_noise, rt_noise,
        fit_seed,
    ) = task
    observed_e, observed_r, truth_diagnostics = simulate_offlibrary_observations(
        truth, spec, panel_size, elastic_noise, rt_noise,
    )
    fixed = _fixed_for_inverse(truth)
    elastic = fit_panel(
        observed_e, observed_r, fixed, "dual_porosity", False, panel_size,
        np.random.default_rng(fit_seed), n_starts=1,
    )
    fits = []
    for family in CANDIDATE_FAMILIES:
        prior = None
        for coupling in COUPLING_MODES:
            fitted = fit_candidate(
                observed_e, observed_r, fixed, family, coupling, panel_size,
                elastic["estimate"], prior,
            )
            fits.append(fitted)
            prior = fitted["estimate"]
    return {
        "case": spec.name,
        "panel": panel_index,
        "replicate": replicate,
        "truth": dict(truth),
        "elastic": elastic,
        "fits": fits,
        "truth_diagnostics": truth_diagnostics,
    }


def _calibrated_interval(
    result: Mapping[str, float], target: str, c90: float,
) -> dict[str, float]:
    lower_bound, upper_bound = _range(target)
    lower = max(lower_bound, float(result["estimate"]) - c90 * float(result["sd"]))
    upper = min(upper_bound, float(result["estimate"]) + c90 * float(result["sd"]))
    return {"lower90": lower, "upper90": upper, "interval_width90": upper - lower}


def _validate_weight_order(fits: Sequence[Mapping]) -> None:
    keys = tuple((fit["family"], fit["coupling"]) for fit in fits)
    if keys != CANDIDATE_KEYS:
        raise ValueError("Candidate fit order differs from the frozen training order")


def _append_recovery(
    rows: list[dict], base: Mapping, strategy: str, target: str,
    truth_value: float, result: Mapping[str, float], c90: float,
    raw_success_fraction: float, surviving_prior_mass: float,
    critical_bound_mass: float, target_bound_mass: float,
    best_reduced_chi2: float,
    alarm_threshold: float,
) -> None:
    interval = _calibrated_interval(result, target, c90)
    rows.append({
        "case": base["case"], "panel": base["panel"],
        "replicate": base["replicate"], "strategy": strategy,
        "target": target, "truth": truth_value,
        "estimate": result["estimate"], "sd": result["sd"], "c90": c90,
        **interval,
        "covered90": interval["lower90"] <= truth_value <= interval["upper90"],
        "weighted_reduced_chi2": result["weighted_reduced_chi2"],
        "effective_models": result["effective_models"],
        "max_weight": result["max_weight"],
        "raw_candidate_success_fraction": raw_success_fraction,
        "surviving_prior_mass": surviving_prior_mass,
        "critical_bound_mass": critical_bound_mass,
        "target_bound_mass": target_bound_mass,
        "best_reduced_chi2": best_reduced_chi2,
        "ood_alarm": bool(best_reduced_chi2 > alarm_threshold),
        "alarm_threshold": alarm_threshold,
    })


def truth_library_distance(
    design: Sequence[Mapping[str, float]], cases: Sequence[OffLibraryCase],
    panel_size: int,
) -> pd.DataFrame:
    """Noise-free standardized distance at the true parameter vector."""
    sigma_e = np.array([DATA_SIGMA["vp"], DATA_SIGMA["vs"], DATA_SIGMA["rho"]])
    whitener = np.linalg.inv(_rt_cholesky())
    rows = []
    for spec in cases:
        for panel_index, panel in enumerate(design):
            truth_e, truth_r, diagnostics = offlibrary_truth_vectors(
                panel, spec, panel_size,
            )
            state = dict(panel)
            state["q_conn"] = _base_connectivity(panel)
            for family, coupling in CANDIDATE_KEYS:
                pred_e, pred_r = _candidate_vectors(
                    state, family, coupling, panel_size,
                )
                residual_e = ((pred_e - truth_e) / sigma_e).ravel()
                residual_r = np.hstack([
                    whitener @ row for row in pred_r - truth_r
                ])
                residual = np.concatenate([residual_e, residual_r])
                rows.append({
                    "case": spec.name, "is_ood": spec.is_ood,
                    "atomic": spec.atomic, "panel": panel_index,
                    "family": family, "coupling": coupling,
                    "fixed_truth_standardized_rms": float(np.sqrt(np.mean(residual**2))),
                    **diagnostics,
                })
    return pd.DataFrame(rows)


def summarize_phase12c(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, target, strategy), group in raw.groupby(
        ["case", "target", "strategy"], sort=True,
    ):
        error = group["estimate"] - group["truth"]
        rows.append({
            "case": case, "target": target, "strategy": strategy,
            "n": len(group), "bias": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "coverage90": float(group["covered90"].mean()),
            "median_interval_width": float(group["interval_width90"].median()),
            "median_reduced_chi2": float(group["weighted_reduced_chi2"].median()),
            "alarm_fraction": float(group["ood_alarm"].mean()),
            "raw_candidate_success_fraction": float(
                group["raw_candidate_success_fraction"].mean()
            ),
            "mean_surviving_prior_mass": float(group["surviving_prior_mass"].mean()),
            "mean_critical_bound_mass": float(group["critical_bound_mass"].mean()),
        })
    return pd.DataFrame(rows)


def _paired_strategy(raw: pd.DataFrame, case: str, target: str, strategy: str) -> pd.DataFrame:
    subset = raw.loc[
        raw["case"].eq(case)
        & raw["target"].eq(target)
        & raw["strategy"].isin(["elastic", strategy])
    ]
    counts = subset.groupby(["panel", "replicate", "strategy"]).size()
    if not counts.eq(1).all():
        raise ValueError("Phase-1.2C recovery pairing contains duplicates")
    paired = subset.pivot(
        index=["panel", "replicate"], columns="strategy",
        values=[
            "truth", "estimate", "interval_width90", "covered90",
            "weighted_reduced_chi2", "raw_candidate_success_fraction",
            "surviving_prior_mass", "critical_bound_mass", "ood_alarm",
        ],
    )
    for name in ("elastic", strategy):
        if ("truth", name) not in paired:
            raise ValueError(f"Missing paired strategy: {name}")
    if not np.allclose(
        paired[("truth", "elastic")].astype(float).to_numpy(),
        paired[("truth", strategy)].astype(float).to_numpy(),
    ):
        raise ValueError("Paired strategies disagree on synthetic truth")
    return paired


def build_phase12c_gates(
    raw: pd.DataFrame, strategies: Sequence[str] = ("frozen_press", "adapted_average"),
    n_bootstrap: int = 4000,
) -> pd.DataFrame:
    """Confirmatory panel-cluster gates for every scenario and target."""
    rows = []
    cases = [item.name for item in OFF_LIBRARY_CASES if item.name in set(raw["case"])]
    for case in cases:
        spec = _case_lookup(case)
        for target in TARGETS:
            for strategy in strategies:
                paired = _paired_strategy(raw, case, target, strategy)
                joint_error = (
                    paired[("estimate", strategy)].astype(float)
                    - paired[("truth", strategy)].astype(float)
                )
                elastic_error = (
                    paired[("estimate", "elastic")].astype(float)
                    - paired[("truth", "elastic")].astype(float)
                )
                width_values = (
                    paired[("interval_width90", strategy)].astype(float)
                    / paired[("interval_width90", "elastic")].astype(float).clip(lower=1e-12)
                )
                joint_covered = paired[("covered90", strategy)].astype(bool)
                elastic_covered = paired[("covered90", "elastic")].astype(bool)
                false_confidence = width_values.le(0.80) & ~joint_covered
                harmful_false = false_confidence & elastic_covered
                panels = np.array(sorted(paired.index.get_level_values("panel").unique()))
                panel_index = paired.index.get_level_values("panel").to_numpy()
                rng = np.random.default_rng(_stable_seed(
                    case, target, strategy, "phase12c_cluster",
                ))
                coverage_draws, ratio_draws, width_draws = [], [], []
                false_draws, harmful_draws, bias_draws = [], [], []
                truth_sd = max(float(
                    pd.Series(
                        paired[("truth", strategy)].astype(float).to_numpy(),
                        index=paired.index,
                    )
                    .groupby(level="panel").first().std(ddof=0)
                ), 1e-12)
                for _ in range(n_bootstrap):
                    sampled = rng.choice(panels, size=len(panels), replace=True)
                    indices = np.concatenate([
                        np.flatnonzero(panel_index == panel) for panel in sampled
                    ])
                    je = joint_error.to_numpy()[indices]
                    ee = elastic_error.to_numpy()[indices]
                    coverage_draws.append(float(joint_covered.to_numpy()[indices].mean()))
                    ratio_draws.append(
                        np.sqrt(np.mean(je**2)) / max(np.sqrt(np.mean(ee**2)), 1e-12)
                    )
                    width_draws.append(float(np.median(width_values.to_numpy()[indices])))
                    false_draws.append(float(false_confidence.to_numpy()[indices].mean()))
                    harmful_draws.append(float(harmful_false.to_numpy()[indices].mean()))
                    bias_draws.append(abs(float(je.mean())) / truth_sd)

                coverage = float(joint_covered.mean())
                rmse_ratio = float(
                    np.sqrt(np.mean(joint_error**2))
                    / max(np.sqrt(np.mean(elastic_error**2)), 1e-12)
                )
                width_ratio = float(np.median(width_values))
                standardized_bias = abs(float(joint_error.mean())) / truth_sd
                coverage_low = float(np.quantile(coverage_draws, 0.025))
                coverage_high = float(np.quantile(coverage_draws, 0.975))
                ratio_upper = float(np.quantile(ratio_draws, 0.975))
                width_upper = float(np.quantile(width_draws, 0.975))
                false_upper = float(np.quantile(false_draws, 0.975))
                harmful_upper = float(np.quantile(harmful_draws, 0.975))
                bias_upper = float(np.quantile(bias_draws, 0.975))
                reduced_chi2 = float(np.median(
                    paired[("weighted_reduced_chi2", strategy)].astype(float)
                ))
                raw_success = float(np.mean(
                    paired[("raw_candidate_success_fraction", strategy)].astype(float)
                ))
                surviving_mass = float(np.mean(
                    paired[("surviving_prior_mass", strategy)].astype(float)
                ))
                critical_mass = float(np.mean(
                    paired[("critical_bound_mass", strategy)].astype(float)
                ))
                alarm_fraction = float(np.mean(
                    paired[("ood_alarm", strategy)].astype(bool)
                ))

                pass_checks = {
                    "coverage_0.85_to_0.97": 0.85 <= coverage <= 0.97,
                    "coverage_cluster_low_ge_0.80": coverage_low >= 0.80,
                    "rmse_ratio_le_0.90": rmse_ratio <= 0.90,
                    "rmse_ratio_upper_lt_1": ratio_upper < 1.0,
                    "width_ratio_le_0.80": width_ratio <= 0.80,
                    "width_ratio_upper_lt_1": width_upper < 1.0,
                    "false_confidence_upper_le_0.10": false_upper <= 0.10,
                    "harmful_false_upper_le_0.10": harmful_upper <= 0.10,
                    "standardized_bias_le_0.10": standardized_bias <= 0.10,
                    "bias_upper_le_0.20": bias_upper <= 0.20,
                    "reduced_chi2_le_1.50": reduced_chi2 <= 1.50,
                    "candidate_success_ge_0.99": raw_success >= 0.99,
                    "surviving_prior_mass_ge_0.99": surviving_mass >= 0.99,
                    "critical_bound_mass_le_0.10": critical_mass <= 0.10,
                }
                conditional_checks = {
                    "coverage_ge_0.80": coverage >= 0.80,
                    "coverage_cluster_low_ge_0.70": coverage_low >= 0.70,
                    "rmse_ratio_le_1.05": rmse_ratio <= 1.05,
                    "rmse_ratio_upper_le_1.10": ratio_upper <= 1.10,
                    "width_ratio_le_1": width_ratio <= 1.00,
                    "width_ratio_upper_le_1.10": width_upper <= 1.10,
                    "false_confidence_upper_le_0.20": false_upper <= 0.20,
                    "standardized_bias_le_0.20": standardized_bias <= 0.20,
                    "bias_upper_le_0.35": bias_upper <= 0.35,
                    "reduced_chi2_le_2.50": reduced_chi2 <= 2.50,
                    "candidate_success_ge_0.95": raw_success >= 0.95,
                    "surviving_prior_mass_ge_0.95": surviving_mass >= 0.95,
                    "critical_bound_mass_le_0.25": critical_mass <= 0.25,
                }
                if all(pass_checks.values()):
                    verdict = "PASS"
                elif all(conditional_checks.values()):
                    verdict = "CONDITIONAL"
                else:
                    verdict = "FAIL"
                inferential_harm = (
                    coverage < 0.85 or coverage_low < 0.80
                    or ratio_upper >= 1.0
                    or false_upper > 0.10 or harmful_upper > 0.10
                    or standardized_bias > 0.10 or bias_upper > 0.20
                )
                if verdict == "PASS":
                    safety = "ROBUST"
                elif alarm_fraction >= 0.50:
                    safety = "DETECTED_OOD_ABSTAIN"
                elif inferential_harm:
                    safety = "SILENT_FAILURE"
                else:
                    safety = "NONROBUST_GATE_FAILURE"
                rows.append({
                    "case": case, "case_label": spec.label,
                    "is_ood": spec.is_ood, "atomic": spec.atomic,
                    "target": target, "strategy": strategy,
                    "verdict": verdict, "safety_class": safety,
                    "failed_pass_criteria": ";".join(
                        name for name, passed in pass_checks.items() if not passed
                    ),
                    "failed_conditional_criteria": ";".join(
                        name for name, passed in conditional_checks.items() if not passed
                    ),
                    "coverage90": coverage,
                    "coverage90_cluster_low": coverage_low,
                    "coverage90_cluster_high": coverage_high,
                    "rmse_ratio_over_elastic": rmse_ratio,
                    "rmse_ratio_cluster_upper": ratio_upper,
                    "median_width_ratio": width_ratio,
                    "width_ratio_cluster_upper": width_upper,
                    "false_confidence_fraction": float(false_confidence.mean()),
                    "false_confidence_cluster_upper": false_upper,
                    "harmful_false_confidence_fraction": float(harmful_false.mean()),
                    "harmful_false_confidence_cluster_upper": harmful_upper,
                    "standardized_bias": standardized_bias,
                    "standardized_bias_cluster_upper": bias_upper,
                    "median_reduced_chi2": reduced_chi2,
                    "raw_candidate_success_fraction": raw_success,
                    "mean_surviving_prior_mass": surviving_mass,
                    "mean_critical_bound_mass": critical_mass,
                    "ood_alarm_fraction": alarm_fraction,
                })
    return pd.DataFrame(rows)


def overall_phi_decision(gates: pd.DataFrame) -> pd.DataFrame:
    """Predeclared overall decision; adapted weights can never upgrade it."""
    selected = gates.loc[
        gates["target"].eq("phi") & gates["strategy"].eq("frozen_press")
    ].copy()
    expected = {case.name for case in OFF_LIBRARY_CASES}
    if set(selected["case"]) != expected:
        raise ValueError("Overall decision requires the matched control and all OOD cases")
    control = selected.loc[selected["case"].eq("matched_ema_control")].iloc[0]
    atomic = selected.loc[selected["atomic"].astype(bool)]
    combined = selected.loc[selected["case"].eq("combined_stress")].iloc[0]
    if control.verdict == "PASS" and atomic.verdict.eq("PASS").all() and combined.verdict == "PASS":
        decision = "GO"
        rationale = "Matched control and every predeclared off-library stress passed."
    elif (
        control.verdict in ("PASS", "CONDITIONAL")
        and ~atomic.verdict.eq("FAIL").any()
        and combined.verdict in ("PASS", "CONDITIONAL")
    ):
        decision = "CONDITIONAL"
        rationale = "No atomic stress failed, but at least one confirmatory gate was not a PASS."
    else:
        decision = "STOP"
        rationale = "The frozen estimator failed the matched control or at least one off-library stress."
    return pd.DataFrame([{
        "target": "phi", "primary_strategy": "frozen_press",
        "decision": decision, "rationale": rationale,
        "matched_control_verdict": control.verdict,
        "atomic_pass_count": int(atomic.verdict.eq("PASS").sum()),
        "atomic_conditional_count": int(atomic.verdict.eq("CONDITIONAL").sum()),
        "atomic_fail_count": int(atomic.verdict.eq("FAIL").sum()),
        "combined_verdict": combined.verdict,
    }])


def run_phase12c(
    training_recoveries: pd.DataFrame,
    training_components: pd.DataFrame,
    n_panels: int = 24,
    replicates: int = 3,
    panel_size: int = 8,
    seed: int = 20260917,
    cases: Sequence[OffLibraryCase] = OFF_LIBRARY_CASES,
    workers: int = 1,
    n_bootstrap: int = 4000,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    """Execute the independent Phase-1.2C falsification benchmark."""
    if n_panels < 3:
        raise ValueError("Phase 1.2C requires at least three independent panels")
    if replicates < 1 or panel_size < 2:
        raise ValueError("Replicates must be positive and panel_size at least two")
    cases = tuple(_case_lookup(case) for case in cases)
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Phase-1.2C case names must be unique")
    frozen_weights, calibration, training_reconstruction, alarm_threshold = (
        prepare_frozen_training_rule(training_recoveries, training_components)
    )
    calibration_lookup = {
        (row.strategy, row.target): float(row.c90)
        for row in calibration.itertuples(index=False)
    }
    design = phase11_design(n_panels, seed)
    noise_bank = {}
    for panel_index in range(n_panels):
        for replicate in range(replicates):
            rng = np.random.default_rng(seed + 17011 * panel_index + 223 * replicate)
            noise_bank[(panel_index, replicate)] = (
                rng.normal(size=(panel_size, 3)),
                rng.normal(size=(panel_size, 2)),
            )
    tasks = []
    for case_index, spec in enumerate(cases):
        for panel_index, truth in enumerate(design):
            for replicate in range(replicates):
                tasks.append((
                    spec, panel_index, replicate, truth, panel_size,
                    *noise_bank[(panel_index, replicate)],
                    seed + 700001 * case_index + 13007 * panel_index + 401 * replicate,
                ))
    completed = []
    if workers == 1:
        completed = [_fit_phase12c_observation(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fit_phase12c_observation, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
    completed.sort(key=lambda item: (item["case"], item["panel"], item["replicate"]))

    by_case: dict[str, dict[tuple[int, int], Sequence[Mapping]]] = {}
    for item in completed:
        by_case.setdefault(item["case"], {})[(item["panel"], item["replicate"])] = item["fits"]
    adapted_weights = {}
    for spec in cases:
        bank = by_case[spec.name]
        for panel_index in range(n_panels):
            adapted_weights[(spec.name, panel_index)] = _crossfit_weights(
                bank, panel_index,
            )

    recovery_rows, component_rows, diagnostic_rows = [], [], []
    for item in completed:
        fits = item["fits"]
        _validate_weight_order(fits)
        success_mask = np.array([bool(fit["success"]) for fit in fits])
        raw_success = float(success_mask.mean())
        frozen_surviving = float(np.dot(frozen_weights, success_mask))
        frozen = frozen_press_weights(fits, frozen_weights)
        adapted_prior = adapted_weights[(item["case"], item["panel"])]
        adapted_surviving = float(np.dot(adapted_prior, success_mask))
        adapted = converged_weights(fits, adapted_prior)
        best_chi2 = float(min(
            fit["reduced_chi2"] for fit in fits if fit["success"]
        ))
        spec = _case_lookup(item["case"])
        diagnostic_rows.append({
            "case": item["case"], "case_label": spec.label,
            "is_ood": spec.is_ood, "atomic": spec.atomic,
            "panel": item["panel"], "replicate": item["replicate"],
            "raw_candidate_success_fraction": raw_success,
            "best_reduced_chi2": best_chi2,
            "ood_alarm": bool(best_chi2 > alarm_threshold),
            "alarm_threshold": alarm_threshold,
            **item["truth_diagnostics"],
        })
        for index, fit in enumerate(fits):
            bound_hit = fit["bound_hit"]
            component_rows.append({
                "case": item["case"], "panel": item["panel"],
                "replicate": item["replicate"], "family": fit["family"],
                "coupling": fit["coupling"], "success": fit["success"],
                "predictive_score": fit["predictive_score"],
                "reduced_chi2": fit["reduced_chi2"], "nfev": fit["nfev"],
                "total_nfev": fit["total_nfev"],
                "convergence_fallback": fit["convergence_fallback"],
                "frozen_prior_weight": float(frozen_weights[index]),
                "frozen_weight": float(frozen[index]),
                "adapted_prior_weight": float(adapted_prior[index]),
                "adapted_weight": float(adapted[index]),
                "any_bound_hit": any(bound_hit.values()),
                "critical_bound_hit": any(
                    bound_hit.get(name, False)
                    for name in ("phi", "sw", "aspect", "secondary_fraction", "q_conn")
                ),
                **{
                    f"bound_hit_{name}": bound_hit.get(name, False)
                    for name in (
                        "phi", "sw", "aspect", "secondary_fraction", "q_conn",
                        "log_rw", "archie_m", "archie_n", "surface_cond",
                        "invasion", "rmf_ratio",
                    )
                },
                **{
                    f"estimate_{target}": fit["estimate"][target]
                    for target in TARGETS
                },
                **{
                    f"sd_{target}": fit["sd"][target]
                    for target in TARGETS
                },
            })

        elastic = item["elastic"]
        elastic_component = {
            "estimate": elastic["estimate"], "sd": elastic["sd"],
            "reduced_chi2": elastic["cost"] / max(3 * panel_size - len(TARGETS), 1),
            "success": elastic["success"],
        }
        for target in TARGETS:
            truth_value = float(item["truth"][target])
            elastic_result = mixture_recovery(
                [elastic_component], np.array([1.0]), target,
            )
            _append_recovery(
                recovery_rows, item, "elastic", target, truth_value,
                elastic_result, calibration_lookup[("elastic", target)],
                float(elastic["success"]), 1.0,
                float(any(elastic["bound_hit"].get(name, False) for name in TARGETS)),
                float(elastic["bound_hit"].get(target, False)),
                best_chi2, alarm_threshold,
            )
            for strategy, weights, surviving in (
                ("frozen_press", frozen, frozen_surviving),
                ("adapted_average", adapted, adapted_surviving),
            ):
                result = mixture_recovery(fits, weights, target)
                target_bound = np.array([
                    fit["bound_hit"].get(
                        "secondary_fraction" if target == "secondary" else target,
                        False,
                    )
                    for fit in fits
                ], dtype=float)
                critical_bound = np.array([
                    any(
                        fit["bound_hit"].get(name, False)
                        for name in (
                            "phi", "sw", "aspect", "secondary_fraction", "q_conn",
                        )
                    )
                    for fit in fits
                ], dtype=float)
                _append_recovery(
                    recovery_rows, item, strategy, target, truth_value, result,
                    calibration_lookup[(strategy, target)], raw_success,
                    surviving, float(np.dot(weights, critical_bound)),
                    float(np.dot(weights, target_bound)),
                    best_chi2, alarm_threshold,
                )

    raw = pd.DataFrame(recovery_rows).sort_values(
        ["case", "panel", "replicate", "strategy", "target"], ignore_index=True,
    )
    components = pd.DataFrame(component_rows).sort_values(
        ["case", "panel", "replicate", "family", "coupling"], ignore_index=True,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["case", "panel", "replicate"], ignore_index=True,
    )
    summary = summarize_phase12c(raw)
    gates = build_phase12c_gates(raw, n_bootstrap=n_bootstrap)
    if {case.name for case in cases} == {case.name for case in OFF_LIBRARY_CASES}:
        overall = overall_phi_decision(gates)
    else:
        overall = pd.DataFrame()
    distance = truth_library_distance(design, cases, panel_size)
    weights_frame = pd.DataFrame([
        {"family": family, "coupling": coupling, "frozen_weight": weight}
        for (family, coupling), weight in zip(CANDIDATE_KEYS, frozen_weights)
    ])
    return (
        raw, components, diagnostics, summary, gates, overall, calibration,
        weights_frame, training_reconstruction, distance,
    )
