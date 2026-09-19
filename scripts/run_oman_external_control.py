#!/usr/bin/env python3
"""Reproduce the Oman GT3A external porosity-information control.

The implementation follows the prospectively frozen protocol recorded in
``data/oman/provenance/phase14e_protocol_lock.json``.  It operates on one row
per physical core sample; measurement directions are never treated as
independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "oman" / "oman_gt3a_sample_level.csv"
DEFAULT_REFERENCE = ROOT / "results" / "oman" / "phase14e_result.json"
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def rmse(values: Sequence[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


def mae(values: Sequence[float]) -> float:
    return mean([abs(value) for value in values])


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Gauss--Jordan solver matching the locked JavaScript implementation."""
    size = len(rhs)
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            augmented[pivot][column] = 1.0e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [row[size] for row in augmented]


@dataclass(frozen=True)
class RidgeModel:
    keys: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    target_mean: float
    coefficients: tuple[float, ...]

    def predict(self, row: dict[str, object]) -> float:
        standardized = [
            (float(row[key]) - self.means[index]) / self.scales[index]
            for index, key in enumerate(self.keys)
        ]
        return self.target_mean + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )


def fit_ridge(rows: Sequence[dict[str, object]], keys: Sequence[str], alpha: float) -> RidgeModel:
    x_means = [mean([float(row[key]) for row in rows]) for key in keys]
    x_scales = []
    for key, center in zip(keys, x_means):
        scale = math.sqrt(mean([(float(row[key]) - center) ** 2 for row in rows]))
        x_scales.append(scale if scale > 1.0e-12 else 1.0)
    target_mean = mean([float(row["phi"]) for row in rows])
    size = len(keys)
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row in rows:
        standardized = [
            (float(row[key]) - x_means[index]) / x_scales[index]
            for index, key in enumerate(keys)
        ]
        centered_target = float(row["phi"]) - target_mean
        for left in range(size):
            rhs[left] += standardized[left] * centered_target
            for right in range(size):
                matrix[left][right] += standardized[left] * standardized[right]
    for index in range(size):
        matrix[index][index] += alpha
    coefficients = solve_linear(matrix, rhs)
    return RidgeModel(
        tuple(keys), tuple(x_means), tuple(x_scales), target_mean, tuple(coefficients)
    )


def choose_alpha(rows: Sequence[dict[str, object]], keys: Sequence[str]) -> tuple[float, float]:
    blocks = sorted({int(row["block"]) for row in rows})
    best_alpha = ALPHA_GRID[0]
    best_score = math.inf
    for alpha in ALPHA_GRID:
        block_errors = []
        for block in blocks:
            training = [row for row in rows if int(row["block"]) != block]
            testing = [row for row in rows if int(row["block"]) == block]
            if not training or not testing:
                continue
            model = fit_ridge(training, keys, alpha)
            block_errors.append(
                mean([(float(row["phi"]) - model.predict(row)) ** 2 for row in testing])
            )
        score = mean(block_errors)
        if score < best_score - 1.0e-12 or (
            abs(score - best_score) <= 1.0e-12 and alpha > best_alpha
        ):
            best_alpha, best_score = alpha, score
    return best_alpha, best_score


def outer_cv(
    rows: Sequence[dict[str, object]], keys: Sequence[str], label: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for block in range(1, 9):
        training = [row for row in rows if int(row["block"]) != block]
        testing = [row for row in rows if int(row["block"]) == block]
        if not testing:
            continue
        alpha, inner_mse = choose_alpha(training, keys)
        model = fit_ridge(training, keys, alpha)
        fold_predictions = []
        for row in testing:
            prediction = dict(row)
            prediction.update(model=label, alpha=alpha, prediction=model.predict(row))
            fold_predictions.append(prediction)
        predictions.extend(fold_predictions)
        errors = [float(row["phi"]) - float(row["prediction"]) for row in fold_predictions]
        folds.append(
            {
                "model": label,
                "block": block,
                "n_train": len(training),
                "n_test": len(testing),
                "alpha": alpha,
                "inner_mse": inner_mse,
                "fold_rmse": rmse(errors),
                "fold_mae": mae(errors),
            }
        )
    return predictions, folds


def paired_metrics(
    elastic: Sequence[dict[str, object]], joint: Sequence[dict[str, object]]
) -> dict[str, object]:
    elastic_by_id = {str(row["id"]): row for row in elastic}
    pairs = [(elastic_by_id[str(row["id"])], row) for row in joint]
    elastic_errors = [float(row["phi"]) - float(row["prediction"]) for row, _ in pairs]
    joint_errors = [float(row["phi"]) - float(row["prediction"]) for _, row in pairs]
    elastic_rmse, joint_rmse = rmse(elastic_errors), rmse(joint_errors)
    elastic_mae, joint_mae = mae(elastic_errors), mae(joint_errors)
    return {
        "pairs": pairs,
        "rmse_elastic": elastic_rmse,
        "rmse_joint": joint_rmse,
        "rmse_ratio": joint_rmse / elastic_rmse,
        "mae_elastic": elastic_mae,
        "mae_joint": joint_mae,
        "mae_ratio": joint_mae / elastic_mae,
        "mean_delta_squared_error": mean(
            [
                (float(e["phi"]) - float(e["prediction"])) ** 2
                - (float(j["phi"]) - float(j["prediction"])) ** 2
                for e, j in pairs
            ]
        ),
    }


class Mulberry32:
    """Bit-exact port of the generator used for the frozen analysis."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    @staticmethod
    def _imul(left: int, right: int) -> int:
        value = ((left & 0xFFFFFFFF) * (right & 0xFFFFFFFF)) & 0xFFFFFFFF
        return value

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        value = self.state
        value = self._imul(value ^ (value >> 15), 1 | value)
        previous = value
        value = (
            ((previous + self._imul(previous ^ (previous >> 7), 61 | previous))
             & 0xFFFFFFFF)
            ^ previous
        )
        value ^= value >> 14
        return (value & 0xFFFFFFFF) / 4294967296.0


def bootstrap_ratios(metrics: dict[str, object], draws: int, seed: int) -> list[float]:
    generator = Mulberry32(seed)
    pairs = metrics["pairs"]
    assert isinstance(pairs, list)
    blocks = sorted({int(pair[0]["block"]) for pair in pairs})
    by_block = {block: [pair for pair in pairs if int(pair[0]["block"]) == block] for block in blocks}
    ratios = []
    for _ in range(draws):
        sampled = []
        for _ in blocks:
            block = blocks[math.floor(generator.random() * len(blocks))]
            sampled.extend(by_block[block])
        joint_errors = [float(j["phi"]) - float(j["prediction"]) for _, j in sampled]
        elastic_errors = [float(e["phi"]) - float(e["prediction"]) for e, _ in sampled]
        ratios.append(rmse(joint_errors) / rmse(elastic_errors))
    return sorted(ratios)


def permute_within_sequence(
    rows: Sequence[dict[str, object]], key: str, generator: Mulberry32
) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(output):
        groups.setdefault(str(row["sequence"]), []).append(index)
    for indices in groups.values():
        values = [output[index][key] for index in indices]
        for index in range(len(values) - 1, 0, -1):
            other = math.floor(generator.random() * (index + 1))
            values[index], values[other] = values[other], values[index]
        for index, value in zip(indices, values):
            output[index][key] = value
    return output


def permutation_test(
    rows: Sequence[dict[str, object]],
    elastic_predictions: Sequence[dict[str, object]],
    joint_keys: Sequence[str],
    electrical_key: str,
    draws: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[float], float]:
    generator = Mulberry32(seed)
    joint_predictions, joint_folds = outer_cv(rows, joint_keys, f"J_{electrical_key}")
    observed = paired_metrics(elastic_predictions, joint_predictions)
    observed_difference = float(observed["rmse_elastic"]) - float(observed["rmse_joint"])
    permutation_differences = []
    exceedances = 0
    for _ in range(draws):
        permuted = permute_within_sequence(rows, electrical_key, generator)
        permuted_predictions, _ = outer_cv(permuted, joint_keys, f"J_perm_{electrical_key}")
        permuted_metrics = paired_metrics(elastic_predictions, permuted_predictions)
        difference = float(permuted_metrics["rmse_elastic"]) - float(permuted_metrics["rmse_joint"])
        permutation_differences.append(difference)
        if difference >= observed_difference - 1.0e-12:
            exceedances += 1
    p_value = (1 + exceedances) / (draws + 1)
    return observed, joint_predictions, joint_folds, sorted(permutation_differences), p_value


def parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for source in csv.DictReader(stream):
            row = {
                "source_row": int(source["Source row"]),
                "core": source["Core"],
                "section": source["Section"],
                "depth": parse_optional_float(source["Depth (m)"]),
                "lithology": source["Lithology"],
                "sequence": source["Sequence"],
                "density": parse_optional_float(source["Bulk density"]),
                "phi": parse_optional_float(source["Porosity (%)"]),
                "vp_wet": parse_optional_float(source["Wet Vp mean"]),
                "vs_wet": parse_optional_float(source["Wet Vs mean"]),
                "log_r35": parse_optional_float(source["log10 R35"]),
                "log_f": parse_optional_float(source["log10 F"]),
                "block": int(source["Depth block"]),
                "primary_complete": source["Primary complete"].lower() == "true",
                "f_complete": source["F complete"].lower() == "true",
            }
            row["id"] = f"{row['core']}|{row['section']}|{row['depth']}"
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])


def summarize_variant(
    name: str,
    rows: Sequence[dict[str, object]],
    electrical_key: str,
    bootstrap_draws: int,
    permutation_draws: int,
    seed_offset: int,
) -> dict[str, object]:
    elastic_keys = ("vp_wet", "vs_wet", "density")
    joint_keys = (*elastic_keys, electrical_key)
    elastic_predictions, elastic_folds = outer_cv(rows, elastic_keys, f"{name}_E")
    metrics, joint_predictions, joint_folds, permuted, p_value = permutation_test(
        rows,
        elastic_predictions,
        joint_keys,
        electrical_key,
        permutation_draws,
        14052027 + seed_offset,
    )
    ratios = bootstrap_ratios(metrics, bootstrap_draws, 14052026 + seed_offset)
    return {
        "name": name,
        "n": len(rows),
        "metrics": metrics,
        "elastic_predictions": elastic_predictions,
        "joint_predictions": joint_predictions,
        "elastic_folds": elastic_folds,
        "joint_folds": joint_folds,
        "bootstrap": {
            "draws": bootstrap_draws,
            "lower": quantile(ratios, 0.025),
            "median": quantile(ratios, 0.5),
            "upper": quantile(ratios, 0.975),
        },
        "permutation": {
            "draws": permutation_draws,
            "p_value": p_value,
            "lower": quantile(permuted, 0.025),
            "median": quantile(permuted, 0.5),
            "upper": quantile(permuted, 0.975),
        },
    }


def serializable_summary(variant: dict[str, object]) -> dict[str, object]:
    metrics = variant["metrics"]
    assert isinstance(metrics, dict)
    return {
        "n": variant["n"],
        "rmse_elastic": metrics["rmse_elastic"],
        "rmse_joint": metrics["rmse_joint"],
        "rmse_ratio": metrics["rmse_ratio"],
        "mae_elastic": metrics["mae_elastic"],
        "mae_joint": metrics["mae_joint"],
        "mae_ratio": metrics["mae_ratio"],
        "bootstrap_rmse_ratio_l95": variant["bootstrap"]["lower"],
        "bootstrap_rmse_ratio_u95": variant["bootstrap"]["upper"],
        "permutation_p": variant["permutation"]["p_value"],
        "permutation_draws": variant["permutation"]["draws"],
        "bootstrap_draws": variant["bootstrap"]["draws"],
    }


def verify(reference_path: Path, result: dict[str, object], tolerance: float = 1.0e-11) -> None:
    """Verify every frozen statistic emitted by the default full run.

    The reference JSON also contains interpretation and decision fields that
    are not recalculated by this numerical script.  Verification therefore
    covers the protocol identifier plus all sample counts, draw counts, point
    metrics, confidence limits, and permutation probabilities shared with the
    generated summary.
    """
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if result.get("protocol_id") != reference.get("protocol_id"):
        raise RuntimeError(
            "verification failed for protocol_id: "
            f"{result.get('protocol_id')} != {reference.get('protocol_id')}"
        )

    integer_checks = {
        "primary.n": (result["primary"]["n"], reference["primary"]["n"]),
        "primary.bootstrap_draws": (
            result["primary"]["bootstrap_draws"],
            reference["primary"]["bootstrap_draws"],
        ),
        "primary.permutation_draws": (
            result["primary"]["permutation_draws"],
            reference["primary"]["permutation_draws"],
        ),
        "formation_factor_sensitivity.n": (
            result["formation_factor_sensitivity"]["n"],
            reference["formation_factor_sensitivity"]["n"],
        ),
    }
    for field, (actual, expected) in integer_checks.items():
        if int(actual) != int(expected):
            raise RuntimeError(f"verification failed for {field}: {actual} != {expected}")

    float_fields = {
        "primary": (
            "rmse_elastic",
            "rmse_joint",
            "rmse_ratio",
            "mae_elastic",
            "mae_joint",
            "mae_ratio",
            "bootstrap_rmse_ratio_l95",
            "bootstrap_rmse_ratio_u95",
            "permutation_p",
        ),
        "formation_factor_sensitivity": (
            "rmse_ratio",
            "mae_ratio",
            "bootstrap_rmse_ratio_l95",
            "bootstrap_rmse_ratio_u95",
            "permutation_p",
        ),
    }
    for variant, fields in float_fields.items():
        for field in fields:
            actual = float(result[variant][field])
            expected = float(reference[variant][field])
            label = f"{variant}.{field}"
            if math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
                continue
            raise RuntimeError(
                f"verification failed for {label}: {actual} != {expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "oman" / "reproduced")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--verify-reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"input table not found: {args.input}. Download the official Oman "
            "workbook and run scripts/prepare_oman_data.py first; see "
            "data/oman/README.md"
        )
    all_rows = load_rows(args.input)
    primary_rows = [row for row in all_rows if bool(row["primary_complete"])]
    formation_rows = [row for row in all_rows if bool(row["f_complete"])]
    primary = summarize_variant(
        "Primary_R35", primary_rows, "log_r35", args.bootstrap, args.permutations, 0
    )
    formation = summarize_variant(
        "Sensitivity_F", formation_rows, "log_f", args.bootstrap, args.permutations, 1000
    )
    result = {
        "protocol_id": "phase14e_oman_external_transfer_v1",
        "primary": serializable_summary(primary),
        "formation_factor_sensitivity": serializable_summary(formation),
    }
    if args.bootstrap == 10000 and args.permutations == 999:
        verify(args.verify_reference, result)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_rows = []
    for label, variant in (("primary", primary), ("formation_factor", formation)):
        joint_by_id = {str(row["id"]): row for row in variant["joint_predictions"]}
        for elastic in variant["elastic_predictions"]:
            joint = joint_by_id[str(elastic["id"])]
            prediction_rows.append(
                {
                    "variant": label,
                    "id": elastic["id"],
                    "block": elastic["block"],
                    "sequence": elastic["sequence"],
                    "lithology": elastic["lithology"],
                    "porosity": elastic["phi"],
                    "elastic_prediction": elastic["prediction"],
                    "joint_prediction": joint["prediction"],
                    "elastic_alpha": elastic["alpha"],
                    "joint_alpha": joint["alpha"],
                }
            )
    write_csv(
        args.output / "outer_predictions.csv",
        prediction_rows,
        (
            "variant",
            "id",
            "block",
            "sequence",
            "lithology",
            "porosity",
            "elastic_prediction",
            "joint_prediction",
            "elastic_alpha",
            "joint_alpha",
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
