#!/usr/bin/env python3
"""Rebuild Phase-1.2C derived artifacts without rerunning nonlinear fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from jointinv.phase12c import (
    build_phase12c_gates, overall_phi_decision, summarize_phase12c,
)
from run_phase12c_falsification import (
    ROOT, plot_alarm_and_prior, plot_phi_performance, plot_truth_distance,
    report_text, summarize_alarm_selective_risk, summarize_bound_mass,
    summarize_control_verdicts,
)


DEFAULT_OUTPUT = ROOT / "outputs/phase12c_offlibrary"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    output = args.output
    raw = pd.read_csv(output / "recovery_raw.csv")
    components = pd.read_csv(output / "candidate_components.csv")
    diagnostics = pd.read_csv(output / "observation_diagnostics.csv")
    calibration = pd.read_csv(output / "frozen_calibration.csv")
    weights = pd.read_csv(output / "frozen_prior_weights.csv")
    reconstruction = pd.read_csv(output / "training_rule_reconstruction.csv")
    distance = pd.read_csv(output / "truth_library_distance.csv")
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))

    calibration["calibration_source"] = calibration["calibration_source"].replace({
        "phase11_elastic_outer_panel": "phase11_elastic_empirical_bank",
        "phase11_model_average_outer_panel": "phase11_model_average_empirical_bank",
        "phase11_nested_frozen_reconstruction": (
            "phase11_outer_panel_frozen_reconstruction_empirical"
        ),
    })
    summary = summarize_phase12c(raw)
    gates = build_phase12c_gates(raw, n_bootstrap=args.bootstrap)
    overall = overall_phi_decision(gates)
    bound_mass = summarize_bound_mass(components)
    control_verdicts = summarize_control_verdicts(gates)
    selective_risk = summarize_alarm_selective_risk(raw)
    calibration.to_csv(output / "frozen_calibration.csv", index=False)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    gates.to_csv(output / "decision_gates.csv", index=False)
    overall.to_csv(output / "overall_decision.csv", index=False)
    bound_mass.to_csv(output / "bound_mass_decomposition.csv", index=False)
    control_verdicts.to_csv(output / "control_verdict_summary.csv", index=False)
    selective_risk.to_csv(output / "alarm_selective_risk.csv", index=False)

    plot_phi_performance(gates, output / "phi_offlibrary_performance.png")
    plot_truth_distance(distance, output / "truth_library_distance.png")
    plot_alarm_and_prior(
        diagnostics, weights, output / "ood_alarm_and_frozen_prior.png",
    )
    report = report_text(
        gates, overall, calibration, diagnostics, distance,
        bound_mass, control_verdicts,
        selective_risk,
        int(config["n_panels"]), int(config["replicates"]),
        int(config["panel_size"]), int(config["seed"]),
    )
    (output / "PHASE12C_OFFLIBRARY_REPORT.md").write_text(report, encoding="utf-8")

    config["cluster_bootstrap_draws"] = args.bootstrap
    config["runtime_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8",
    )

    governed = [
        ROOT / "src/jointinv/analysis.py",
        ROOT / "src/jointinv/families.py",
        ROOT / "src/jointinv/phase05.py",
        ROOT / "src/jointinv/phase1.py",
        ROOT / "src/jointinv/phase11.py",
        ROOT / "src/jointinv/phase12.py",
        ROOT / "src/jointinv/phase12c.py",
        ROOT / "scripts/run_phase12c_falsification.py",
        ROOT / "scripts/build_phase12c_pdf.py",
        Path(__file__),
        ROOT / "tests/test_phase12c.py",
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
    ]
    artifacts = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "phase": "1.2C",
        "decision": overall.decision.iloc[0],
        "counts": {
            "recoveries": len(raw), "candidate_fits": len(components),
            "observations": len(diagnostics), "gates": len(gates),
            "training_reconstruction_rows": len(reconstruction),
        },
        "artifacts": {
            path.relative_to(output).as_posix(): sha256(path)
            for path in artifacts
        },
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in governed},
        "inputs": config["training_inputs"],
        "runtime_versions": config["runtime_versions"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(overall.to_string(index=False))
    print(f"Finalized {output}")


if __name__ == "__main__":
    main()
