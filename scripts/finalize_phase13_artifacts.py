#!/usr/bin/env python3
"""Audit Phase 1.3 outputs and refresh their portable integrity manifest.

This finalizer never refits a model or changes a scientific result.  It checks
the persisted relational contract, policy identity, observation-bank alignment
and decision, then re-hashes the report and complete local source closure.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase13_selective"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def normalize_legacy_alarm_columns() -> None:
    """Disambiguate inherited q95 diagnostics without changing their values."""
    names = {
        "ood_alarm": "legacy_phase12c_ood_alarm_q95",
        "alarm_threshold": "legacy_phase12c_alarm_threshold_q95",
    }
    for filename in ("recovery_raw.csv", "observation_diagnostics.csv"):
        path = OUT / filename
        frame = pd.read_csv(path)
        if any(name in frame.columns for name in names):
            frame = frame.rename(columns=names)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False,
                newline="",
            ) as stream:
                frame.to_csv(stream, index=False, lineterminator="\n")
                temporary = Path(stream.name)
            temporary.replace(path)


def audit_contract() -> dict[str, int | str]:
    policy = json.loads((OUT / "policy_lock.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run_config.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(OUT / "recovery_raw.csv")
    components = pd.read_csv(OUT / "candidate_components.csv")
    diagnostics = pd.read_csv(OUT / "observation_diagnostics.csv")
    selective = pd.read_csv(OUT / "selective_observations.csv")
    curve = pd.read_csv(OUT / "risk_coverage_curve.csv")
    gates = pd.read_csv(OUT / "decision_gates.csv")
    controls = pd.read_csv(OUT / "control_summary.csv")
    overall = pd.read_csv(OUT / "overall_decision.csv")
    index = pd.read_csv(OUT / "observation_bank_index.csv")

    assert run["policy_hash"] == policy["policy_hash"]
    assert run["policy_id"] == policy["policy"]["policy_id"]
    assert run["policy_locked_before_test_generation"] is True
    assert len(diagnostics) == len(selective) == len(index) == 1152
    assert len(components) == 10368
    assert len(raw) == 13824
    assert len(curve) == 54 and len(gates) == 6 and len(controls) == 18
    assert overall.iloc[0]["decision"] == "STOP"
    assert not diagnostics.duplicated(["case", "panel", "replicate"]).any()
    assert not components.duplicated(
        ["case", "panel", "replicate", "family", "coupling"]
    ).any()
    assert not raw.duplicated(
        ["case", "panel", "replicate", "strategy", "target"]
    ).any()
    assert components.groupby(["case", "panel", "replicate"]).size().eq(9).all()
    assert raw.groupby(["case", "panel", "replicate"]).size().eq(12).all()
    assert set(gates.set_index("case")["verdict"]) == {
        "SAFE_JOINT", "SILENT_FAILURE", "SAFE_DOMAIN_REJECT"
    }
    assert gates.set_index("case")["verdict"].to_dict() == {
        "matched_ema_control": "SAFE_JOINT",
        "hybrid_law": "SILENT_FAILURE",
        "connectivity_drift": "SAFE_JOINT",
        "invasion_mismatch": "SILENT_FAILURE",
        "patchy_saturation": "SAFE_JOINT",
        "combined_stress": "SAFE_DOMAIN_REJECT",
    }

    with np.load(OUT / "observation_bank.npz", allow_pickle=False) as bank:
        assert bank["elastic"].shape == (1152, 8, 3)
        assert bank["log_rt"].shape == (1152, 8, 2)
        assert bank["truth_targets"].shape == (1152, 4)
        for name in ("elastic", "log_rt", "truth_targets"):
            assert np.isfinite(bank[name]).all()

    return {
        "test_observations": len(diagnostics),
        "candidate_fits": len(components),
        "successful_candidate_fits": int(components["success"].sum()),
        "recoveries": len(raw),
        "selective_observations": len(selective),
        "risk_curve_rows": len(curve),
        "gates": len(gates),
        "control_rows": len(controls),
        "decision": str(overall.iloc[0]["decision"]),
        "policy_id": str(run["policy_id"]),
        "policy_hash": str(run["policy_hash"]),
    }


def refresh_manifest(audit: dict[str, int | str]) -> Path:
    old = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    artifacts = {
        path.name: sha256(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    source_paths = sorted((ROOT / "src/jointinv").glob("*.py"))
    source_paths.extend(
        path for path in (
            ROOT / "scripts/run_phase13_selective.py",
            ROOT / "scripts/finalize_phase13_artifacts.py",
            ROOT / "scripts/build_phase13_pdf.py",
            ROOT / "scripts/package_phase13.py",
            ROOT / "tests/test_phase13.py",
            ROOT / "README.md",
            ROOT / "CHANGELOG.md",
            ROOT / "pyproject.toml",
        ) if path.exists()
    )
    inputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            ROOT / "outputs/phase11_carbonate/recovery_strategies.csv",
            ROOT / "outputs/phase11_carbonate/fit_components.csv",
        )
    }
    manifest = {
        **{key: value for key, value in old.items()
           if key not in {"artifacts", "sources", "inputs", "counts", "decision"}},
        "artifacts": artifacts,
        "sources": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "inputs": inputs,
        "counts": {
            key: value for key, value in audit.items()
            if key not in {"decision", "policy_id", "policy_hash"}
        },
        "decision": audit["decision"],
        "policy_id": audit["policy_id"],
        "policy_hash": audit["policy_hash"],
    }
    path = OUT / "manifest.json"
    atomic_json(path, manifest)
    return path


def verify_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        assert sha256(OUT / name) == expected, name
    for name, expected in manifest["sources"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in manifest["inputs"].items():
        assert sha256(ROOT / name) == expected, name


def main() -> None:
    normalize_legacy_alarm_columns()
    audit = audit_contract()
    path = refresh_manifest(audit)
    verify_manifest(path)
    print(path)
    print(f"sha256={sha256(path)}")


if __name__ == "__main__":
    main()
