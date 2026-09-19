#!/usr/bin/env python3
"""Audit and deterministically package the complete Phase 1.2C project."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase12c_offlibrary"
PDF = ROOT / "output/pdf/PHASE12C_OFFLIBRARY_REPORT.pdf"
ARCHIVE = ROOT.parent / "joint_elastic_electrical_phase12c_offlibrary_v0_7.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        assert sha256(OUT / name) == expected, name
    for name, expected in manifest["sources"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in manifest["inputs"].items():
        assert sha256(Path(name)) == expected, name

    raw = pd.read_csv(OUT / "recovery_raw.csv")
    components = pd.read_csv(OUT / "candidate_components.csv")
    diagnostics = pd.read_csv(OUT / "observation_diagnostics.csv")
    gates = pd.read_csv(OUT / "decision_gates.csv")
    overall = pd.read_csv(OUT / "overall_decision.csv")
    reconstruction = pd.read_csv(OUT / "training_rule_reconstruction.csv")
    assert len(raw) == 5184
    assert len(components) == 3888
    assert len(diagnostics) == 432
    assert len(gates) == 48
    assert len(reconstruction) == 960
    assert components["success"].all()
    assert not raw.duplicated(
        ["case", "panel", "replicate", "strategy", "target"]
    ).any()
    assert not components.duplicated(
        ["case", "panel", "replicate", "family", "coupling"]
    ).any()
    candidate_counts = components.groupby(
        ["case", "panel", "replicate"]
    ).size()
    assert candidate_counts.eq(9).all()
    weight_sums = components.groupby(
        ["case", "panel", "replicate"]
    )[["frozen_weight", "adapted_weight"]].sum()
    assert np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-12)
    assert overall.decision.iloc[0] == "STOP"
    phi = gates[
        gates.target.eq("phi") & gates.strategy.eq("frozen_press")
    ].set_index("case")
    assert phi.verdict.to_dict() == {
        "matched_ema_control": "FAIL",
        "hybrid_law": "CONDITIONAL",
        "connectivity_drift": "FAIL",
        "invasion_mismatch": "FAIL",
        "patchy_saturation": "FAIL",
        "combined_stress": "FAIL",
    }
    reader = PdfReader(str(PDF))
    assert len(reader.pages) == 4
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in (
        "DECISION: STOP", "Why the formal gate fails",
        "Exploratory selective-risk audit", "Recommended next gate",
    ):
        assert phrase in text
    return raw, components, diagnostics, gates, overall, reconstruction


def write_manifest(raw, components, diagnostics, gates, overall, reconstruction):
    important = [
        ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "pyproject.toml",
        ROOT / "src/jointinv/phase12c.py",
        ROOT / "scripts/run_phase12c_falsification.py",
        ROOT / "scripts/finalize_phase12c_artifacts.py",
        ROOT / "scripts/build_phase12c_pdf.py",
        ROOT / "scripts/package_phase12c.py",
        ROOT / "tests/test_phase12c.py",
        OUT / "manifest.json", OUT / "PHASE12C_OFFLIBRARY_REPORT.md", PDF,
    ]
    manifest = {
        "project": "joint elastic-electrical rock-physics inversion",
        "phase": "1.2C", "package_version": "0.7.0",
        "unit_tests_passed": 35,
        "candidate_fit_rows": len(components),
        "recovery_rows": len(raw),
        "observation_rows": len(diagnostics),
        "gate_rows": len(gates),
        "training_reconstruction_rows": len(reconstruction),
        "overall_decision": overall.decision.iloc[0],
        "phi_decisions": gates[
            gates.target.eq("phi") & gates.strategy.eq("frozen_press")
        ].set_index("case")["verdict"].to_dict(),
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in important
        },
    }
    path = ROOT / "PHASE12C_PACKAGE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def package():
    excluded_parts = {
        "__pycache__", "tmp", ".pytest_cache", "jointinv_phase0.egg-info",
    }
    files = [
        path for path in ROOT.rglob("*") if path.is_file()
        and not excluded_parts.intersection(path.relative_to(ROOT).parts)
        and path.suffix != ".pyc"
    ]
    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as archive:
        for path in sorted(files):
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(str(relative), date_time=(2026, 8, 29, 17, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        assert bad is None, bad
        assert str(Path(ROOT.name) / "PHASE12C_PACKAGE_MANIFEST.json") in archive.namelist()
    print(ARCHIVE)
    print(f"size={ARCHIVE.stat().st_size}")
    print(f"sha256={sha256(ARCHIVE)}")


def main():
    audited = audit()
    write_manifest(*audited)
    package()


if __name__ == "__main__":
    main()
