#!/usr/bin/env python3
"""Audit and deterministically package the complete Phase 1.3 project."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase13_selective"
PDF = ROOT / "output/pdf/PHASE13_SELECTIVE_REPORT.pdf"
PACKAGE_MANIFEST = ROOT / "PHASE13_PACKAGE_MANIFEST.json"
ARCHIVE = ROOT.parent / "joint_elastic_electrical_phase13_selective_v0_8.zip"
EXCLUDED_PARTS = {
    "__pycache__", "tmp", ".pytest_cache", "jointinv_phase0.egg-info",
}
ARCHIVE_TIMESTAMP = (2026, 8, 29, 18, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(*, include_package_manifest: bool) -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        if not include_package_manifest and path == PACKAGE_MANIFEST:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def refresh_scientific_manifest() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/finalize_phase13_artifacts.py")],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )


def audit() -> dict[str, object]:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        assert sha256(OUT / name) == expected, name
    for name, expected in manifest["sources"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in manifest["inputs"].items():
        assert sha256(ROOT / name) == expected, name

    raw = pd.read_csv(OUT / "recovery_raw.csv")
    components = pd.read_csv(OUT / "candidate_components.csv")
    diagnostics = pd.read_csv(OUT / "observation_diagnostics.csv")
    selective = pd.read_csv(OUT / "selective_observations.csv")
    gates = pd.read_csv(OUT / "decision_gates.csv")
    controls = pd.read_csv(OUT / "control_summary.csv")
    overall = pd.read_csv(OUT / "overall_decision.csv")
    assert len(raw) == 13824 and len(components) == 10368
    assert len(diagnostics) == len(selective) == 1152
    assert len(gates) == 6 and len(controls) == 18
    assert int(components["success"].sum()) == 10365
    assert int(selective["accepted"].sum()) == 817
    assert not raw.duplicated(
        ["case", "panel", "replicate", "strategy", "target"]
    ).any()
    assert not components.duplicated(
        ["case", "panel", "replicate", "family", "coupling"]
    ).any()
    assert components.groupby(["case", "panel", "replicate"]).size().eq(9).all()
    weight_sums = components.groupby(
        ["case", "panel", "replicate"]
    )[["frozen_weight", "adapted_weight"]].sum()
    assert np.allclose(weight_sums.to_numpy(), 1.0, atol=2e-12)
    assert overall.iloc[0]["decision"] == "STOP"
    verdicts = gates.set_index("case")["verdict"].to_dict()
    assert verdicts == {
        "matched_ema_control": "SAFE_JOINT",
        "hybrid_law": "SILENT_FAILURE",
        "connectivity_drift": "SAFE_JOINT",
        "invasion_mismatch": "SILENT_FAILURE",
        "patchy_saturation": "SAFE_JOINT",
        "combined_stress": "SAFE_DOMAIN_REJECT",
    }
    for frame in (raw, diagnostics):
        assert "ood_alarm" not in frame.columns
        assert "alarm_threshold" not in frame.columns
        assert "legacy_phase12c_ood_alarm_q95" in frame.columns
        assert "legacy_phase12c_alarm_threshold_q95" in frame.columns

    reader = PdfReader(str(PDF))
    assert len(reader.pages) == 5
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in (
        "DECISION: STOP", "SAFE_DOMAIN_REJECT", "Hybrid law",
        "Invasion mismatch", "Policy provenance and reproducibility",
    ):
        assert phrase in pdf_text, phrase
    report_text = (OUT / "PHASE13_SELECTIVE_REPORT.md").read_text(encoding="utf-8")
    for phrase in (
        "marginal,", "structural failure", "Merely changing the chi-square cutoff",
    ):
        assert phrase in report_text, phrase
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--output outputs/phase13_reproduced" in readme_text
    assert "must not be overwritten" in readme_text

    return {
        "scientific_manifest_sha256": sha256(OUT / "manifest.json"),
        "pdf_sha256": sha256(PDF),
        "policy_hash": manifest["policy_hash"],
        "overall_decision": str(overall.iloc[0]["decision"]),
        "verdicts": verdicts,
        "negative_control_alerts": int(
            controls["promoted_by_phi_guard"].astype(bool).sum()
        ),
        "test_observations": len(diagnostics),
        "candidate_fits": len(components),
        "successful_candidate_fits": int(components["success"].sum()),
        "recovery_rows": len(raw),
        "accepted_observations": int(selective["accepted"].sum()),
        "abstained_observations": int(selective["abstained"].sum()),
    }


def write_package_manifest(audit_result: dict[str, object]) -> None:
    members = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in included_files(include_package_manifest=False)
    }
    payload = {
        "project": "joint elastic-electrical rock-physics inversion",
        "phase": "1.3",
        "package_version": "0.8.0",
        "archive_name": ARCHIVE.name,
        "unit_tests_passed": 45,
        **audit_result,
        "member_count_excluding_this_manifest": len(members),
        "members_sha256": members,
    }
    PACKAGE_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_archive() -> None:
    files = included_files(include_package_manifest=True)
    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as archive:
        for path in files:
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def verify_archive() -> None:
    package_manifest = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    prefix = f"{ROOT.name}/"
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert prefix + PACKAGE_MANIFEST.name in names
        expected_names = {
            prefix + path.relative_to(ROOT).as_posix()
            for path in included_files(include_package_manifest=True)
        }
        assert names == expected_names
        for relative, expected in package_manifest["members_sha256"].items():
            actual = hashlib.sha256(archive.read(prefix + relative)).hexdigest()
            assert actual == expected, relative


def main() -> None:
    refresh_scientific_manifest()
    audit_result = audit()
    write_package_manifest(audit_result)
    write_archive()
    verify_archive()
    print(PACKAGE_MANIFEST)
    print(f"package_manifest_sha256={sha256(PACKAGE_MANIFEST)}")
    print(ARCHIVE)
    print(f"archive_size={ARCHIVE.stat().st_size}")
    print(f"archive_sha256={sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
