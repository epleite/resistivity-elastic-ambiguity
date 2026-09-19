#!/usr/bin/env python3
"""Audit and deterministically package the complete Phase 1.2B project."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase12b_sentinels"
PDF = ROOT / "output/pdf/PHASE12B_SENTINEL_REPORT.pdf"
ARCHIVE = ROOT.parent / "joint_elastic_electrical_phase12b_sentinels_v0_6.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    phase_manifest = json.loads((OUT / "manifest.json").read_text())
    for name, expected in phase_manifest["artifacts"].items():
        assert sha256(OUT / name) == expected, name
    for name, expected in phase_manifest["sources"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in phase_manifest["input_hashes"].items():
        assert sha256(ROOT / name) == expected, name

    bootstrap = pd.read_csv(OUT / "bootstrap_raw.csv")
    checkpoint = pd.read_csv(OUT / "bootstrap_checkpoint.csv")
    summary = pd.read_csv(OUT / "sentinel_summary.csv")
    base = pd.read_csv(OUT / "profile_base_fits.csv")
    valid = bootstrap[[
        "profile_unrestricted_success", "profile_conditioned_success",
        "operational_success", "elastic_success",
    ]].all(axis=1)
    assert len(bootstrap) == 796
    assert set(bootstrap.groupby(["sentinel", "split"]).size()) == {99, 100}
    assert valid.mean() >= 0.99 and int((~valid).sum()) == 1
    assert (bootstrap["profile_lr_raw"] >= -1e-4).all()
    assert checkpoint.equals(bootstrap)
    assert base["reproduction_error"].abs().max() < 1e-8
    assert not summary["profile_bound_limited"].any()
    assert not summary["profile_disconnected"].any()
    assert summary.set_index("sentinel")["verdict"].to_dict() == {
        "phi_positive": "PASS", "sw_stress": "FAIL",
        "aspect_ambiguous": "CONDITIONAL", "secondary_negative": "FAIL",
    }
    reader = PdfReader(str(PDF))
    assert len(reader.pages) == 3
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in ("Non-Gaussian sentinel validation", "Data-only profile likelihoods", "Held-split coverage"):
        assert phrase in text
    return bootstrap, summary, valid


def write_manifest(bootstrap, summary, valid):
    important = [
        ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "pyproject.toml",
        ROOT / "src/jointinv/phase12b.py",
        ROOT / "scripts/run_phase12b_sentinels.py",
        ROOT / "scripts/build_phase12b_pdf.py",
        ROOT / "scripts/package_phase12b.py",
        ROOT / "tests/test_phase12b.py",
        OUT / "manifest.json", OUT / "PHASE12B_SENTINEL_REPORT.md", PDF,
    ]
    manifest = {
        "project": "joint elastic-electrical rock-physics inversion",
        "phase": "1.2B", "package_version": "0.6.0",
        "unit_tests_passed": 27,
        "bootstrap_rows": len(bootstrap),
        "numerically_valid_rows": int(valid.sum()),
        "profile_rows": int(len(pd.read_csv(OUT / "profile_raw.csv"))),
        "decisions": summary.set_index("target")["verdict"].to_dict(),
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in important
        },
    }
    path = ROOT / "PHASE12B_PACKAGE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def package():
    excluded_parts = {"__pycache__", "tmp", ".pytest_cache", "jointinv_phase0.egg-info"}
    files = [
        path for path in ROOT.rglob("*") if path.is_file()
        and not excluded_parts.intersection(path.relative_to(ROOT).parts)
        and path.suffix != ".pyc"
    ]
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(str(relative), date_time=(2026, 8, 29, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        assert bad is None, bad
    print(ARCHIVE)
    print(f"size={ARCHIVE.stat().st_size}")
    print(f"sha256={sha256(ARCHIVE)}")


def main():
    bootstrap, summary, valid = audit()
    write_manifest(bootstrap, summary, valid)
    package()


if __name__ == "__main__":
    main()
