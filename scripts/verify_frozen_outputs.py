#!/usr/bin/env python3
"""Verify hashes for frozen numerical artifacts included in this release.

Historical package manifests also pin the README and source files that shipped
with earlier stage archives. Those source hashes are intentionally not checked
here because the public v1.0 wrapper adds documentation, tests, and a portable
Oman runner. Numerical artifact hashes remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> str | None:
    if not path.is_file():
        return f"missing: {path}"
    actual = sha256(path)
    if actual != expected:
        return f"hash mismatch: {path} ({actual} != {expected})"
    return None


def verify(root: Path) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []

    for manifest_path in sorted((root / "outputs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for relative_path, expected in sorted(artifacts.items()):
            checked += 1
            failure = verify_file(manifest_path.parent / relative_path, str(expected))
            if failure:
                failures.append(failure)

    oman_manifest_path = root / "results/oman/phase14e_release_manifest.json"
    oman_manifest = json.loads(oman_manifest_path.read_text(encoding="utf-8"))
    for item in oman_manifest["files"]:
        checked += 1
        failure = verify_file(root / item["path"], item["sha256"])
        if failure:
            failures.append(failure)

    return checked, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    checked, failures = verify(args.root.resolve())
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(f"FAILED: {len(failures)} of {checked} artifact checks")
    print(f"OK: {checked} frozen artifact hashes verified")


if __name__ == "__main__":
    main()

