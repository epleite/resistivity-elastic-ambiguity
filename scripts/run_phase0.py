#!/usr/bin/env python3
"""Run all Phase-0 scenarios and write tables, figures and JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jointinv.analysis import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

results, table = run_benchmark()
table.to_csv(OUT / "phase0_summary.csv", index=False)
(OUT / "phase0_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
for ax, result in zip(axes, results):
    broad = result["cases"]["deep+shallow|broad"]["parameter_variance_reduction"]
    tight = result["cases"]["deep+shallow|tight"]["parameter_variance_reduction"]
    names = list(broad)
    y = np.arange(len(names))
    ax.barh(y + 0.17, np.array(list(broad.values())) * 100, 0.34,
            label="Broad nuisance priors")
    ax.barh(y - 0.17, np.array(list(tight.values())) * 100, 0.34,
            label="Tight nuisance priors")
    ax.axvline(20, color="black", ls="--", lw=0.8)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 100)
    ax.set_title(result["label"])
    ax.set_xlabel("Variance reduction (%)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(OUT / "parameter_variance_reduction.png", dpi=220)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
for ax, result in zip(axes, results):
    direction = result["cases"]["deep+shallow|broad"]["most_rescued_null_direction"]
    names, values = list(direction), list(direction.values())
    ax.barh(names, values, color=["#2f6b8a" if v >= 0 else "#c45a4a" for v in values])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title(result["label"])
    ax.set_xlim(-1, 1)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "most_rescued_null_directions.png", dpi=220)
plt.close(fig)

print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nVerdicts:")
for result in results:
    print(f"- {result['label']}: {result['verdict']} {result['target_verdicts']}")
