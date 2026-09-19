#!/usr/bin/env python3
"""Regenerate the GMD manuscript figures from frozen result tables.

The inverse problems are not rerun.  Figures 2--6 retain the GJI panel
structure while replacing internal development-stage titles with descriptive
geophysical language.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jointinv.phase13 import SelectivePolicy  # noqa: E402
from run_oman_external_control import (  # noqa: E402
    bootstrap_ratios,
    load_rows,
    outer_cv,
    paired_metrics,
)
from run_phase11_carbonate import gate_plot, weight_plot  # noqa: E402
from run_phase12b_sentinels import plot_profiles  # noqa: E402
from run_phase12c_falsification import (  # noqa: E402
    plot_alarm_and_prior,
    plot_phi_performance,
)
from run_phase13_selective import (  # noqa: E402
    plot_policy_diagnostics,
    plot_risk_coverage_curve,
)


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
)


def save_both(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")


def framework_figure(stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.6))
    panels = [
        (
            "(a) Elastic view",
            "Observations: $V_P$, $V_S$, and bulk density\n\n"
            "Main sensitivity: load-bearing frame, compliant cracks, contacts, "
            "fluid substitution, and total pore volume.",
            "#eef2ff",
            "#4f46e5",
        ),
        (
            "(b) Electrical view",
            "Observations: deep and shallow resistivity\n\n"
            "Main sensitivity: connected brine-filled pores, saturation, "
            "tortuosity, surface conduction, and invasion.",
            "#ecfeff",
            "#0891b2",
        ),
        (
            "(c) Shared state, alternative geometry",
            "Porosity $\\phi$ and water saturation $S_w$ are shared.\n\n"
            "Elastic pore shape $a$ and electrical connectivity $q$ are tested "
            "as rigidly tied, partially coupled, or independent.",
            "#f0fdf4",
            "#16a34a",
        ),
        (
            "(d) Independent assessment sequence",
            "Retained information after uncertain electrical properties are "
            "accounted for  →  nonlinear recovery and interval checks  →  tests "
            "with omitted transport processes  →  pre-specified quality rule  →  "
            "measured-rock control.",
            "#fff7ed",
            "#ea580c",
        ),
    ]
    for axis, (title, body, fill, edge) in zip(axes.ravel(), panels):
        axis.set_axis_off()
        box = FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.92,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.8,
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(
            0.07,
            0.87,
            title,
            color=edge,
            fontsize=13,
            fontweight="bold",
            va="top",
            transform=axis.transAxes,
        )
        lines = []
        for paragraph in body.split("\n"):
            lines.append(textwrap.fill(paragraph, 55) if paragraph else "")
        axis.text(
            0.07,
            0.69,
            "\n".join(lines),
            color="#243447",
            fontsize=10.3,
            linespacing=1.35,
            va="top",
            transform=axis.transAxes,
        )
    fig.subplots_adjust(wspace=0.08, hspace=0.10)
    save_both(fig, stem)
    plt.close(fig)


def composite_figure(
    stem: Path,
    sources: list[Path],
    width_fractions: list[float] | None = None,
    panel_labels: list[str] | None = None,
) -> None:
    images = [Image.open(path).convert("RGB") for path in sources]
    widths = width_fractions or [1.0] * len(images)
    labels = panel_labels or [""] * len(images)
    figure_width = 12.0
    panel_heights = [
        figure_width * fraction * source.height / source.width
        for source, fraction in zip(images, widths)
    ]
    label_space = 0.26 if any(labels) else 0.0
    gap = 0.28 if len(images) > 1 else 0.0
    figure_height = sum(panel_heights) + gap * (len(images) - 1) + label_space * len(images)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    top = 1.0
    for source, fraction, height, label in zip(images, widths, panel_heights, labels):
        height_fraction = height / figure_height
        label_fraction = label_space / figure_height
        top -= label_fraction
        left = (1.0 - fraction) / 2.0
        axis = fig.add_axes([left, top - height_fraction, fraction, height_fraction])
        axis.imshow(source)
        axis.set_axis_off()
        if label:
            fig.text(left, top + 0.01 / figure_height, label, fontsize=12, fontweight="bold")
        top -= height_fraction + gap / figure_height
    save_both(fig, stem)
    plt.close(fig)
    for source in images:
        source.close()


def identifiability_figure(stem: Path, component_directory: Path) -> None:
    labels = {
        "phi": "Porosity",
        "sw": "Water saturation",
        "cement": "Cement fraction",
        "coord": "Coordination number",
        "aspect": "Aspect ratio",
        "secondary": "Secondary porosity",
    }

    results = json.loads((ROOT / "outputs" / "phase0_results.json").read_text(encoding="utf-8"))
    variance_path = component_directory / "identifiability_variance.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    for axis, result in zip(axes, results):
        broad = result["cases"]["deep+shallow|broad"]["parameter_variance_reduction"]
        tight = result["cases"]["deep+shallow|tight"]["parameter_variance_reduction"]
        names = list(broad)
        y = np.arange(len(names))
        axis.barh(
            y + 0.17, np.asarray([broad[name] for name in names]) * 100, 0.34,
            label="Broad nuisance priors",
        )
        axis.barh(
            y - 0.17, np.asarray([tight[name] for name in names]) * 100, 0.34,
            label="Tight nuisance priors",
        )
        axis.axvline(20, color="black", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, [labels[name] for name in names])
        axis.set_xlim(0, 100)
        axis.set_title(result["label"])
        axis.set_xlabel("Variance reduction (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(variance_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    surfaces = pd.read_csv(ROOT / "outputs" / "phase05" / "profile_surfaces.csv")
    surfaces = surfaces[
        surfaces.scenario.eq("multimodal_carbonate")
        & surfaces["pair"].eq("secondary|aspect")
    ]
    profile_path = component_directory / "identifiability_profiles.png"
    modes = ["matched_elastic", "matched_joint", "mismatch_joint"]
    mode_labels = ["Elastic only", "Compatible joint", "Mismatched joint"]
    levels = [0.0, 2.30, 5.991, 9.21, 20.0]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), constrained_layout=True)
    image = None
    for axis, mode, title in zip(axes, modes, mode_labels):
        data = surfaces[surfaces["mode"].eq(mode)]
        pivot = data.pivot(index="iy", columns="ix", values="delta_q")
        x = np.sort(data.x.unique())
        y = np.sort(data.y.unique())
        clipped = np.minimum(pivot.to_numpy(), 20.0)
        image = axis.contourf(x, y, clipped, levels=levels, cmap="viridis", extend="max")
        axis.contour(x, y, clipped, levels=[2.30, 5.991, 9.21], colors="white", linewidths=0.8)
        axis.set_title(title)
        axis.set_xlabel("Secondary porosity")
        axis.set_ylabel("Aspect ratio")
    fig.colorbar(
        image,
        ax=axes,
        label=r"Increase in misfit from best fit, $\Delta Q$",
        shrink=0.82,
    )
    fig.savefig(profile_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    composite_figure(
        stem,
        [variance_path, profile_path],
        [1.0, 1.0],
        ["(a)", "(b)"],
    )


def model_averaging_figure(stem: Path, component_directory: Path) -> None:
    directory = ROOT / "outputs" / "phase11_carbonate"
    first = component_directory / "model_averaging_weights.png"
    second = component_directory / "model_averaging_gates.png"
    weight_plot(
        pd.read_csv(directory / "family_weights.csv"),
        pd.read_csv(directory / "coupling_weights.csv"),
        first,
    )
    gate_plot(pd.read_csv(directory / "decision_gates.csv"), second)
    composite_figure(
        stem,
        [first, second],
        [1.0, 0.86],
        ["(a)", "(b)"],
    )


def profile_figure(stem: Path) -> None:
    directory = ROOT / "outputs" / "phase12b_sentinels"
    profiles = pd.read_csv(directory / "profile_raw.csv")
    summary = pd.read_csv(directory / "sentinel_summary.csv")
    plot_profiles(profiles, summary, stem.with_suffix(".png"))
    plot_profiles(profiles, summary, stem.with_suffix(".pdf"))


def model_error_figure(stem: Path, component_directory: Path) -> None:
    directory = ROOT / "outputs" / "phase12c_offlibrary"
    first = component_directory / "model_error_performance.png"
    second = component_directory / "model_error_alarm.png"
    plot_phi_performance(pd.read_csv(directory / "decision_gates.csv"), first)
    plot_alarm_and_prior(
        pd.read_csv(directory / "observation_diagnostics.csv"),
        pd.read_csv(directory / "frozen_prior_weights.csv"),
        second,
    )
    composite_figure(stem, [first, second], [1.0, 1.0], ["(a)", "(b)"])


def selective_figure(stem: Path, component_directory: Path) -> None:
    directory = ROOT / "outputs" / "phase13_selective"
    first = component_directory / "selective_risk.png"
    second = component_directory / "selective_diagnostics.png"
    plot_risk_coverage_curve(pd.read_csv(directory / "risk_coverage_curve.csv"), first)
    payload = json.loads((directory / "policy_lock.json").read_text(encoding="utf-8"))["policy"]
    payload["deployable_features"] = tuple(payload["deployable_features"])
    policy = SelectivePolicy(**payload)
    plot_policy_diagnostics(
        pd.read_csv(directory / "selective_observations.csv"),
        pd.read_csv(directory / "selective_summary.csv"),
        policy,
        second,
    )
    composite_figure(stem, [first, second], [1.0, 1.0], ["(a)", "(b)"])


def oman_figure(stem: Path) -> None:
    input_path = ROOT / "data" / "oman" / "oman_gt3a_sample_level.csv"
    if not input_path.is_file():
        published_stem = ROOT / "manuscript_figures" / "fig07_oman_control"
        if stem.resolve() == published_stem.resolve():
            print(
                "Oman table not present; retaining the included aggregate-data "
                "Figure 7. Run scripts/prepare_oman_data.py to regenerate it."
            )
            return
        for suffix in (".pdf", ".png"):
            source = published_stem.with_suffix(suffix)
            if not source.is_file():
                raise FileNotFoundError(
                    "Figure 7 cannot be regenerated without the locally prepared "
                    "Oman table; see data/oman/README.md"
                )
            shutil.copy2(source, stem.with_suffix(suffix))
        return
    rows = [
        row
        for row in load_rows(input_path)
        if bool(row["primary_complete"])
    ]
    elastic, _ = outer_cv(rows, ("vp_wet", "vs_wet", "density"), "elastic")
    joint, _ = outer_cv(rows, ("vp_wet", "vs_wet", "density", "log_r35"), "joint")
    metrics = paired_metrics(elastic, joint)
    joint_by_id = {str(row["id"]): row for row in joint}
    pairs = [(row, joint_by_id[str(row["id"])]) for row in elastic]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    truth = np.array([float(e["phi"]) for e, _ in pairs])
    elastic_prediction = np.array([float(e["prediction"]) for e, _ in pairs])
    joint_prediction = np.array([float(j["prediction"]) for _, j in pairs])
    limits = (min(truth.min(), elastic_prediction.min(), joint_prediction.min()) - 0.4,
              max(truth.max(), elastic_prediction.max(), joint_prediction.max()) + 0.4)
    axes[0].plot(limits, limits, color="#64748b", linestyle="--", linewidth=1)
    axes[0].scatter(truth, elastic_prediction, s=24, facecolors="none", edgecolors="#2563eb", label="Elastic")
    axes[0].scatter(truth, joint_prediction, s=24, color="#d95f02", alpha=0.75, label="Elastic + resistivity")
    axes[0].set(xlim=limits, ylim=limits, xlabel="Measured porosity (%)", ylabel="Held-out prediction (%)", title="(a) Outer-block predictions")
    axes[0].legend(frameon=False, fontsize=8)

    ratios = []
    for block in range(1, 9):
        block_pairs = [pair for pair in pairs if int(pair[0]["block"]) == block]
        error_e = [float(e["phi"]) - float(e["prediction"]) for e, _ in block_pairs]
        error_j = [float(j["phi"]) - float(j["prediction"]) for _, j in block_pairs]
        ratios.append(np.sqrt(np.mean(np.square(error_j))) / np.sqrt(np.mean(np.square(error_e))))
    axes[1].bar(np.arange(1, 9), ratios, color=["#0f766e" if value < 1 else "#b91c1c" for value in ratios])
    axes[1].axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Contiguous depth block", ylabel="Joint / elastic RMSE", title="(b) Block-level effect")
    axes[1].set_xticks(range(1, 9))

    bootstrap = bootstrap_ratios(metrics, 10000, 14052026)
    axes[2].hist(bootstrap, bins=40, color="#93c5fd", edgecolor="white")
    axes[2].axvline(float(metrics["rmse_ratio"]), color="#075985", linewidth=2, label=f"Observed = {float(metrics['rmse_ratio']):.3f}")
    axes[2].axvline(0.90, color="#d95f02", linestyle="--", linewidth=1.4, label="Strong-effect gate = 0.90")
    axes[2].axvline(1.0, color="#64748b", linestyle=":", linewidth=1.4, label="No improvement")
    axes[2].set(xlabel="Joint / elastic RMSE", ylabel="Whole-block bootstrap count", title="(c) Effect-size uncertainty")
    axes[2].legend(frameon=False, fontsize=7.5)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_both(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript_figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    components = args.output / "_components"
    components.mkdir(exist_ok=True)

    framework_figure(args.output / "fig01_framework")
    identifiability_figure(args.output / "fig02_identifiability", components)
    model_averaging_figure(args.output / "fig03_model_averaging", components)
    profile_figure(args.output / "fig04_profiles")
    model_error_figure(args.output / "fig05_model_error", components)
    selective_figure(args.output / "fig06_selective_test", components)
    oman_figure(args.output / "fig07_oman_control")
    shutil.rmtree(components)
    print(f"Wrote seven manuscript figures to {args.output}")


if __name__ == "__main__":
    main()
