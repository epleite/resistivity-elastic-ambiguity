#!/usr/bin/env python3
"""Build the Phase-1.3 prospective selective-inversion decision report.

The runner writes its numerical artifacts to ``outputs/phase13_selective``.
This builder intentionally tolerates a partially populated output directory:
available tables and figures are rendered, while missing optional artifacts are
identified in restrained placeholders.  Re-running the script after the
runner completes replaces those placeholders without changing the report
layout or filename.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/phase13_selective"
DEFAULT_OUTPUT = ROOT / "output/pdf/PHASE13_SELECTIVE_REPORT.pdf"


CASE_ORDER = (
    "matched_ema_control",
    "hybrid_law",
    "connectivity_drift",
    "invasion_mismatch",
    "patchy_saturation",
    "combined_stress",
)
CASE_LABELS = {
    "matched_ema_control": "Matched EMA",
    "hybrid_law": "Hybrid law",
    "connectivity_drift": "Connectivity drift",
    "invasion_mismatch": "Invasion mismatch",
    "patchy_saturation": "Patchy saturation",
    "combined_stress": "Combined stress",
    "__atomic_macro__": "Atomic stresses (pooled)",
    "__all__": "All scenarios",
}
TARGET_LABELS = {
    "phi": "Porosity",
    "sw": "Water saturation",
    "aspect": "Aspect ratio",
    "secondary": "Secondary porosity",
}
DECISION_COLORS = {
    "SELECTIVE_GO": colors.HexColor("#15803d"),
    "GO": colors.HexColor("#15803d"),
    "CONDITIONAL": colors.HexColor("#b45309"),
    "STOP": colors.HexColor("#991b1b"),
    "PENDING OUTPUTS": colors.HexColor("#475569"),
}
VERDICT_COLORS = {
    "SELECTIVE_PASS": "#15803d",
    "SAFE_JOINT": "#0369a1",
    "SAFE_DOMAIN_REJECT": "#2563eb",
    "INCONCLUSIVE_LOW_SUPPORT": "#b45309",
    "SILENT_FAILURE": "#b91c1c",
    "PASS": "#15803d",
    "CONDITIONAL": "#b45309",
    "FAIL": "#b91c1c",
}


def _register_fonts() -> tuple[str, str]:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(regular)))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold)))
        return "DejaVu", "DejaVu-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(
        18 * mm, 9 * mm,
        "Joint elastic-electrical inversion - Phase 1.3",
    )
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _read_csv(source: Path, names: Sequence[str]) -> tuple[pd.DataFrame, str | None]:
    for name in names:
        path = source / name
        if path.exists():
            try:
                return pd.read_csv(path), name
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                continue
    return pd.DataFrame(), None


def _read_json(source: Path, names: Sequence[str]) -> tuple[dict, str | None]:
    for name in names:
        path = source / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload, name
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
    return {}, None


def _markdown_sections(path: Path) -> dict[str, str]:
    """Extract short prose fallbacks from the generated Markdown report."""
    if not path.exists():
        return {}
    current = "preamble"
    sections: dict[str, list[str]] = {current: []}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            current = line.lstrip("#").strip().lower()
            sections.setdefault(current, [])
        elif line and not line.startswith(("|", "```", "- ")):
            sections.setdefault(current, []).append(line)
    cleaned = {}
    for key, values in sections.items():
        text = " ".join(values)
        text = text.replace("_", " ")
        text = re.sub(r"[*`]", "", text)
        text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cleaned[key] = text
    return cleaned


def _first_value(row: Mapping | pd.Series, names: Iterable[str], default=None):
    for name in names:
        try:
            value = row[name]
        except (KeyError, TypeError):
            continue
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return default


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value)


def _fmt(value, digits: int = 3, missing: str = "NA") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(numeric):
        return missing
    return f"{numeric:.{digits}f}"


def _fmt_int(value, missing: str = "NA") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(numeric):
        return missing
    return str(int(round(numeric)))


def _case_label(case: object) -> str:
    text = str(case)
    return CASE_LABELS.get(text, text.replace("_", " ").title())


def _verdict_markup(value: object) -> str:
    verdict = str(value) if value is not None else "PENDING"
    color = VERDICT_COLORS.get(verdict, "#475569")
    return f"<font color='{color}'><b>{escape(verdict.replace('_', ' '))}</b></font>"


def _make_table(
    rows: Sequence[Sequence[object]],
    widths: Sequence[float],
    *,
    header: bool = True,
    font_size: float = 7.4,
    left_columns: Sequence[int] = (0,),
) -> Table:
    base = getSampleStyleSheet()
    center = ParagraphStyle(
        "P13TableCell", parent=base["BodyText"], fontName=FONT,
        fontSize=font_size, leading=font_size + 2.0, alignment=TA_CENTER,
        textColor=colors.HexColor("#1e293b"),
    )
    left = ParagraphStyle("P13TableLeft", parent=center, alignment=TA_LEFT)
    head = ParagraphStyle(
        "P13TableHead", parent=center, fontName=FONT_BOLD,
        textColor=colors.white,
    )
    wrapped = []
    for row_index, row in enumerate(rows):
        output_row = []
        for column_index, value in enumerate(row):
            style = head if header and row_index == 0 else (
                left if column_index in left_columns else center
            )
            output_row.append(Paragraph(str(value), style))
        wrapped.append(output_row)
    table = Table(wrapped, colWidths=list(widths), repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("BACKGROUND", (0, 1 if header else 0), (-1, -1), colors.HexColor("#f8fafc")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    table.setStyle(TableStyle(commands))
    return table


def _placeholder(label: str, height: float = 48 * mm) -> Table:
    base = getSampleStyleSheet()
    style = ParagraphStyle(
        "P13Placeholder", parent=base["BodyText"], fontName=FONT,
        fontSize=9, leading=13, alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )
    table = Table(
        [[Paragraph(
            f"Artifact pending: <b>{escape(label)}</b><br/>"
            "Re-run this builder after the Phase-1.3 runner completes.",
            style,
        )]],
        colWidths=[174 * mm], rowHeights=[height],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _figure(
    source: Path,
    names: Sequence[str],
    *,
    max_width: float = 174 * mm,
    max_height: float = 82 * mm,
) -> Image | Table:
    """Return the first available figure, preserving its native aspect ratio."""
    for name in names:
        path = source / name
        if not path.exists():
            continue
        try:
            width_px, height_px = ImageReader(str(path)).getSize()
            scale = min(max_width / width_px, max_height / height_px)
            return Image(
                str(path), width=width_px * scale, height=height_px * scale,
            )
        except Exception:
            continue
    return _placeholder(" or ".join(names), height=max_height)


def _ordered_cases(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "case" not in frame:
        return list(CASE_ORDER)
    present = set(frame["case"].astype(str))
    ordered = [case for case in CASE_ORDER if case in present]
    ordered.extend(sorted(present.difference(ordered).difference({"__atomic_macro__", "__all__"})))
    return ordered


def _decision_rows(gates: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Scenario", "Retention<br/>(low)", "Coverage<br/>(low)",
        "RMSE ratio<br/>(upper)", "vs random", "Verdict",
    ]]
    if gates.empty or "case" not in gates:
        return rows + [["Pending runner outputs", "NA", "NA", "NA", "NA", "PENDING"]]
    indexed = gates.drop_duplicates("case").set_index("case")
    for case in _ordered_cases(gates):
        row = indexed.loc[case]
        retention = _fmt(_first_value(row, ("retention",)))
        retention_low = _fmt(_first_value(row, ("retention_cluster_low",)))
        coverage = _fmt(_first_value(row, ("accepted_coverage90",)))
        coverage_low = _fmt(_first_value(row, ("accepted_coverage90_cluster_low",)))
        rmse = _fmt(_first_value(row, ("accepted_rmse_ratio",)))
        rmse_upper = _fmt(_first_value(
            row, ("accepted_rmse_ratio_cluster_upper95", "accepted_rmse_ratio_cluster_high"),
        ))
        enriched = _first_value(row, ("selection_enriched",), False)
        random_text = "yes" if _as_bool(enriched) else "no"
        rows.append([
            escape(_case_label(case)),
            f"{retention}<br/>({retention_low})",
            f"{coverage}<br/>({coverage_low})",
            f"{rmse}<br/>({rmse_upper})",
            random_text,
            _verdict_markup(_first_value(row, ("verdict",), "PENDING")),
        ])
    return rows


def _performance_rows(summary: pd.DataFrame, gates: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Scenario", "Accepted", "Excess MSE<br/>(upper)",
        "System RMSE<br/>(upper)", "System coverage", "Fallback utility",
    ]]
    source = summary if not summary.empty else gates
    if source.empty or "case" not in source:
        return rows + [["Pending runner outputs", "NA", "NA", "NA", "NA", "NA"]]
    gate_lookup = (
        gates.drop_duplicates("case").set_index("case")
        if not gates.empty and "case" in gates else pd.DataFrame()
    )
    indexed = source.drop_duplicates("case").set_index("case")
    for case in _ordered_cases(source):
        row = indexed.loc[case]
        gate = gate_lookup.loc[case] if case in gate_lookup.index else row
        accepted = _fmt_int(_first_value(row, ("accepted_n",)))
        total = _fmt_int(_first_value(row, ("n",)))
        excess = _fmt(_first_value(row, ("accepted_excess_mse",)), 6)
        excess_upper = _fmt(_first_value(
            row, ("accepted_excess_mse_cluster_upper95", "accepted_excess_mse_cluster_high"),
        ), 6)
        system_ratio = _fmt(_first_value(row, ("system_rmse_ratio",)))
        system_upper = _fmt(_first_value(
            row, ("system_rmse_ratio_cluster_upper95", "system_rmse_ratio_cluster_high"),
        ))
        fallback = _first_value(gate, ("fallback_pass",), None)
        fallback_text = "passes" if fallback is True or str(fallback).lower() == "true" else (
            "fails" if fallback is False or str(fallback).lower() == "false" else "NA"
        )
        rows.append([
            escape(_case_label(case)), f"{accepted}/{total}",
            f"{excess}<br/>({excess_upper})",
            f"{system_ratio}<br/>({system_upper})",
            _fmt(_first_value(row, ("system_coverage90",))),
            fallback_text,
        ])
    return rows


def _random_rows(comparator: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Scenario", "Retention", "Accepted MSE", "Random MSE<br/>(95% range)",
        "Delta upper", "p", "Enriched",
    ]]
    if comparator.empty or "case" not in comparator:
        return rows + [["Pending runner outputs", "NA", "NA", "NA", "NA", "NA", "NA"]]
    indexed = comparator.drop_duplicates("case").set_index("case")
    for case in _ordered_cases(comparator):
        row = indexed.loc[case]
        random_mean = _fmt(_first_value(row, ("random_mse_mean",)), 6)
        random_low = _fmt(_first_value(row, ("random_mse_low",)), 6)
        random_high = _fmt(_first_value(row, ("random_mse_high",)), 6)
        enriched = _first_value(row, ("selection_enriched",), False)
        rows.append([
            escape(_case_label(case)),
            _fmt(_first_value(row, ("retention",))),
            _fmt(_first_value(row, ("actual_accepted_mse",)), 6),
            f"{random_mean}<br/>({random_low}, {random_high})",
            _fmt(_first_value(row, ("actual_minus_random_upper95",)), 6),
            _fmt(_first_value(row, ("enrichment_p_value",)), 3),
            "yes" if _as_bool(enriched) else "no",
        ])
    return rows


def _capture_rows(gates: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Scenario", "Abstention", "False-conf.<br/>events", "Captured",
        "Capture lift", "System harm<br/>(upper)",
    ]]
    if gates.empty or "case" not in gates:
        return rows + [["Pending runner outputs", "NA", "NA", "NA", "NA", "NA"]]
    indexed = gates.drop_duplicates("case").set_index("case")
    for case in _ordered_cases(gates):
        row = indexed.loc[case]
        captured = _fmt(_first_value(row, ("false_confidence_capture",)))
        rows.append([
            escape(_case_label(case)),
            _fmt(_first_value(row, ("abstention",))),
            _fmt_int(_first_value(row, ("false_confidence_count",))),
            captured,
            _fmt(_first_value(row, ("false_confidence_capture_lift",))),
            f"{_fmt(_first_value(row, ('system_harmful_false_confidence',)))}"
            f"<br/>({_fmt(_first_value(row, ('system_harmful_false_confidence_cluster_upper95',)))})",
        ])
    return rows


def _control_rows(controls: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Target", "Scenarios", "Median retention", "Worst coverage low",
        "Worst excess upper", "Promoted",
    ]]
    if controls.empty or "target" not in controls:
        return rows + [["Pending runner outputs", "NA", "NA", "NA", "NA", "NA"]]
    for target, group in controls.groupby("target", sort=True):
        promoted_column = next(
            (name for name in ("promoted_by_phi_guard", "promoted") if name in group),
            None,
        )
        promoted = int(group[promoted_column].fillna(False).astype(bool).sum()) if promoted_column else 0
        retention = pd.to_numeric(group.get("retention"), errors="coerce")
        coverage = pd.to_numeric(
            group.get("accepted_coverage90_low", group.get("accepted_coverage90_cluster_low")),
            errors="coerce",
        )
        excess = pd.to_numeric(
            group.get("accepted_excess_mse_upper95", group.get("accepted_excess_mse_cluster_upper95")),
            errors="coerce",
        )
        rows.append([
            escape(TARGET_LABELS.get(str(target), str(target))),
            str(len(group)),
            _fmt(retention.median() if retention is not None else None),
            _fmt(coverage.min() if coverage is not None else None),
            _fmt(excess.max() if excess is not None else None, 6),
            str(promoted),
        ])
    return rows


def _threshold_rows(grid: pd.DataFrame, policy: Mapping) -> list[list[str]]:
    rows = [["Quantile", "Chi-square threshold", "Role", "Training observations"]]
    if grid.empty:
        threshold = _first_value(policy, ("chi2_threshold",))
        if threshold is None:
            return rows + [["NA", "NA", "Pending runner outputs", "NA"]]
        return rows + [[
            _fmt(_first_value(policy, ("threshold_quantile",)), 3),
            _fmt(threshold, 6), "confirmatory", _fmt_int(_first_value(policy, ("training_observations",))),
        ]]
    selected = grid.copy()
    if len(selected) > 6 and "threshold_quantile" in selected:
        quantiles = pd.to_numeric(selected["threshold_quantile"], errors="coerce")
        preferred = quantiles.isin([0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
        selected = selected.loc[preferred]
    primary = _first_value(policy, ("threshold_quantile",), 0.90)
    for row in selected.itertuples(index=False):
        quantile = getattr(row, "threshold_quantile", math.nan)
        role = "confirmatory" if math.isclose(float(quantile), float(primary), abs_tol=1e-12) else "descriptive curve"
        rows.append([
            _fmt(quantile, 3),
            _fmt(getattr(row, "chi2_threshold", math.nan), 6),
            role,
            _fmt_int(getattr(row, "training_observations", math.nan)),
        ])
    return rows


def _integrity_text(config: Mapping, manifest: Mapping) -> str:
    bits = []
    for key, label in (
        ("seed", "seed"),
        ("n_panels", "panels"),
        ("replicates", "noise replicates"),
        ("panel_size", "depths"),
        ("cluster_bootstrap_draws", "cluster bootstrap draws"),
        ("random_rejection_permutations", "random-rejection permutations"),
    ):
        if key in config:
            bits.append(f"{label}: {escape(str(config[key]))}")
    counts = manifest.get("counts", {}) if isinstance(manifest, Mapping) else {}
    if isinstance(counts, Mapping):
        observation_count = _first_value(
            counts, ("test_observations", "observations"), None,
        )
        if observation_count is not None:
            bits.append(f"observations: {escape(str(observation_count))}")
        for key, label in (
            ("candidate_fits", "joint candidate fits"),
            ("recoveries", "target recoveries"),
        ):
            if key in counts:
                bits.append(f"{label}: {escape(str(counts[key]))}")
    if not bits:
        return "Run metadata and SHA-256 manifest will appear after the Phase-1.3 runner completes."
    return "; ".join(bits) + "."


def _policy_contract_text(policy: Mapping) -> str:
    threshold = _fmt(_first_value(policy, ("chi2_threshold",)), 6)
    bound = _fmt(_first_value(policy, ("phi_bound_mass_limit",)), 2)
    success = _fmt(_first_value(policy, ("candidate_success_min",)), 2)
    mass = _fmt(_first_value(policy, ("surviving_prior_mass_min",)), 2)
    if "NA" in {threshold, bound, success, mass}:
        return (
            "Release joint porosity only when inference diagnostics are finite, "
            "the frozen chi-square and porosity-bound checks pass, and candidate "
            "success and surviving prior mass meet their locked minima."
        )
    return (
        f"Release joint porosity only when diagnostics are finite, best-candidate "
        f"reduced chi-square is at most {threshold}, phi-bound mass is at most "
        f"{bound}, candidate success is at least {success}, and surviving prior "
        f"mass is at least {mass}."
    )


def build(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> Path:
    summary, _ = _read_csv(source, ("selective_summary.csv", "selective_case_summary.csv"))
    comparator, _ = _read_csv(source, ("random_rejection_comparator.csv",))
    curve, _ = _read_csv(source, ("risk_coverage_curve.csv",))
    gates, _ = _read_csv(source, ("decision_gates.csv", "selective_decision_gates.csv"))
    overall_frame, _ = _read_csv(source, ("overall_decision.csv",))
    controls, _ = _read_csv(source, ("control_summary.csv", "negative_control_summary.csv"))
    threshold_grid, _ = _read_csv(source, ("threshold_grid.csv",))
    policy_lock, _ = _read_json(source, ("policy_lock.json",))
    config, _ = _read_json(source, ("run_config.json",))
    manifest, _ = _read_json(source, ("manifest.json",))
    markdown = _markdown_sections(source / "PHASE13_SELECTIVE_REPORT.md")

    # The runner stores policy details inside a durable lock envelope.  Accept
    # both that canonical layout and a flat policy JSON for forward compatibility.
    nested_policy = policy_lock.get("policy", policy_lock)
    policy = dict(nested_policy) if isinstance(nested_policy, Mapping) else {}
    if "policy_hash" in policy_lock:
        policy.setdefault("policy_hash", policy_lock["policy_hash"])
    if threshold_grid.empty and isinstance(policy_lock.get("threshold_grid"), list):
        threshold_grid = pd.DataFrame(policy_lock["threshold_grid"])
    if not config and isinstance(policy_lock.get("test_design_locked"), Mapping):
        config = dict(policy_lock["test_design_locked"])

    overall = overall_frame.iloc[0] if not overall_frame.empty else {}
    decision = str(_first_value(overall, ("decision",), "PENDING OUTPUTS"))
    rationale = _first_value(overall, ("rationale",), None)
    if not rationale:
        rationale = next(
            (text for heading, text in markdown.items() if "decision" in heading),
            "The report is structurally complete; numerical conclusions will populate after the prospective runner finishes.",
        )
    if decision == "STOP" and not gates.empty and {"case", "verdict"}.issubset(gates):
        verdicts = dict(zip(gates["case"].astype(str), gates["verdict"].astype(str)))
        silent = [
            label for case, label in CASE_LABELS.items()
            if verdicts.get(case) == "SILENT_FAILURE"
        ]
        if silent:
            rationale = (
                "Matched transport was SAFE_JOINT and the combined stress was "
                "SAFE_DOMAIN_REJECT. However, " + " and ".join(silent)
                + " remained SILENT_FAILURE, and no scenario showed risk "
                "enrichment beyond random rejection at equal retention."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="Phase 1.3 prospective selective inversion",
        author="Joint elastic-electrical inversion project",
        subject="Selective risk, abstention, and elastic fallback under off-library stress",
    )
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "P13Title", parent=base["Title"], fontName=FONT_BOLD,
        fontSize=18.5, leading=22.5, textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT, spaceAfter=4.5 * mm,
    )
    subtitle = ParagraphStyle(
        "P13Subtitle", parent=base["BodyText"], fontName=FONT,
        fontSize=10.0, leading=14.0, textColor=colors.HexColor("#475569"),
        spaceAfter=4.5 * mm,
    )
    heading = ParagraphStyle(
        "P13Heading", parent=base["Heading2"], fontName=FONT_BOLD,
        fontSize=13.0, leading=16.5, textColor=colors.HexColor("#0f172a"),
        spaceBefore=2.5 * mm, spaceAfter=2.5 * mm,
    )
    body = ParagraphStyle(
        "P13Body", parent=base["BodyText"], fontName=FONT,
        fontSize=8.9, leading=12.6, textColor=colors.HexColor("#1e293b"),
        spaceAfter=2.5 * mm,
    )
    small = ParagraphStyle(
        "P13Small", parent=body, fontSize=8.0, leading=11.2,
        textColor=colors.HexColor("#475569"),
    )
    decision_style = ParagraphStyle(
        "P13Decision", parent=body, fontName=FONT_BOLD, fontSize=14.5,
        leading=18, alignment=TA_CENTER, textColor=colors.white,
    )
    decision_color = DECISION_COLORS.get(decision, colors.HexColor("#475569"))
    decision_box = Table(
        [[Paragraph(f"DECISION: {escape(decision.replace('_', ' '))}", decision_style)]],
        colWidths=[174 * mm], rowHeights=[15 * mm],
    )
    decision_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.8, decision_color),
    ]))

    policy_id = _first_value(policy, ("policy_id",), _first_value(overall, ("primary_policy",), "pending"))
    feature_list = _first_value(policy, ("deployable_features",), [])
    if isinstance(feature_list, (list, tuple)):
        feature_labels = {
            "best_reduced_chi2": "best reduced chi-square",
            "target_bound_mass": "phi-bound mass",
            "raw_candidate_success_fraction": "candidate success fraction",
            "surviving_prior_mass": "surviving prior mass",
        }
        feature_text = ", ".join(
            feature_labels.get(str(value), str(value).replace("_", " "))
            for value in feature_list
        )
    else:
        feature_text = str(feature_list).replace("_", " ")
    threshold_source = str(
        _first_value(policy, ("threshold_source",), "pending")
    ).replace("phase11", "Phase-1.1").replace("_", " ")
    policy_rows = [
        ["Policy", escape(str(policy_id)), "Threshold source", escape(threshold_source)],
        ["Chi-square threshold", _fmt(_first_value(policy, ("chi2_threshold",)), 6), "Phi-bound limit", _fmt(_first_value(policy, ("phi_bound_mass_limit",)), 3)],
        ["Operating quantile", _fmt(_first_value(policy, ("threshold_quantile",)), 3), "Training observations", _fmt_int(_first_value(policy, ("training_observations",)))],
        ["Deployable features", escape(feature_text or "pending"), "Policy hash", escape(str(_first_value(policy, ("policy_hash", "canonical_hash"), "see policy_lock.json")))[:28] + "..."],
    ]

    story = [
        Paragraph("Phase 1.3<br/>Prospective selective inversion", title),
        Paragraph(
            "A frozen, target-aware guard decides when joint porosity may be released and when the operational system must abstain or fall back to elastic-only inference.",
            subtitle,
        ),
        decision_box,
        Spacer(1, 3.5 * mm),
        Paragraph("Scientific decision", heading),
        Paragraph(escape(str(rationale)), body),
        Paragraph(
            "Selective benefit is credited only when accepted cases outperform both elastic-only inference and random rejection at the same retention. Near-total rejection is reported as domain safety, never as accurate selective inversion.",
            body,
        ),
        Paragraph("Frozen operating policy", heading),
        _make_table(policy_rows, [32 * mm, 55 * mm, 38 * mm, 49 * mm], header=False, font_size=7.5),
        Paragraph("Confirmatory scenario verdicts", heading),
        _make_table(
            _decision_rows(gates),
            [34 * mm, 26 * mm, 26 * mm, 28 * mm, 24 * mm, 36 * mm],
            font_size=6.8,
        ),
        PageBreak(),

        Paragraph("Selective performance at the frozen gate", title),
        Paragraph(
            "The accepted-joint result and the operational joint-or-elastic fallback are distinct estimands. Low-support accepted subsets are not assigned perfect performance.",
            subtitle,
        ),
        _figure(
            source,
            ("selective_case_performance.png", "selective_case_decisions.png"),
            max_height=82 * mm,
        ),
        Spacer(1, 3 * mm),
        _make_table(
            _performance_rows(summary, gates),
            [35 * mm, 23 * mm, 34 * mm, 31 * mm, 27 * mm, 24 * mm],
            font_size=6.8,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Excess MSE is paired within the exact accepted subset: joint squared error minus elastic squared error. A negative upper confidence limit is stronger evidence than a favorable RMSE ratio alone. The system columns include elastic fallback for every abstained observation.",
            small,
        ),
        Paragraph(
            "Fallback utility compares the joint-or-elastic system with elastic-only inference. It is distinct from safe domain rejection: the combined stress can be rejected safely even when fallback utility fails.",
            small,
        ),
        PageBreak(),

        Paragraph("Risk versus retention", title),
        Paragraph(
            "Only the frozen operating point is confirmatory. Other thresholds describe the risk-retention geometry and cannot promote the decision after the test truths are revealed.",
            subtitle,
        ),
        _figure(
            source,
            ("risk_coverage_curve.png", "risk_coverage_curves.png"),
            max_height=92 * mm,
        ),
        Spacer(1, 3 * mm),
        Paragraph("Matched-retention random-rejection test", heading),
        _make_table(
            _random_rows(comparator),
            [30 * mm, 20 * mm, 25 * mm, 39 * mm, 24 * mm, 17 * mm, 19 * mm],
            font_size=6.5,
        ),
        Paragraph(
            "The random comparator permutes the observed acceptance pattern among whole geological panels. Selection is enriched only when the one-sided upper limit of actual-minus-random MSE is below zero. The oracle curve is explanatory and never deployable.",
            small,
        ),
        PageBreak(),

        Paragraph("Safety, abstention and negative controls", title),
        Paragraph(
            "Failure capture must be interpreted jointly with the abstention fraction. Capturing nearly every failure by rejecting nearly every observation has capture lift near one and is classified as safe domain rejection, not selective inference.",
            subtitle,
        ),
        _figure(
            source,
            ("policy_diagnostics.png", "failure_capture.png"),
            max_height=74 * mm,
        ),
        Spacer(1, 3 * mm),
        _make_table(
            _capture_rows(gates),
            [37 * mm, 25 * mm, 28 * mm, 24 * mm, 27 * mm, 33 * mm],
            font_size=6.9,
        ),
        Paragraph("Phi-mask transfer to stress and negative controls", heading),
        _make_table(
            _control_rows(controls),
            [36 * mm, 23 * mm, 30 * mm, 31 * mm, 34 * mm, 20 * mm],
            font_size=6.8,
        ),
        Paragraph(
            "The porosity acceptance mask is applied unchanged to water saturation, aspect ratio, and secondary porosity. These targets cannot tune the guard or upgrade the primary porosity claim.",
            small,
        ),
        PageBreak(),

        Paragraph("Policy provenance and reproducibility", title),
        Paragraph(
            "The policy is frozen from the Phase-1.1 matched-library bank. Phase-1.2C and Phase-1.3 truths, scenario labels, realized errors, coverage indicators, and generator-only diagnostics are excluded from the scorer.",
            subtitle,
        ),
        Paragraph("Threshold grid", heading),
        _make_table(
            _threshold_rows(threshold_grid, policy),
            [34 * mm, 45 * mm, 55 * mm, 40 * mm],
            font_size=7.2,
        ),
        Paragraph("Policy contract", heading),
        Paragraph(escape(_policy_contract_text(policy)), body),
        Paragraph("Run integrity", heading),
        Paragraph(_integrity_text(config, manifest), body),
        Paragraph(
            "All uncertainty intervals resample whole geological panels while keeping noise replicates and paired joint/elastic outputs together. The threshold remains fixed inside the evaluation bootstrap. Artifact and source hashes are recorded in manifest.json.",
            body,
        ),
        Paragraph("Interpretation boundary", heading),
        Paragraph(
            escape(next(
                (text for heading_name, text in markdown.items() if "interpretation" in heading_name or "boundary" in heading_name),
                "This remains synthetic constitutive-stress validation. A favorable decision supports a controlled blind-well or experimental pilot with the abstention guard active; it does not establish general field validity.",
            )),
            body,
        ),
        Paragraph("Next gate", heading),
        Paragraph(
            escape({
                "SELECTIVE_GO": "Proceed to a controlled blind-well or experimental pilot, preserving the frozen policy and reporting every abstention.",
                "GO": "Proceed to a controlled blind-well or experimental pilot, preserving the frozen policy and reporting every abstention.",
                "CONDITIONAL": "Run a Phase-1.3B synthetic expansion before field use; no threshold may be retuned on this test bank.",
                "STOP": "Keep field deployment blocked and redesign the guard around the documented silent or low-support failure modes.",
            }.get(decision, "Complete the Phase-1.3 runner and rebuild this report before making a scientific decision.")),
            body,
        ),
    ]

    # Accessing ``curve`` here is intentional: a malformed curve file is caught
    # by the loader, while its visual summary remains optional and figure-based.
    _ = curve
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
