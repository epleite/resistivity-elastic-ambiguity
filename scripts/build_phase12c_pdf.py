#!/usr/bin/env python3
"""Build the visually verified Phase-1.2C scientific decision report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase12c_offlibrary"
OUTPUT = ROOT / "output/pdf/PHASE12C_OFFLIBRARY_REPORT.pdf"


CASE_LABELS = {
    "matched_ema_control": "Matched EMA",
    "hybrid_law": "Hybrid law",
    "connectivity_drift": "Connectivity drift",
    "invasion_mismatch": "Invasion mismatch",
    "patchy_saturation": "Patchy saturation",
    "combined_stress": "Combined stress",
}
TARGET_LABELS = {
    "phi": "Porosity", "sw": "Water saturation",
    "aspect": "Aspect ratio", "secondary": "Secondary porosity",
}
VERDICT_COLORS = {
    "PASS": colors.HexColor("#15803d"),
    "CONDITIONAL": colors.HexColor("#d97706"),
    "FAIL": colors.HexColor("#b91c1c"),
}


pdfmetrics.registerFont(TTFont(
    "DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
))
pdfmetrics.registerFont(TTFont(
    "DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
))


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 9 * mm, "Joint elastic-electrical inversion - Phase 1.2C")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def make_table(rows, widths, header=True, font_size=7.6):
    styles = getSampleStyleSheet()
    cell = ParagraphStyle(
        "TableCell", parent=styles["BodyText"], fontName="DejaVu",
        fontSize=font_size, leading=font_size + 2, alignment=TA_CENTER,
    )
    head = ParagraphStyle(
        "TableHead", parent=cell, fontName="DejaVu-Bold",
        textColor=colors.white,
    )
    wrapped = []
    for row_index, row in enumerate(rows):
        wrapped.append([
            Paragraph(str(value), head if header and row_index == 0 else cell)
            for value in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 1 if header else 0), (-1, -1), colors.HexColor("#f8fafc")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    table.setStyle(TableStyle(commands))
    return table


def build():
    gates = pd.read_csv(SOURCE / "decision_gates.csv")
    overall = pd.read_csv(SOURCE / "overall_decision.csv").iloc[0]
    calibration = pd.read_csv(SOURCE / "frozen_calibration.csv")
    bound = pd.read_csv(SOURCE / "bound_mass_decomposition.csv")
    controls = pd.read_csv(SOURCE / "control_verdict_summary.csv")
    selective = pd.read_csv(SOURCE / "alarm_selective_risk.csv")
    diagnostics = pd.read_csv(SOURCE / "observation_diagnostics.csv")
    distance = pd.read_csv(SOURCE / "truth_library_distance.csv")
    phi = gates[
        gates.target.eq("phi") & gates.strategy.eq("frozen_press")
    ].set_index("case")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="Phase 1.2C independent off-library falsification",
        author="Joint elastic-electrical inversion project",
    )
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title", parent=base["Title"], fontName="DejaVu-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT, spaceAfter=5 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=base["BodyText"], fontName="DejaVu",
        fontSize=10.2, leading=14.2, textColor=colors.HexColor("#475569"),
        spaceAfter=5 * mm,
    )
    heading = ParagraphStyle(
        "Heading", parent=base["Heading2"], fontName="DejaVu-Bold",
        fontSize=13.5, leading=17, textColor=colors.HexColor("#0f172a"),
        spaceBefore=3 * mm, spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="DejaVu",
        fontSize=9.1, leading=13.0, textColor=colors.HexColor("#1e293b"),
        spaceAfter=2.8 * mm,
    )
    decision_style = ParagraphStyle(
        "Decision", parent=body, fontName="DejaVu-Bold", fontSize=15,
        leading=19, alignment=TA_CENTER, textColor=colors.white,
    )

    decision_box = Table(
        [[Paragraph(f"DECISION: {overall.decision}", decision_style)]],
        colWidths=[174 * mm], rowHeights=[16 * mm],
    )
    decision_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#991b1b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#7f1d1d")),
    ]))

    story = [
        Paragraph("Phase 1.2C<br/>Independent off-library falsification", title),
        Paragraph(
            "A frozen Phase-1.1 deployment rule is challenged by independent carbonate panels and constitutive truths that are absent from the unchanged inverse library.",
            subtitle,
        ),
        decision_box,
        Spacer(1, 4 * mm),
        Paragraph("What the decision means", heading),
        Paragraph(
            "The combined stress fails decisively: porosity coverage is 76.4%, the 95% upper bound on the joint-to-elastic RMSE ratio is 1.33, false-confidence upper bound is 33.3%, and 71 of 72 panels are flagged by the frozen OOD alarm. No adapted strategy produces a PASS. Field deployment remains unauthorized.",
            body,
        ),
        Paragraph(
            "The atomic cases are more nuanced. Matched, hybrid and patchy truths retain strong porosity accuracy, but strict gates reject dependence on any critical parameter bound. Secondary-porosity bounds dominate that penalty. The gate is retained exactly as declared; the decomposition below is sensitivity analysis, not a post-hoc reclassification.",
            body,
        ),
        Paragraph("Confirmatory porosity results", heading),
    ]
    gate_rows = [[
        "Scenario", "Coverage<br/>(cluster low)", "RMSE ratio<br/>(upper)",
        "Width ratio<br/>(upper)", "Alarm", "Verdict",
    ]]
    for case in CASE_LABELS:
        row = phi.loc[case]
        gate_rows.append([
            CASE_LABELS[case],
            f"{row.coverage90:.3f}<br/>({row.coverage90_cluster_low:.3f})",
            f"{row.rmse_ratio_over_elastic:.3f}<br/>({row.rmse_ratio_cluster_upper:.3f})",
            f"{row.median_width_ratio:.3f}<br/>({row.width_ratio_cluster_upper:.3f})",
            f"{row.ood_alarm_fraction:.3f}",
            f"<font color='{VERDICT_COLORS[row.verdict].hexval()}'><b>{row.verdict}</b></font>",
        ])
    story += [
        make_table(gate_rows, [34 * mm, 31 * mm, 30 * mm, 30 * mm, 21 * mm, 28 * mm]),
        PageBreak(),
        Paragraph("Porosity performance across external stresses", title),
        Paragraph(
            "Points are observed metrics; bars are 95% panel-cluster intervals. The adapted average is an optimistic ceiling learned on other panels in the same simulator-defined OOD scenario and cannot upgrade the primary decision.",
            subtitle,
        ),
        Image(str(SOURCE / "phi_offlibrary_performance.png"), width=174 * mm, height=55 * mm),
        Spacer(1, 4 * mm),
        Paragraph("Structural distance from the candidate library", heading),
        Image(str(SOURCE / "truth_library_distance.png"), width=145 * mm, height=77 * mm),
        Paragraph(
            "The matched control has zero structural distance at the true parameters. Hybrid and combined truths are furthest from the library. Distance alone is not a safety signal: the hybrid truth is rarely alarmed even though its median fixed-truth discrepancy exceeds one noise standard deviation RMS.",
            body,
        ),
        PageBreak(),
        Paragraph("Why the formal gate fails", title),
        Paragraph(
            "The operational gate penalizes weighted mass on any of five critical boundaries. The following decomposition shows that secondary porosity, a declared negative control, is the main contributor. Phi itself rarely reaches a bound.",
            subtitle,
        ),
    ]
    aggregate = phi["mean_critical_bound_mass"]
    frozen_bound = bound[bound.strategy.eq("frozen_press")].pivot(
        index="case", columns="bound_name", values="mean_weighted_bound_mass",
    )
    bound_rows = [["Scenario", "Any critical", "Phi", "Secondary", "q-connectivity"]]
    for case in CASE_LABELS:
        row = frozen_bound.loc[case]
        bound_rows.append([
            CASE_LABELS[case], f"{aggregate[case]:.3f}", f"{row.phi:.3f}",
            f"{row.secondary_fraction:.3f}", f"{row.q_conn:.3f}",
        ])
    story += [
        make_table(bound_rows, [46 * mm, 32 * mm, 28 * mm, 34 * mm, 34 * mm]),
        Spacer(1, 5 * mm),
        Image(str(SOURCE / "ood_alarm_and_frozen_prior.png"), width=174 * mm, height=71 * mm),
        Spacer(1, 3 * mm),
        Paragraph("Exploratory selective-risk audit", heading),
    ]
    selective_rows = [["Scenario", "Alarmed", "Coverage if not alarmed", "False confidence caught"]]
    selective = selective.set_index("case")
    for case in CASE_LABELS:
        row = selective.loc[case]
        capture = (
            "n/a" if pd.isna(row.false_confidence_capture_fraction)
            else f"{int(row.false_confidence_alarmed_count)}/{int(row.false_confidence_count)}"
        )
        selective_rows.append([
            CASE_LABELS[case], f"{int(row.alarm_count)}/{int(row.n)}",
            f"{row.nonalarm_coverage90:.3f} (n={int(row.nonalarm_count)})",
            capture,
        ])
    story += [
        make_table(selective_rows, [48 * mm, 31 * mm, 52 * mm, 43 * mm]),
        Paragraph(
            "This audit was added after the confirmatory run and does not change any verdict. The alarm captures all false-confidence events in invasion and combined stress, but weak atomic-case sensitivity remains unsafe for deployment.",
            body,
        ),
        PageBreak(),
        Paragraph("Controls, calibration and reproducibility", title),
        Paragraph("Stress and negative controls", heading),
    ]
    control_rows = [["Target", "Strategy", "PASS", "CONDITIONAL", "FAIL"]]
    for row in controls.itertuples(index=False):
        control_rows.append([
            TARGET_LABELS[row.target], row.strategy.replace("_", " "),
            row.pass_count, row.conditional_count, row.fail_count,
        ])
    story += [
        make_table(control_rows, [48 * mm, 48 * mm, 24 * mm, 30 * mm, 24 * mm]),
        Paragraph(
            "Aspect ratio and secondary porosity fail every scenario; water saturation never passes. The negative controls are not spuriously promoted.",
            body,
        ),
        Paragraph("Frozen empirical inflation factors", heading),
    ]
    calibration_rows = [["Strategy", "Target", "c90", "Training rows"]]
    for row in calibration.itertuples(index=False):
        calibration_rows.append([
            row.strategy.replace("_", " "), TARGET_LABELS[row.target],
            f"{row.c90:.4f}", row.n_training,
        ])
    story += [
        make_table(calibration_rows, [52 * mm, 52 * mm, 34 * mm, 36 * mm], font_size=7.2),
        Spacer(1, 3 * mm),
        Paragraph("Run integrity", heading),
        Paragraph(
            "Independent seed 20260917; 24 geological panels; three repeated-noise realizations; eight depths; six scenarios; 432 elastic fits; 3,888 joint candidate fits; 5,184 target recoveries. All candidate fits converged, all frozen prior mass survived, and every output/source is SHA-256 hashed with runtime versions in the manifest.",
            body,
        ),
        Paragraph(
            "Calibration uses 12 Phase-1.1 geological panels and correlated case/noise repetitions. The c90 factors are empirical outer-panel inflation factors, not formal finite-sample cluster-conformal guarantees. File hashes prove reproducibility of this run but are not a time-stamped preregistration.",
            body,
        ),
        Paragraph("Recommended next gate", heading),
        Paragraph(
            "Do not move to field inversion. Phase 1.3 should predeclare a selective-risk experiment: coverage and RMSE versus abstention rate, with a target-specific phi boundary gate and secondary porosity retained as a separate negative control. The current STOP decision must remain frozen as the Phase-1.2C result.",
            body,
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
