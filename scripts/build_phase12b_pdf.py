#!/usr/bin/env python3
"""Build the mobile-friendly Phase 1.2B summary PDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase12b_sentinels"
OUTPUT = ROOT / "output/pdf/PHASE12B_SENTINEL_REPORT.pdf"


TARGET_NAMES = {
    "phi": "Porosity", "sw": "Water saturation",
    "aspect": "Aspect ratio", "secondary": "Secondary porosity",
}
VERDICT_COLORS = {
    "PASS": colors.HexColor("#15803d"),
    "CONDITIONAL": colors.HexColor("#d97706"),
    "FAIL": colors.HexColor("#b91c1c"),
}

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(18 * mm, 9 * mm, "Joint elastic-electrical inversion - Phase 1.2B")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build():
    summary = pd.read_csv(SOURCE / "sentinel_summary.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm,
        title="Phase 1.2B carbonate sentinel report",
        author="Joint elastic-electrical inversion project",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontName="DejaVu-Bold",
        fontSize=19, leading=23, textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT, spaceAfter=7 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName="DejaVu", fontSize=10.5, leading=15,
        textColor=colors.HexColor("#475569"), spaceAfter=6 * mm,
    )
    heading = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], fontName="DejaVu-Bold",
        fontSize=14, leading=17, textColor=colors.HexColor("#0f172a"),
        spaceBefore=3 * mm, spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=9.4, leading=13.5,
        textColor=colors.HexColor("#1e293b"), spaceAfter=3 * mm,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontName="DejaVu", fontSize=8.0, leading=10.5, alignment=TA_CENTER,
    )
    header_small = ParagraphStyle(
        "HeaderSmall", parent=small, fontName="DejaVu-Bold", textColor=colors.white,
    )
    story = [
        Paragraph("Phase 1.2B<br/>Non-Gaussian sentinel validation", title),
        Paragraph(
            "Data-only profile likelihoods and a split parametric bootstrap for carbonate joint elastic-electrical inversion.",
            subtitle,
        ),
        Paragraph("Decision", heading),
        Paragraph(
            "Porosity is the only endpoint that passes the confirmatory sentinel gate. Water saturation fails despite its point-RMSE gain; aspect ratio remains conditional; secondary porosity fails as expected and confirms the negative control. This is a GO to off-library falsification for porosity only, not authorization for field deployment.",
            body,
        ),
    ]

    rows = [[
        Paragraph("Target", header_small), Paragraph("Coverage", header_small),
        Paragraph("RMSE ratio<br/>(95% upper)", header_small),
        Paragraph("Width ratio<br/>(95% upper)", header_small),
        Paragraph("Decision", header_small),
    ]]
    for row in summary.itertuples(index=False):
        rows.append([
            Paragraph(TARGET_NAMES[row.target], small),
            Paragraph(f"{100 * row.calibrated_wald_coverage90:.0f}%", small),
            Paragraph(f"{row.rmse_ratio_joint_over_elastic:.3f}<br/>({row.rmse_ratio_high:.3f})", small),
            Paragraph(f"{row.median_width_ratio:.3f}<br/>({row.width_ratio_high:.3f})", small),
            Paragraph(f"<font color='{VERDICT_COLORS[row.verdict].hexval()}'><b>{row.verdict}</b></font>", small),
        ])
    table = Table(rows, colWidths=[43 * mm, 25 * mm, 36 * mm, 36 * mm, 32 * mm], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for index, verdict in enumerate(summary.verdict, start=1):
        commands.append(("TEXTCOLOR", (-1, index), (-1, index), VERDICT_COLORS[verdict]))
    table.setStyle(TableStyle(commands))
    story += [table, Spacer(1, 5 * mm), Paragraph("Why the decisions differ", heading)]

    bullets = [
        "<b>Porosity:</b> calibrated coverage 97% (95% lower bound 91.5%); RMSE-ratio upper bound 0.64; width-ratio upper bound 0.53; all non-Gaussian intervals contain the truth.",
        "<b>Water saturation:</b> the held-panel multiplier 1.60 is only half the bootstrap requirement 3.19; the original data-only profile and the bootstrap basic interval miss the truth.",
        "<b>Aspect ratio:</b> profile coverage is acceptable, but both the RMSE and width intervals cross one. There is no demonstrated advantage over elastic-only uncertainty.",
        "<b>Secondary porosity:</b> intervals can be honest, but RMSE and width are worse than elastic-only. That is the expected negative-control outcome.",
    ]
    for item in bullets:
        story.append(Paragraph(f"- {item}", body))
    story += [
        Spacer(1, 2 * mm),
        Paragraph("Numerical integrity", heading),
        Paragraph(
            "The run contains 796 bootstrap fits, split into 99 calibration and 100 validation replicates per sentinel. The operational estimator was reproduced to 1.2e-12. There were no materially negative likelihood-ratio values, no disconnected confidence sets and no profile-bound contacts. One conditioned calibration fit reached the iteration limit and was retained: 795/796 fits were valid (99.87%).",
            body,
        ),
        PageBreak(),
        Paragraph("Data-only profile likelihoods", title),
        Paragraph(
            "All nuisance parameters are reoptimized at each target value. The shaded region is the bootstrap-calibrated 90% interval; the green line is the synthetic truth. Predictive model weights are not interpreted as likelihood probabilities.",
            subtitle,
        ),
        Image(str(SOURCE / "profile_likelihoods.png"), width=174 * mm, height=121 * mm),
        Spacer(1, 4 * mm),
        Paragraph(
            "The water-saturation profile is the decisive failure: its optimum is strongly displaced from the truth in the predeclared replicate. Secondary porosity has a broad profile, consistent with weak identifiability rather than false precision.",
            body,
        ),
        PageBreak(),
        Paragraph("Held-split coverage", title),
        Paragraph(
            "The first 99 simulations calibrate the profile threshold. The following 100 are held out for coverage evaluation. Phase-1.2A multipliers are frozen and are never relearned from these simulations.",
            subtitle,
        ),
        Image(str(SOURCE / "sentinel_coverages.png"), width=174 * mm, height=89 * mm),
        Spacer(1, 5 * mm),
        Paragraph("Next scientific gate", heading),
        Paragraph(
            "Proceed to off-library falsification for porosity: hybrid electrical laws, depth-varying connectivity, invasion mismatch and patchy saturation. Water saturation and secondary porosity remain STOP under the present formulation. Aspect ratio should be expanded beyond 199 replicates only if it becomes a primary paper claim.",
            body,
        ),
        Paragraph(
            "Scope: all current truths are present in the candidate constitutive library. Even the porosity PASS is a matched-library ceiling and cannot yet be interpreted as field generalization.",
            body,
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
