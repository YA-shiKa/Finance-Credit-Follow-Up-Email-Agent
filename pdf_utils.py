"""
pdf_utils.py
────────────
Generates a professional PDF report of all emails produced by the
Finance Credit Follow-Up Email Agent.

Uses ReportLab (Platypus) — already in requirements.txt.
Call generate_emails_pdf(results) → bytes  (ready for st.download_button).
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour palette (matches Streamlit UI) ─────────────────────────────────────
DARK_NAVY   = colors.HexColor("#1a1a2e")
MID_GREY    = colors.HexColor("#555e7a")
LIGHT_BG    = colors.HexColor("#f4f6f9")
BORDER      = colors.HexColor("#dde1ea")
GREEN_BG    = colors.HexColor("#d1fae5")
GREEN_FG    = colors.HexColor("#065f46")
YELLOW_BG   = colors.HexColor("#fef3c7")
YELLOW_FG   = colors.HexColor("#78350f")
RED_BG      = colors.HexColor("#fee2e2")
RED_FG      = colors.HexColor("#7f1d1d")
PURPLE_BG   = colors.HexColor("#f3e8ff")
PURPLE_FG   = colors.HexColor("#581c87")
GREY_BG     = colors.HexColor("#f3f4f6")
GREY_FG     = colors.HexColor("#374151")

STAGE_COLORS = {
    1: (GREEN_BG,  GREEN_FG,  "Warm & Friendly"),
    2: (YELLOW_BG, YELLOW_FG, "Polite but Firm"),
    3: (RED_BG,    RED_FG,    "Formal & Serious"),
    4: (PURPLE_BG, PURPLE_FG, "Stern & Urgent"),
    5: (GREY_BG,   GREY_FG,   "Legal Flag"),
}


def _styles():
    """Return a dict of ParagraphStyles."""
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1",
            fontSize=20,
            fontName="Helvetica-Bold",
            textColor=DARK_NAVY,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=10,
            fontName="Helvetica",
            textColor=MID_GREY,
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=8,
            fontName="Helvetica-Bold",
            textColor=MID_GREY,
            spaceBefore=14,
            spaceAfter=6,
            textTransform="uppercase",
        ),
        "label": ParagraphStyle(
            "label",
            fontSize=9,
            fontName="Helvetica-Bold",
            textColor=DARK_NAVY,
        ),
        "value": ParagraphStyle(
            "value",
            fontSize=9,
            fontName="Helvetica",
            textColor=GREY_FG,
        ),
        "subject": ParagraphStyle(
            "subject",
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=DARK_NAVY,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=9,
            fontName="Helvetica",
            textColor=GREY_FG,
            leading=14,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontSize=8,
            fontName="Helvetica",
            textColor=MID_GREY,
            alignment=1,  # centre
        ),
    }


def _stage_badge_table(stage: int, tone: str) -> Table:
    """Return a small coloured badge Table for the stage label."""
    bg, fg, _ = STAGE_COLORS.get(stage, (GREY_BG, GREY_FG, "Unknown"))
    label = f"  Stage {stage}: {tone}  "
    t = Table([[label]], colWidths=[120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TEXTCOLOR",  (0, 0), (-1, -1), fg),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [6]),
        ("BOX",        (0, 0), (-1, -1), 0.5, fg),
    ]))
    return t


def _summary_table(results: list[dict]) -> Table:
    """Compact summary table at the top of the PDF."""
    header = ["Invoice", "Client", "Amount (Rs)", "Days OD", "Stage", "Status"]
    rows = [header]
    for r in results:
        rows.append([
            r.get("invoice_no", "-"),
            r.get("client_name", "-"),
            f"{int(r.get('amount', 0)):,}",
            str(r.get("days_overdue", "-")),
            f"Stage {r.get('stage', '-')}",
            r.get("send_status", "-"),
        ])

    col_widths = [70, 90, 70, 45, 50, 55]
    t = Table(rows, colWidths=col_widths, repeatRows=1)

    style = [
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 8),
        ("ROWPADDING",  (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ROWPADDING",  (0, 1), (-1, -1), 5),
        ("TEXTCOLOR",   (0, 1), (-1, -1), GREY_FG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, BORDER),
        ("ALIGN",       (2, 0), (3, -1), "RIGHT"),
    ]
    t.setStyle(TableStyle(style))
    return t


def generate_emails_pdf(results: list[dict]) -> bytes:
    """
    Generate a PDF report for a list of email result dicts.

    Parameters
    ──────────
    results : list of dicts as returned by generate_and_send() / run_agent()

    Returns
    ───────
    bytes — PDF content ready to pass to st.download_button(data=...)
    """
    if not results:
        raise ValueError("No email results to export.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Credit Follow-Up Email Report",
        author="Acme Finance Ltd — Credit Agent",
    )

    st = _styles()
    story = []
    now   = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")

    # ── Cover / header ────────────────────────────────────────────────────────
    story.append(Paragraph("Finance Credit Follow-Up", st["h1"]))
    story.append(Paragraph("Email Report — Generated by the LangGraph Agent", st["subtitle"]))
    story.append(Paragraph(f"Generated: {now}  |  Total emails: {len(results)}", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=DARK_NAVY, spaceAfter=10))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Summary", st["section"]))
    story.append(_summary_table(results))
    story.append(Spacer(1, 14))

    # ── Individual email cards ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Paragraph("Generated Emails", st["section"]))

    for i, res in enumerate(results, 1):
        stage = res.get("stage", 1)
        bg, fg, stage_label = STAGE_COLORS.get(stage, (GREY_BG, GREY_FG, "Unknown"))

        # ── Card header ───────────────────────────────────────────────────────
        header_data = [[
            Paragraph(
                f"<b>{i}. {res.get('invoice_no', '?')} — {res.get('client_name', '?')}</b>",
                ParagraphStyle("ch", fontSize=10, fontName="Helvetica-Bold",
                               textColor=DARK_NAVY),
            ),
            Paragraph(
                f"Stage {stage}: {res.get('tone', stage_label)}",
                ParagraphStyle("ct", fontSize=9, fontName="Helvetica-Bold",
                               textColor=fg, alignment=2),
            ),
        ]]
        card_header = Table(header_data, colWidths=[300, 170])
        card_header.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), LIGHT_BG),
            ("ROWPADDING",  (0, 0), (-1, -1), 8),
            ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        story.append(card_header)

        # ── Meta row ──────────────────────────────────────────────────────────
        meta_data = [[
            Paragraph(f"To: {res.get('client_email', '-')}", st["value"]),
            Paragraph(f"Amount: Rs {int(res.get('amount', 0)):,}", st["value"]),
            Paragraph(f"Days overdue: {res.get('days_overdue', '-')}", st["value"]),
            Paragraph(f"Status: {res.get('send_status', '-')}", st["value"]),
            Paragraph(f"{res.get('timestamp', '')[:19]} UTC", st["value"]),
        ]]
        meta_table = Table(meta_data, colWidths=[130, 80, 80, 80, 100])
        meta_table.setStyle(TableStyle([
            ("ROWPADDING",  (0, 0), (-1, -1), 5),
            ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
            ("LINEBELOW",   (0, 0), (-1, 0),  0.5, BORDER),
            ("BACKGROUND",  (0, 0), (-1, -1), colors.white),
        ]))
        story.append(meta_table)

        # ── Subject ───────────────────────────────────────────────────────────
        subj_table = Table(
            [[Paragraph(f"Subject: {res.get('subject', '')}", st["subject"])]],
            colWidths=[470],
        )
        subj_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), colors.white),
            ("ROWPADDING",  (0, 0), (-1, -1), 8),
            ("LINEBELOW",   (0, 0), (-1, -1), 0.5, BORDER),
            ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        story.append(subj_table)

        # ── Body ──────────────────────────────────────────────────────────────
        body_text = res.get("body", "").replace("\n", "<br/>")
        body_para = Paragraph(body_text, st["body"])
        body_table = Table([[body_para]], colWidths=[470])
        body_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), colors.white),
            ("ROWPADDING",  (0, 0), (-1, -1), 10),
            ("BOX",         (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        story.append(body_table)
        story.append(Spacer(1, 12))

        # Divider between cards (not after the last one)
        if i < len(results):
            story.append(HRFlowable(width="100%", thickness=0.4, color=BORDER, spaceAfter=8))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Acme Finance Ltd — Accounts Receivable Team | "
        "Generated by Finance Credit Follow-Up Agent v3.0 (LangGraph Edition) | "
        "CONFIDENTIAL",
        st["footer"],
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
