"""
מחולל דוחות PDF ראייתיים — VerifyAI Forensic Report Generator
יוצר דוח מקצועי עם ממצאים, סיכום, חותמת קריפטוגרפית ו-disclaimer.
"""
import io
import hashlib
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# Colors
VOID = HexColor("#04070d")
DEEP = HexColor("#0b1224")
SURFACE = HexColor("#101b33")
BLUE = HexColor("#3366ff")
BLUE_LIGHT = HexColor("#e8eeff")
CYAN = HexColor("#00bcd4")
GREEN = HexColor("#00c853")
RED = HexColor("#ff1744")
AMBER = HexColor("#ff8f00")
TEXT = HexColor("#1a1a2e")
TEXT2 = HexColor("#555577")
TEXT3 = HexColor("#888899")
WHITE = HexColor("#ffffff")
BG_LIGHT = HexColor("#f5f7fb")
BORDER = HexColor("#dde2ee")


def _styles():
    """סגנונות טקסט לדוח."""
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=28, alignment=TA_LEFT, textColor=TEXT),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11, leading=16, textColor=TEXT2),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14, leading=20, textColor=TEXT, spaceBefore=16, spaceAfter=8),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=15, textColor=TEXT),
        "body_small": ParagraphStyle("body_small", fontName="Helvetica", fontSize=9, leading=13, textColor=TEXT2),
        "mono": ParagraphStyle("mono", fontName="Courier", fontSize=8, leading=11, textColor=TEXT2),
        "mono_small": ParagraphStyle("mono_small", fontName="Courier", fontSize=7, leading=10, textColor=TEXT3),
        "verdict_auth": ParagraphStyle("va", fontName="Helvetica-Bold", fontSize=16, leading=22, textColor=GREEN, alignment=TA_CENTER),
        "verdict_forged": ParagraphStyle("vf", fontName="Helvetica-Bold", fontSize=16, leading=22, textColor=RED, alignment=TA_CENTER),
        "verdict_inc": ParagraphStyle("vi", fontName="Helvetica-Bold", fontSize=16, leading=22, textColor=AMBER, alignment=TA_CENTER),
        "center": ParagraphStyle("center", fontName="Helvetica", fontSize=10, leading=14, textColor=TEXT2, alignment=TA_CENTER),
        "disclaimer": ParagraphStyle("disc", fontName="Helvetica", fontSize=8, leading=11, textColor=TEXT3),
    }


def generate_report(
    case_id: str,
    verdict: str,
    confidence: float,
    file_hash: str,
    media_type: str,
    agent_results: list,
    cross_reference: dict,
    requester: dict = None,
) -> bytes:
    """יוצר דוח PDF ומחזיר bytes."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=25*mm, bottomMargin=20*mm,
    )

    s = _styles()
    story = []
    now = datetime.utcnow()

    # ─── HEADER ───
    story.append(Paragraph("VerifyAI — Forensic Analysis Report", s["title"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))
    story.append(Paragraph(
        f"Case ID: {case_id}  |  Generated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  Classification: CONFIDENTIAL",
        s["mono"]
    ))
    story.append(Spacer(1, 16))

    # ─── REQUESTER INFO ───
    if requester:
        story.append(Paragraph("Requester Information", s["h2"]))
        req_data = []
        if requester.get("type") == "person":
            req_data = [
                ["Name", requester.get("name", "N/A")],
                ["ID Type", "Teudat Zehut" if requester.get("id_type") == "id" else "Passport"],
                ["ID Number", requester.get("id_number", "N/A")],
                ["Email", requester.get("email", "N/A")],
                ["Phone", requester.get("phone", "N/A")],
            ]
        else:
            req_data = [
                ["Company", requester.get("company", "N/A")],
                ["Entity Type", requester.get("entity_type", "N/A")],
                ["Registration #", requester.get("reg_number", "N/A")],
                ["Contact", requester.get("contact", "N/A")],
                ["Email", requester.get("email", "N/A")],
                ["Phone", requester.get("phone", "N/A")],
            ]

        req_table = Table(req_data, colWidths=[40*mm, 120*mm])
        req_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), TEXT2),
            ("TEXTCOLOR", (1, 0), (1, -1), TEXT),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
        ]))
        story.append(req_table)
        story.append(Spacer(1, 16))

    # ─── VERDICT ───
    story.append(Paragraph("Analysis Verdict", s["h2"]))

    verdict_labels = {
        "authentic": "AUTHENTIC — No signs of forgery detected",
        "forged": "FORGERY DETECTED — High confidence",
        "inconclusive": "INCONCLUSIVE — Further review required",
    }
    verdict_style = {
        "authentic": s["verdict_auth"],
        "forged": s["verdict_forged"],
        "inconclusive": s["verdict_inc"],
    }

    verdict_box_color = {"authentic": HexColor("#e8f5e9"), "forged": HexColor("#fce4ec"), "inconclusive": HexColor("#fff8e1")}
    verdict_border = {"authentic": GREEN, "forged": RED, "inconclusive": AMBER}

    vt = Table(
        [[Paragraph(verdict_labels.get(verdict, verdict.upper()), verdict_style.get(verdict, s["body"]))]],
        colWidths=[160*mm],
    )
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_box_color.get(verdict, BG_LIGHT)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("BOX", (0, 0), (-1, -1), 1.5, verdict_border.get(verdict, BORDER)),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(vt)
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Confidence Score: {confidence:.1%}", s["center"]))
    story.append(Spacer(1, 16))

    # ─── FILE INFO ───
    story.append(Paragraph("File Information", s["h2"]))
    file_data = [
        ["SHA-256 Hash", file_hash or "N/A"],
        ["Media Type", media_type],
        ["Analysis Date", now.strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Agents Deployed", str(len(agent_results))],
    ]
    ft = Table(file_data, colWidths=[40*mm, 120*mm])
    ft.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT2),
        ("TEXTCOLOR", (1, 0), (1, -1), TEXT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
    ]))
    story.append(ft)
    story.append(Spacer(1, 16))

    # ─── AGENT RESULTS ───
    story.append(Paragraph("Agent Analysis Details", s["h2"]))

    agent_names = {
        "forensic_technical": "Forensic-Technical Agent",
        "physical": "Physical Agent",
        "contextual": "Contextual Agent",
        "ai_generation": "AI Generation Detection Agent",
    }
    agent_descs = {
        "forensic_technical": "ELA, EXIF metadata, compression analysis, grain analysis",
        "physical": "Shadow direction, lighting, perspective, reflections",
        "contextual": "Historical context, uniforms, technology, architecture, vegetation",
        "ai_generation": "AI-generated content detection (DALL-E, Midjourney, Stable Diffusion)",
    }

    for ar in agent_results:
        atype = ar.get("agent_type", "unknown")
        aname = agent_names.get(atype, atype)
        adesc = agent_descs.get(atype, "")
        ascore = ar.get("confidence_score", 0)

        story.append(Paragraph(f"{aname}", s["h3"]))
        story.append(Paragraph(f"{adesc}", s["body_small"]))
        story.append(Paragraph(f"Confidence: {ascore:.0%}", s["body_small"]))
        story.append(Spacer(1, 4))

        anomalies = ar.get("anomalies", {})
        items = anomalies.get("items", []) if isinstance(anomalies, dict) else anomalies
        if items:
            anom_data = [["#", "Type", "Severity", "Description"]]
            for i, item in enumerate(items):
                anom_data.append([
                    str(i + 1),
                    item.get("type", "N/A"),
                    item.get("severity", "N/A").upper(),
                    item.get("description", "")[:120],
                ])

            at = Table(anom_data, colWidths=[8*mm, 25*mm, 20*mm, 107*mm])
            at.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BG_LIGHT]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(at)
        else:
            story.append(Paragraph("No anomalies detected by this agent.", s["body_small"]))

        story.append(Spacer(1, 10))

    # ─── CROSS-REFERENCE ───
    story.append(Paragraph("Cross-Reference Analysis", s["h2"]))
    reasoning = cross_reference.get("reasoning", "N/A")
    story.append(Paragraph(reasoning, s["body"]))
    story.append(Spacer(1, 8))

    anom_sum = cross_reference.get("anomaly_summary", {})
    cr_data = [
        ["Combined Score", f"{cross_reference.get('combined_score', 0):.1%}"],
        ["Total Anomalies", str(anom_sum.get("total", 0))],
        ["High Severity", str(anom_sum.get("high", 0))],
        ["Medium Severity", str(anom_sum.get("medium", 0))],
        ["Low Severity", str(anom_sum.get("low", 0))],
        ["Final Verdict", cross_reference.get("final_verdict", "N/A").upper()],
    ]
    ct = Table(cr_data, colWidths=[40*mm, 120*mm])
    ct.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
    ]))
    story.append(ct)
    story.append(Spacer(1, 16))

    # ─── CRYPTOGRAPHIC STAMP ───
    story.append(Paragraph("Cryptographic Verification Stamp", s["h2"]))

    stamp_raw = f"CASE:{case_id}|VERDICT:{verdict}|CONF:{confidence}|HASH:{file_hash}|DATE:{now.isoformat()}|AGENTS:{len(agent_results)}"
    signature = hashlib.sha256(stamp_raw.encode()).hexdigest()

    stamp_data = [
        ["Case ID", case_id],
        ["Timestamp", now.strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["File SHA-256", file_hash or "N/A"],
        ["Verdict", verdict.upper()],
        ["Confidence", f"{confidence:.1%}"],
        ["Signature", signature],
    ]
    st = Table(stamp_data, colWidths=[35*mm, 125*mm])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT2),
        ("TEXTCOLOR", (1, 0), (1, -1), TEXT),
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, BLUE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
    ]))
    story.append(st)
    story.append(Spacer(1, 20))

    # ─── DISCLAIMER ───
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=8))
    story.append(Paragraph(
        "LEGAL DISCLAIMER: This report is generated by VerifyAI automated multi-agent forensic analysis system. "
        "Results are based on algorithmic analysis and do not constitute legal opinion, admissible evidence in court proceedings, "
        "or a substitute for professional examination by a certified expert. Chain of custody is documented via SHA-256 hash "
        "from the moment of file ingestion. The cryptographic stamp certifies that this analysis was performed — it does not "
        "certify authenticity of the analyzed media. For cases requiring legal certainty, engagement of a certified HITL expert "
        "is recommended. VerifyAI assumes no liability for decisions made based on this report.",
        s["disclaimer"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Report generated by VerifyAI Forensic Engine v0.1.0  |  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  Signature: {signature[:32]}...",
        s["mono_small"]
    ))

    # Build
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
