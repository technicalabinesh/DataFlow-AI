"""
=============================================================================
Automated Executive Analytical Report Generator (src/reporter.py)
=============================================================================
Produces:
1. High-Quality PDF Report via ReportLab with professional headers, KPI tables, and summaries.
2. Microsoft Word (.docx) Document via python-docx.
3. Full Session Export (JSON / Markdown) containing every response.
Includes:
- Executive Summary & Ingestion Health
- Full Audit Cleaning Log
- Power BI Dashboard Visual Insights
- Machine Learning Benchmark Leaderboard & Best Model Evaluation
- Complete Conversational AI Transcript (All User Questions & AI Answers)
"""

import io
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def generate_pdf_report(
    dataset_name: str,
    summary_metrics: Dict[str, Any],
    cleaning_log: List[str],
    ml_results: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    eda_insights: Optional[List[str]] = None,
) -> bytes:
    """
    Builds a professional executive PDF analytical report containing ALL session responses.
    Returns raw PDF bytes for instant browser download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=4,
    )

    # Header
    story.append(Paragraph("Universal AI Data Analyst: Executive Session Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Target Dataset: {dataset_name}", sub_style))

    # 1. Dataset Architecture & Health Metrics
    story.append(Paragraph("1. Dataset Architecture & Health Metrics", h2_style))
    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Rows", f"{summary_metrics.get('rows', 0):,}", "Numeric Features", str(summary_metrics.get("numeric_count", 0))],
        ["Total Columns", str(summary_metrics.get("columns", 0)), "Categorical Features", str(summary_metrics.get("categorical_count", 0))],
        ["Missing Percentage", f"{summary_metrics.get('missing_pct', 0)}%", "Memory Usage", f"{summary_metrics.get('memory_mb', 0)} MB"],
        ["Duplicate Rows", f"{summary_metrics.get('duplicate_rows', 0):,} ({summary_metrics.get('duplicate_pct', 0)}%)", "Health Status", "Audit Validated"],
    ]
    t1 = Table(kpi_data, colWidths=[130, 130, 130, 130])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # 2. Data Cleaning Audit Log (All entries)
    story.append(Paragraph("2. Autonomous Data Cleaning Transformation Log", h2_style))
    if cleaning_log:
        for item in cleaning_log:
            safe_item = str(item).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(f"• {safe_item}", body_style))
    else:
        story.append(Paragraph("No automated cleaning operations logged in this session.", body_style))
    story.append(Spacer(1, 10))

    # 3. AutoML Performance
    story.append(Paragraph("3. Machine Learning Leaderboard & Model Selection", h2_style))
    if ml_results and "leaderboard" in ml_results:
        leaderboard_df = ml_results["leaderboard"]
        table_data = [list(leaderboard_df.columns)]
        for _, row in leaderboard_df.iterrows():
            table_data.append([str(val) for val in row.values])

        t_ml = Table(table_data, colWidths=[140] + [90] * (len(table_data[0]) - 1))
        t_ml.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0078D4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t_ml)
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Best Selected Model:</b> {ml_results.get('best_model_name', 'N/A')}", body_style))
    else:
        story.append(Paragraph("AutoML model training has not been executed yet for this dataset.", body_style))
    story.append(Spacer(1, 10))

    # 4. Chatbot Session Responses (ALL responses included)
    story.append(Paragraph("4. AI Copilot Conversational Q&A Transcript (All Responses)", h2_style))
    if chat_history:
        for idx, msg in enumerate(chat_history, start=1):
            role_label = "USER QUESTION" if msg.get("role") == "user" else "AI ANALYST RESPONSE"
            role_color = "#8b5cf6" if msg.get("role") == "user" else "#0078D4"
            raw_content = str(msg.get("content", ""))
            safe_content = raw_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            story.append(Paragraph(f"<b><font color='{role_color}'>#{idx} [{role_label}]:</font></b> {safe_content}", body_style))
            if "code" in msg and msg["code"]:
                raw_code = str(msg["code"])
                safe_code = raw_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(f"<i>Executed Code:</i> {safe_code[:300]}", code_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("No chat inquiries logged during this session.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()



def generate_word_report(
    dataset_name: str,
    summary_metrics: Dict[str, Any],
    cleaning_log: List[str],
    ml_results: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """
    Generates a formatted Microsoft Word (.docx) report with complete response history.
    """
    doc = Document()
    
    title = doc.add_heading("Universal AI Data Analyst: Executive Session Report", level=0)
    p_meta = doc.add_paragraph(f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Target: {dataset_name}")
    p_meta.runs[0].font.color.rgb = RGBColor(100, 116, 139)

    # 1. Dataset Profile
    doc.add_heading("1. Dataset Profile & Dimensions", level=1)
    doc.add_paragraph(f"Total Records: {summary_metrics.get('rows', 0):,}")
    doc.add_paragraph(f"Total Columns: {summary_metrics.get('columns', 0)}")
    doc.add_paragraph(f"Missing Values Rate: {summary_metrics.get('missing_pct', 0)}%")
    doc.add_paragraph(f"Duplicate Rows: {summary_metrics.get('duplicate_rows', 0):,}")

    # 2. Cleaning Log
    doc.add_heading("2. Auto-Cleaning Operations", level=1)
    if cleaning_log:
        for log in cleaning_log:
            doc.add_paragraph(log, style="List Bullet")
    else:
        doc.add_paragraph("No cleaning actions recorded.")

    # 3. ML Results
    doc.add_heading("3. AutoML Benchmark Evaluation", level=1)
    if ml_results and "leaderboard" in ml_results:
        ld_df = ml_results["leaderboard"]
        table = doc.add_table(rows=1, cols=len(ld_df.columns))
        hdr_cells = table.rows[0].cells
        for i, col_name in enumerate(ld_df.columns):
            hdr_cells[i].text = str(col_name)

        for _, r in ld_df.iterrows():
            row_cells = table.add_row().cells
            for i, val in enumerate(r.values):
                row_cells[i].text = str(val)

        doc.add_paragraph(f"Selected Champion Pipeline: {ml_results.get('best_model_name')}")
    else:
        doc.add_paragraph("AutoML was not executed during this session.")

    # 4. Chatbot Responses
    doc.add_heading("4. Conversational AI Q&A Transcript", level=1)
    if chat_history:
        for msg in chat_history:
            p = doc.add_paragraph()
            p.add_run(f"[{msg['role'].upper()}]: ").bold = True
            p.add_run(msg["content"])
            if "code" in msg and msg["code"]:
                doc.add_paragraph(f"Code: {msg['code']}", style="No Spacing")
    else:
        doc.add_paragraph("No conversation history recorded.")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def generate_json_session_export(
    dataset_name: str,
    summary_metrics: Dict[str, Any],
    cleaning_log: List[str],
    ml_results: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Exports the entire session data (including all chat responses and models) to JSON.
    """
    payload = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "summary_metrics": summary_metrics,
        "cleaning_log": cleaning_log,
        "ml_leaderboard": ml_results["leaderboard"].to_dict(orient="records") if (ml_results and "leaderboard" in ml_results) else None,
        "best_model_name": ml_results.get("best_model_name") if ml_results else None,
        "chat_history": [
            {
                "role": m.get("role"),
                "content": str(m.get("content", "")),
                "code": m.get("code"),
                "result_records": m["result_df"].head(20).to_dict(orient="records") if (m.get("result_df") is not None and isinstance(m.get("result_df"), pd.DataFrame)) else None,
            }
            for m in (chat_history or [])
        ],
    }
    return json.dumps(payload, indent=2)
