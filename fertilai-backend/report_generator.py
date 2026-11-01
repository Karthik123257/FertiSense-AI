"""
report_generator.py
Generates downloadable clinical PDF reports with patient-friendly explainability
and a clear Final Prediction Verdict box at the end using ReportLab.
"""

import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_patient_pdf(patient_name, gender, probability, strengths, concerns, recommendations, metrics_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=14
    )
    section_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155')
    )
    bullet_style = ParagraphStyle(
        'BulletTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=12
    )
    verdict_style = ParagraphStyle(
        'VerdictTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        alignment=1, # Center aligned
        textColor=colors.HexColor('#1e293b')
    )

    elements = []

    is_fertile = probability < 0.40
    readiness_pct = round((1.0 - probability) * 100, 1)
    risk_pct = round(probability * 100, 1)

    # Title & Subtitle
    elements.append(Paragraph("FertilAI — Clinical Evaluation & Explainability Summary", title_style))
    elements.append(Paragraph(f"Patient: <b>{patient_name}</b> &nbsp;|&nbsp; Sex: <b>{gender.capitalize()}</b> &nbsp;|&nbsp; Diagnostic System v2.4", subtitle_style))

    # Top Score Card Banner
    status_label = "Favorable Conception Baseline" if is_fertile else "Subfertility / Physiological Blockers Identified"
    banner_color = colors.HexColor('#ecfdf5') if is_fertile else colors.HexColor('#fff7ed')
    banner_border = colors.HexColor('#10b981') if is_fertile else colors.HexColor('#f97316')
    banner_text_color = '#047857' if is_fertile else '#c2410c'

    banner_html = f"""
    <font color="{banner_text_color}"><b>{status_label}</b></font><br/>
    <font size="8.5" color="#475569">Estimated Conception Readiness: <b>{readiness_pct}%</b> | Subfertility Risk Score: <b>{risk_pct}%</b></font>
    """
    banner_table = Table([[Paragraph(banner_html, body_style)]], colWidths=[530])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_color),
        ('BOX', (0,0), (-1,-1), 1, banner_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 10))

    # Extracted Lab Biomarkers Table
    elements.append(Paragraph("Report Biomarkers Extracted", section_h2))
    table_data = [["Biomarker", "Measured Value", "Clinical Reference"]]
    
    ref_map = {
        'Volume(mL)': '≥ 1.4 mL',
        'Sperm_Concentration(M/mL)': '≥ 16.0 M/mL',
        'Progressive_Motility(%)': '≥ 30.0 %',
        'Total_Motility(%)': '≥ 42.0 %',
        'Normal_Morphology(%)': '≥ 4.0 %',
        'Total_Testosterone(ng/dL)': '300 - 850 ng/dL',
        'Sitting_Hours_Per_Day': '< 8.0 hrs/day',
        'AMH(ng/mL)': '1.2 - 4.0 ng/mL',
        'TSH (mIU/L)': '0.5 - 2.5 mIU/L',
        'FSH(mIU/mL)': '3.5 - 10.5 mIU/mL',
        'LH(mIU/mL)': '2.4 - 12.6 mIU/mL',
        'PRL(ng/mL)': '< 25.0 ng/mL',
        'Follicle No. (Total)': '10 - 20 follicles',
        'BMI': '18.5 - 24.9',
        'Cycle length(days)': '26 - 32 days'
    }

    for k, v in metrics_dict.items():
        clean_name = k.replace('_', ' ')
        ref_str = ref_map.get(k, "Evaluated")
        table_data.append([clean_name, str(v), ref_str])

    biomarker_table = Table(table_data, colWidths=[230, 140, 160])
    biomarker_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#334155')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('TOPPADDING', (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    elements.append(biomarker_table)
    elements.append(Spacer(1, 10))

    # What Looks Great (Positive SHAP factors)
    elements.append(Paragraph("What Looks Great (Protective Baseline Strengths)", section_h2))
    if strengths:
        for s in strengths:
            elements.append(Paragraph(f"• <b>{s}</b>", bullet_style))
            elements.append(Spacer(1, 2))
    else:
        elements.append(Paragraph("• Baseline volume and general secondary markers maintained.", bullet_style))
    elements.append(Spacer(1, 8))

    # Areas Needing Attention (Negative SHAP / Blockers)
    elements.append(Paragraph("Areas Requiring Attention (Root Causes & Blockers)", section_h2))
    if concerns:
        for c in concerns:
            elements.append(Paragraph(f"• <b>{c}</b>", bullet_style))
            elements.append(Spacer(1, 2))
    else:
        elements.append(Paragraph("• No acute pathological bottlenecks detected in current screening panel.", bullet_style))
    elements.append(Spacer(1, 8))

    # Action Plan & Medical Roadmap
    elements.append(Paragraph("Actionable Care Pathway & Clinical Roadmap", section_h2))
    for r in recommendations:
        elements.append(Paragraph(f"✓ {r}", bullet_style))
        elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 14))

    # =========================================================================
    # FINAL PREDICTION VERDICT BOX AT THE END OF THE PDF
    # =========================================================================
    if is_fertile:
        verdict_title = "FINAL AI PREDICTION RESULT: FERTILE"
        verdict_sub = f"Based on your clinical biomarkers, your overall profile is <b>FERTILE</b> (Conception Readiness: <b>{readiness_pct}%</b>). Your key reproductive indicators meet healthy clinical benchmarks for natural conception."
        v_bg = colors.HexColor('#ecfdf5')
        v_border = colors.HexColor('#059669')
        v_title_color = '#065f46'
    else:
        verdict_title = "FINAL AI PREDICTION RESULT: INFERTILE (SUBFERTILITY IDENTIFIED)"
        verdict_sub = f"Based on your clinical biomarkers, your current profile indicates <b>INFERTILITY / SUBFERTILITY</b> (Risk Score: <b>{risk_pct}%</b> | Readiness: <b>{readiness_pct}%</b>). Please review the treatable root causes and clinical action plan above with your doctor."
        v_bg = colors.HexColor('#fef2f2')
        v_border = colors.HexColor('#dc2626')
        v_title_color = '#991b1b'

    verdict_html = f"""
    <font size="13" color="{v_title_color}"><b>{verdict_title}</b></font><br/><br/>
    <font size="9" color="#334155">{verdict_sub}</font>
    """
    verdict_table = Table([[Paragraph(verdict_html, verdict_style)]], colWidths=[530])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), v_bg),
        ('BOX', (0,0), (-1,-1), 1.5, v_border),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
    ]))
    elements.append(verdict_table)

    elements.append(Spacer(1, 12))
    disclaimer = Paragraph(
        "<font size='7' color='#94a3b8'>* This automated evaluation is generated using calibrated machine learning models (XGBoost & SHAP) trained on clinical reproductive datasets. It serves as an assistive triage tool and must be correlated with clinical judgment by a licensed reproductive endocrinologist or urologist.</font>",
        body_style
    )
    elements.append(disclaimer)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()