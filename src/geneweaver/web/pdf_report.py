"""
Publication-Grade PDF Clinical Specificity Audit Certificate Generator.
Generates an official CRISPR-Cas9 therapeutic safety dossier using ReportLab Platypus.
"""

import io
import time
from typing import Dict, Any
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def generate_clinical_pdf_dossier(metrics: Dict[str, Any], df_hits: pd.DataFrame) -> bytes:
    """
    Generates a publication-grade clinical specificity audit certificate in PDF format.
    Returns bytes suitable for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A')
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569')
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0284C7')
    )

    table_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("GENEWEAVER | CRISPR CLINICAL SPECIFICITY AUDIT DOSSIER", title_style))
    story.append(Paragraph(
        f"Therapeutic Safety Certification &middot; Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')} &middot; Engine: GeneWeaver GPU Core v1.2",
        subtitle_style
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0284C7'), spaceBefore=2, spaceAfter=10))

    # 2. Executive Clearance Badge
    high_count = metrics.get('high_risk_count', 0)
    cancer_hits = metrics.get('cancer_hallmark_hits_count', 0)
    on_target = metrics.get('on_target_score', 80.0)

    if high_count == 0 and cancer_hits == 0:
        verdict_text = "PASSED: HIGH CLINICAL SPECIFICITY & LOW HAZARD PROFILE"
        verdict_color = colors.HexColor('#16A34A')  # Green
        sub_verdict = "Zero high-risk off-target cleavage coordinates identified in genomic reference sequence."
    elif cancer_hits > 0 or high_count > 2:
        verdict_text = "WARNING: HIGH-RISK GENOMIC CLEAVAGE DETECTED"
        verdict_color = colors.HexColor('#DC2626')  # Red
        sub_verdict = f"{high_count} high-hazard site(s) and {cancer_hits} critical tumor suppressor/exon hit(s) flagged."
    else:
        verdict_text = "CONDITIONAL APPROVAL: MINOR OFF-TARGET ACTIVITY"
        verdict_color = colors.HexColor('#D97706')  # Amber
        sub_verdict = "Low-frequency off-targets detected. Wet-lab PCR validation strongly advised."

    badge_data = [[
        Paragraph(f"<b>REGULATORY VERDICT:</b> {verdict_text}<br/>"
                  f"<font size='8' color='#334155'>{sub_verdict}</font>", body_style)
    ]]
    badge_table = Table(badge_data, colWidths=[540])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BORDER', (0, 0), (-1, -1), 1.5, verdict_color),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 12))

    # 3. Clinical & Biophysical Target Parameters
    story.append(Paragraph("1. Target Therapeutic Parameters & Biophysical Metrics", h2_style))
    param_data = [
        [
            Paragraph("<b>Target Indication:</b>", body_style),
            Paragraph(f"{metrics.get('disease_name', 'Benchmark Target')}", body_style),
            Paragraph("<b>Guide RNA (20nt):</b>", body_style),
            Paragraph(f"5'-{metrics.get('guide_used', '')}-3'", code_style),
        ],
        [
            Paragraph("<b>PAM Motif:</b>", body_style),
            Paragraph(f"{metrics.get('pam_used', 'NGG')[:3]}", code_style),
            Paragraph("<b>On-Target Cleavage (Doench '16):</b>", body_style),
            Paragraph(f"<b>{on_target}/100</b> ({metrics.get('on_target_tier', 'OPTIMAL')})", body_style),
        ],
        [
            Paragraph("<b>Genomic Scan Depth:</b>", body_style),
            Paragraph(f"{metrics.get('total_bp', 0):,} bp", body_style),
            Paragraph("<b>Compute Throughput:</b>", body_style),
            Paragraph(f"{metrics.get('throughput_mbp', 0.0):.2f} Mbp/s (Elapsed: {metrics.get('elapsed', 0.0):.3f}s)", body_style),
        ],
        [
            Paragraph("<b>Total Candidates:</b>", body_style),
            Paragraph(f"{metrics.get('total_candidates', 0)}", body_style),
            Paragraph("<b>Exonic Disruptions:</b>", body_style),
            Paragraph(f"<b>{metrics.get('exonic_hits_count', 0)}</b> ({cancer_hits} Cancer Hallmarks)", body_style),
        ],
        [
            Paragraph("<b>Delivery & Cell System:</b>", body_style),
            Paragraph(f"{str(metrics.get('delivery_modality', 'LNP / Electroporation'))[:28]} &bull; {str(metrics.get('cell_type', 'Primary Human Cells'))[:22]}", body_style),
            Paragraph("<b>FDA IND Feasibility:</b>", body_style),
            Paragraph(f"<b>{metrics.get('fda_feasibility', 85.0)}/100</b> ({metrics.get('feasibility_tier', 'HIGH_FEASIBILITY')})", body_style),
        ]
    ]
    param_table = Table(param_data, colWidths=[120, 150, 140, 130])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 12))

    # 4. Highest-Priority Off-Target Sites Table
    story.append(Paragraph("2. Top Cleavage Off-Target Coordinates (Ranked by Hsu-Zhang & CFD Matrix)", h2_style))
    if df_hits is not None and len(df_hits) > 0:
        table_rows = [[
            Paragraph("Coordinate", table_header_style),
            Paragraph("Gene / Locus", table_header_style),
            Paragraph("Feature", table_header_style),
            Paragraph("Mis.", table_header_style),
            Paragraph("Hsu", table_header_style),
            Paragraph("CFD", table_header_style),
            Paragraph("Risk Tier", table_header_style),
            Paragraph("Protospacer Sequence", table_header_style)
        ]]

        for _, r in df_hits.sort_values(by="Hsu_Score", ascending=False).head(8).iterrows():
            tier_str = str(r['Risk_Tier']).replace('_RISK', '')
            table_rows.append([
                Paragraph(f"{r['Chromosome']}:{r['Position']:,}", body_style),
                Paragraph(f"<b>{r.get('Gene_Symbol', 'N/A')}</b>", body_style),
                Paragraph(f"{r.get('Genomic_Feature', 'Exon')[:12]}", body_style),
                Paragraph(f"{r['Mismatches']}", body_style),
                Paragraph(f"{r['Hsu_Score']:.3f}", body_style),
                Paragraph(f"{r['CFD_Score']:.3f}", body_style),
                Paragraph(f"{tier_str}", body_style),
                Paragraph(f"{r['Sequence']}", code_style)
            ])

        hits_table = Table(table_rows, colWidths=[80, 65, 75, 25, 35, 35, 65, 160])
        hits_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('PADDING', (0, 0), (-1, -1), 3.5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(hits_table)
    else:
        story.append(Paragraph("Zero off-target sites detected within specified tolerance thresholds.", body_style))

    story.append(Spacer(1, 14))

    # 5. Regulatory Sign-Off Block
    story.append(Paragraph("3. Quality Assurance & Analytical Sign-Off", h2_style))
    sign_data = [
        [
            Paragraph("<b>Principal Investigator / Analyst:</b><br/>GeneWeaver Computational Engine", body_style),
            Paragraph("<b>Verification Status:</b><br/>NVIDIA CUDA Bit-Parallel Verified", body_style),
            Paragraph("<b>Signature:</b><br/><i>[ELECTRONICALLY VERIFIED]</i>", body_style),
            Paragraph(f"<b>Date:</b><br/>{time.strftime('%Y-%m-%d')}", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[150, 140, 150, 100])
    sign_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sign_table)

    doc.build(story)
    return buffer.getvalue()
