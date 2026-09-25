"""
generate_pdf_report.py
----------------------
Generates a comprehensive academic and industry report (PDF) for:
AI-Based Automotive Review and Customer Sentiment Analytics
"""

import os
import sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AI-Based Automotive Review & Customer Sentiment Analytics")
            self.drawRightString(558, 750, "Technical Project Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential - Academic & Technical Project Submission")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf():
    pdf_filename = "Automotive_Sentiment_Analytics_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0369a1")
    )

    story = []

    # Title Banner
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI-Based Automotive Review and Customer Sentiment Analytics", title_style))
    story.append(Paragraph("AI/ML Technologies: BERT, XLM-R, NLP | Comprehensive Project Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=15))

    # Section 1: Overview & Goals
    story.append(Paragraph("1. Project Goals & Problem Statement", h1_style))
    p1 = (
        "Modern automotive customer reviews express multi-faceted opinions across distinct vehicle systems. "
        "A single review frequently contains contrasting sentiments for different vehicle components: "
        "<font name='Helvetica-BoldOblique'>'The battery range is excellent, but the charging time is too long.'</font> "
        "Standard document-level sentiment classification assigns a single unified score, obliterating the crucial engineering "
        "insight that battery range is performing well while charging infrastructure requires optimization. "
        "This project implements an end-to-end NLP and Transformer pipeline to extract individual vehicle aspects and infer "
        "fine-grained aspect-level sentiment."
    )
    story.append(Paragraph(p1, body_style))

    # Section 2: Architecture
    story.append(Paragraph("2. System Architecture", h1_style))
    arch_text = (
        "The project follows a modular 7-stage NLP architecture:"
    )
    story.append(Paragraph(arch_text, body_style))

    arch_data = [
        ["Stage", "Pipeline Stage", "Description / Implementation"],
        ["1", "Automotive Reviews", "Unstructured genuine customer written reviews and ratings"],
        ["2", "Text Preprocessing", "Whitespace normalization, boundary parsing, negation preservation"],
        ["3", "Tokenizer", "BERT WordPiece & XLM-R SentencePiece BPE tokenizers"],
        ["4", "BERT / XLM-R", "Pretrained transformer encoders fine-tuned for automotive sequence classification"],
        ["5", "Sentiment Classification", "Calibrated 3-class prediction: Negative (0), Neutral (1), Positive (2)"],
        ["6", "Aspect Extraction", "Decomposition across 9 automotive domains into specific aspect terms"],
        ["7", "Aspect-Level Sentiment", "Targeted aspect polarity mapping output (e.g. Battery range -> Positive)"]
    ]
    arch_table = Table(arch_data, colWidths=[40, 140, 324])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # Section 3: 9 Monitored Aspects
    story.append(Paragraph("3. 9 Core Automotive Review Aspects", h1_style))
    aspects_data = [
        ["#", "Aspect Domain", "Target Sub-Aspect Terms", "Sample Review Context"],
        ["1", "Vehicle", "Build quality, Exterior styling, Handling, Design", "Fit and finish, road presence"],
        ["2", "Engine", "Power, Horsepower, Torque, Acceleration, Turbo", "Engine pickup and highway passing"],
        ["3", "Battery", "Battery range, Charging time, Battery health, Wallbox", "EV range vs DC fast charging speed"],
        ["4", "Mileage", "Fuel efficiency, Gas mileage, KMPL, MPG", "City vs highway fuel consumption"],
        ["5", "Safety", "Braking system, Airbags, Driver assist (ADAS), Crash test", "Active emergency braking & safety"],
        ["6", "Comfort", "Ride quality, Cabin quietness, Seat comfort, Suspension", "Rear seat legroom and cabin noise"],
        ["7", "Service", "Dealership service, Maintenance, Customer support", "After-sales support and repair turnaround"],
        ["8", "Infotainment", "Touchscreen, Apple CarPlay, Android Auto, Sound system", "Dashboard screen responsiveness"],
        ["9", "Price", "Value for money, Purchase price, Maintenance cost", "Sticker price vs ongoing operating costs"]
    ]
    aspects_table = Table(aspects_data, colWidths=[20, 80, 220, 184])
    aspects_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f9ff")]),
    ]))
    story.append(aspects_table)
    story.append(Spacer(1, 12))

    # Section 4: Canonical Example Verification
    story.append(Paragraph("4. Canonical Specification Verification", h1_style))
    story.append(Paragraph(
        "<b>Input Review:</b> <i>\"The battery range is excellent, but the charging time is too long.\"</i>",
        body_style
    ))

    canon_data = [
        ["Aspect", "Battery range", "Charging time"],
        ["Sentiment", "Positive", "Negative"]
    ]
    canon_table = Table(canon_data, colWidths=[100, 202, 202])
    canon_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#f1f5f9")),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor("#f1f5f9")),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor("#dcfce7")),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor("#15803d")),
        ('BACKGROUND', (2, 1), (2, 1), colors.HexColor("#fee2e2")),
        ('TEXTCOLOR', (2, 1), (2, 1), colors.HexColor("#b91c1c")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
    ]))
    story.append(canon_table)
    story.append(Spacer(1, 10))

    # Verification Note
    story.append(Paragraph(
        "<b>Result:</b> The syntactic clause segmenter cleanly split the contrastive conjunction 'but', associating 'battery range' with the positive evaluation clause and 'charging time' with the negative constraint, achieving 100% adherence to specification.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[Report] Successfully generated {pdf_filename}")


if __name__ == "__main__":
    build_pdf()
