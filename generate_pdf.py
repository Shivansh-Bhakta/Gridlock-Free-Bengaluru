import os
import sys
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Define path constants
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_CSV = os.path.join(DATA_DIR, "patrol_schedule.csv")
PDF_OUTPUT = os.path.join(DATA_DIR, "patrol_schedule.pdf")

# Custom Canvas class for Page Numbering and Running Headers/Footers
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
        
        # Header - Top bar accent
        self.setFillColor(colors.HexColor("#1e272e"))
        self.rect(0, 770, 612, 22, fill=True, stroke=False)
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 8)
        self.drawString(36, 777, "BENGALURU TRAFFIC POLICE -- PARKING INTELLIGENCE PIPELINE")
        self.drawRightString(576, 777, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Divider line below header
        self.setStrokeColor(colors.HexColor("#3d4d5e"))
        self.setLineWidth(1)
        self.line(36, 765, 576, 765)
        
        # Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))
        self.drawString(36, 30, "CONFIDENTIAL -- FOR INTERNAL BTP ENFORCEMENT USE ONLY")
        self.drawRightString(576, 30, f"Page {self._pageNumber} of {page_count}")
        
        self.restoreState()


def build_pdf():
    print("[PDF] Generating ReportLab PDF schedule...")
    if not os.path.exists(SCHEDULE_CSV):
        print(f"[ERROR] Schedule CSV not found at {SCHEDULE_CSV}. Run pipeline first!")
        sys.exit(1)
        
    df = pd.read_csv(SCHEDULE_CSV)
    
    # Document Setup (Letter page size with 36pt (0.5 inch) margins to maximize print layout space)
    doc = SimpleDocTemplate(
        PDF_OUTPUT,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=15
    )
    
    desc_style = ParagraphStyle(
        'DocDesc',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=15
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=1 # Centered
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#2c3e50")
    )
    
    table_cell_bold_style = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#2c3e50")
    )
    
    # -- DOCUMENT HEADER --
    story.append(Spacer(1, 10))
    story.append(Paragraph("DAILY SHIFT PATROL SCHEDULE", title_style))
    story.append(Paragraph("Targeted Enforcement Priority Scheduler | Top 50 High-Risk Hotspots & Patrol Assets", subtitle_style))
    
    narrative_text = (
        "<b>Notice:</b> This schedule ranks the top 50 parking violation hotspots in Bengaluru using "
        "an upgraded composite risk score incorporating a spatial Congestion Spillover Index (inflating scores "
        "near major intersections), XGBoost temporal density forecast, and Isolation Forest anomaly flagging. "
        "Patrol assets are optimized and assigned to cells based on local vehicle-type modes (Heavy Tow Trucks for tankers/buses, "
        "Patrol Jeeps for cars/autos, Interceptor Bikes for motorcycles/scooters). "
        "Officers must prioritize critical ranks and coordinate enforcers to the designated shifts and patrol units."
    )
    story.append(Paragraph(narrative_text, desc_style))
    
    # -- DATA TABLE PREPARATION --
    # Column headings (10 columns)
    table_data = [[
        Paragraph("Rank", table_header_style),
        Paragraph("Canonical Station", table_header_style),
        Paragraph("Latitude", table_header_style),
        Paragraph("Longitude", table_header_style),
        Paragraph("Risk Score", table_header_style),
        Paragraph("Morning (6a-2p)", table_header_style),
        Paragraph("Evening (2p-10p)", table_header_style),
        Paragraph("Night (10p-6a)", table_header_style),
        Paragraph("Assigned Patrol Shift", table_header_style),
        Paragraph("Assigned Patrol Unit", table_header_style)
    ]]
    
    # Table Widths: Total available width = 612 - 72 = 540
    # 30 + 85 + 40 + 40 + 45 + 45 + 45 + 45 + 75 + 90 = 540
    col_widths = [30, 85, 40, 40, 45, 45, 45, 45, 75, 90]
    
    # Color definition for shifts
    shift_colors = {
        'Morning': '#3498db',  # Blue
        'Evening': '#e67e22',  # Orange
        'Night': '#9b59b6'     # Purple
    }
    
    t_styles = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e272e")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dcdde1")),
    ]
    
    for idx, row in df.iterrows():
        # Clean text
        rank = f"{int(row['rank'])}"
        station = str(row['canonical_station'])
        lat = f"{row['grid_lat']:.4f}"
        lon = f"{row['grid_lon']:.4f}"
        
        # Check if anomaly exists and add an asterisk or highlight
        is_anom = int(row.get('is_anomaly', 0))
        if is_anom:
            station += " *"
            
        risk = f"{row['risk_score']:.4f}"
        m_count = f"{int(row['Morning'])}"
        e_count = f"{int(row['Evening'])}"
        n_count = f"{int(row['Night'])}"
        shift = str(row['assigned_shift'])
        patrol_unit = str(row.get('assigned_patrol_unit', 'Patrol Jeep'))
        
        # Color highlight for top 10 or anomalies
        bg_color = colors.HexColor("#f8f9fa") if idx % 2 == 1 else colors.white
        if idx < 10:
            bg_color = colors.HexColor("#fff2f2") # Highlight top 10 with very light red
        elif is_anom:
            bg_color = colors.HexColor("#fef5e7") # Highlight anomalies with very light orange
            
        t_styles.append(('BACKGROUND', (0, idx+1), (-1, idx+1), bg_color))
        
        shift_color = shift_colors.get(shift, "#2c3e50")
        shift_html = f"<font color='{shift_color}'><b>{shift}</b></font>"
        
        table_data.append([
            Paragraph(rank, table_cell_bold_style if idx < 10 else table_cell_style),
            Paragraph(station, table_cell_bold_style if idx < 10 or is_anom else table_cell_style),
            Paragraph(lat, table_cell_style),
            Paragraph(lon, table_cell_style),
            Paragraph(risk, table_cell_bold_style if idx < 10 else table_cell_style),
            Paragraph(m_count, table_cell_style),
            Paragraph(e_count, table_cell_style),
            Paragraph(n_count, table_cell_style),
            Paragraph(shift_html, table_cell_bold_style),
            Paragraph(patrol_unit, table_cell_bold_style if idx < 10 else table_cell_style)
        ])
        
    # Build Table
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(t_styles))
    
    story.append(t)
    
    # Add footnote for anomalies
    story.append(Spacer(1, 10))
    story.append(Paragraph("<font color='#e67e22'>* Note: Stations marked with asterisk (*) represent Spatial-Temporal Anomaly zones flagged by Isolation Forest.</font>", desc_style))
    
    # Save document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] ReportLab PDF schedule generated at {PDF_OUTPUT}")


if __name__ == "__main__":
    build_pdf()
