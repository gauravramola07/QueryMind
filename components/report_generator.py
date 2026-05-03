# components/report_generator.py
"""
QueryMind PDF Report Generator
Fixes applied vs v2:
  - FIX 1: matplotlib.use('Agg') moved to module level so it takes effect
            before pyplot is ever imported, regardless of import order.
  - FIX 2: _matplotlib_chart() now accepts explicit `col`, `group_col`,
            and `chart_type` params so each chart spec plots different data.
  - FIX 3: Removed the premature "Generating charts from dataset…" message
            that appeared alongside "No charts could be generated" when all
            fallbacks failed, causing doubled/overlapping text on Page 3.
  - FIX 4: Added PageBreak before Numeric Summary so Page 1 no longer
            overflows ~846 pts of content into the 694 pt frame (which was
            causing the visual overlap / content bleeding into footer).
  - FIX 5: Reduced grade-pill fontSize from 52 → 36 to reclaim ~60 pts of
            vertical space on Page 1 while still looking prominent.
  - FIX 6: Exception in _matplotlib_chart now prints a traceback so failures
            are diagnosable from the Streamlit console.
"""

# ── matplotlib MUST be configured before pyplot is imported anywhere ──────────
import matplotlib
matplotlib.use('Agg')                      # non-interactive PNG backend
import matplotlib.pyplot as plt            # safe to import here now

import io
import re
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ── Colour palette mirrors the app's dark theme ──────────────────────────────
PURPLE    = colors.HexColor('#667eea')
PURPLE_LT = colors.HexColor('#a78bfa')
GREEN     = colors.HexColor('#48bb78')
ORANGE    = colors.HexColor('#ed8936')
BLUE      = colors.HexColor('#4facfe')
RED_CLR   = colors.HexColor('#fc8181')
BG_DARK   = colors.HexColor('#0d0d2b')
BG_CARD   = colors.HexColor('#1a1a3e')
BG_ROW    = colors.HexColor('#161630')
BG_BAR    = colors.HexColor('#2d3748')
TEXT_WHT  = colors.HexColor('#f7fafc')
TEXT_MUT  = colors.HexColor('#a0aec0')
BORDER    = colors.HexColor('#2d3748')


def _grade_color(grade):
    return {'A': GREEN, 'B': BLUE, 'C': ORANGE, 'D': RED_CLR}.get(grade, PURPLE)


# ── Shared paragraph styles ───────────────────────────────────────────────────
def _styles():
    def s(name, **kw):
        return ParagraphStyle(name, **kw)
    return {
        'h1':   s('h1',   fontSize=28, textColor=TEXT_WHT, fontName='Helvetica-Bold', spaceAfter=4),
        'h2':   s('h2',   fontSize=13, textColor=PURPLE_LT, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6),
        'sub':  s('sub',  fontSize=11, textColor=PURPLE_LT, fontName='Helvetica', spaceAfter=3),
        'body': s('body', fontSize=9,  textColor=TEXT_WHT,  fontName='Helvetica', spaceAfter=5, leading=15),
        'muted':s('muted',fontSize=8,  textColor=TEXT_MUT,  fontName='Helvetica', spaceAfter=3),
        'cap':  s('cap',  fontSize=8,  textColor=TEXT_MUT,  fontName='Helvetica-Oblique', alignment=TA_CENTER),
        'kv':   s('kv',   fontSize=20, textColor=PURPLE,    fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=1),
        'kl':   s('kl',   fontSize=7,  textColor=TEXT_MUT,  fontName='Helvetica', alignment=TA_CENTER),
    }


# ── Custom doc template: dark background + header/footer ─────────────────────
class _Doc(BaseDocTemplate):
    def __init__(self, buf, dataset_name, **kw):
        super().__init__(buf, **kw)
        self._dataset = dataset_name
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id='main')
        self.addPageTemplates([
            PageTemplate(id='main', frames=[frame], onPage=self._decorate)
        ])

    def _decorate(self, canv, doc):
        canv.saveState()
        w, h = A4

        # Dark page background
        canv.setFillColor(BG_DARK)
        canv.rect(0, 0, w, h, fill=1, stroke=0)

        # Purple top accent bar
        canv.setFillColor(PURPLE)
        canv.rect(0, h - 3, w, 3, fill=1, stroke=0)

        # Header text
        canv.setFont('Helvetica', 7.5)
        canv.setFillColor(TEXT_MUT)
        canv.drawString(doc.leftMargin, h - 1.5*cm, 'QueryMind Analytics Report')
        canv.drawRightString(w - doc.rightMargin, h - 1.5*cm, self._dataset)

        # Header rule
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.4)
        canv.line(doc.leftMargin, h - 1.65*cm, w - doc.rightMargin, h - 1.65*cm)

        # Footer
        canv.line(doc.leftMargin, 1.4*cm, w - doc.rightMargin, 1.4*cm)
        canv.setFont('Helvetica-Oblique', 7.5)
        ts = datetime.now().strftime('%B %d, %Y  %H:%M')
        canv.drawString(doc.leftMargin, 0.8*cm, f'Generated by QueryMind  |  {ts}')
        canv.drawRightString(w - doc.rightMargin, 0.8*cm, f'Page {doc.page}')
        canv.restoreState()


# ── Real drawn progress bar ───────────────────────────────────────────────────
def _progress_bar(score, bar_w, bar_h=10, fill_color=None):
    if fill_color is None:
        if score >= 90:   fill_color = GREEN
        elif score >= 75: fill_color = BLUE
        elif score >= 60: fill_color = ORANGE
        else:             fill_color = RED_CLR

    d = Drawing(bar_w, bar_h)
    d.add(Rect(0, 0, bar_w, bar_h, fillColor=BG_BAR, strokeColor=None, rx=3, ry=3))
    fill_w = max(0, (score / 100) * bar_w)
    if fill_w > 0:
        d.add(Rect(0, 0, fill_w, bar_h, fillColor=fill_color, strokeColor=None, rx=3, ry=3))
    return d


# ── FIX 1 + 2: Matplotlib chart — explicit column params, module-level backend ─
def _matplotlib_chart(df, title, usable_w, chart_type='bar', col=None, group_col=None):
    """
    Generate a PNG chart via matplotlib and return a ReportLab Image flowable.

    Parameters
    ----------
    df         : pd.DataFrame  — source data
    title      : str           — chart title (displayed on the chart)
    usable_w   : float         — available page width in points
    chart_type : 'bar' | 'hist'
    col        : str | None    — numeric column to plot (auto-detect if None)
    group_col  : str | None    — categorical column to group by (bar charts)
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None

        # Resolve column defaults
        if col is None or col not in df.columns:
            col = numeric_cols[0]

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if group_col is None and cat_cols:
            group_col = cat_cols[0]

        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor='#0d0d2b')
        ax.set_facecolor('#1a1a3e')

        if chart_type == 'bar' and group_col and group_col in df.columns:
            grp = df.groupby(group_col)[col].sum().nlargest(10)
            ax.bar(
                grp.index.astype(str), grp.values,
                color='#667eea', edgecolor='#2d3748', linewidth=0.5
            )
            ax.set_xlabel(group_col, color='#a0aec0', fontsize=8)
            ax.set_ylabel(f'Sum of {col}', color='#a0aec0', fontsize=8)
        else:
            # Histogram / distribution
            ax.hist(
                df[col].dropna(), bins=20,
                color='#667eea', edgecolor='#2d3748', linewidth=0.5
            )
            ax.set_xlabel(col, color='#a0aec0', fontsize=8)
            ax.set_ylabel('Frequency', color='#a0aec0', fontsize=8)

        ax.set_title(title, color='#f7fafc', fontsize=10, pad=8)
        ax.tick_params(colors='#a0aec0', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3748')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#0d0d2b', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        img_h = usable_w * (3.5 / 9)
        return Image(buf, width=usable_w, height=img_h)

    except Exception:
        # FIX 6: Print traceback so the failure is visible in the Streamlit console
        print(f'[QueryMind PDF] Chart "{title}" failed:\n{traceback.format_exc()}')
        return None


def _plotly_image(fig, usable_w):
    """Try plotly/kaleido; return None so matplotlib fallback kicks in."""
    try:
        import plotly.io as pio
        png = pio.to_image(fig, format='png', width=900, height=420, scale=2)
        img_h = usable_w * (420 / 900)
        return Image(io.BytesIO(png), width=usable_w, height=img_h)
    except Exception:
        return None


# ── Reusable table builders ───────────────────────────────────────────────────
def _kpi_row(pairs, styles, usable_w):
    n   = len(pairs)
    cw  = [usable_w / n] * n
    hdr = [Paragraph(lbl.upper(), styles['kl']) for lbl, _ in pairs]
    val = [Paragraph(str(v),      styles['kv']) for _,   v in pairs]
    t   = Table([hdr, val], colWidths=cw)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), BG_CARD),
        ('BOX',           (0, 0), (-1, -1), 0.4, BORDER),
        ('INNERGRID',     (0, 0), (-1, -1), 0.4, BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def _data_table(headers, rows, usable_w, col_widths=None, font_size=8):
    if col_widths is None:
        col_widths = [usable_w / len(headers)] * len(headers)

    th_style = ParagraphStyle('th', fontSize=font_size,   textColor=PURPLE_LT,
                               fontName='Helvetica-Bold')
    td_style = ParagraphStyle('td', fontSize=font_size-1, textColor=TEXT_WHT,
                               fontName='Helvetica', leading=12)

    data = [[Paragraph(str(h), th_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(cell), td_style) for cell in row])

    row_bgs = [
        ('BACKGROUND', (0, i), (-1, i), BG_CARD if i % 2 == 0 else BG_ROW)
        for i in range(1, len(rows) + 1)
    ]

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1e1e4a')),
        ('LINEBELOW',     (0, 0), (-1, 0), 0.5, PURPLE),
        ('BOX',           (0, 0), (-1, -1), 0.4, BORDER),
        ('INNERGRID',     (0, 0), (-1, -1), 0.3, BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ] + row_bgs))
    return t


# ── Main entry point ──────────────────────────────────────────────────────────
def generate_pdf_report(df, fi, health, kpis, data_summary,
                         cleaning_report=None, figures=None,
                         file_name='dataset'):
    """
    Build the QueryMind PDF report and return raw bytes.

    Parameters
    ----------
    df              : pd.DataFrame
    fi              : dict          — file_info from session state
    health          : dict          — from get_data_health_score()
    kpis            : list          — from session state kpis
    data_summary    : str | None    — AI executive summary text
    cleaning_report : dict | None   — from generate_cleaning_report()
    figures         : list | None   — Plotly Figure objects (optional)
    file_name       : str           — original uploaded filename
    """
    buf      = io.BytesIO()
    st       = _styles()
    story    = []
    usable_w = A4[0] - 4*cm

    doc = _Doc(
        buf, dataset_name=file_name,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=3.0*cm, bottomMargin=2.2*cm,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — COVER + DATA QUALITY + DATASET STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph('QueryMind', st['h1']))
    story.append(Paragraph('Data Analytics Report', st['sub']))
    story.append(HRFlowable(width=usable_w, thickness=0.5, color=PURPLE, spaceAfter=10))
    story.append(Spacer(1, 0.2*cm))

    for label, value in [
        ('Dataset',   fi.get('file_name', file_name)),
        ('Rows',      f"{fi.get('num_rows', len(df)):,}"),
        ('Columns',   str(fi.get('num_cols', len(df.columns)))),
        ('Generated', datetime.now().strftime('%B %d, %Y  %H:%M')),
    ]:
        story.append(Paragraph(
            f'<font color="#a78bfa"><b>{label}:</b></font>  {value}',
            st['body']
        ))

    # ── Data Quality ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph('Data Quality', st['h2']))

    grade     = health.get('grade', 'C')
    gc        = _grade_color(grade)
    grade_lbl = health.get('label', 'Fair')
    overall   = health.get('overall', 0)

    # FIX 5: Reduced grade fontSize 52 → 36 to save ~60 pts of vertical space
    grade_pill = Table([[
        Paragraph(grade, ParagraphStyle(
            'gp', fontSize=36, textColor=gc,
            fontName='Helvetica-Bold', alignment=TA_CENTER)),
        Paragraph(
            f'<font size="14" color="{gc.hexval()}"><b>{grade_lbl}</b></font><br/>'
            f'<font size="10" color="#a0aec0">Overall Score: {overall}/100</font>',
            ParagraphStyle('gl', fontSize=14, textColor=gc,
                            fontName='Helvetica-Bold', leading=22)),
    ]], colWidths=[3*cm, usable_w - 3*cm])
    grade_pill.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), BG_CARD),
        ('BOX',           (0, 0), (-1, -1), 0.5, gc),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING',   (0, 0), (-1, -1), 16),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(grade_pill)
    story.append(Spacer(1, 0.25*cm))

    # Real progress bars for metric breakdown
    breakdown = health.get('breakdown', {})
    if breakdown:
        bar_w    = usable_w - 5.5*cm
        bar_rows = []
        for metric, score in breakdown.items():
            bar = _progress_bar(score, bar_w, bar_h=9)
            bar_rows.append([
                Paragraph(metric.title(), ParagraphStyle(
                    'bm', fontSize=8, textColor=TEXT_MUT, fontName='Helvetica')),
                bar,
                Paragraph(f'{score}/100', ParagraphStyle(
                    'bs', fontSize=8, textColor=TEXT_WHT,
                    fontName='Helvetica-Bold', alignment=TA_RIGHT)),
            ])

        bar_table = Table(bar_rows, colWidths=[3*cm, bar_w, 2*cm])
        bar_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), BG_CARD),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID',     (0, 0), (-1, -1), 0.3, BORDER),
            ('BOX',           (0, 0), (-1, -1), 0.4, BORDER),
        ]))
        story.append(bar_table)

    # ── Dataset Statistics ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph('Dataset Statistics', st['h2']))

    null_c = int(df.isnull().sum().sum())
    num_c  = int(df.select_dtypes(include=[np.number]).shape[1])
    mem_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

    story.append(_kpi_row([
        ('Total Rows',     f"{len(df):,}"),
        ('Total Columns',  len(df.columns)),
        ('Numeric Cols',   num_c),
        ('Missing Values', null_c),
        ('Memory',         f'{mem_mb} MB'),
    ], st, usable_w))

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4: PAGE BREAK before Numeric Summary
    # Without this, Page 1 accumulates ~846 pts of content vs a 694 pt frame,
    # causing the grade pill, progress bars, stats and table to visually
    # overlap each other and bleed into the footer area.
    # ─────────────────────────────────────────────────────────────────────────
    story.append(PageBreak())

    # ── Numeric Summary ───────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        story.append(Paragraph('Numeric Summary', st['h2']))

        desc = numeric_df.describe().round(2)

        # Abbreviate column names to max 9 chars so they fit in cells
        abbrev = {}
        for col in desc.columns:
            abbrev[col] = (col[:8] + '…') if len(col) > 9 else col
        desc.columns = [abbrev[c] for c in desc.columns]

        n_cols     = len(desc.columns)
        stat_w     = 2.2*cm
        col_w      = (usable_w - stat_w) / n_cols
        col_widths = [stat_w] + [col_w] * n_cols

        hdr    = ['Stat'] + list(desc.columns)
        d_rows = [[str(idx)] + [str(v) for v in row]
                  for idx, row in desc.iterrows()]

        story.append(_data_table(
            hdr, d_rows, usable_w,
            col_widths=col_widths,
            font_size=7
        ))

    # ── Key KPIs ─────────────────────────────────────────────────────────────
    if kpis:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph('Key Metrics', st['h2']))
        pairs = [(k['label'], k['formatted_value']) for k in kpis[:6]]
        for i in range(0, len(pairs), 3):
            story.append(_kpi_row(pairs[i:i+3], st, usable_w))
            story.append(Spacer(1, 0.12*cm))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — AI EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    if data_summary:
        story.append(PageBreak())
        story.append(Paragraph('AI Executive Summary', st['h2']))
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', data_summary)
        clean = re.sub(r'#{1,6}\s*', '', clean)
        clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
        for para in clean.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), st['body']))
                story.append(Spacer(1, 0.12*cm))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — VISUAL ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('Visual Analytics', st['h2']))

    charts_added = 0

    # ── Attempt Plotly/kaleido first ─────────────────────────────────────────
    if figures:
        for fig in figures[:4]:
            img = _plotly_image(fig, usable_w)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.35*cm))
                charts_added += 1

    # ── FIX 1+2: Matplotlib fallback with explicit column routing ────────────
    # Each chart_spec now carries the exact col + group_col to plot, so all
    # four charts show DIFFERENT data instead of all defaulting to col[0].
    # The premature "Generating charts…" message (FIX 3) is removed — it was
    # always followed by "No charts could be generated" when charts failed,
    # producing two overlapping status lines on the Visual Analytics page.
    if charts_added < 4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Build specs — each entry describes exactly what to plot
        chart_specs = []

        if cat_cols and numeric_cols:
            chart_specs.append(dict(
                title=f'{numeric_cols[0]} by {cat_cols[0]}',
                chart_type='bar',
                col=numeric_cols[0],
                group_col=cat_cols[0],
            ))

        if numeric_cols:
            chart_specs.append(dict(
                title=f'Distribution of {numeric_cols[0]}',
                chart_type='hist',
                col=numeric_cols[0],
                group_col=None,
            ))

        if len(numeric_cols) >= 2:
            chart_specs.append(dict(
                title=f'Distribution of {numeric_cols[1]}',
                chart_type='hist',
                col=numeric_cols[1],
                group_col=None,
            ))

        if len(cat_cols) >= 2 and len(numeric_cols) >= 2:
            chart_specs.append(dict(
                title=f'{numeric_cols[1]} by {cat_cols[1]}',
                chart_type='bar',
                col=numeric_cols[1],
                group_col=cat_cols[1],
            ))

        needed = 4 - charts_added
        for spec in chart_specs[:needed]:
            img = _matplotlib_chart(
                df, spec['title'], usable_w,
                chart_type=spec['chart_type'],
                col=spec['col'],
                group_col=spec['group_col'],
            )
            if img:
                story.append(img)
                story.append(Paragraph(spec['title'], st['cap']))
                story.append(Spacer(1, 0.5*cm))
                charts_added += 1

    # FIX 3: Only ONE status message, only when truly no charts were produced
    if charts_added == 0:
        story.append(Paragraph(
            'No charts could be generated for this dataset.',
            st['muted']
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — SMART CLEANING REPORT (only if cleaning was run)
    # ─────────────────────────────────────────────────────────────────────────
    if cleaning_report:
        cr = cleaning_report
        story.append(PageBreak())
        story.append(Paragraph('Smart Cleaning Report', st['h2']))
        story.append(_kpi_row([
            ('Nulls Filled',       cr['total_nulls_filled']),
            ('Duplicates Removed', cr['duplicates_removed']),
            ('Columns Changed',    cr['columns_changed']),
            ('Rows After Clean',   f"{cr['rows_after']:,}"),
        ], st, usable_w))

        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph('Per-Column Changes', st['h2']))

        cr_rows = []
        for col in cr['columns']:
            ns = (f"{col['nulls_before']} -> {col['nulls_after']} (-{col['nulls_filled']})"
                  if col['nulls_filled'] > 0 else str(col['nulls_before']))
            ts = (f"{col['dtype_before']} -> {col['dtype_after']}"
                  if col['dtype_before'] != col['dtype_after'] else col['dtype_before'])
            us = (f"{col['unique_before']} -> {col['unique_after']}"
                  if col['unique_before'] != col['unique_after']
                  else str(col['unique_before']))
            ac = ' | '.join(col['actions']) if col['actions'] else 'No changes'
            cr_rows.append([col['name'], ts, ns, us, ac,
                            'Yes' if col['changed'] else '-'])

        story.append(_data_table(
            ['Column', 'Type', 'Nulls', 'Unique', 'Actions', 'Changed'],
            cr_rows, usable_w,
            col_widths=[3*cm, 2.5*cm, 2.5*cm, 2*cm, 4.5*cm, 1.5*cm],
            font_size=7
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # LAST PAGE — COLUMN SCHEMA
    # ─────────────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('Column Schema', st['h2']))
    schema_rows = [
        [col,
         str(df[col].dtype),
         str(df[col].count()),
         str(int(df[col].isnull().sum())),
         str(df[col].nunique())]
        for col in df.columns
    ]
    story.append(_data_table(
        ['Column', 'Type', 'Non-Null', 'Nulls', 'Unique Values'],
        schema_rows, usable_w,
        col_widths=[4*cm, 3*cm, 3*cm, 2.5*cm, 3.5*cm]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()