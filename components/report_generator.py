# components/report_generator.py
"""
QueryMind PDF Report Generator — PIXEL-PERFECT LAYOUT VERSION
Fixes:
  - Cover drawn 100% on canvas (no Platypus paragraphs) → zero overlap
  - Grade pill drawn on canvas with perfect vertical centering  
  - 2-column chart layout
  - Consistent spacing throughout
"""

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
    PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie

# ── Optional deps ─────────────────────────────────────────────────────────────
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    pass

_KALEIDO_AVAILABLE = False
try:
    import plotly.io as pio
    _KALEIDO_AVAILABLE = True
except ImportError:
    pass

# ── Colours ───────────────────────────────────────────────────────────────────
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
BG_HEAD   = colors.HexColor('#1e1e4a')
TEXT_WHT  = colors.HexColor('#f7fafc')
TEXT_MUT  = colors.HexColor('#a0aec0')
BORDER    = colors.HexColor('#2d3748')

CHART_PAL = [
    colors.HexColor('#667eea'), colors.HexColor('#a78bfa'),
    colors.HexColor('#43e97b'), colors.HexColor('#4facfe'),
    colors.HexColor('#f093fb'), colors.HexColor('#ed8936'),
    colors.HexColor('#fc8181'), colors.HexColor('#48bb78'),
]

W, H = A4          # 595.27 x 841.89 pts
LM   = 2 * cm      # left margin
RM   = 2 * cm      # right margin
TM   = 3 * cm      # top margin (space for header)
BM   = 2.2 * cm    # bottom margin
UW   = W - LM - RM # usable width  ≈ 453 pts


def _grade_color(g):
    return {'A': GREEN, 'B': BLUE, 'C': ORANGE, 'D': RED_CLR}.get(g, PURPLE)


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    def s(n, **kw): return ParagraphStyle(n, **kw)
    return {
        'h2':   s('h2',   fontSize=13, textColor=PURPLE_LT,
                  fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6),
        'body': s('body', fontSize=9,  textColor=TEXT_WHT,
                  fontName='Helvetica', spaceAfter=5, leading=15),
        'muted':s('muted',fontSize=8,  textColor=TEXT_MUT,
                  fontName='Helvetica', spaceAfter=3),
        'cap':  s('cap',  fontSize=7,  textColor=TEXT_MUT,
                  fontName='Helvetica-Oblique',
                  alignment=TA_CENTER, spaceAfter=4),
        'kv':   s('kv',   fontSize=20, textColor=PURPLE,
                  fontName='Helvetica-Bold',
                  alignment=TA_CENTER, spaceAfter=1),
        'kl':   s('kl',   fontSize=7,  textColor=TEXT_MUT,
                  fontName='Helvetica', alignment=TA_CENTER),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT TEMPLATE  — draws header/footer on every page
# Page 1 cover is drawn via onFirstPage callback stored in _page1_data
# ─────────────────────────────────────────────────────────────────────────────
class _Doc(BaseDocTemplate):
    def __init__(self, buf, dataset_name, page1_data, **kw):
        super().__init__(buf, **kw)
        self._ds   = dataset_name
        self._p1   = page1_data   # dict with fi, health, breakdown
        frame = Frame(LM, BM, UW, H - TM - BM, id='main')
        self.addPageTemplates([PageTemplate(
            id='main', frames=[frame], onPage=self._decorate
        )])

    # ── called for EVERY page ─────────────────────────────────────────────────
    def _decorate(self, canv, doc):
        canv.saveState()

        # Dark background
        canv.setFillColor(BG_DARK)
        canv.rect(0, 0, W, H, fill=1, stroke=0)

        # Purple top bar
        canv.setFillColor(PURPLE)
        canv.rect(0, H - 3, W, 3, fill=1, stroke=0)

        # Header text
        canv.setFont('Helvetica', 7.5)
        canv.setFillColor(TEXT_MUT)
        canv.drawString(LM, H - 1.45*cm, 'QueryMind Analytics Report')
        canv.drawRightString(W - RM, H - 1.45*cm, self._ds)

        # Header rule
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.4)
        canv.line(LM, H - 1.6*cm, W - RM, H - 1.6*cm)

        # Footer rule + text
        canv.line(LM, 1.35*cm, W - RM, 1.35*cm)
        canv.setFont('Helvetica-Oblique', 7.5)
        ts = datetime.now().strftime('%B %d, %Y  %H:%M')
        canv.drawString(LM, 0.75*cm,
                        f'Generated by QueryMind  |  {ts}')
        canv.drawRightString(W - RM, 0.75*cm, f'Page {doc.page}')

        # ── Page 1 only: draw cover + quality section on canvas ───────────────
        if doc.page == 1:
            self._draw_page1(canv)

        canv.restoreState()

    def _draw_page1(self, canv):
        """
        Draw the entire Page 1 content directly on the canvas.
        Pixel-perfect — no Platypus paragraph stacking involved.
        """
        p1  = self._p1
        fi  = p1['fi']
        h   = p1['health']
        now = datetime.now().strftime('%B %d, %Y  %H:%M')

        # ── Starting Y (just below the header rule) ───────────────────────────
        # Header rule is at H - 1.6cm, so we start 0.8cm below that
        y = H - TM - 0.3*cm     # ≈ 750 pts from bottom

        # ── Brand name ────────────────────────────────────────────────────────
        canv.setFont('Helvetica-Bold', 36)
        canv.setFillColor(TEXT_WHT)
        canv.drawString(LM, y, 'QueryMind')
        y -= 0.55*cm

        # ── Thin purple rule under brand ──────────────────────────────────────
        canv.setStrokeColor(PURPLE)
        canv.setLineWidth(1.2)
        canv.line(LM, y, W - RM, y)
        y -= 0.55*cm

        # ── Tagline ───────────────────────────────────────────────────────────
        canv.setFont('Helvetica', 13)
        canv.setFillColor(PURPLE_LT)
        canv.drawString(LM, y, 'Data Analytics Report')
        y -= 0.9*cm

        # ── Metadata grid ─────────────────────────────────────────────────────
        meta = [
            ('Dataset',   fi.get('file_name', '—')),
            ('Rows',      f"{fi.get('num_rows', 0):,}"),
            ('Columns',   str(fi.get('num_cols', 0))),
            ('Generated', now),
        ]
        label_x = LM
        value_x = LM + 2.1*cm
        for label, val in meta:
            canv.setFont('Helvetica-Bold', 9)
            canv.setFillColor(PURPLE_LT)
            canv.drawString(label_x, y, f'{label}:')
            canv.setFont('Helvetica', 9)
            canv.setFillColor(TEXT_WHT)
            canv.drawString(value_x, y, val)
            y -= 0.5*cm

        y -= 0.3*cm   # gap before quality section

        # ── "Data Quality" section heading ────────────────────────────────────
        canv.setFont('Helvetica-Bold', 13)
        canv.setFillColor(PURPLE_LT)
        canv.drawString(LM, y, 'Data Quality')
        y -= 0.6*cm

        # ── Grade pill (drawn as filled rect + centred text) ──────────────────
        grade     = h.get('grade', 'C')
        gc        = _grade_color(grade)
        grade_lbl = h.get('label', 'Fair')
        overall   = h.get('overall', 0)

        pill_h = 1.6*cm
        pill_y = y - pill_h

        # Pill background
        canv.setFillColor(BG_CARD)
        canv.roundRect(LM, pill_y, UW, pill_h,
                        radius=4, fill=1, stroke=1)
        canv.setStrokeColor(gc)
        canv.setLineWidth(0.8)
        canv.roundRect(LM, pill_y, UW, pill_h,
                        radius=4, fill=0, stroke=1)

        # Grade letter — perfectly centred in left 2.6cm of pill
        letter_box_w = 2.6*cm
        letter_cx    = LM + letter_box_w / 2
        letter_cy    = pill_y + pill_h / 2   # true vertical centre

        canv.setFont('Helvetica-Bold', 38)
        canv.setFillColor(gc)
        # ascent of 38pt Helvetica-Bold ≈ 28pt, descent ≈ 8pt
        # visual centre offset ≈ (ascent - descent)/2 ≈ 10pt
        canv.drawCentredString(letter_cx, letter_cy - 10, grade)

        # Vertical divider
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.5)
        canv.line(LM + letter_box_w,
                  pill_y + 6,
                  LM + letter_box_w,
                  pill_y + pill_h - 6)

        # Grade label + score (right side)
        text_x = LM + letter_box_w + 0.4*cm
        canv.setFont('Helvetica-Bold', 14)
        canv.setFillColor(gc)
        canv.drawString(text_x, pill_y + pill_h * 0.58, grade_lbl)

        canv.setFont('Helvetica', 9)
        canv.setFillColor(TEXT_MUT)
        canv.drawString(text_x, pill_y + pill_h * 0.25,
                        f'Overall Score: {overall}/100')

        y = pill_y - 0.25*cm

        # ── Progress bars ─────────────────────────────────────────────────────
        breakdown = h.get('breakdown', {})
        bar_area_w = UW - 5.5*cm
        label_col  = LM
        bar_col    = LM + 3*cm
        score_col  = bar_col + bar_area_w + 0.15*cm
        row_h      = 0.72*cm

        # Table background
        total_bar_h = len(breakdown) * row_h + 0.1*cm
        canv.setFillColor(BG_CARD)
        canv.rect(LM, y - total_bar_h, UW, total_bar_h,
                  fill=1, stroke=0)
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.4)
        canv.rect(LM, y - total_bar_h, UW, total_bar_h,
                  fill=0, stroke=1)

        row_y = y - row_h * 0.5
        for i, (metric, score) in enumerate(breakdown.items()):
            cy = row_y - i * row_h

            # Row separator
            if i > 0:
                canv.setStrokeColor(BORDER)
                canv.setLineWidth(0.3)
                canv.line(LM, cy + row_h / 2,
                          LM + UW, cy + row_h / 2)

            # Metric label
            canv.setFont('Helvetica', 8)
            canv.setFillColor(TEXT_MUT)
            canv.drawString(label_col + 0.3*cm, cy - 3,
                            metric.title())

            # Bar track
            track_h = 9
            track_y = cy - track_h / 2
            canv.setFillColor(BG_BAR)
            canv.roundRect(bar_col, track_y,
                           bar_area_w, track_h,
                           radius=3, fill=1, stroke=0)

            # Bar fill
            if score >= 90:   fc = GREEN
            elif score >= 75: fc = BLUE
            elif score >= 60: fc = ORANGE
            else:             fc = RED_CLR
            fill_w = max(0, score / 100 * bar_area_w)
            if fill_w > 0:
                canv.setFillColor(fc)
                canv.roundRect(bar_col, track_y,
                               fill_w, track_h,
                               radius=3, fill=1, stroke=0)

            # Score label
            canv.setFont('Helvetica-Bold', 8)
            canv.setFillColor(TEXT_WHT)
            canv.drawString(score_col, cy - 3, f'{score}/100')


# ── Progress bar Drawing (for non-page-1 use) ────────────────────────────────
def _progress_bar(score, bar_w, bar_h=10, fill_color=None):
    if fill_color is None:
        fill_color = (GREEN if score >= 90 else BLUE if score >= 75
                      else ORANGE if score >= 60 else RED_CLR)
    d = Drawing(bar_w, bar_h)
    d.add(Rect(0, 0, bar_w, bar_h, fillColor=BG_BAR,
               strokeColor=None, rx=3, ry=3))
    fw = max(0, score / 100 * bar_w)
    if fw:
        d.add(Rect(0, 0, fw, bar_h, fillColor=fill_color,
                   strokeColor=None, rx=3, ry=3))
    return d


# ── Number formatter ──────────────────────────────────────────────────────────
def _fmt(val):
    try:
        av = abs(val)
        if av >= 1_000_000: return f'{val/1_000_000:.1f}M'
        if av >= 1_000:     return f'{val/1_000:.1f}K'
        return f'{val:.1f}'
    except Exception:
        return str(val)


# ── Tables ────────────────────────────────────────────────────────────────────
def _kpi_row(pairs, styles, usable_w):
    n  = len(pairs)
    cw = [usable_w / n] * n
    t  = Table(
        [[Paragraph(l.upper(), styles['kl']) for l, _ in pairs],
         [Paragraph(str(v),    styles['kv']) for _, v in pairs]],
        colWidths=cw,
    )
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), BG_CARD),
        ('BOX',           (0,0),(-1,-1), 0.4, BORDER),
        ('INNERGRID',     (0,0),(-1,-1), 0.4, BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 8),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


def _data_table(headers, rows, usable_w, col_widths=None, font_size=8):
    if col_widths is None:
        col_widths = [usable_w / len(headers)] * len(headers)
    th = ParagraphStyle('th', fontSize=font_size,
                        textColor=PURPLE_LT, fontName='Helvetica-Bold')
    td = ParagraphStyle('td', fontSize=font_size - 1,
                        textColor=TEXT_WHT,  fontName='Helvetica', leading=12)
    data = [[Paragraph(str(h), th) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), td) for c in row])
    bgs = [('BACKGROUND', (0, i), (-1, i),
            BG_CARD if i % 2 == 0 else BG_ROW)
           for i in range(1, len(rows) + 1)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), BG_HEAD),
        ('LINEBELOW',     (0,0),(-1,0), 0.5, PURPLE),
        ('BOX',           (0,0),(-1,-1), 0.4, BORDER),
        ('INNERGRID',     (0,0),(-1,-1), 0.3, BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 7),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ] + bgs))
    return t


# ── Native bar chart ──────────────────────────────────────────────────────────
def _native_bar(df, title, w, col=None, group_col=None):
    try:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        cc = df.select_dtypes(include=['object','category']).columns.tolist()
        if not nc: return None
        if col is None or col not in df.columns: col = nc[0]
        if group_col is None and cc: group_col = cc[0]

        if group_col and group_col in df.columns:
            grp    = df.groupby(group_col)[col].sum().nlargest(8)
            labels = [str(x)[:12] for x in grp.index]
            values = [float(v) for v in grp.values]
        else:
            counts, edges = np.histogram(df[col].dropna(), bins=8)
            labels = [f'{e:.0f}' for e in edges[:-1]]
            values = [float(c) for c in counts]

        if not values or max(values) == 0: return None

        cw_  = float(w); ch = 180
        pl   = 52; pb = 30; pt = 24; pr = 8
        pw   = cw_ - pl - pr; ph = ch - pb - pt

        d = Drawing(cw_, ch)
        d.add(Rect(0, 0, cw_, ch, fillColor=BG_CARD,
                   strokeColor=BORDER, strokeWidth=0.5))

        mx    = max(values)
        n     = len(values)
        sw    = pw / n
        bw    = sw * 0.65
        gap   = (sw - bw) / 2

        for i in range(5):
            gy = pb + ph * i / 4
            d.add(Line(pl, gy, cw_-pr, gy,
                       strokeColor=colors.HexColor('#2d3748'),
                       strokeWidth=0.3))
            d.add(String(pl - 3, gy - 3, _fmt(mx * i / 4),
                         fontSize=5.5, fillColor=TEXT_MUT,
                         textAnchor='end'))

        for i, (lbl, val) in enumerate(zip(labels, values)):
            bx  = pl + i * sw + gap
            bh  = max(2, val / mx * ph)
            clr = CHART_PAL[i % len(CHART_PAL)]
            d.add(Rect(bx, pb, bw, bh, fillColor=clr, strokeColor=None))
            d.add(String(bx + bw/2, pb + bh + 2, _fmt(val),
                         fontSize=5, fillColor=TEXT_WHT,
                         textAnchor='middle'))
            d.add(String(bx + bw/2, pb - 11, lbl[:10],
                         fontSize=5.5, fillColor=TEXT_MUT,
                         textAnchor='middle'))

        d.add(String(cw_/2, ch - 13, title,
                     fontSize=8, fillColor=TEXT_WHT,
                     fontName='Helvetica-Bold', textAnchor='middle'))
        return d
    except Exception:
        print(f'[PDF] bar "{title}":\n{traceback.format_exc()}')
        return None


# ── Native pie chart ──────────────────────────────────────────────────────────
def _native_pie(df, title, w, col=None, group_col=None):
    try:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        cc = df.select_dtypes(include=['object','category']).columns.tolist()
        if not nc or not cc: return None
        if col is None or col not in df.columns: col = nc[0]
        if group_col is None: group_col = cc[0]
        if group_col not in df.columns: return None

        grp    = df.groupby(group_col)[col].sum().nlargest(6)
        if grp.empty: return None
        labels = [str(x)[:16] for x in grp.index]
        values = [float(v) for v in grp.values]
        total  = sum(values)

        cw_, ch = float(w), 180
        d = Drawing(cw_, ch)
        d.add(Rect(0, 0, cw_, ch, fillColor=BG_CARD,
                   strokeColor=BORDER, strokeWidth=0.5))
        d.add(String(cw_/2, ch - 13, title,
                     fontSize=8, fillColor=TEXT_WHT,
                     fontName='Helvetica-Bold', textAnchor='middle'))

        pie = Pie()
        pie.x = 28; pie.y = 22
        pie.width = 118; pie.height = 118
        pie.data   = values
        pie.labels = [f'{v/total*100:.1f}%' for v in values]
        pie.sideLabels       = True
        pie.sideLabelsOffset = 0.08
        pie.simpleLabels     = False
        for i in range(len(values)):
            pie.slices[i].fillColor   = CHART_PAL[i % len(CHART_PAL)]
            pie.slices[i].strokeColor = BG_DARK
            pie.slices[i].strokeWidth = 0.5
            pie.slices[i].fontSize    = 5.5
            pie.slices[i].fontColor   = TEXT_WHT
        d.add(pie)

        lx = 185
        for i, (lbl, val) in enumerate(zip(labels, values)):
            ly = ch - 35 - i * 20
            if ly < 15: break
            d.add(Rect(lx, ly, 9, 9,
                       fillColor=CHART_PAL[i % len(CHART_PAL)],
                       strokeColor=None))
            d.add(String(lx + 13, ly + 1,
                         f'{lbl}: {_fmt(val)}',
                         fontSize=6.5, fillColor=TEXT_WHT,
                         textAnchor='start'))
        return d
    except Exception:
        print(f'[PDF] pie "{title}":\n{traceback.format_exc()}')
        return None


def _plotly_img(fig, usable_w):
    if not _KALEIDO_AVAILABLE: return None
    try:
        png   = pio.to_image(fig, format='png', width=900, height=400, scale=2)
        return Image(io.BytesIO(png), width=usable_w,
                     height=usable_w * 400/900)
    except Exception:
        return None


def _mpl_chart(df, title, usable_w, chart_type='bar', col=None, group_col=None):
    if not _MPL_AVAILABLE: return None
    try:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        if not nc: return None
        if col is None or col not in df.columns: col = nc[0]
        cc = df.select_dtypes(include=['object','category']).columns.tolist()
        if group_col is None and cc: group_col = cc[0]

        fig, ax = plt.subplots(figsize=(9, 3.2), facecolor='#0d0d2b')
        ax.set_facecolor('#1a1a3e')
        if chart_type == 'bar' and group_col and group_col in df.columns:
            grp = df.groupby(group_col)[col].sum().nlargest(10)
            ax.bar(grp.index.astype(str), grp.values,
                   color='#667eea', edgecolor='#2d3748', linewidth=0.4)
        else:
            ax.hist(df[col].dropna(), bins=20,
                    color='#667eea', edgecolor='#2d3748', linewidth=0.4)
        ax.set_title(title, color='#f7fafc', fontsize=9, pad=6)
        ax.tick_params(colors='#a0aec0', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140,
                    facecolor='#0d0d2b', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image(buf, width=usable_w, height=usable_w * 3.2/9)
    except Exception:
        print(f'[PDF] mpl "{title}":\n{traceback.format_exc()}')
        return None


# ── 2-column chart section ────────────────────────────────────────────────────
def _build_charts(df, story, st_s, usable_w, figures=None):
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    cc = df.select_dtypes(include=['object','category']).columns.tolist()

    items = []  # (drawing_or_image, caption)

    # Plotly
    if figures and _KALEIDO_AVAILABLE:
        for fig in figures[:4]:
            img = _plotly_img(fig, usable_w)
            if img: items.append((img, ''))

    # matplotlib
    if len(items) < 4 and _MPL_AVAILABLE:
        specs = []
        if cc and nc:
            specs.append(('bar', nc[0], cc[0], f'{nc[0]} by {cc[0]}'))
        if nc:
            specs.append(('hist', nc[0], None, f'Distribution of {nc[0]}'))
        if len(nc) >= 2:
            specs.append(('hist', nc[1], None, f'Distribution of {nc[1]}'))
        if len(cc) >= 2 and len(nc) >= 2:
            specs.append(('bar', nc[1], cc[1], f'{nc[1]} by {cc[1]}'))
        for ct, c, gc, ttl in specs[:4 - len(items)]:
            img = _mpl_chart(df, ttl, usable_w, chart_type=ct,
                             col=c, group_col=gc)
            if img: items.append((img, ttl))

    # ReportLab native
    if len(items) < 4:
        specs = []
        if cc and nc:
            specs.append(('bar', nc[0], cc[0], f'{nc[0]} by {cc[0]}'))
        if cc and nc:
            specs.append(('pie', nc[0], cc[0],
                          f'{cc[0]} Share of {nc[0]}'))
        if len(nc) >= 2 and cc:
            specs.append(('bar', nc[1], cc[0], f'{nc[1]} by {cc[0]}'))
        if len(cc) >= 2 and len(nc) >= 2:
            specs.append(('pie', nc[1], cc[1],
                          f'{cc[1]} Share of {nc[1]}'))
        for ct, c, gc, ttl in specs[:4 - len(items)]:
            drw = (_native_bar(df, ttl, usable_w, col=c, group_col=gc)
                   if ct == 'bar'
                   else _native_pie(df, ttl, usable_w, col=c, group_col=gc))
            if drw is not None:
                items.append((drw, ttl))

    if not items:
        story.append(Paragraph(
            'No charts could be generated.', st_s['muted']))
        return

    # Pad to even count
    if len(items) % 2:
        items.append((Spacer(1, 1), ''))

    cap_s = ParagraphStyle('cap2', fontSize=7, textColor=TEXT_MUT,
                           fontName='Helvetica-Oblique',
                           alignment=TA_CENTER, spaceAfter=0)
    half  = (usable_w - 0.3*cm) / 2

    for i in range(0, len(items), 2):
        lf, lc = items[i]
        rf, rc = items[i + 1]

        # Scale native Drawings to half-width
        for fl in (lf, rf):
            if isinstance(fl, Drawing) and fl.width > 1:
                scale       = half / fl.width
                fl.width    = half
                fl.height   = fl.height * scale
                fl.transform = (scale, 0, 0, scale, 0, 0)

        tbl = Table(
            [[lf,                        rf],
             [Paragraph(lc, cap_s),      Paragraph(rc, cap_s)]],
            colWidths=[half, half],
            hAlign='LEFT',
        )
        tbl.setStyle(TableStyle([
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
            ('TOPPADDING',    (0,0),(-1,-1), 0),
            ('BOTTOMPADDING', (0,0),(-1,-1), 3),
            ('LEFTPADDING',   (0,0),(-1,-1), 0),
            ('RIGHTPADDING',  (0,0),(-1,-1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.25*cm))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(df, fi, health, kpis, data_summary,
                         cleaning_report=None, figures=None,
                         file_name='dataset'):
    buf  = io.BytesIO()
    st_s = _styles()

    # Page 1 data bundle — passed into _Doc so _decorate can draw it
    page1_data = {'fi': fi, 'health': health}

    doc = _Doc(
        buf,
        dataset_name=fi.get('file_name', file_name),
        page1_data=page1_data,
        pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM,  bottomMargin=BM,
    )

    story = []

    # ── PAGE 1 spacer (actual content drawn by _decorate/_draw_page1) ─────────
    # We need enough Spacer height to fill page 1 so Platypus emits a page.
    # Page 1 content occupies roughly 18cm of the ~24.5cm frame — push rest.
    story.append(Spacer(1, H - TM - BM))   # fills the entire frame → page break

    # =========================================================================
    # PAGE 2 — DATASET STATISTICS + NUMERIC SUMMARY + KEY METRICS
    # =========================================================================
    story.append(Paragraph('Dataset Statistics', st_s['h2']))

    null_c = int(df.isnull().sum().sum())
    num_c  = int(df.select_dtypes(include=[np.number]).shape[1])
    mem_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    story.append(_kpi_row([
        ('Total Rows',     f'{len(df):,}'),
        ('Total Columns',  len(df.columns)),
        ('Numeric Cols',   num_c),
        ('Missing Values', null_c),
        ('Memory',         f'{mem_mb} MB'),
    ], st_s, UW))
    story.append(Spacer(1, 0.4*cm))

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        story.append(Paragraph('Numeric Summary', st_s['h2']))
        desc = num_df.describe().round(2)
        abbr = {c: (c[:8]+'…' if len(c) > 9 else c) for c in desc.columns}
        desc.columns = [abbr[c] for c in desc.columns]
        sw  = 2.2*cm
        cw_ = (UW - sw) / len(desc.columns)
        story.append(_data_table(
            ['Stat'] + list(desc.columns),
            [[str(idx)] + [str(v) for v in row]
             for idx, row in desc.iterrows()],
            UW, col_widths=[sw] + [cw_]*len(desc.columns), font_size=7,
        ))
        story.append(Spacer(1, 0.4*cm))

    if kpis:
        story.append(Paragraph('Key Metrics', st_s['h2']))
        pairs = [(k['label'], k['formatted_value']) for k in kpis[:6]]
        for i in range(0, len(pairs), 3):
            story.append(_kpi_row(pairs[i:i+3], st_s, UW))
            story.append(Spacer(1, 0.1*cm))

    # =========================================================================
    # PAGE 3 — AI EXECUTIVE SUMMARY
    # =========================================================================
    if data_summary:
        story.append(PageBreak())
        story.append(Paragraph('AI Executive Summary', st_s['h2']))
        story.append(HRFlowable(width=UW, thickness=0.4,
                                color=BORDER, spaceAfter=8))
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', data_summary)
        clean = re.sub(r'#{1,6}\s*', '', clean)
        clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
        for para in clean.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), st_s['body']))
                story.append(Spacer(1, 0.1*cm))

    # =========================================================================
    # PAGE 4 — VISUAL ANALYTICS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph('Visual Analytics', st_s['h2']))
    story.append(Paragraph(
        'Automated charts generated from your dataset '
        'by the QueryMind analytics engine.', st_s['muted']))
    story.append(Spacer(1, 0.2*cm))
    _build_charts(df, story, st_s, UW, figures=figures)

    # =========================================================================
    # PAGE 5 — SMART CLEANING REPORT
    # =========================================================================
    if cleaning_report:
        cr = cleaning_report
        story.append(PageBreak())
        story.append(Paragraph('Smart Cleaning Report', st_s['h2']))
        story.append(_kpi_row([
            ('Nulls Filled',       cr['total_nulls_filled']),
            ('Duplicates Removed', cr['duplicates_removed']),
            ('Columns Changed',    cr['columns_changed']),
            ('Rows After Clean',   f"{cr['rows_after']:,}"),
        ], st_s, UW))
        story.append(Spacer(1, 0.35*cm))
        story.append(Paragraph('Per-Column Changes', st_s['h2']))
        cr_rows = []
        for col in cr['columns']:
            ns = (f"{col['nulls_before']}→{col['nulls_after']} (−{col['nulls_filled']})"
                  if col['nulls_filled'] > 0 else str(col['nulls_before']))
            ts = (f"{col['dtype_before']}→{col['dtype_after']}"
                  if col['dtype_before'] != col['dtype_after']
                  else col['dtype_before'])
            us = (f"{col['unique_before']}→{col['unique_after']}"
                  if col['unique_before'] != col['unique_after']
                  else str(col['unique_before']))
            cr_rows.append([col['name'], ts, ns, us,
                            ' | '.join(col['actions']) or '—',
                            'Yes' if col['changed'] else '—'])
        story.append(_data_table(
            ['Column','Type','Nulls','Unique','Actions','Changed'],
            cr_rows, UW,
            col_widths=[3*cm,2.5*cm,2.5*cm,2*cm,4.5*cm,1.5*cm],
            font_size=7,
        ))

    # =========================================================================
    # LAST PAGE — COLUMN SCHEMA
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph('Column Schema', st_s['h2']))
    story.append(_data_table(
        ['Column','Type','Non-Null','Nulls','Unique Values'],
        [[col, str(df[col].dtype),
          str(df[col].count()),
          str(int(df[col].isnull().sum())),
          str(df[col].nunique())]
         for col in df.columns],
        UW, col_widths=[4*cm,3*cm,3*cm,2.5*cm,3.5*cm],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()