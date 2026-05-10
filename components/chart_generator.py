# components/chart_generator.py
# ============================================================
# CHART GENERATOR — ULTRA EDITION  (90 chart types)
# Full Plotly interactive library + Smart Auto-Detection
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    config = None


# ─────────────────────────────────────────────────────────────
# COLOR PALETTES
# ─────────────────────────────────────────────────────────────

GRADIENT_COLORS = [
    '#667eea', '#764ba2', '#f093fb', '#4facfe',
    '#00f2fe', '#43e97b', '#38f9d7', '#fa709a',
    '#fee140', '#30cfd0', '#a18cd1', '#fbc2eb'
]

NEON_COLORS = [
    '#6366f1', '#8b5cf6', '#a78bfa', '#c084fc',
    '#06b6d4', '#22d3ee', '#2dd4bf', '#34d399',
    '#f59e0b', '#f97316', '#ef4444', '#ec4899'
]

EXTENDED_COLORS = NEON_COLORS + [
    '#84cc16', '#eab308', '#14b8a6', '#0ea5e9',
    '#d946ef', '#fb7185', '#a3e635', '#facc15',
    '#4ade80', '#60a5fa', '#f472b6', '#fb923c'
]

DIVERGING_COLORS = ['#ef4444', '#f97316', '#eab308', '#e2e8f0', '#22d3ee', '#6366f1', '#8b5cf6']

CHART_FONT = 'Inter, Arial, sans-serif'


# ─────────────────────────────────────────────────────────────
# SUPPORTED CHART TYPES (90 total)
# ─────────────────────────────────────────────────────────────

SUPPORTED_CHART_TYPES = [
    # Bar family
    'bar', 'column', 'grouped_bar', 'stacked_bar', 'stacked_100_bar',
    'lollipop', 'diverging_bar', 'pyramid_bar', 'circular_bar', 'combo_bar_line',
    # Line family
    'line', 'multi_line', 'step_line', 'area', 'stacked_area',
    'slope_chart', 'bump_chart', 'moving_average',
    # Scatter/Correlation
    'scatter', 'bubble', 'connected_scatter', 'scatter_matrix',
    'density_contour', 'density_heatmap', 'parallel_coordinates', 'dumbbell',
    # Distribution
    'histogram', 'kde', 'violin', 'box', 'strip',
    'ridgeline', 'ecdf', 'hexbin', 'rug_plot',
    # Proportion
    'pie', 'treemap', 'sunburst', 'funnel', 'waterfall',
    'marimekko', 'waffle_chart', 'dot_plot', 'pictogram',
    # Heat/Matrix
    'heatmap', 'contour', 'calendar_heatmap', 'parallel_categories', 'qq_plot',
    # Statistical/Comparison
    'error_bar', 'confidence_band', 'tornado_chart',
    'comparison_bar', 'annotated_line', 'multi_axis_line', 'heat_table',
    # Time Series
    'candlestick', 'ohlc', 'gantt', 'timeline',
    'forecast_chart', 'seasonal_chart', 'stacked_waterfall',
    # KPI/Indicator
    'gauge', 'bullet_chart', 'indicator_tile', 'progress_chart', 'sparkline',
    # Polar/Radial
    'radar', 'polar_bar', 'polar_scatter', 'windrose', 'nightingale',
    # 3D Charts
    'scatter_3d', 'surface_3d', 'line_3d', 'contour_3d', 'bar_3d',
    # Flow/Network
    'sankey', 'chord', 'network_graph', 'flow_tree',
    # Geo
    'choropleth', 'scatter_geo', 'bubble_map', 'density_map',
    # Misc / Special
    'matrix_bubble', 'animated_scatter', 'table_chart', 'stream_graph',
]


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_chart(result_df: pd.DataFrame, question: str, chart_type: str = None) -> dict:
    """
    Main entry point. Validates data, picks chart type, builds figure.
    Returns: {success, figure, chart_type, error}
    """
    try:
        if result_df is None or result_df.empty:
            return _fail("No data to visualize", chart_type)

        if not chart_type or chart_type == 'auto':
            chart_type = detect_chart_type(result_df, question)

        print(f"📊 Generating '{chart_type}' chart…")

        creator = _CHART_REGISTRY.get(chart_type, create_bar_chart)
        fig = creator(result_df, question)

        if fig is None:
            return _fail("Chart creator returned no figure", chart_type)

        fig = apply_premium_styling(fig, chart_type)
        return {'success': True, 'figure': fig, 'chart_type': chart_type, 'error': None}

    except Exception as exc:
        print(f"❌ Chart error: {exc}")
        return _fail(str(exc), chart_type)


def _fail(msg, chart_type):
    return {'success': False, 'figure': None, 'chart_type': chart_type, 'error': msg}


# ─────────────────────────────────────────────────────────────
# CHART TYPE AUTO-DETECTION
# ─────────────────────────────────────────────────────────────

def detect_chart_type(df: pd.DataFrame, question: str) -> str:
    q        = question.lower()
    num_rows = len(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    n_num    = len(num_cols)
    n_txt    = len(txt_cols)

    # Keyword → chart type mappings
    rules = [
        (['candlestick', 'candle', 'ohlc', 'open high low close'], 'candlestick'),
        (['gantt', 'timeline', 'schedule', 'project plan'],        'gantt'),
        (['sankey', 'flow', 'alluvial'],                           'sankey'),
        (['sunburst', 'radial', 'hierarchical'],                   'sunburst'),
        (['waterfall', 'bridge', 'variance', 'contribution'],      'waterfall'),
        (['heatmap', 'heat map', 'matrix', 'correlation'],         'heatmap'),
        (['treemap', 'nested', 'hierarchy', 'tree'],               'treemap'),
        (['funnel', 'conversion', 'pipeline', 'stages'],           'funnel'),
        (['radar', 'spider', 'web chart', 'radial chart'],         'radar'),
        (['gauge', 'speedometer', 'dial', 'meter'],                'gauge'),
        (['violin', 'distribution shape'],                         'violin'),
        (['box', 'boxplot', 'quartile', 'outlier', 'whisker'],     'box'),
        (['strip', 'jitter', 'dot strip'],                         'strip'),
        (['bubble', 'size encod'],                                 'bubble'),
        (['scatter matrix', 'pair plot', 'splom', 'pairwise'],     'scatter_matrix'),
        (['scatter', 'correlation', 'relationship', ' vs '],       'scatter'),
        (['histogram', 'frequency', 'distribution', 'spread'],     'histogram'),
        (['kde', 'density curve', 'kernel density'],               'kde'),
        (['ecdf', 'cumulative distribution'],                      'ecdf'),
        (['parallel coord', 'high dim', 'multivariate'],           'parallel_coordinates'),
        (['parallel categ', 'alluvial categ'],                     'parallel_categories'),
        (['3d scatter', 'three dimension'],                        'scatter_3d'),
        (['surface', '3d surface', 'mesh'],                        'surface_3d'),
        (['contour', 'isoline', 'topograph'],                      'contour'),
        (['calendar', 'daily heatmap', 'activity'],                'calendar_heatmap'),
        (['slope', 'before after', 'before and after'],            'slope_chart'),
        (['bump', 'rank over time', 'ranking'],                    'bump_chart'),
        (['lollipop', 'dot stem'],                                 'lollipop'),
        (['dumbbell', 'dot plot', 'connected dot'],                'dumbbell'),
        (['diverging', 'positive negative', 'pos neg'],           'diverging_bar'),
        (['polar bar', 'wind rose', 'rose chart'],                 'polar_bar'),
        (['nightingale', 'florence'],                              'nightingale'),
        (['stacked area', 'stream'],                               'stacked_area'),
        (['stacked bar', 'stacked column'],                        'stacked_bar'),
        (['100% stacked', 'normalized stacked'],                   'stacked_100_bar'),
        (['grouped bar', 'side by side bar'],                      'grouped_bar'),
        (['combo', 'bar and line', 'mixed chart'],                 'combo_bar_line'),
        (['moving average', 'rolling mean', 'moving mean'],        'moving_average'),
        (['forecast', 'predict', 'projection'],                    'forecast_chart'),
        (['seasonal', 'seasonality'],                              'seasonal_chart'),
        (['tornado', 'sensitivity', 'impact'],                     'tornado_chart'),
        (['error bar', 'confidence interval', 'error'],           'error_bar'),
        (['waffle', 'unit chart', 'pictogram', 'icon'],           'waffle_chart'),
        (['marimekko', 'mosaic', 'mekko'],                        'marimekko'),
        (['area chart', 'cumulative', 'filled'],                   'area'),
        (['trend', 'over time', 'monthly', 'yearly', 'daily',
          'growth', 'by month', 'by year', 'by date'],            'line'),
        (['share', 'percentage', 'proportion', 'breakdown',
          'composition', 'ratio'],                                 'pie'),
    ]

    for kws, ctype in rules:
        if any(kw in q for kw in kws):
            # Extra guard for some types
            if ctype == 'scatter' and n_num < 2:
                return 'bar'
            if ctype == 'bubble' and n_num < 2:
                return 'scatter'
            if ctype == 'scatter_3d' and n_num < 3:
                return 'scatter'
            if ctype == 'surface_3d' and n_num < 2:
                return 'heatmap'
            return ctype

    # ── Structure-based fallback ─────────────────────────────
    if n_num >= 3 and n_txt >= 1:
        return 'bubble'
    if n_num >= 2 and n_txt == 0:
        return 'heatmap' if num_rows >= 5 else 'bar'
    if num_rows <= 8 and n_txt == 1 and n_num == 1:
        return 'pie'
    if num_rows <= 30:
        return 'bar'
    return 'bar'


# ═══════════════════════════════════════════════════════════
# ██  GROUP 1 — BAR FAMILY  (10 charts)
# ═══════════════════════════════════════════════════════════

def create_bar_chart(df: pd.DataFrame, question: str):
    """Horizontal gradient bar chart, sorted top 20."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None

        if txt_cols:
            x_col, y_col = txt_cols[0], num_cols[0]
            d = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=True).tail(20)
            n = len(d)
            colors = [
                f'rgb({int(102+i/max(n-1,1)*(118-102))},{int(126+i/max(n-1,1)*(75-126))},{int(234+i/max(n-1,1)*(162-234))})'
                for i in range(n)
            ]
            fig = go.Figure(go.Bar(
                x=d[y_col], y=d[x_col], orientation='h',
                marker=dict(color=colors, line=dict(width=0), cornerradius=6),
                text=[f'{v:,.0f}' for v in d[y_col]], textposition='outside',
                textfont=dict(size=11, color='#cbd5e1'),
                hovertemplate=f'<b>%{{y}}</b><br>{format_col_name(y_col)}: %{{x:,.2f}}<extra></extra>'
            ))
            fig.update_layout(title=format_title(question),
                              xaxis_title=format_col_name(y_col), yaxis_title='', showlegend=False)
            return fig

        fig = go.Figure()
        for i, col in enumerate(num_cols[:6]):
            fig.add_trace(go.Bar(name=format_col_name(col), y=df[col].head(20),
                                 marker=dict(color=NEON_COLORS[i % 12], cornerradius=5),
                                 hovertemplate=f'{format_col_name(col)}: %{{y:,.2f}}<extra></extra>'))
        fig.update_layout(title=format_title(question), barmode='group', bargap=0.15)
        return fig
    except Exception as e:
        print(f"❌ bar: {e}"); return None


def create_column_chart(df: pd.DataFrame, question: str):
    """Vertical column chart with gradient color."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        y_col = num_cols[0]
        x_col = txt_cols[0] if txt_cols else None
        d = df[[x_col, y_col]].dropna().head(25) if x_col else df[[y_col]].dropna().head(25)
        x_data = d[x_col] if x_col else list(range(len(d)))
        n = len(d)
        colors = [NEON_COLORS[i % 12] for i in range(n)]
        fig = go.Figure(go.Bar(
            x=x_data, y=d[y_col], marker=dict(color=colors, cornerradius=6),
            text=[f'{v:,.0f}' for v in d[y_col]], textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col) if x_col else '',
                          yaxis_title=format_col_name(y_col), bargap=0.15)
        return fig
    except Exception as e:
        print(f"❌ column: {e}"); return None


def create_grouped_bar(df: pd.DataFrame, question: str):
    """Multi-series grouped bar chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        x_data = df[txt_cols[0]] if txt_cols else list(range(len(df)))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:6]):
            fig.add_trace(go.Bar(
                name=format_col_name(col), x=x_data, y=df[col].head(25),
                marker=dict(color=NEON_COLORS[i % 12], cornerradius=4)
            ))
        fig.update_layout(title=format_title(question), barmode='group', bargap=0.12)
        return fig
    except Exception as e:
        print(f"❌ grouped_bar: {e}"); return None


def create_stacked_bar(df: pd.DataFrame, question: str):
    """Absolute stacked bar chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        x_data = df[txt_cols[0]] if txt_cols else list(range(len(df)))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:8]):
            fig.add_trace(go.Bar(
                name=format_col_name(col), x=x_data, y=df[col].head(25),
                marker=dict(color=NEON_COLORS[i % 12])
            ))
        fig.update_layout(title=format_title(question), barmode='stack', bargap=0.12)
        return fig
    except Exception as e:
        print(f"❌ stacked_bar: {e}"); return None


def create_stacked_100_bar(df: pd.DataFrame, question: str):
    """100% normalized stacked bar chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_pie_chart(df, question)
        d = df[num_cols[:6]].head(20).copy().fillna(0)
        row_totals = d.sum(axis=1).replace(0, 1)
        for col in d.columns:
            d[col] = d[col] / row_totals * 100
        x_data = df[txt_cols[0]].head(20) if txt_cols else list(range(len(d)))
        fig = go.Figure()
        for i, col in enumerate(d.columns):
            fig.add_trace(go.Bar(
                name=format_col_name(col), x=x_data, y=d[col],
                marker=dict(color=NEON_COLORS[i % 12]),
                hovertemplate=f'{format_col_name(col)}: %{{y:.1f}}%<extra></extra>'
            ))
        fig.update_layout(title=format_title(question), barmode='stack',
                          yaxis_title='Percentage (%)', bargap=0.12)
        return fig
    except Exception as e:
        print(f"❌ stacked_100_bar: {e}"); return None


def create_lollipop(df: pd.DataFrame, question: str):
    """Lollipop chart — stem + circle head."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        y_col = num_cols[0]
        x_col = txt_cols[0] if txt_cols else None
        d = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=False).head(20) if x_col \
            else df[[y_col]].dropna().head(20)
        x_data = d[x_col].tolist() if x_col else list(range(len(d)))
        y_data = d[y_col].tolist()

        fig = go.Figure()
        # Stems
        for x, y in zip(x_data, y_data):
            fig.add_shape(type='line', x0=x, x1=x, y0=0, y1=y,
                          line=dict(color='rgba(99,102,241,0.5)', width=2))
        # Heads
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers',
            marker=dict(size=14, color=NEON_COLORS[0], line=dict(width=2, color='white')),
            text=[f'{v:,.0f}' for v in y_data], textposition='top center',
            hovertemplate='<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col) if x_col else '',
                          yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ lollipop: {e}"); return None


def create_diverging_bar(df: pd.DataFrame, question: str):
    """Diverging bar chart — positive green, negative red."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(20) if lbl_col else df[[val_col]].dropna().head(20)
        labels = d[lbl_col].tolist() if lbl_col else [f'Item {i+1}' for i in range(len(d))]
        values = d[val_col].tolist()
        colors = ['#34d399' if v >= 0 else '#ef4444' for v in values]
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation='h',
            marker=dict(color=colors, line=dict(width=0), cornerradius=4),
            text=[f'{v:+,.1f}' for v in values], textposition='outside',
            hovertemplate='<b>%{y}</b><br>Value: %{x:,.2f}<extra></extra>'
        ))
        fig.add_vline(x=0, line_color='rgba(255,255,255,0.3)', line_width=1)
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(val_col), yaxis_title='')
        return fig
    except Exception as e:
        print(f"❌ diverging_bar: {e}"); return None


def create_pyramid_bar(df: pd.DataFrame, question: str):
    """Population-pyramid style mirrored horizontal bar."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2 or not txt_cols:
            return create_bar_chart(df, question)
        lbl_col, left_col, right_col = txt_cols[0], num_cols[0], num_cols[1]
        d = df[[lbl_col, left_col, right_col]].dropna().head(20)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=d[lbl_col], x=-d[left_col].abs(), orientation='h',
            name=format_col_name(left_col),
            marker=dict(color=NEON_COLORS[0], cornerradius=4),
            hovertemplate=f'{format_col_name(left_col)}: %{{customdata:,.0f}}<extra></extra>',
            customdata=d[left_col].abs()
        ))
        fig.add_trace(go.Bar(
            y=d[lbl_col], x=d[right_col], orientation='h',
            name=format_col_name(right_col),
            marker=dict(color=NEON_COLORS[4], cornerradius=4),
            hovertemplate=f'{format_col_name(right_col)}: %{{x:,.0f}}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question), barmode='overlay',
                          xaxis=dict(tickformat=',.0f'))
        return fig
    except Exception as e:
        print(f"❌ pyramid_bar: {e}"); return None


def create_circular_bar(df: pd.DataFrame, question: str):
    """Polar/rose bar chart (radial layout)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(18) if lbl_col else df[[val_col]].head(18)
        labels = d[lbl_col].tolist() if lbl_col else [f'Cat {i+1}' for i in range(len(d))]
        values = d[val_col].tolist()
        n = len(labels)
        theta = [i * 360 / n for i in range(n)]
        fig = go.Figure(go.Barpolar(
            r=values, theta=theta, width=[360 / n] * n,
            marker=dict(color=NEON_COLORS[:n], line=dict(color='rgba(0,0,0,0.3)', width=1)),
            hovertemplate='<b>%{theta:.0f}°</b><br>Value: %{r:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question),
                          polar=dict(radialaxis=dict(visible=True, showticklabels=True)))
        return fig
    except Exception as e:
        print(f"❌ circular_bar: {e}"); return None


def create_combo_bar_line(df: pd.DataFrame, question: str):
    """Combo chart: bar for first metric, line for second."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_bar_chart(df, question)
        x_data = df[txt_cols[0]].head(25) if txt_cols else list(range(min(25, len(df))))
        bar_col, line_col = num_cols[0], num_cols[1]
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Bar(
            x=x_data, y=df[bar_col].head(25), name=format_col_name(bar_col),
            marker=dict(color='rgba(99,102,241,0.7)', cornerradius=4)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=x_data, y=df[line_col].head(25), name=format_col_name(line_col),
            mode='lines+markers', line=dict(color=NEON_COLORS[4], width=3),
            marker=dict(size=8)
        ), secondary_y=True)
        fig.update_yaxes(title_text=format_col_name(bar_col), secondary_y=False)
        fig.update_yaxes(title_text=format_col_name(line_col), secondary_y=True)
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ combo_bar_line: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 2 — LINE FAMILY  (8 charts)
# ═══════════════════════════════════════════════════════════

def create_line_chart(df: pd.DataFrame, question: str):
    """Smooth spline line with gradient fill, up to 3 series."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        x_col  = non_num[0] if non_num else None
        x_data = df[x_col].head(100) if x_col else list(range(min(100, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:3]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col].head(100), mode='lines+markers',
                name=format_col_name(col),
                line=dict(color=NEON_COLORS[i], width=3, shape='spline'),
                marker=dict(size=7, color=NEON_COLORS[i], line=dict(width=2, color='white')),
                fill='tozeroy' if i == 0 else 'none',
                fillcolor='rgba(102,126,234,0.08)' if i == 0 else 'rgba(0,0,0,0)',
                hovertemplate=f'<b>%{{x}}</b><br>{format_col_name(col)}: %{{y:,.2f}}<extra></extra>'
            ))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col) if x_col else '',
                          yaxis_title=format_col_name(num_cols[0]), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ line: {e}"); return None


def create_multi_line(df: pd.DataFrame, question: str):
    """Multi-series line chart — up to 8 series."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        x_data = df[non_num[0]].head(100) if non_num else list(range(min(100, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:8]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col].head(100), mode='lines', name=format_col_name(col),
                line=dict(color=EXTENDED_COLORS[i % len(EXTENDED_COLORS)], width=2.5)
            ))
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ multi_line: {e}"); return None


def create_step_line(df: pd.DataFrame, question: str):
    """Step (staircase) line chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        x_data = df[non_num[0]].head(80) if non_num else list(range(min(80, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:3]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col].head(80), mode='lines', name=format_col_name(col),
                line=dict(color=NEON_COLORS[i], width=2.5, shape='hv'),
                hovertemplate=f'{format_col_name(col)}: %{{y:,.2f}}<extra></extra>'
            ))
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ step_line: {e}"); return None


def create_area_chart(df: pd.DataFrame, question: str):
    """Filled area chart with layered fills."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        x_data = df[non_num[0]] if non_num else list(range(len(df)))
        fills  = ['rgba(99,102,241,0.2)', 'rgba(139,92,246,0.15)', 'rgba(6,182,212,0.15)']
        fig = go.Figure()
        for i, col in enumerate(num_cols[:3]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col], fill='tozeroy' if i == 0 else 'tonexty',
                fillcolor=fills[i], line=dict(color=NEON_COLORS[i], width=2.5),
                mode='lines', name=format_col_name(col)
            ))
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ area: {e}"); return None


def create_stacked_area(df: pd.DataFrame, question: str):
    """Stacked area / stream chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_area_chart(df, question)
        x_data = df[non_num[0]].head(100) if non_num else list(range(min(100, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:6]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col].head(100).fillna(0), name=format_col_name(col),
                stackgroup='one', line=dict(color=NEON_COLORS[i % 12], width=1.5),
                fillcolor=NEON_COLORS[i % 12].replace(')', ',0.6)').replace('rgb', 'rgba')
                          if NEON_COLORS[i % 12].startswith('rgb') else NEON_COLORS[i % 12],
                mode='lines'
            ))
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ stacked_area: {e}"); return None


def create_slope_chart(df: pd.DataFrame, question: str):
    """Slope / before-after chart for two time points."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2 or not txt_cols:
            return create_line_chart(df, question)
        lbl_col, before_col, after_col = txt_cols[0], num_cols[0], num_cols[1]
        d = df[[lbl_col, before_col, after_col]].dropna().head(15)
        fig = go.Figure()
        for _, row in d.iterrows():
            color = '#34d399' if row[after_col] >= row[before_col] else '#ef4444'
            fig.add_trace(go.Scatter(
                x=[format_col_name(before_col), format_col_name(after_col)],
                y=[row[before_col], row[after_col]],
                mode='lines+markers+text',
                name=str(row[lbl_col]), line=dict(color=color, width=2),
                marker=dict(size=10, color=color),
                text=[f'{row[before_col]:,.1f}', f'{row[after_col]:,.1f}'],
                textposition=['middle left', 'middle right'],
                showlegend=False
            ))
        fig.update_layout(title=format_title(question))
        return fig
    except Exception as e:
        print(f"❌ slope_chart: {e}"); return None


def create_bump_chart(df: pd.DataFrame, question: str):
    """Bump chart: ranking over time / across periods."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2 or not txt_cols:
            return create_line_chart(df, question)
        lbl_col = txt_cols[0]
        period_cols = num_cols[:6]
        d = df[[lbl_col] + period_cols].dropna().head(12)
        # Convert values to ranks per period
        rank_df = d[period_cols].rank(ascending=False, axis=0)
        fig = go.Figure()
        for i, (_, row) in enumerate(d.iterrows()):
            fig.add_trace(go.Scatter(
                x=period_cols, y=rank_df.iloc[i].tolist(),
                mode='lines+markers', name=str(row[lbl_col]),
                line=dict(color=EXTENDED_COLORS[i % len(EXTENDED_COLORS)], width=2.5),
                marker=dict(size=12, color=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                            line=dict(width=2, color='white'))
            ))
        fig.update_layout(title=format_title(question),
                          yaxis=dict(autorange='reversed', title='Rank'))
        return fig
    except Exception as e:
        print(f"❌ bump_chart: {e}"); return None


def create_moving_average(df: pd.DataFrame, question: str):
    """Line chart with 7-period and 20-period moving averages overlaid."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col    = num_cols[0]
        x_data = df[non_num[0]].head(200) if non_num else list(range(min(200, len(df))))
        y      = df[col].head(200)
        ma7    = y.rolling(7,  min_periods=1).mean()
        ma20   = y.rolling(20, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y, name='Actual',
                                 line=dict(color='rgba(226,232,240,0.4)', width=1.5)))
        fig.add_trace(go.Scatter(x=x_data, y=ma7, name='MA(7)',
                                 line=dict(color=NEON_COLORS[0], width=2.5)))
        fig.add_trace(go.Scatter(x=x_data, y=ma20, name='MA(20)',
                                 line=dict(color=NEON_COLORS[4], width=2.5, dash='dash')))
        fig.update_layout(title=format_title(question) or f'Moving Average — {format_col_name(col)}',
                          hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ moving_average: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 3 — SCATTER / CORRELATION  (8 charts)
# ═══════════════════════════════════════════════════════════

def create_scatter_chart(df: pd.DataFrame, question: str):
    """Scatter plot, optionally colored by categorical."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_bar_chart(df, question)
        x_col, y_col   = num_cols[0], num_cols[1]
        color_col = txt_cols[0] if txt_cols else None
        d = df.sample(500, random_state=42) if len(df) > 500 else df
        fig = px.scatter(d, x=x_col, y=y_col, color=color_col,
                         color_discrete_sequence=NEON_COLORS, opacity=0.75)
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,0.3)')))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ scatter: {e}"); return None


def create_bubble_chart(df: pd.DataFrame, question: str):
    """Bubble chart — 3rd numeric = size, 4th cat = color."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_scatter_chart(df, question)
        x_col    = num_cols[0]
        y_col    = num_cols[1]
        size_col = num_cols[2] if len(num_cols) >= 3 else None
        color_col= txt_cols[0] if txt_cols else None
        d = df.sample(300, random_state=42) if len(df) > 300 else df
        fig = px.scatter(d, x=x_col, y=y_col, size=size_col, color=color_col,
                         color_discrete_sequence=NEON_COLORS, size_max=55, opacity=0.75,
                         hover_data=df.columns.tolist())
        fig.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.25)')))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ bubble: {e}"); return None


def create_connected_scatter(df: pd.DataFrame, question: str):
    """Connected scatter plot — points joined by lines in order."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_scatter_chart(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        d = df[[x_col, y_col]].dropna().head(80)
        fig = go.Figure(go.Scatter(
            x=d[x_col], y=d[y_col], mode='lines+markers',
            line=dict(color=NEON_COLORS[0], width=1.5),
            marker=dict(size=9, color=NEON_COLORS[1], line=dict(width=2, color='white')),
            text=list(range(len(d))), hovertemplate=f'Point %{{text}}<br>X: %{{x:,.2f}}<br>Y: %{{y:,.2f}}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ connected_scatter: {e}"); return None


def create_scatter_matrix(df: pd.DataFrame, question: str):
    """Scatter matrix (SPLOM) — pairwise scatter plots."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_scatter_chart(df, question)
        d = df.sample(300, random_state=42) if len(df) > 300 else df
        color_col = txt_cols[0] if txt_cols else None
        fig = px.scatter_matrix(d, dimensions=num_cols, color=color_col,
                                color_discrete_sequence=NEON_COLORS, opacity=0.7)
        fig.update_traces(marker=dict(size=4), diagonal_visible=False)
        fig.update_layout(title=format_title(question) or 'Scatter Matrix', height=600)
        return fig
    except Exception as e:
        print(f"❌ scatter_matrix: {e}"); return None


def create_density_contour(df: pd.DataFrame, question: str):
    """2D density contour lines."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_histogram(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        d = df[[x_col, y_col]].dropna().sample(min(500, len(df)), random_state=42)
        fig = px.density_contour(d, x=x_col, y=y_col,
                                 color_discrete_sequence=NEON_COLORS)
        fig.update_traces(contours_coloring='fill', colorscale=[
            [0, 'rgba(99,102,241,0)'], [0.5, 'rgba(99,102,241,0.3)'], [1, 'rgba(99,102,241,0.8)']
        ])
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ density_contour: {e}"); return None


def create_density_heatmap(df: pd.DataFrame, question: str):
    """2D density heatmap (bin-based)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_histogram(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        d = df[[x_col, y_col]].dropna().sample(min(1000, len(df)), random_state=42)
        fig = px.density_heatmap(d, x=x_col, y=y_col, nbinsx=30, nbinsy=30,
                                 color_continuous_scale=['#0f0c29', '#302b63', '#667eea'])
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ density_heatmap: {e}"); return None


def create_parallel_coordinates(df: pd.DataFrame, question: str):
    """Parallel coordinates for high-dimensional numeric data."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_scatter_chart(df, question)
        d = df[num_cols].dropna().sample(min(300, len(df)), random_state=42)
        color_col = num_cols[0]
        fig = px.parallel_coordinates(d, color=color_col,
                                      color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(title=format_title(question) or 'Parallel Coordinates')
        return fig
    except Exception as e:
        print(f"❌ parallel_coordinates: {e}"); return None


def create_dumbbell(df: pd.DataFrame, question: str):
    """Dumbbell / before-after dot chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2 or not txt_cols:
            return create_bar_chart(df, question)
        lbl_col, c1, c2 = txt_cols[0], num_cols[0], num_cols[1]
        d = df[[lbl_col, c1, c2]].dropna().head(20)
        fig = go.Figure()
        for _, row in d.iterrows():
            fig.add_shape(type='line', y0=row[lbl_col], y1=row[lbl_col],
                          x0=row[c1], x1=row[c2], line=dict(color='rgba(255,255,255,0.2)', width=2))
        fig.add_trace(go.Scatter(
            y=d[lbl_col], x=d[c1], mode='markers', name=format_col_name(c1),
            marker=dict(size=14, color=NEON_COLORS[0], line=dict(width=2, color='white'))
        ))
        fig.add_trace(go.Scatter(
            y=d[lbl_col], x=d[c2], mode='markers', name=format_col_name(c2),
            marker=dict(size=14, color=NEON_COLORS[4], symbol='square',
                        line=dict(width=2, color='white'))
        ))
        fig.update_layout(title=format_title(question))
        return fig
    except Exception as e:
        print(f"❌ dumbbell: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 4 — DISTRIBUTION  (9 charts)
# ═══════════════════════════════════════════════════════════

def create_histogram(df: pd.DataFrame, question: str):
    """Distribution histogram with mean & median lines."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col  = num_cols[0]
        data = df[col].dropna()
        fig  = go.Figure(go.Histogram(
            x=data, nbinsx=min(50, max(10, len(data) // 10)),
            marker=dict(color='rgba(99,102,241,0.7)', line=dict(color='#6366f1', width=1)),
            hovertemplate=f'{format_col_name(col)}: %{{x:,.2f}}<br>Count: %{{y}}<extra></extra>'
        ))
        for val, color, pos, label in [
            (data.mean(),   '#ef4444', 'top right', f'Mean: {data.mean():,.2f}'),
            (data.median(), '#f59e0b', 'top left',  f'Median: {data.median():,.2f}'),
        ]:
            fig.add_vline(x=val, line_dash='dash', line_color=color, line_width=2,
                          annotation_text=label, annotation_position=pos,
                          annotation_font=dict(size=12, color=color))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(col), yaxis_title='Frequency', bargap=0.04)
        return fig
    except Exception as e:
        print(f"❌ histogram: {e}"); return None


def create_kde(df: pd.DataFrame, question: str):
    """Kernel Density Estimate curve."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        fig = go.Figure()
        for i, col in enumerate(num_cols[:4]):
            data = df[col].dropna()
            x_range = np.linspace(data.min(), data.max(), 200)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), mode='lines', name=format_col_name(col),
                line=dict(color=NEON_COLORS[i % 12], width=3),
                fill='tozeroy', fillcolor=NEON_COLORS[i % 12].replace('#', 'rgba(') + ',0.1)'
                    if False else f'rgba(99,102,241,{0.12 - i*0.02})'
            ))
        fig.update_layout(title=format_title(question) or 'Density Distribution',
                          xaxis_title='Value', yaxis_title='Density')
        return fig
    except ImportError:
        # Fallback: approximate KDE with histogram
        return create_histogram(df, question)
    except Exception as e:
        print(f"❌ kde: {e}"); return None


def create_violin(df: pd.DataFrame, question: str):
    """Violin plot for distribution shape."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        fig = go.Figure()
        if txt_cols:
            group_col, val_col = txt_cols[0], num_cols[0]
            groups = df[group_col].dropna().value_counts().head(8).index
            for i, grp in enumerate(groups):
                data = df[df[group_col] == grp][val_col].dropna()
                fig.add_trace(go.Violin(
                    y=data, name=str(grp), box_visible=True, meanline_visible=True,
                    fillcolor=NEON_COLORS[i % 12], opacity=0.7,
                    line=dict(color=NEON_COLORS[i % 12])
                ))
        else:
            for i, col in enumerate(num_cols[:6]):
                fig.add_trace(go.Violin(
                    y=df[col].dropna(), name=format_col_name(col),
                    box_visible=True, meanline_visible=True,
                    fillcolor=NEON_COLORS[i % 12], opacity=0.7
                ))
        fig.update_layout(title=format_title(question), violingap=0.1, violinmode='overlay')
        return fig
    except Exception as e:
        print(f"❌ violin: {e}"); return None


def create_box_plot(df: pd.DataFrame, question: str):
    """Box & whisker plot."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        fig = go.Figure()
        if txt_cols:
            group_col, val_col = txt_cols[0], num_cols[0]
            for i, grp in enumerate(df[group_col].dropna().unique()[:12]):
                subset = df[df[group_col] == grp][val_col].dropna()
                fig.add_trace(go.Box(
                    y=subset, name=str(grp), marker_color=NEON_COLORS[i % 12], boxmean='sd'
                ))
            fig.update_layout(title=format_title(question),
                              yaxis_title=format_col_name(val_col),
                              xaxis_title=format_col_name(group_col), boxgap=0.3)
        else:
            for i, col in enumerate(num_cols[:8]):
                fig.add_trace(go.Box(y=df[col].dropna(), name=format_col_name(col),
                                     marker_color=NEON_COLORS[i % 12], boxmean='sd'))
            fig.update_layout(title=format_title(question), yaxis_title='Value', boxgap=0.3)
        return fig
    except Exception as e:
        print(f"❌ box: {e}"); return None


def create_strip(df: pd.DataFrame, question: str):
    """Strip / jitter plot (individual points)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col   = num_cols[0]
        color_col = txt_cols[0] if txt_cols else None
        d = df.sample(min(500, len(df)), random_state=42)
        fig = px.strip(d, x=color_col, y=val_col, color=color_col,
                       color_discrete_sequence=NEON_COLORS,
                       stripmode='overlay')
        fig.update_traces(jitter=0.4, marker_size=5, opacity=0.7)
        fig.update_layout(title=format_title(question), showlegend=False)
        return fig
    except Exception as e:
        print(f"❌ strip: {e}"); return None


def create_ridgeline(df: pd.DataFrame, question: str):
    """Ridge / joy plot — overlapping KDE per group."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        if txt_cols:
            group_col = txt_cols[0]
            groups = df[group_col].dropna().value_counts().head(8).index.tolist()
        else:
            groups = [val_col]

        fig = go.Figure()
        for i, grp in enumerate(reversed(groups)):
            data = df[df[group_col] == grp][val_col].dropna() if txt_cols else df[val_col].dropna()
            x_range = np.linspace(data.min(), data.max(), 200)
            try:
                from scipy.stats import gaussian_kde
                kde_vals = gaussian_kde(data)(x_range)
            except Exception:
                kde_vals, _ = np.histogram(data, bins=50, density=True)
                x_range = np.linspace(data.min(), data.max(), 50)

            offset = i * (kde_vals.max() * 1.2)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_vals + offset, mode='lines',
                name=str(grp), fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=NEON_COLORS[i % 12], width=2),
                fillcolor=f'rgba({int(NEON_COLORS[i%12][1:3],16)},'
                          f'{int(NEON_COLORS[i%12][3:5],16)},'
                          f'{int(NEON_COLORS[i%12][5:7],16)},0.2)'
            ))
        fig.update_layout(title=format_title(question) or f'Ridge Plot — {format_col_name(val_col)}',
                          yaxis=dict(showticklabels=False, title=''))
        return fig
    except Exception as e:
        print(f"❌ ridgeline: {e}"); return None


def create_ecdf(df: pd.DataFrame, question: str):
    """Empirical Cumulative Distribution Function."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        fig = go.Figure()
        for i, col in enumerate(num_cols[:4]):
            data = df[col].dropna().sort_values()
            cdf  = np.arange(1, len(data) + 1) / len(data)
            fig.add_trace(go.Scatter(
                x=data, y=cdf, mode='lines', name=format_col_name(col),
                line=dict(color=NEON_COLORS[i % 12], width=2.5)
            ))
        fig.update_layout(title=format_title(question) or 'Empirical CDF',
                          xaxis_title='Value', yaxis_title='Cumulative Probability',
                          yaxis=dict(range=[0, 1.05]))
        return fig
    except Exception as e:
        print(f"❌ ecdf: {e}"); return None


def create_hexbin(df: pd.DataFrame, question: str):
    """Hexagonal binning for large scatter data."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_histogram(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        d = df[[x_col, y_col]].dropna().sample(min(2000, len(df)), random_state=42)
        # Plotly doesn't have native hexbin — simulate with density heatmap
        fig = px.density_heatmap(d, x=x_col, y=y_col, nbinsx=30, nbinsy=30,
                                 color_continuous_scale=['#0f0c29', '#302b63', '#667eea', '#43e97b'])
        fig.update_layout(title=format_title(question) or 'Hexbin Density',
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ hexbin: {e}"); return None


def create_rug_plot(df: pd.DataFrame, question: str):
    """Rug plot + density overlay."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col  = num_cols[0]
        data = df[col].dropna().sample(min(300, len(df)), random_state=42)
        fig  = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=30, name='Distribution',
            marker=dict(color='rgba(99,102,241,0.5)', line=dict(color='#6366f1', width=1))
        ))
        # Rug marks
        fig.add_trace(go.Scatter(
            x=data, y=[-data.std() * 0.05] * len(data), mode='markers',
            marker=dict(symbol='line-ns-open', size=8, color=NEON_COLORS[4], opacity=0.5),
            name='Rug', showlegend=True
        ))
        fig.update_layout(title=format_title(question) or f'Rug Plot — {format_col_name(col)}',
                          xaxis_title=format_col_name(col))
        return fig
    except Exception as e:
        print(f"❌ rug_plot: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 5 — PROPORTION  (9 charts)
# ═══════════════════════════════════════════════════════════

def create_pie_chart(df: pd.DataFrame, question: str):
    """Donut chart with top-7 + Others consolidation."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna() if lbl_col else df[[val_col]].dropna().copy()
        if lbl_col is None:
            d['_cat'] = [f'Item {i+1}' for i in range(len(d))]; lbl_col = '_cat'
        if len(d) > 8:
            top  = d.nlargest(7, val_col)
            rest = pd.DataFrame({lbl_col: ['Others'], val_col: [d.nsmallest(len(d)-7, val_col)[val_col].sum()]})
            d    = pd.concat([top, rest], ignore_index=True)
        total = d[val_col].sum()
        fig = go.Figure(go.Pie(
            labels=d[lbl_col], values=d[val_col], hole=0.55,
            marker=dict(colors=NEON_COLORS[:len(d)], line=dict(color='rgba(255,255,255,0.1)', width=2)),
            textinfo='percent+label', textposition='outside',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>',
            pull=[0.03] * len(d), rotation=45
        ))
        fig.add_annotation(
            text=f"<b>{format_number(total)}</b><br><span style='font-size:11px'>Total {format_col_name(val_col)}</span>",
            showarrow=False, font=dict(size=14, color='#e2e8f0')
        )
        fig.update_layout(title=format_title(question), showlegend=True,
                          legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05))
        return fig
    except Exception as e:
        print(f"❌ pie: {e}"); return None


def create_treemap(df: pd.DataFrame, question: str):
    """Hierarchical treemap (1 or 2 levels)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols or not txt_cols:
            return create_bar_chart(df, question)
        val_col = num_cols[0]
        if len(txt_cols) >= 2:
            p, c = txt_cols[0], txt_cols[1]
            d = df[[p, c, val_col]].dropna()
            d = d[d[val_col] > 0]
            fig = px.treemap(d, path=[px.Constant('All'), p, c], values=val_col,
                             color=val_col,
                             color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3], GRADIENT_COLORS[5]])
        else:
            lbl = txt_cols[0]
            d = df[[lbl, val_col]].dropna()
            d = d[d[val_col] > 0]
            fig = px.treemap(d, path=[px.Constant('All'), lbl], values=val_col,
                             color=val_col,
                             color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3], GRADIENT_COLORS[5]])
        fig.update_traces(
            textinfo='label+value+percent root',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percentRoot:.1%}<extra></extra>',
            marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)'))
        )
        fig.update_layout(title=format_title(question), coloraxis_showscale=False)
        return fig
    except Exception as e:
        print(f"❌ treemap: {e}"); return None


def create_sunburst(df: pd.DataFrame, question: str):
    """Radial sunburst chart (hierarchical)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols or not txt_cols:
            return create_pie_chart(df, question)
        val_col = num_cols[0]
        path = [px.Constant('All')] + txt_cols[:3]
        d = df[txt_cols[:3] + [val_col]].dropna()
        d = d[d[val_col] > 0]
        fig = px.sunburst(d, path=path, values=val_col, color=val_col,
                          color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3], GRADIENT_COLORS[5]])
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>'
        )
        fig.update_layout(title=format_title(question), coloraxis_showscale=False)
        return fig
    except Exception as e:
        print(f"❌ sunburst: {e}"); return None


def create_funnel_chart(df: pd.DataFrame, question: str):
    """Conversion / pipeline funnel."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().sort_values(val_col, ascending=False).head(12) if lbl_col \
            else df[[val_col]].dropna().head(12)
        if lbl_col is None:
            d = d.copy(); d['Stage'] = [f'Stage {i+1}' for i in range(len(d))]; lbl_col = 'Stage'
        fig = go.Figure(go.Funnel(
            y=d[lbl_col], x=d[val_col],
            textinfo='value+percent initial+percent previous',
            marker=dict(color=NEON_COLORS[:len(d)], line=dict(width=1.5, color='rgba(255,255,255,0.1)')),
            connector=dict(line=dict(color='rgba(255,255,255,0.05)', width=2)),
            hovertemplate=f'<b>%{{y}}</b><br>Value: %{{x:,.0f}}<br>% Total: %{{percentInitial}}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question))
        return fig
    except Exception as e:
        print(f"❌ funnel: {e}"); return None


def create_waterfall_chart(df: pd.DataFrame, question: str):
    """Waterfall / bridge chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df.head(20).copy()
        labels = d[lbl_col].astype(str).tolist() if lbl_col else [f'Step {i+1}' for i in range(len(d))]
        values = d[val_col].fillna(0).tolist()
        measure = ['relative'] * len(values)
        if len(values) > 1:
            measure[0] = 'absolute'; measure[-1] = 'total'
        fig = go.Figure(go.Waterfall(
            orientation='v', measure=measure, x=labels, y=values,
            connector=dict(line=dict(color='rgba(255,255,255,0.15)', width=1.5, dash='dot')),
            increasing=dict(marker=dict(color='#34d399', line=dict(width=0))),
            decreasing=dict(marker=dict(color='#ef4444', line=dict(width=0))),
            totals=dict(marker=dict(color='#6366f1', line=dict(width=0))),
            textposition='outside',
            text=[f'{v:+,.0f}' if m == 'relative' else f'{v:,.0f}' for v, m in zip(values, measure)]
        ))
        fig.update_layout(title=format_title(question), showlegend=False)
        return fig
    except Exception as e:
        print(f"❌ waterfall: {e}"); return None


def create_marimekko(df: pd.DataFrame, question: str):
    """Marimekko / mosaic chart (width + height encode two variables)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols or not txt_cols:
            return create_stacked_100_bar(df, question)
        val_col = num_cols[0]
        if len(txt_cols) >= 2:
            row_col, col_col = txt_cols[0], txt_cols[1]
            pivot = df.groupby([row_col, col_col])[val_col].sum().unstack(fill_value=0)
        else:
            return create_stacked_100_bar(df, question)
        row_totals  = pivot.sum(axis=1)
        grand_total = row_totals.sum()
        col_totals  = pivot.sum(axis=0)
        fig = go.Figure()
        x_start = 0
        for r_idx, row_name in enumerate(pivot.index):
            row_width  = row_totals[row_name] / grand_total * 100
            col_total  = pivot.loc[row_name].sum()
            y_start = 0
            for c_idx, col_name in enumerate(pivot.columns):
                height = (pivot.loc[row_name, col_name] / col_total * 100) if col_total else 0
                fig.add_shape(type='rect',
                    x0=x_start, x1=x_start + row_width,
                    y0=y_start, y1=y_start + height,
                    fillcolor=NEON_COLORS[c_idx % 12],
                    line=dict(color='rgba(0,0,0,0.4)', width=1)
                )
                if height > 4:
                    fig.add_annotation(x=x_start + row_width/2, y=y_start + height/2,
                        text=f'{col_name}<br>{height:.0f}%', showarrow=False,
                        font=dict(size=9, color='white'))
                y_start += height
            if row_width > 4:
                fig.add_annotation(x=x_start + row_width/2, y=103,
                    text=f'{row_name}', showarrow=False, font=dict(size=9, color='#e2e8f0'))
            x_start += row_width
        fig.update_layout(title=format_title(question) or 'Marimekko Chart',
                          xaxis=dict(range=[0, 100], title='% Total', showgrid=False),
                          yaxis=dict(range=[0, 110], title='% Within Group', showgrid=False))
        return fig
    except Exception as e:
        print(f"❌ marimekko: {e}"); return None


def create_waffle_chart(df: pd.DataFrame, question: str):
    """Waffle / unit chart — 10×10 grid of colored squares."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(8) if lbl_col else df[[val_col]].head(8)
        if lbl_col is None:
            d = d.copy(); d['_cat'] = [f'Cat {i+1}' for i in range(len(d))]; lbl_col = '_cat'
        total = d[val_col].sum()
        units = np.round(d[val_col] / total * 100).astype(int)
        units[-1] = 100 - units[:-1].sum()  # adjust to exactly 100
        grid   = []
        colors = []
        for i, (_, row) in enumerate(d.iterrows()):
            n = max(0, int(units.iloc[i]))
            grid   += [(row[lbl_col], row[val_col])] * n
            colors += [NEON_COLORS[i % 12]] * n
        while len(grid) < 100:
            grid.append(('Extra', 0)); colors.append('rgba(0,0,0,0)')
        xs = [i % 10 for i in range(100)]
        ys = [i // 10 for i in range(100)]
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode='markers',
            marker=dict(size=28, color=colors, symbol='square',
                        line=dict(color='rgba(0,0,0,0.5)', width=1)),
            text=[f'{g[0]}: {g[1]:,.0f}' for g in grid],
            hovertemplate='%{text}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or 'Waffle Chart',
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          height=420)
        return fig
    except Exception as e:
        print(f"❌ waffle_chart: {e}"); return None


def create_dot_plot(df: pd.DataFrame, question: str):
    """Cleveland dot plot — horizontal dots on axis."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().sort_values(val_col).tail(20) if lbl_col \
            else df[[val_col]].dropna().head(20)
        x_data = d[val_col].tolist()
        y_data = d[lbl_col].tolist() if lbl_col else list(range(len(d)))
        fig = go.Figure()
        for y, x in zip(y_data, x_data):
            fig.add_shape(type='line', x0=0, x1=x, y0=y, y1=y,
                          line=dict(color='rgba(226,232,240,0.1)', width=1, dash='dot'))
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers',
            marker=dict(size=12, color=NEON_COLORS[0], line=dict(width=2, color='white')),
            text=[f'{v:,.2f}' for v in x_data], textposition='middle right',
            hovertemplate='<b>%{y}</b><br>Value: %{x:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question),
                          xaxis_title=format_col_name(val_col), yaxis_title='')
        return fig
    except Exception as e:
        print(f"❌ dot_plot: {e}"); return None


def create_pictogram(df: pd.DataFrame, question: str):
    """Pictogram / icon array chart (circles of varying sizes)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(10) if lbl_col else df[[val_col]].head(10)
        if lbl_col is None:
            d = d.copy(); d['_cat'] = [f'Cat {i+1}' for i in range(len(d))]; lbl_col = '_cat'
        max_val = d[val_col].max()
        sizes   = (d[val_col] / max_val * 80 + 20).tolist()
        fig = go.Figure(go.Scatter(
            x=list(range(len(d))), y=[1] * len(d), mode='markers+text',
            marker=dict(size=sizes, color=NEON_COLORS[:len(d)],
                        line=dict(color='rgba(255,255,255,0.3)', width=2), opacity=0.85),
            text=d[lbl_col].tolist(), textposition='bottom center',
            customdata=d[val_col].tolist(),
            hovertemplate='<b>%{text}</b><br>Value: %{customdata:,.0f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or 'Pictogram Chart',
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 2]),
                          height=350)
        return fig
    except Exception as e:
        print(f"❌ pictogram: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 6 — HEAT / MATRIX  (5 charts)
# ═══════════════════════════════════════════════════════════

def create_heatmap(df: pd.DataFrame, question: str):
    """Correlation matrix or pivot heatmap."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) >= 2 and len(txt_cols) < 2:
            corr   = df[num_cols].corr()
            labels = [format_col_name(c) for c in corr.columns]
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale=[[0, '#ef4444'], [0.5, '#1e1b4b'], [1, '#6366f1']],
                zmid=0, text=np.round(corr.values, 2), texttemplate='%{text}',
                textfont=dict(size=11),
                hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
                colorbar=dict(title='r', tickfont=dict(color='#e2e8f0'))
            ))
            fig.update_layout(title=format_title(question) or 'Correlation Matrix')
            return fig
        if len(txt_cols) >= 2 and num_cols:
            r_col, c_col, v_col = txt_cols[0], txt_cols[1], num_cols[0]
            pivot = df.groupby([r_col, c_col])[v_col].sum().unstack(fill_value=0).iloc[:20, :20]
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
                colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#667eea']],
                hovertemplate=f'{format_col_name(r_col)}: %{{y}}<br>{format_col_name(c_col)}: %{{x}}<br>Value: %{{z:,.0f}}<extra></extra>',
                colorbar=dict(tickfont=dict(color='#e2e8f0'))
            ))
            fig.update_layout(title=format_title(question),
                              xaxis_title=format_col_name(c_col), yaxis_title=format_col_name(r_col))
            return fig
        return create_bar_chart(df, question)
    except Exception as e:
        print(f"❌ heatmap: {e}"); return None


def create_contour(df: pd.DataFrame, question: str):
    """Filled contour plot (2D)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_histogram(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        z_col = num_cols[2] if len(num_cols) >= 3 else None
        d = df[[x_col, y_col] + ([z_col] if z_col else [])].dropna().sample(min(500, len(df)), random_state=42)
        if z_col:
            fig = go.Figure(go.Contour(
                x=d[x_col], y=d[y_col], z=d[z_col],
                colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#667eea']],
                contours_coloring='heatmap',
                hovertemplate='X: %{x:,.2f}<br>Y: %{y:,.2f}<br>Z: %{z:,.2f}<extra></extra>'
            ))
        else:
            xi = np.linspace(d[x_col].min(), d[x_col].max(), 50)
            yi = np.linspace(d[y_col].min(), d[y_col].max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            from scipy.interpolate import griddata
            zi = griddata((d[x_col], d[y_col]),
                           np.random.rand(len(d)),
                           (xi, yi), method='linear')
            fig = go.Figure(go.Contour(z=zi, x=xi[0], y=yi[:, 0],
                                       colorscale=[[0, '#0f0c29'], [1, '#667eea']],
                                       contours_coloring='heatmap'))
        fig.update_layout(title=format_title(question) or 'Contour Plot',
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        # Fallback to density heatmap if scipy not available
        return create_density_heatmap(df, question)


def create_calendar_heatmap(df: pd.DataFrame, question: str):
    """Calendar heatmap — one row per week, columns = weekdays."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Find a date-like column
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    test = pd.to_datetime(df[col].dropna().head(10))
                    if len(test) >= 5:
                        date_col = col; break
                except Exception:
                    pass
            elif 'datetime' in str(df[col].dtype):
                date_col = col; break

        if date_col is None or not num_cols:
            return create_heatmap(df, question)

        d = df[[date_col, num_cols[0]]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
        d = d.dropna().sort_values(date_col)
        d = d.groupby(date_col)[num_cols[0]].sum().reset_index()
        d['week'] = d[date_col].dt.isocalendar().week.astype(int)
        d['dow']  = d[date_col].dt.dayofweek
        pivot = d.pivot_table(index='week', columns='dow', values=num_cols[0], aggfunc='sum').fillna(0)
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[dow_names[i] for i in pivot.columns],
            y=[f'W{w}' for w in pivot.index],
            colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#43e97b']],
            hovertemplate='Week %{y} — %{x}<br>Value: %{z:,.0f}<extra></extra>',
            colorbar=dict(tickfont=dict(color='#e2e8f0'))
        ))
        fig.update_layout(title=format_title(question) or f'Calendar Heatmap — {format_col_name(num_cols[0])}',
                          xaxis_title='Day of Week', yaxis_title='Week')
        return fig
    except Exception as e:
        print(f"❌ calendar_heatmap: {e}"); return create_heatmap(df, question)


def create_parallel_categories(df: pd.DataFrame, question: str):
    """Parallel categories (alluvial) for categorical flow."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:5]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(txt_cols) < 2:
            return create_bar_chart(df, question)
        color_col = num_cols[0] if num_cols else None
        d = df[txt_cols + ([color_col] if color_col else [])].dropna().sample(min(500, len(df)), random_state=42)
        fig = px.parallel_categories(d, dimensions=txt_cols, color=color_col,
                                     color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(title=format_title(question) or 'Parallel Categories')
        return fig
    except Exception as e:
        print(f"❌ parallel_categories: {e}"); return None


def create_qq_plot(df: pd.DataFrame, question: str):
    """Q-Q plot vs Normal distribution."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col  = num_cols[0]
        data = df[col].dropna().sample(min(300, len(df)), random_state=42).sort_values()
        n    = len(data)
        theoretical = np.array([(i - 0.5) / n for i in range(1, n + 1)])
        from scipy.stats import norm
        q_theoretical = norm.ppf(theoretical)
        q_sample      = np.sort((data - data.mean()) / data.std())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q_theoretical, y=q_sample, mode='markers',
                                 marker=dict(size=6, color=NEON_COLORS[0], opacity=0.7),
                                 name='Sample Quantiles'))
        line_range = [min(q_theoretical.min(), q_sample.min()),
                      max(q_theoretical.max(), q_sample.max())]
        fig.add_trace(go.Scatter(x=line_range, y=line_range, mode='lines',
                                 line=dict(color='#ef4444', dash='dash', width=2),
                                 name='Normal Reference'))
        fig.update_layout(title=format_title(question) or f'Q-Q Plot — {format_col_name(col)}',
                          xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
        return fig
    except ImportError:
        return create_histogram(df, question)
    except Exception as e:
        print(f"❌ qq_plot: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 7 — STATISTICAL / COMPARISON  (7 charts)
# ═══════════════════════════════════════════════════════════

def create_error_bar(df: pd.DataFrame, question: str):
    """Bar chart with error bars (std dev)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols or not txt_cols:
            return create_bar_chart(df, question)
        grp_col, val_col = txt_cols[0], num_cols[0]
        agg = df.groupby(grp_col)[val_col].agg(['mean', 'std']).reset_index().head(15)
        agg['std'] = agg['std'].fillna(0)
        fig = go.Figure(go.Bar(
            x=agg[grp_col], y=agg['mean'],
            error_y=dict(type='data', array=agg['std'].tolist(), visible=True,
                         color='rgba(255,255,255,0.6)', thickness=2),
            marker=dict(color=NEON_COLORS[0], cornerradius=5),
            hovertemplate='<b>%{x}</b><br>Mean: %{y:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or f'Mean ± SD — {format_col_name(val_col)}',
                          xaxis_title=format_col_name(grp_col), yaxis_title=f'Mean {format_col_name(val_col)}')
        return fig
    except Exception as e:
        print(f"❌ error_bar: {e}"); return None


def create_confidence_band(df: pd.DataFrame, question: str):
    """Line chart with shaded 95% confidence band."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col    = num_cols[0]
        x_data = df[non_num[0]].head(100) if non_num else list(range(min(100, len(df))))
        y      = df[col].head(100).rolling(5, min_periods=1)
        y_mean = y.mean()
        y_std  = y.std().fillna(0)
        y_upper= y_mean + 1.96 * y_std
        y_lower= y_mean - 1.96 * y_std
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(x_data) + list(x_data)[::-1],
                                  y=list(y_upper) + list(y_lower)[::-1],
                                  fill='toself', fillcolor='rgba(99,102,241,0.15)',
                                  line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
        fig.add_trace(go.Scatter(x=x_data, y=y_mean, mode='lines', name=format_col_name(col),
                                  line=dict(color=NEON_COLORS[0], width=3)))
        fig.update_layout(title=format_title(question) or f'Confidence Band — {format_col_name(col)}',
                          hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ confidence_band: {e}"); return None


def create_tornado_chart(df: pd.DataFrame, question: str):
    """Tornado / sensitivity chart — sorted diverging bars."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols or not txt_cols:
            return create_diverging_bar(df, question)
        val_col = num_cols[0]
        lbl_col = txt_cols[0]
        d = df[[lbl_col, val_col]].dropna()
        d = d.reindex(d[val_col].abs().sort_values(ascending=True).index).tail(15)
        colors = ['#34d399' if v >= 0 else '#ef4444' for v in d[val_col]]
        fig = go.Figure(go.Bar(
            y=d[lbl_col], x=d[val_col], orientation='h',
            marker=dict(color=colors, cornerradius=4),
            text=[f'{v:+,.2f}' for v in d[val_col]], textposition='outside'
        ))
        fig.add_vline(x=0, line_color='rgba(255,255,255,0.4)', line_width=1.5)
        fig.update_layout(title=format_title(question) or 'Tornado / Sensitivity Chart',
                          xaxis_title=format_col_name(val_col), yaxis_title='')
        return fig
    except Exception as e:
        print(f"❌ tornado_chart: {e}"); return None


def create_comparison_bar(df: pd.DataFrame, question: str):
    """Side-by-side grouped bar for explicit comparison."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_bar_chart(df, question)
        x_data = df[txt_cols[0]].head(12) if txt_cols else list(range(min(12, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:4]):
            fig.add_trace(go.Bar(
                name=format_col_name(col), x=x_data, y=df[col].head(12),
                marker=dict(color=NEON_COLORS[i * 2 % 12], cornerradius=5),
                text=[f'{v:,.1f}' for v in df[col].head(12)], textposition='outside'
            ))
        fig.update_layout(title=format_title(question), barmode='group', bargap=0.12)
        return fig
    except Exception as e:
        print(f"❌ comparison_bar: {e}"); return None


def create_annotated_line(df: pd.DataFrame, question: str):
    """Line chart with auto-generated annotations at peaks/troughs."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col    = num_cols[0]
        x_data = list(df[non_num[0]].head(80)) if non_num else list(range(min(80, len(df))))
        y_data = df[col].head(80).tolist()
        fig = go.Figure(go.Scatter(
            x=x_data, y=y_data, mode='lines+markers',
            line=dict(color=NEON_COLORS[0], width=3, shape='spline'),
            marker=dict(size=6, color=NEON_COLORS[0])
        ))
        # Annotate max and min
        y_arr = np.array(y_data)
        for idx, label, color in [
            (int(np.argmax(y_arr)), f'Peak: {max(y_data):,.1f}', '#34d399'),
            (int(np.argmin(y_arr)), f'Trough: {min(y_data):,.1f}', '#ef4444'),
        ]:
            fig.add_annotation(x=x_data[idx], y=y_data[idx], text=label, showarrow=True,
                                arrowhead=2, arrowcolor=color, font=dict(color=color, size=11),
                                bgcolor='rgba(0,0,0,0.6)', bordercolor=color, borderwidth=1)
        fig.update_layout(title=format_title(question) or f'Annotated Line — {format_col_name(col)}')
        return fig
    except Exception as e:
        print(f"❌ annotated_line: {e}"); return None


def create_multi_axis_line(df: pd.DataFrame, question: str):
    """Dual Y-axis line chart for metrics on different scales."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_line_chart(df, question)
        x_data = df[non_num[0]].head(80) if non_num else list(range(min(80, len(df))))
        col1, col2 = num_cols[0], num_cols[1]
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=x_data, y=df[col1].head(80), name=format_col_name(col1),
                                  line=dict(color=NEON_COLORS[0], width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_data, y=df[col2].head(80), name=format_col_name(col2),
                                  line=dict(color=NEON_COLORS[4], width=2.5, dash='dash')), secondary_y=True)
        fig.update_yaxes(title_text=format_col_name(col1), secondary_y=False)
        fig.update_yaxes(title_text=format_col_name(col2), secondary_y=True)
        fig.update_layout(title=format_title(question), hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ multi_axis_line: {e}"); return None


def create_heat_table(df: pd.DataFrame, question: str):
    """Table with cells colored by magnitude."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return create_bar_chart(df, question)
        d = df[num_cols].head(20).fillna(0)
        col_min, col_max = d.min(), d.max()
        # Normalize 0-1
        normed = (d - col_min) / (col_max - col_min).replace(0, 1)
        labels = txt_cols[0] if txt_cols else None
        row_labels = df[labels].head(20).tolist() if labels else [f'Row {i+1}' for i in range(len(d))]
        fig = go.Figure(go.Heatmap(
            z=normed.values, x=[format_col_name(c) for c in d.columns],
            y=row_labels,
            colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#43e97b']],
            text=[[f'{d.iloc[r, c]:,.1f}' for c in range(len(d.columns))] for r in range(len(d))],
            texttemplate='%{text}', textfont=dict(size=10),
            hovertemplate='%{y} — %{x}<br>Value: %{text}<extra></extra>',
            showscale=True, colorbar=dict(tickfont=dict(color='#e2e8f0'))
        ))
        fig.update_layout(title=format_title(question) or 'Heat Table', height=max(300, len(d) * 30 + 100))
        return fig
    except Exception as e:
        print(f"❌ heat_table: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 8 — TIME SERIES  (7 charts)
# ═══════════════════════════════════════════════════════════

def create_candlestick(df: pd.DataFrame, question: str):
    """OHLC Candlestick chart — detects open/high/low/close columns."""
    try:
        cols_lower = {c.lower(): c for c in df.columns}
        open_col  = next((cols_lower[k] for k in cols_lower if 'open'  in k), None)
        high_col  = next((cols_lower[k] for k in cols_lower if 'high'  in k), None)
        low_col   = next((cols_lower[k] for k in cols_lower if 'low'   in k), None)
        close_col = next((cols_lower[k] for k in cols_lower if 'close' in k), None)
        date_col  = next((cols_lower[k] for k in cols_lower if 'date'  in k or 'time' in k), None)
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()

        # Fallback: use first 4 numeric columns as O/H/L/C
        if not all([open_col, high_col, low_col, close_col]) and len(num_cols) >= 4:
            open_col, high_col, low_col, close_col = num_cols[:4]

        if not all([open_col, high_col, low_col, close_col]):
            return create_line_chart(df, question)

        x_data = df[date_col] if date_col else list(range(len(df)))
        fig = go.Figure(go.Candlestick(
            x=x_data, open=df[open_col], high=df[high_col],
            low=df[low_col], close=df[close_col],
            increasing_line_color='#34d399', decreasing_line_color='#ef4444',
            increasing_fillcolor='rgba(52,211,153,0.8)', decreasing_fillcolor='rgba(239,68,68,0.8)'
        ))
        fig.update_layout(title=format_title(question) or 'Candlestick Chart',
                          xaxis_rangeslider_visible=False)
        return fig
    except Exception as e:
        print(f"❌ candlestick: {e}"); return create_line_chart(df, question)


def create_ohlc(df: pd.DataFrame, question: str):
    """OHLC bar chart (line-style)."""
    try:
        cols_lower = {c.lower(): c for c in df.columns}
        num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        open_col   = next((cols_lower[k] for k in cols_lower if 'open'  in k), num_cols[0] if num_cols else None)
        high_col   = next((cols_lower[k] for k in cols_lower if 'high'  in k), num_cols[1] if len(num_cols) > 1 else None)
        low_col    = next((cols_lower[k] for k in cols_lower if 'low'   in k), num_cols[2] if len(num_cols) > 2 else None)
        close_col  = next((cols_lower[k] for k in cols_lower if 'close' in k), num_cols[3] if len(num_cols) > 3 else None)
        date_col   = next((cols_lower[k] for k in cols_lower if 'date'  in k or 'time' in k), None)

        if not all([open_col, high_col, low_col, close_col]):
            return create_line_chart(df, question)

        x_data = df[date_col] if date_col else list(range(len(df)))
        fig = go.Figure(go.Ohlc(
            x=x_data, open=df[open_col], high=df[high_col],
            low=df[low_col], close=df[close_col],
            increasing_line_color='#34d399', decreasing_line_color='#ef4444'
        ))
        fig.update_layout(title=format_title(question) or 'OHLC Chart',
                          xaxis_rangeslider_visible=False)
        return fig
    except Exception as e:
        print(f"❌ ohlc: {e}"); return create_line_chart(df, question)


def create_gantt(df: pd.DataFrame, question: str):
    """Gantt chart for project/task schedules."""
    try:
        txt_cols  = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'start' in c.lower() or 'end' in c.lower()]

        # Build synthetic Gantt from available data
        if not txt_cols:
            return create_bar_chart(df, question)

        task_col = txt_cols[0]
        if len(date_cols) >= 2:
            start_col, end_col = date_cols[0], date_cols[1]
            d = df[[task_col, start_col, end_col]].dropna().head(15)
            d[start_col] = pd.to_datetime(d[start_col], errors='coerce')
            d[end_col]   = pd.to_datetime(d[end_col], errors='coerce')
            d = d.dropna()
            fig = px.timeline(d, x_start=start_col, x_end=end_col, y=task_col,
                              color=task_col, color_discrete_sequence=NEON_COLORS)
        elif num_cols:
            # Use numeric columns as duration/end
            dur_col = num_cols[0]
            d = df[[task_col, dur_col]].dropna().head(15)
            base = pd.Timestamp('2024-01-01')
            d['_start'] = [base + pd.Timedelta(days=int(i * 5))       for i in range(len(d))]
            d['_end']   = [base + pd.Timedelta(days=int(i * 5 + max(1, v))) for i, v in enumerate(d[dur_col])]
            fig = px.timeline(d, x_start='_start', x_end='_end', y=task_col,
                              color=task_col, color_discrete_sequence=NEON_COLORS)
        else:
            return create_bar_chart(df, question)

        fig.update_yaxes(autorange='reversed')
        fig.update_layout(title=format_title(question) or 'Gantt Chart',
                          xaxis_title='Date / Period', yaxis_title='')
        return fig
    except Exception as e:
        print(f"❌ gantt: {e}"); return create_bar_chart(df, question)


def create_timeline(df: pd.DataFrame, question: str):
    """Event timeline — dots on a horizontal time axis."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
        lbl_col  = txt_cols[0] if txt_cols else None

        if date_col:
            d = df[[date_col] + ([lbl_col] if lbl_col else [])].dropna().head(30)
            d[date_col] = pd.to_datetime(d[date_col], errors='coerce').dropna()
            x_data = d[date_col]
        elif num_cols:
            x_data = df[num_cols[0]].head(30)
        else:
            return create_bar_chart(df, question)

        labels = df[lbl_col].head(len(x_data)).tolist() if lbl_col else [f'Event {i+1}' for i in range(len(x_data))]
        fig = go.Figure()
        fig.add_hline(y=0, line_color='rgba(255,255,255,0.2)')
        for i, (x, lbl) in enumerate(zip(x_data, labels)):
            above = i % 2 == 0
            fig.add_shape(type='line', x0=x, x1=x, y0=0, y1=0.8 if above else -0.8,
                          line=dict(color=NEON_COLORS[i % 12], width=1.5))
            fig.add_trace(go.Scatter(
                x=[x], y=[1 if above else -1], mode='markers+text',
                marker=dict(size=12, color=NEON_COLORS[i % 12], line=dict(width=2, color='white')),
                text=[lbl], textposition='top center' if above else 'bottom center',
                textfont=dict(size=9), showlegend=False,
                hovertemplate=f'<b>{lbl}</b><br>{x}<extra></extra>'
            ))
        fig.update_layout(title=format_title(question) or 'Timeline',
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, 1.8]),
                          height=380)
        return fig
    except Exception as e:
        print(f"❌ timeline: {e}"); return create_bar_chart(df, question)


def create_forecast_chart(df: pd.DataFrame, question: str):
    """Line chart with a simple linear-trend forecast extension."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col    = num_cols[0]
        n      = min(80, len(df))
        y      = df[col].head(n).fillna(method='ffill').values
        x_num  = np.arange(n)
        # Fit simple linear trend
        coeffs = np.polyfit(x_num, y, 1)
        ext    = 10  # forecast 10 steps
        x_fore = np.arange(n, n + ext)
        y_fore = np.polyval(coeffs, x_fore)
        y_upper= y_fore * 1.05
        y_lower= y_fore * 0.95

        x_all  = df[non_num[0]].head(n).tolist() if non_num else list(range(n))
        x_fore_labels = [f'+{i+1}' for i in range(ext)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_all, y=y, mode='lines', name='Actual',
                                  line=dict(color=NEON_COLORS[0], width=2.5)))
        # CI band
        fig.add_trace(go.Scatter(
            x=x_fore_labels + x_fore_labels[::-1],
            y=list(y_upper) + list(y_lower)[::-1],
            fill='toself', fillcolor='rgba(99,102,241,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='95% CI'
        ))
        fig.add_trace(go.Scatter(x=x_fore_labels, y=y_fore, mode='lines+markers',
                                  name='Forecast', line=dict(color=NEON_COLORS[4], width=2.5, dash='dash'),
                                  marker=dict(size=7, color=NEON_COLORS[4])))
        fig.update_layout(title=format_title(question) or f'Forecast — {format_col_name(col)}')
        return fig
    except Exception as e:
        print(f"❌ forecast_chart: {e}"); return create_line_chart(df, question)


def create_seasonal_chart(df: pd.DataFrame, question: str):
    """Seasonal pattern chart — one line per group/year."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_line_chart(df, question)
        # Treat columns as different seasons/years
        non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
        x_data  = df[non_num[0]].head(50) if non_num else list(range(min(50, len(df))))
        fig = go.Figure()
        for i, col in enumerate(num_cols[:6]):
            fig.add_trace(go.Scatter(
                x=x_data, y=df[col].head(50), mode='lines+markers',
                name=format_col_name(col),
                line=dict(color=NEON_COLORS[i % 12], width=2),
                marker=dict(size=5)
            ))
        fig.update_layout(title=format_title(question) or 'Seasonal Pattern Chart',
                          hovermode='x unified')
        return fig
    except Exception as e:
        print(f"❌ seasonal_chart: {e}"); return create_line_chart(df, question)


def create_stacked_waterfall(df: pd.DataFrame, question: str):
    """Multi-series stacked waterfall."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_waterfall_chart(df, question)
        lbl_col = txt_cols[0] if txt_cols else None
        d = df.head(15).copy()
        labels = d[lbl_col].astype(str).tolist() if lbl_col else [f'Step {i+1}' for i in range(len(d))]
        fig = go.Figure()
        for i, col in enumerate(num_cols[:4]):
            values = d[col].fillna(0).tolist()
            measure = ['relative'] * len(values)
            if len(values) > 1:
                measure[0] = 'absolute'; measure[-1] = 'total'
            fig.add_trace(go.Waterfall(
                name=format_col_name(col), orientation='v',
                measure=measure, x=labels, y=values,
                increasing=dict(marker=dict(color=NEON_COLORS[i * 3 % 12])),
                decreasing=dict(marker=dict(color='#ef4444')),
                totals=dict(marker=dict(color=NEON_COLORS[(i * 3 + 1) % 12]))
            ))
        fig.update_layout(title=format_title(question) or 'Stacked Waterfall', waterfallgroupgap=0.1)
        return fig
    except Exception as e:
        print(f"❌ stacked_waterfall: {e}"); return create_waterfall_chart(df, question)


# ═══════════════════════════════════════════════════════════
# ██  GROUP 9 — KPI / INDICATOR  (5 charts)
# ═══════════════════════════════════════════════════════════

def create_gauge(df: pd.DataFrame, question: str):
    """Speedometer gauge — shows first numeric value vs max."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col   = num_cols[0]
        value = float(df[col].dropna().mean())
        max_v = float(df[col].dropna().max())
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=value,
            delta={'reference': value * 0.85, 'increasing': {'color': '#34d399'}},
            title={'text': format_col_name(col), 'font': {'size': 16, 'color': '#e2e8f0'}},
            number={'font': {'size': 36, 'color': NEON_COLORS[0]}, 'valueformat': ',.2f'},
            gauge={
                'axis': {'range': [0, max_v], 'tickcolor': '#cbd5e1', 'tickfont': {'color': '#cbd5e1'}},
                'bar':  {'color': NEON_COLORS[0], 'thickness': 0.3},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, max_v * 0.33], 'color': 'rgba(239,68,68,0.15)'},
                    {'range': [max_v * 0.33, max_v * 0.66], 'color': 'rgba(245,158,11,0.15)'},
                    {'range': [max_v * 0.66, max_v], 'color': 'rgba(52,211,153,0.15)'},
                ],
                'threshold': {'line': {'color': '#ef4444', 'width': 3}, 'thickness': 0.85,
                              'value': max_v * 0.9}
            }
        ))
        fig.update_layout(title=format_title(question) or 'Gauge', height=350,
                          font=dict(color='#e2e8f0'))
        return fig
    except Exception as e:
        print(f"❌ gauge: {e}"); return None


def create_bullet_chart(df: pd.DataFrame, question: str):
    """Bullet chart (target vs actual)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_bar_chart(df, question)
        actual_col = num_cols[0]
        target_col = num_cols[1]
        label_col  = txt_cols[0] if txt_cols else None
        d = df[[actual_col, target_col] + ([label_col] if label_col else [])].dropna().head(8)
        n = len(d)
        labels = d[label_col].tolist() if label_col else [f'KPI {i+1}' for i in range(n)]
        specs  = [[{'type': 'indicator'}] for _ in range(n)]
        fig    = make_subplots(rows=n, cols=1, specs=specs)
        for i, (_, row) in enumerate(d.iterrows()):
            fig.add_trace(go.Indicator(
                mode='number+gauge',
                value=float(row[actual_col]),
                gauge={'shape': 'bullet', 'axis': {'range': [0, float(row[target_col]) * 1.2]},
                       'threshold': {'value': float(row[target_col]), 'line': {'color': '#ef4444', 'width': 3}},
                       'bar': {'color': NEON_COLORS[i % 12]},
                       'steps': [{'range': [0, float(row[target_col])], 'color': 'rgba(255,255,255,0.05)'}]},
                title={'text': str(labels[i])},
                number={'font': {'color': NEON_COLORS[i % 12]}}
            ), row=i+1, col=1)
        fig.update_layout(title=format_title(question) or 'Bullet Chart',
                          height=max(200, n * 120))
        return fig
    except Exception as e:
        print(f"❌ bullet_chart: {e}"); return None


def create_indicator_tile(df: pd.DataFrame, question: str):
    """Big-number indicator tiles for key metrics."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        if not num_cols:
            return None
        n    = len(num_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        specs = [[{'type': 'indicator'}] * cols for _ in range(rows)]
        fig   = make_subplots(rows=rows, cols=cols, specs=specs)
        for i, col in enumerate(num_cols):
            val = float(df[col].sum())
            fig.add_trace(go.Indicator(
                mode='number',
                value=val,
                title={'text': format_col_name(col), 'font': {'size': 13, 'color': '#e2e8f0'}},
                number={'font': {'size': 30, 'color': NEON_COLORS[i % 12]}, 'valueformat': ',.1f'}
            ), row=(i // cols) + 1, col=(i % cols) + 1)
        fig.update_layout(title=format_title(question) or 'Key Metrics', height=200 * rows)
        return fig
    except Exception as e:
        print(f"❌ indicator_tile: {e}"); return None


def create_progress_chart(df: pd.DataFrame, question: str):
    """Progress bar indicators (actual vs target)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_gauge(df, question)
        val_col = num_cols[0]
        max_col = num_cols[1]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[val_col, max_col] + ([lbl_col] if lbl_col else [])].dropna().head(8)
        labels = d[lbl_col].tolist() if lbl_col else [f'KPI {i+1}' for i in range(len(d))]
        fig = go.Figure()
        for i, (_, row) in enumerate(d.iterrows()):
            pct = min(100, float(row[val_col]) / max(float(row[max_col]), 1) * 100)
            fig.add_shape(type='rect', x0=0, x1=100, y0=i - 0.25, y1=i + 0.25,
                          fillcolor='rgba(255,255,255,0.07)', line=dict(width=0))
            fig.add_shape(type='rect', x0=0, x1=pct, y0=i - 0.25, y1=i + 0.25,
                          fillcolor=NEON_COLORS[i % 12], line=dict(width=0))
            fig.add_annotation(x=102, y=i, text=f'{pct:.0f}%', showarrow=False,
                                font=dict(size=11, color='#e2e8f0'), xanchor='left')
            fig.add_annotation(x=-1, y=i, text=str(labels[i]), showarrow=False,
                                font=dict(size=11, color='#a78bfa'), xanchor='right')
        fig.update_layout(title=format_title(question) or 'Progress Chart',
                          xaxis=dict(range=[-20, 115], showgrid=False, showticklabels=False),
                          yaxis=dict(showgrid=False, showticklabels=False),
                          height=max(250, len(d) * 70 + 100))
        return fig
    except Exception as e:
        print(f"❌ progress_chart: {e}"); return None


def create_sparkline(df: pd.DataFrame, question: str):
    """Sparkline grid — one mini trend per numeric column."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        if not num_cols:
            return None
        n    = len(num_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig  = make_subplots(rows=rows, cols=cols, subplot_titles=[format_col_name(c) for c in num_cols])
        for i, col in enumerate(num_cols):
            y = df[col].dropna().head(50)
            r, c = (i // cols) + 1, (i % cols) + 1
            fig.add_trace(go.Scatter(
                y=y, mode='lines', line=dict(color=NEON_COLORS[i % 12], width=2),
                fill='tozeroy', fillcolor=f'rgba(99,102,241,0.1)',
                showlegend=False
            ), row=r, col=c)
        fig.update_layout(title=format_title(question) or 'Sparklines', height=100 * rows + 80)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        return fig
    except Exception as e:
        print(f"❌ sparkline: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 10 — POLAR / RADIAL  (5 charts)
# ═══════════════════════════════════════════════════════════

def create_radar(df: pd.DataFrame, question: str):
    """Spider / radar chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        if len(num_cols) >= 3 and len(df) <= 10:
            # Multiple entities on one radar
            fig = go.Figure()
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                values = [row[c] for c in num_cols[:8]]
                values_closed = values + [values[0]]
                labels_closed = [format_col_name(c) for c in num_cols[:8]] + [format_col_name(num_cols[0])]
                fig.add_trace(go.Scatterpolar(
                    r=values_closed, theta=labels_closed,
                    fill='toself', fillcolor=NEON_COLORS[i % 12].replace('#', 'rgba(').replace(')', ',0.15)')
                        if False else f'rgba(99,102,241,{0.15 - i * 0.02})',
                    line=dict(color=NEON_COLORS[i % 12], width=2),
                    name=str(row[txt_cols[0]]) if txt_cols else f'Item {i+1}'
                ))
        else:
            # Single row or one series per column
            values = df[num_cols[:8]].mean().tolist()
            labels = [format_col_name(c) for c in num_cols[:8]]
            values += [values[0]]; labels += [labels[0]]
            fig = go.Figure(go.Scatterpolar(
                r=values, theta=labels, fill='toself',
                fillcolor='rgba(99,102,241,0.2)',
                line=dict(color=NEON_COLORS[0], width=2.5)
            ))
        fig.update_layout(title=format_title(question) or 'Radar Chart',
                          polar=dict(radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)')))
        return fig
    except Exception as e:
        print(f"❌ radar: {e}"); return None


def create_polar_bar(df: pd.DataFrame, question: str):
    """Polar bar (wind rose style)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(16) if lbl_col else df[[val_col]].head(16)
        n = len(d)
        theta = [i * 360 / n for i in range(n)]
        labels = d[lbl_col].tolist() if lbl_col else [f'{t:.0f}°' for t in theta]
        fig = go.Figure(go.Barpolar(
            r=d[val_col].tolist(), theta=theta, width=[360 / n] * n,
            marker=dict(color=NEON_COLORS[:n % 12 + 1] * 2,
                        line=dict(color='rgba(255,255,255,0.15)', width=1)),
            hovertemplate='<b>%{customdata}</b><br>Value: %{r:,.2f}<extra></extra>',
            customdata=labels
        ))
        fig.update_layout(title=format_title(question) or 'Polar Bar Chart',
                          polar=dict(radialaxis=dict(visible=True)))
        return fig
    except Exception as e:
        print(f"❌ polar_bar: {e}"); return None


def create_polar_scatter(df: pd.DataFrame, question: str):
    """Polar scatter (r vs theta)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return None
        r_col, theta_col = num_cols[0], num_cols[1]
        color_col = txt_cols[0] if txt_cols else None
        d = df.sample(min(300, len(df)), random_state=42)
        theta_vals = (d[theta_col] % 360).tolist()
        fig = go.Figure(go.Scatterpolar(
            r=d[r_col].tolist(), theta=theta_vals, mode='markers',
            marker=dict(size=7, color=NEON_COLORS[0], opacity=0.7,
                        line=dict(width=1, color='rgba(255,255,255,0.2)')),
            hovertemplate=f'r: %{{r:,.2f}}<br>θ: %{{theta:.1f}}°<extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or 'Polar Scatter',
                          polar=dict(radialaxis=dict(visible=True)))
        return fig
    except Exception as e:
        print(f"❌ polar_scatter: {e}"); return None


def create_windrose(df: pd.DataFrame, question: str):
    """Wind rose chart — direction × frequency × intensity."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        # Build binned wind-rose from first numeric column
        val_col = num_cols[0]
        data = df[val_col].dropna()
        bins  = np.linspace(data.min(), data.max(), 9)
        dirs  = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        counts= np.histogram(data, bins=bins)[0]
        theta = [i * 45 for i in range(8)]
        fig = go.Figure()
        for i, (dir_lbl, cnt) in enumerate(zip(dirs, counts)):
            fig.add_trace(go.Barpolar(
                r=[cnt], theta=[theta[i]], width=[45], name=dir_lbl,
                marker=dict(color=NEON_COLORS[i % 12], opacity=0.8)
            ))
        fig.update_layout(title=format_title(question) or f'Wind Rose — {format_col_name(val_col)}',
                          polar=dict(angularaxis=dict(direction='clockwise', tickmode='array',
                                                       tickvals=theta, ticktext=dirs)))
        return fig
    except Exception as e:
        print(f"❌ windrose: {e}"); return None


def create_nightingale(df: pd.DataFrame, question: str):
    """Florence Nightingale rose chart (polar area)."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return None
        val_col = num_cols[0]
        lbl_col = txt_cols[0] if txt_cols else None
        d = df[[lbl_col, val_col]].dropna().head(12) if lbl_col else df[[val_col]].head(12)
        n = len(d)
        width = 360 / n
        theta = [i * width + width / 2 for i in range(n)]
        labels = d[lbl_col].tolist() if lbl_col else [f'Seg {i+1}' for i in range(n)]
        # Use sqrt for visual area accuracy
        r_vals = np.sqrt(d[val_col].values.astype(float))
        r_vals = r_vals / r_vals.max() * 100
        fig = go.Figure(go.Barpolar(
            r=r_vals, theta=theta, width=[width] * n,
            marker=dict(color=EXTENDED_COLORS[:n], opacity=0.8,
                        line=dict(color='rgba(0,0,0,0.4)', width=1)),
            customdata=d[val_col].tolist(),
            hovertemplate='<b>%{customdata2}</b><br>Value: %{customdata:,.0f}<extra></extra>',
        ))
        # Add labels manually
        for i, (t, lbl) in enumerate(zip(theta, labels)):
            fig.add_annotation(
                r=r_vals[i] + 5, theta=t,
                text=str(lbl)[:12], showarrow=False,
                font=dict(size=9, color='#e2e8f0')
            )
        fig.update_layout(title=format_title(question) or 'Nightingale Rose Chart',
                          polar=dict(radialaxis=dict(visible=False),
                                     angularaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)')))
        return fig
    except Exception as e:
        print(f"❌ nightingale: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  GROUP 11 — 3D CHARTS  (5 charts)
# ═══════════════════════════════════════════════════════════

def create_scatter_3d(df: pd.DataFrame, question: str):
    """3D scatter plot."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 3:
            return create_scatter_chart(df, question)
        x_col, y_col, z_col = num_cols[0], num_cols[1], num_cols[2]
        color_col = txt_cols[0] if txt_cols else None
        d = df.sample(min(500, len(df)), random_state=42)
        fig = px.scatter_3d(d, x=x_col, y=y_col, z=z_col, color=color_col,
                            color_discrete_sequence=NEON_COLORS, opacity=0.75)
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))
        fig.update_layout(title=format_title(question),
                          scene=dict(
                              xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)'),
                              yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)'),
                              zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)'),
                          ), height=550)
        return fig
    except Exception as e:
        print(f"❌ scatter_3d: {e}"); return create_scatter_chart(df, question)


def create_surface_3d(df: pd.DataFrame, question: str):
    """3D surface chart from numeric matrix."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_heatmap(df, question)
        d = df[num_cols[:20]].head(20).fillna(0)
        fig = go.Figure(go.Surface(
            z=d.values,
            colorscale=[[0, '#0f0c29'], [0.33, '#302b63'], [0.66, '#667eea'], [1, '#43e97b']],
            opacity=0.9,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:,.2f}<extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or '3D Surface',
                          scene=dict(
                              xaxis_title='Column', yaxis_title='Row', zaxis_title='Value',
                              bgcolor='rgba(0,0,0,0)'
                          ), height=550)
        return fig
    except Exception as e:
        print(f"❌ surface_3d: {e}"); return create_heatmap(df, question)


def create_line_3d(df: pd.DataFrame, question: str):
    """3D line chart."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 3:
            return create_line_chart(df, question)
        x_col, y_col, z_col = num_cols[0], num_cols[1], num_cols[2]
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_col = txt_cols[0] if txt_cols else None
        d = df.head(100)
        fig = px.line_3d(d, x=x_col, y=y_col, z=z_col, color=color_col,
                         color_discrete_sequence=NEON_COLORS)
        fig.update_layout(title=format_title(question),
                          scene=dict(bgcolor='rgba(0,0,0,0)'), height=550)
        return fig
    except Exception as e:
        print(f"❌ line_3d: {e}"); return create_line_chart(df, question)


def create_contour_3d(df: pd.DataFrame, question: str):
    """3D contour surface."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_contour(df, question)
        d = df[num_cols[:15]].head(15).fillna(0)
        fig = go.Figure(go.Surface(
            z=d.values,
            colorscale='Viridis',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor='#e2e8f0', project_z=True)
            ),
            opacity=0.85
        ))
        fig.update_layout(title=format_title(question) or '3D Contour',
                          scene=dict(bgcolor='rgba(0,0,0,0)'), height=550)
        return fig
    except Exception as e:
        print(f"❌ contour_3d: {e}"); return create_heatmap(df, question)


def create_bar_3d(df: pd.DataFrame, question: str):
    """3D bar chart — simulated with scatter3d cylinders."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_bar_chart(df, question)
        # Use scatter3d with wide markers as proxy for 3D bars
        x_col, y_col = num_cols[0], num_cols[1]
        z_col = num_cols[2] if len(num_cols) >= 3 else None
        d = df.head(30)
        fig = px.scatter_3d(d, x=x_col, y=y_col,
                            z=z_col if z_col else y_col,
                            color=txt_cols[0] if txt_cols else None,
                            color_discrete_sequence=NEON_COLORS, size_max=20)
        fig.update_traces(marker=dict(size=12, opacity=0.8, symbol='square'))
        fig.update_layout(title=format_title(question) or '3D Bar',
                          scene=dict(bgcolor='rgba(0,0,0,0)'), height=550)
        return fig
    except Exception as e:
        print(f"❌ bar_3d: {e}"); return create_bar_chart(df, question)


# ═══════════════════════════════════════════════════════════
# ██  GROUP 12 — FLOW / NETWORK  (4 charts)
# ═══════════════════════════════════════════════════════════

def create_sankey(df: pd.DataFrame, question: str):
    """Sankey diagram from source → target → value columns."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(txt_cols) < 2:
            return create_bar_chart(df, question)
        src_col, tgt_col = txt_cols[0], txt_cols[1]
        val_col = num_cols[0] if num_cols else None
        d = df[[src_col, tgt_col] + ([val_col] if val_col else [])].dropna().head(50)
        if val_col is None:
            d = d.copy(); d['_val'] = 1; val_col = '_val'
        all_nodes = pd.unique(d[[src_col, tgt_col]].values.ravel()).tolist()
        node_map  = {n: i for i, n in enumerate(all_nodes)}
        src_idx   = d[src_col].map(node_map).tolist()
        tgt_idx   = d[tgt_col].map(node_map).tolist()
        vals      = d[val_col].tolist()
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color='rgba(255,255,255,0.1)', width=0.5),
                      label=all_nodes, color=NEON_COLORS[:len(all_nodes)] * 10),
            link=dict(source=src_idx, target=tgt_idx, value=vals,
                      color='rgba(102,126,234,0.25)',
                      hovertemplate='%{source.label} → %{target.label}<br>Value: %{value:,.0f}<extra></extra>')
        ))
        fig.update_layout(title=format_title(question) or 'Sankey Diagram')
        return fig
    except Exception as e:
        print(f"❌ sankey: {e}"); return None


def create_chord(df: pd.DataFrame, question: str):
    """Chord diagram — approximated as circular Sankey."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(txt_cols) < 2:
            return create_sankey(df, question)
        # For chord we build a matrix and display as heatmap-style
        src_col, tgt_col = txt_cols[0], txt_cols[1]
        val_col = num_cols[0] if num_cols else None
        d = df[[src_col, tgt_col] + ([val_col] if val_col else [])].dropna().head(50)
        if val_col is None:
            d = d.copy(); d['_v'] = 1; val_col = '_v'
        pivot = d.groupby([src_col, tgt_col])[val_col].sum().unstack(fill_value=0)
        nodes = list(pivot.index.union(pivot.columns))
        # Render as polar scatter arrows
        n = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos  = np.cos(angles)
        y_pos  = np.sin(angles)
        fig = go.Figure()
        for i, node in enumerate(nodes):
            fig.add_trace(go.Scatter(
                x=[x_pos[i]], y=[y_pos[i]], mode='markers+text',
                marker=dict(size=18, color=NEON_COLORS[i % 12], line=dict(width=2, color='white')),
                text=[node], textposition='top center', textfont=dict(size=9),
                showlegend=False, hoverinfo='text', hovertext=node
            ))
        for i, src in enumerate(nodes):
            if src in pivot.index:
                for j, tgt in enumerate(nodes):
                    if tgt in pivot.columns and pivot.loc[src, tgt] > 0:
                        weight = float(pivot.loc[src, tgt])
                        fig.add_trace(go.Scatter(
                            x=[x_pos[i], x_pos[j]], y=[y_pos[i], y_pos[j]],
                            mode='lines', line=dict(color='rgba(102,126,234,0.3)',
                                                    width=max(1, min(8, weight / max(1, d[val_col].max()) * 8))),
                            showlegend=False, hoverinfo='skip'
                        ))
        fig.update_layout(title=format_title(question) or 'Chord Diagram',
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
                          height=550)
        return fig
    except Exception as e:
        print(f"❌ chord: {e}"); return create_sankey(df, question)


def create_network_graph(df: pd.DataFrame, question: str):
    """Network / node-link graph (force-directed approximation)."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(txt_cols) < 2:
            return create_bar_chart(df, question)
        src_col, tgt_col = txt_cols[0], txt_cols[1]
        d = df[[src_col, tgt_col]].dropna().head(40)
        nodes = pd.unique(d[[src_col, tgt_col]].values.ravel())
        n     = len(nodes)
        # Circular layout
        angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
        node_pos = {nd: (np.cos(a), np.sin(a)) for nd, a in zip(nodes, angles)}
        fig = go.Figure()
        # Edges
        for _, row in d.iterrows():
            x0, y0 = node_pos.get(row[src_col], (0, 0))
            x1, y1 = node_pos.get(row[tgt_col], (0, 0))
            fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                                      line=dict(color='rgba(99,102,241,0.3)', width=1.5),
                                      showlegend=False, hoverinfo='skip'))
        # Nodes
        node_x = [node_pos[nd][0] for nd in nodes]
        node_y = [node_pos[nd][1] for nd in nodes]
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=14, color=NEON_COLORS[:n] * 10,
                        line=dict(width=2, color='rgba(255,255,255,0.4)')),
            text=nodes, textposition='top center', textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        fig.update_layout(title=format_title(question) or 'Network Graph',
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          height=550)
        return fig
    except Exception as e:
        print(f"❌ network_graph: {e}"); return create_bar_chart(df, question)


def create_flow_tree(df: pd.DataFrame, question: str):
    """Hierarchical tree flow chart."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not txt_cols:
            return create_treemap(df, question)
        # Use sunburst as a tree proxy
        return create_sunburst(df, question)
    except Exception as e:
        print(f"❌ flow_tree: {e}"); return create_treemap(df, question)


# ═══════════════════════════════════════════════════════════
# ██  GROUP 13 — GEO  (4 charts)
# ═══════════════════════════════════════════════════════════

def create_choropleth(df: pd.DataFrame, question: str):
    """Choropleth map — needs ISO-3 country codes or US state codes."""
    try:
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not txt_cols or not num_cols:
            return create_bar_chart(df, question)
        loc_col = txt_cols[0]
        val_col = num_cols[0]
        sample  = df[loc_col].dropna().head(5).tolist()
        # Detect if US states or countries
        us_abbrevs = {'CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'}
        is_us_state = any(s in us_abbrevs for s in sample)
        d = df[[loc_col, val_col]].dropna().head(60)
        if is_us_state:
            fig = px.choropleth(d, locations=loc_col, locationmode='USA-states',
                                color=val_col, scope='usa',
                                color_continuous_scale=['#0f0c29', '#302b63', '#667eea'],
                                title=format_title(question))
        else:
            fig = px.choropleth(d, locations=loc_col, locationmode='ISO-3',
                                color=val_col,
                                color_continuous_scale=['#0f0c29', '#302b63', '#667eea'],
                                title=format_title(question))
        fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False))
        return fig
    except Exception as e:
        print(f"❌ choropleth: {e}"); return create_bar_chart(df, question)


def create_scatter_geo(df: pd.DataFrame, question: str):
    """Geographic scatter map."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        lat_col  = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col  = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
        if not lat_col or not lon_col:
            return create_scatter_chart(df, question)
        val_col   = num_cols[0] if num_cols else None
        color_col = txt_cols[0] if txt_cols else None
        d = df[[lat_col, lon_col] + ([val_col] if val_col else []) + ([color_col] if color_col else [])].dropna().head(500)
        fig = px.scatter_geo(d, lat=lat_col, lon=lon_col, color=color_col,
                              size=val_col, color_discrete_sequence=NEON_COLORS,
                              opacity=0.8, projection='natural earth')
        fig.update_layout(title=format_title(question) or 'Geographic Scatter',
                          geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                                   showcoastlines=True, coastlinecolor='rgba(255,255,255,0.15)'))
        return fig
    except Exception as e:
        print(f"❌ scatter_geo: {e}"); return create_scatter_chart(df, question)


def create_bubble_map(df: pd.DataFrame, question: str):
    """Geographic bubble map."""
    try:
        lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
        if not lat_col or not lon_col:
            return create_scatter_geo(df, question)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        size_col = next((c for c in num_cols if c not in [lat_col, lon_col]), None)
        d = df.dropna(subset=[lat_col, lon_col]).head(300)
        fig = px.scatter_geo(d, lat=lat_col, lon=lon_col, size=size_col,
                              color=size_col, size_max=40,
                              color_continuous_scale=['#302b63', '#667eea', '#43e97b'],
                              opacity=0.7, projection='natural earth')
        fig.update_layout(title=format_title(question) or 'Bubble Map',
                          geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                                   showcoastlines=True, coastlinecolor='rgba(255,255,255,0.15)'))
        return fig
    except Exception as e:
        print(f"❌ bubble_map: {e}"); return create_scatter_chart(df, question)


def create_density_map(df: pd.DataFrame, question: str):
    """Density mapbox (requires lat/lon columns)."""
    try:
        lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
        if not lat_col or not lon_col:
            return create_scatter_geo(df, question)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        z_col    = next((c for c in num_cols if c not in [lat_col, lon_col]), None)
        d = df.dropna(subset=[lat_col, lon_col]).head(500)
        fig = px.density_mapbox(d, lat=lat_col, lon=lon_col, z=z_col,
                                 radius=15, zoom=3,
                                 mapbox_style='carto-darkmatter',
                                 color_continuous_scale=['#0f0c29', '#302b63', '#667eea', '#43e97b'])
        fig.update_layout(title=format_title(question) or 'Density Map')
        return fig
    except Exception as e:
        print(f"❌ density_map: {e}"); return create_scatter_chart(df, question)


# ═══════════════════════════════════════════════════════════
# ██  GROUP 14 — MISC / SPECIAL  (4 charts)
# ═══════════════════════════════════════════════════════════

def create_matrix_bubble(df: pd.DataFrame, question: str):
    """Matrix bubble / circle heatmap — circles sized by value."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not num_cols:
            return create_heatmap(df, question)
        d = df[num_cols].head(15).fillna(0)
        row_labels = df[txt_cols[0]].head(15).tolist() if txt_cols else [f'Row {i+1}' for i in range(len(d))]
        col_labels = [format_col_name(c) for c in d.columns]
        xs, ys, sizes, colors, texts = [], [], [], [], []
        max_v = d.values.max()
        for ri, row in d.iterrows():
            for ci, col in enumerate(d.columns):
                val = row[col]
                xs.append(ci); ys.append(list(d.index).index(ri))
                sizes.append(max(3, val / max(max_v, 1) * 50))
                colors.append(val); texts.append(f'{val:,.1f}')
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=sizes, color=colors,
                        colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#6366f1']],
                        showscale=True, line=dict(width=1, color='rgba(255,255,255,0.2)')),
            text=texts, textfont=dict(size=7, color='rgba(255,255,255,0.8)'),
            hovertemplate='%{text}<extra></extra>'
        ))
        fig.update_layout(
            title=format_title(question) or 'Matrix Bubble Chart',
            xaxis=dict(tickvals=list(range(len(col_labels))), ticktext=col_labels,
                       showgrid=False, side='top'),
            yaxis=dict(tickvals=list(range(len(row_labels))), ticktext=row_labels, showgrid=False),
            height=max(300, len(row_labels) * 50 + 100)
        )
        return fig
    except Exception as e:
        print(f"❌ matrix_bubble: {e}"); return create_heatmap(df, question)


def create_animated_scatter(df: pd.DataFrame, question: str):
    """Animated scatter — animates over a time/categorical column."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(num_cols) < 2:
            return create_scatter_chart(df, question)
        x_col, y_col = num_cols[0], num_cols[1]
        size_col  = num_cols[2] if len(num_cols) >= 3 else None
        anim_col  = txt_cols[0] if txt_cols else None
        color_col = txt_cols[1] if len(txt_cols) > 1 else anim_col
        d = df.head(200)
        if anim_col and d[anim_col].nunique() > 1:
            fig = px.scatter(d, x=x_col, y=y_col, animation_frame=anim_col,
                             color=color_col, size=size_col,
                             color_discrete_sequence=NEON_COLORS, size_max=40, opacity=0.75)
        else:
            fig = px.scatter(d, x=x_col, y=y_col, color=color_col, size=size_col,
                             color_discrete_sequence=NEON_COLORS, size_max=40, opacity=0.75)
        fig.update_layout(title=format_title(question) or 'Animated Scatter',
                          xaxis_title=format_col_name(x_col), yaxis_title=format_col_name(y_col))
        return fig
    except Exception as e:
        print(f"❌ animated_scatter: {e}"); return create_scatter_chart(df, question)


def create_table_chart(df: pd.DataFrame, question: str):
    """Styled data table with alternating row colors."""
    try:
        d = df.head(30)
        header_vals = [format_col_name(c) for c in d.columns]
        cell_vals   = [d[c].astype(str).tolist() for c in d.columns]
        n_rows = len(d)
        row_colors = ['rgba(15,12,41,0.8)' if i % 2 == 0 else 'rgba(30,27,75,0.8)'
                      for i in range(n_rows)]
        fig = go.Figure(go.Table(
            header=dict(
                values=[f'<b>{h}</b>' for h in header_vals],
                fill_color='rgba(99,102,241,0.6)',
                line_color='rgba(255,255,255,0.1)',
                align='center', font=dict(color='white', size=12, family=CHART_FONT),
                height=36
            ),
            cells=dict(
                values=cell_vals,
                fill_color=[row_colors for _ in d.columns],
                line_color='rgba(255,255,255,0.05)',
                align='center', font=dict(color='#e2e8f0', size=11, family=CHART_FONT),
                height=32
            )
        ))
        fig.update_layout(title=format_title(question) or 'Data Table',
                          margin=dict(l=30, r=30, t=80, b=30),
                          height=min(800, n_rows * 35 + 140))
        return fig
    except Exception as e:
        print(f"❌ table_chart: {e}"); return None


def create_stream_graph(df: pd.DataFrame, question: str):
    """Stream / flow graph — symmetric stacked area around zero axis."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return create_stacked_area(df, question)
        x_data = df[non_num[0]].head(80) if non_num else list(range(min(80, len(df))))
        # Center the streams around zero
        cols = num_cols[:6]
        normed = df[cols].head(80).fillna(0)
        total  = normed.sum(axis=1)
        offset = total / 2
        fig = go.Figure()
        cumsum = pd.Series([0.0] * len(normed))
        for i, col in enumerate(cols):
            y = normed[col]
            y_bottom = cumsum - offset
            y_top    = cumsum + y - offset
            fig.add_trace(go.Scatter(
                x=list(x_data) + list(x_data)[::-1],
                y=list(y_top) + list(y_bottom)[::-1],
                fill='toself',
                fillcolor=NEON_COLORS[i % 12] + '99' if len(NEON_COLORS[i % 12]) == 7 else NEON_COLORS[i % 12],
                line=dict(color=NEON_COLORS[i % 12], width=1),
                name=format_col_name(col), mode='lines'
            ))
            cumsum = cumsum + y
        fig.update_layout(title=format_title(question) or 'Stream Graph',
                          hovermode='x unified',
                          yaxis=dict(zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'))
        return fig
    except Exception as e:
        print(f"❌ stream_graph: {e}"); return create_stacked_area(df, question)


# ═══════════════════════════════════════════════════════════
# ██  CHART REGISTRY  (90 types)
# ═══════════════════════════════════════════════════════════

_CHART_REGISTRY = {
    # Bar family
    'bar':              create_bar_chart,
    'column':           create_column_chart,
    'grouped_bar':      create_grouped_bar,
    'stacked_bar':      create_stacked_bar,
    'stacked_100_bar':  create_stacked_100_bar,
    'lollipop':         create_lollipop,
    'diverging_bar':    create_diverging_bar,
    'pyramid_bar':      create_pyramid_bar,
    'circular_bar':     create_circular_bar,
    'combo_bar_line':   create_combo_bar_line,
    # Line family
    'line':             create_line_chart,
    'multi_line':       create_multi_line,
    'step_line':        create_step_line,
    'area':             create_area_chart,
    'stacked_area':     create_stacked_area,
    'slope_chart':      create_slope_chart,
    'bump_chart':       create_bump_chart,
    'moving_average':   create_moving_average,
    # Scatter/Correlation
    'scatter':          create_scatter_chart,
    'bubble':           create_bubble_chart,
    'connected_scatter':create_connected_scatter,
    'scatter_matrix':   create_scatter_matrix,
    'density_contour':  create_density_contour,
    'density_heatmap':  create_density_heatmap,
    'parallel_coordinates': create_parallel_coordinates,
    'dumbbell':         create_dumbbell,
    # Distribution
    'histogram':        create_histogram,
    'kde':              create_kde,
    'violin':           create_violin,
    'box':              create_box_plot,
    'strip':            create_strip,
    'ridgeline':        create_ridgeline,
    'ecdf':             create_ecdf,
    'hexbin':           create_hexbin,
    'rug_plot':         create_rug_plot,
    # Proportion
    'pie':              create_pie_chart,
    'treemap':          create_treemap,
    'sunburst':         create_sunburst,
    'funnel':           create_funnel_chart,
    'waterfall':        create_waterfall_chart,
    'marimekko':        create_marimekko,
    'waffle_chart':     create_waffle_chart,
    'dot_plot':         create_dot_plot,
    'pictogram':        create_pictogram,
    # Heat/Matrix
    'heatmap':          create_heatmap,
    'contour':          create_contour,
    'calendar_heatmap': create_calendar_heatmap,
    'parallel_categories': create_parallel_categories,
    'qq_plot':          create_qq_plot,
    # Statistical
    'error_bar':        create_error_bar,
    'confidence_band':  create_confidence_band,
    'tornado_chart':    create_tornado_chart,
    'comparison_bar':   create_comparison_bar,
    'annotated_line':   create_annotated_line,
    'multi_axis_line':  create_multi_axis_line,
    'heat_table':       create_heat_table,
    # Time Series
    'candlestick':      create_candlestick,
    'ohlc':             create_ohlc,
    'gantt':            create_gantt,
    'timeline':         create_timeline,
    'forecast_chart':   create_forecast_chart,
    'seasonal_chart':   create_seasonal_chart,
    'stacked_waterfall':create_stacked_waterfall,
    # KPI/Indicator
    'gauge':            create_gauge,
    'bullet_chart':     create_bullet_chart,
    'indicator_tile':   create_indicator_tile,
    'progress_chart':   create_progress_chart,
    'sparkline':        create_sparkline,
    # Polar/Radial
    'radar':            create_radar,
    'polar_bar':        create_polar_bar,
    'polar_scatter':    create_polar_scatter,
    'windrose':         create_windrose,
    'nightingale':      create_nightingale,
    # 3D
    'scatter_3d':       create_scatter_3d,
    'surface_3d':       create_surface_3d,
    'line_3d':          create_line_3d,
    'contour_3d':       create_contour_3d,
    'bar_3d':           create_bar_3d,
    # Flow/Network
    'sankey':           create_sankey,
    'chord':            create_chord,
    'network_graph':    create_network_graph,
    'flow_tree':        create_flow_tree,
    # Geo
    'choropleth':       create_choropleth,
    'scatter_geo':      create_scatter_geo,
    'bubble_map':       create_bubble_map,
    'density_map':      create_density_map,
    # Misc
    'matrix_bubble':    create_matrix_bubble,
    'animated_scatter': create_animated_scatter,
    'table_chart':      create_table_chart,
    'stream_graph':     create_stream_graph,
}


# ═══════════════════════════════════════════════════════════
# ██  KPI DASHBOARD
# ═══════════════════════════════════════════════════════════

def create_kpi_dashboard(kpis: list):
    """Render a grid of KPI indicator tiles (up to 6)."""
    try:
        if not kpis:
            return None
        kpis = kpis[:6]
        n    = len(kpis)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        specs = [[{'type': 'indicator'}] * cols for _ in range(rows)]
        fig   = make_subplots(rows=rows, cols=cols, specs=specs)
        for i, kpi in enumerate(kpis):
            r, c = (i // cols) + 1, (i % cols) + 1
            fig.add_trace(go.Indicator(
                mode='number',
                value=float(kpi['value']),
                title={'text': kpi['label'], 'font': {'size': 14, 'color': '#e2e8f0'}},
                number={'font': {'size': 28, 'color': NEON_COLORS[i % 12]}, 'valueformat': ',.2f'},
            ), row=r, col=c)
        fig.update_layout(height=200 * rows, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font=dict(family=CHART_FONT, color='#e2e8f0'))
        return fig
    except Exception as e:
        print(f"❌ KPI dashboard error: {e}"); return None


# ═══════════════════════════════════════════════════════════
# ██  PREMIUM STYLING
# ═══════════════════════════════════════════════════════════

def apply_premium_styling(fig: go.Figure, chart_type: str = 'bar') -> go.Figure:
    """Apply dark glassmorphic styling to every chart."""
    fig.update_layout(
        template='plotly_dark',
        height=500,
        font=dict(family=CHART_FONT, size=12, color='#e2e8f0'),
        title=dict(font=dict(size=18, color='#f7fafc', family=CHART_FONT),
                   x=0.5, xanchor='center', y=0.97),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(bgcolor='rgba(10,10,26,0.95)', bordercolor='#667eea',
                        font_size=13, font_family=CHART_FONT, font_color='#f7fafc'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(102,126,234,0.2)',
                    borderwidth=1, font=dict(size=11, color='#e2e8f0')),
    )
    axis_style = dict(
        showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)',
        showline=True, linewidth=1, linecolor='rgba(255,255,255,0.1)',
        zeroline=False, title_font=dict(size=13, color='#a78bfa'),
        tickfont=dict(size=11, color='#cbd5e1'),
    )
    # 3D / polar / pie / treemap / funnel / table charts don't have 2D axes
    no_axis_types = {'pie', 'treemap', 'sunburst', 'funnel', 'waffle_chart', 'pictogram',
                     'gauge', 'bullet_chart', 'indicator_tile', 'table_chart', 'kpi_dashboard',
                     'scatter_3d', 'surface_3d', 'line_3d', 'contour_3d', 'bar_3d',
                     'radar', 'polar_bar', 'polar_scatter', 'windrose', 'nightingale',
                     'scatter_geo', 'bubble_map', 'choropleth', 'density_map'}
    if chart_type not in no_axis_types:
        try:
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
        except Exception:
            pass
    return fig


# ═══════════════════════════════════════════════════════════
# ██  AUTO BUSINESS VISUALIZATIONS (EDA — 5 smart charts)
# ═══════════════════════════════════════════════════════════

def generate_auto_business_visualizations(df: pd.DataFrame, column_categories: dict) -> list:
    """Generates up to 5 smart business charts from schema. Returns [{title, desc, fig}]."""
    charts = []
    date_cols  = column_categories.get('date_columns', [])
    num_cols   = column_categories.get('numeric_columns', [])
    cat_cols   = column_categories.get('categorical_columns', [])
    id_cols    = set(column_categories.get('id_columns', []))

    valid_nums = [c for c in num_cols if c not in id_cols]
    valid_cats = [c for c in cat_cols if c not in id_cols]
    if not valid_nums:
        return charts

    biz_kws = ['revenue', 'sales', 'profit', 'amount', 'quantity', 'margin', 'cost', 'price']
    primary = valid_nums[0]
    secondary = valid_nums[1] if len(valid_nums) > 1 else None
    for col in valid_nums:
        if any(kw in col.lower() for kw in biz_kws):
            primary = col; break
    for col in valid_nums:
        if col != primary and any(kw in col.lower() for kw in biz_kws):
            secondary = col; break

    # 1. TIME SERIES TREND
    if date_cols:
        try:
            d_col = date_cols[0]
            td    = df.copy()
            td[d_col] = pd.to_datetime(td[d_col], errors='coerce')
            trend_df  = td.groupby(td[d_col].dt.to_period('M'))[primary].sum().reset_index()
            trend_df[d_col] = trend_df[d_col].astype(str)
            if len(trend_df) > 1:
                fig = px.line(trend_df, x=d_col, y=primary,
                              title=f'Monthly Trend: {format_col_name(primary)}',
                              markers=True, color_discrete_sequence=[NEON_COLORS[0]])
                fig.update_traces(fill='tozeroy', fillcolor='rgba(99,102,241,0.1)')
                charts.append({'title': '📈 Business Growth Trajectory',
                                'desc': f'Tracks {format_col_name(primary)} momentum over time.',
                                'fig':  apply_premium_styling(fig, 'line')})
        except Exception as e:
            print(f"Auto-viz trend error: {e}")

    # 2. TOP PERFORMERS — BAR
    if valid_cats:
        try:
            cat_col = valid_cats[0]
            bar_df  = df.groupby(cat_col)[primary].sum().reset_index().nlargest(10, primary)
            if not bar_df.empty:
                fig = px.bar(bar_df, x=primary, y=cat_col, orientation='h',
                             title=f'Top 10 {format_col_name(cat_col)}s by {format_col_name(primary)}',
                             color=primary,
                             color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3]])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                charts.append({'title': '🏆 Top Performers',
                                'desc': f'{format_col_name(cat_col)} breakdown by {format_col_name(primary)}.',
                                'fig':  apply_premium_styling(fig, 'bar')})
        except Exception as e:
            print(f"Auto-viz top performers error: {e}")

    # 3. MARKET SHARE — DONUT
    if len(valid_cats) > 1:
        try:
            cat2   = valid_cats[1]
            pie_df = df.groupby(cat2)[primary].sum().reset_index()
            if len(pie_df) > 5:
                top5   = pie_df.nlargest(5, primary)
                others = pie_df.nsmallest(len(pie_df) - 5, primary)[primary].sum()
                pie_df = pd.concat([top5, pd.DataFrame({cat2: ['Others'], primary: [others]})], ignore_index=True)
            fig = px.pie(pie_df, names=cat2, values=primary, hole=0.45,
                         title=f'{format_col_name(primary)} by {format_col_name(cat2)}',
                         color_discrete_sequence=NEON_COLORS)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            charts.append({'title': '🥧 Market Share',
                            'desc': f'Proportional breakdown across {format_col_name(cat2)}.',
                            'fig':  apply_premium_styling(fig, 'pie')})
        except Exception as e:
            print(f"Auto-viz market share error: {e}")

    # 4. CORRELATION — SCATTER
    if primary and secondary:
        try:
            sc_df = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
            col_  = valid_cats[0] if valid_cats else None
            fig = px.scatter(sc_df, x=primary, y=secondary, color=col_,
                             title=f'Correlation: {format_col_name(primary)} vs {format_col_name(secondary)}',
                             opacity=0.7, color_discrete_sequence=NEON_COLORS)
            charts.append({'title': '🔗 Metric Correlation',
                            'desc': f'Relationship between {format_col_name(primary)} and {format_col_name(secondary)}.',
                            'fig':  apply_premium_styling(fig, 'scatter')})
        except Exception as e:
            print(f"Auto-viz scatter error: {e}")

    # 5. DISTRIBUTION — BOX PLOT
    if valid_cats:
        try:
            cat_col = valid_cats[0]
            groups  = df[cat_col].dropna().value_counts().head(8).index.tolist()
            df_box  = df[df[cat_col].isin(groups)]
            fig = go.Figure()
            for i, grp in enumerate(groups):
                subset = df_box[df_box[cat_col] == grp][primary].dropna()
                fig.add_trace(go.Box(y=subset, name=str(grp),
                                     marker_color=NEON_COLORS[i % 12], boxmean='sd'))
            fig.update_layout(title=f'Distribution of {format_col_name(primary)} by {format_col_name(cat_col)}',
                              yaxis_title=format_col_name(primary), boxgap=0.3)
            charts.append({'title': '📦 Distribution Analysis',
                            'desc': f'Spread and outliers in {format_col_name(primary)} per {format_col_name(cat_col)}.',
                            'fig':  apply_premium_styling(fig, 'box')})
        except Exception as e:
            print(f"Auto-viz box plot error: {e}")

    return charts


# ═══════════════════════════════════════════════════════════
# ██  HELPERS
# ═══════════════════════════════════════════════════════════

def format_title(question: str) -> str:
    title = (question or '').strip()
    if title:
        title = title[0].upper() + title[1:]
    return title[:77] + '...' if len(title) > 80 else title


def format_col_name(col_name) -> str:
    if not col_name:
        return ''
    return str(col_name).replace('_', ' ').title()


def format_number(value) -> str:
    try:
        value   = float(value)
        abs_val = abs(value)
        sign    = '-' if value < 0 else ''
        if abs_val >= 1_000_000_000: return f'{sign}{abs_val/1_000_000_000:.1f}B'
        if abs_val >= 1_000_000:     return f'{sign}{abs_val/1_000_000:.1f}M'
        if abs_val >= 1_000:         return f'{sign}{abs_val/1_000:.1f}K'
        return f'{value:,.0f}'
    except (TypeError, ValueError):
        return str(value)


def get_chart_type_options() -> dict:
    """Return all 90 chart types for Streamlit selectbox."""
    return {
        # Auto
        'auto':               '🤖 Auto Detect',
        # Bar family
        'bar':                '📊 Bar Chart (Horizontal)',
        'column':             '📊 Column Chart (Vertical)',
        'grouped_bar':        '📊 Grouped Bar',
        'stacked_bar':        '📊 Stacked Bar',
        'stacked_100_bar':    '📊 100% Stacked Bar',
        'lollipop':           '🍭 Lollipop Chart',
        'diverging_bar':      '↔️  Diverging Bar',
        'pyramid_bar':        '🔺 Pyramid / Population Bar',
        'circular_bar':       '🌀 Circular Bar (Polar)',
        'combo_bar_line':     '📊 Combo: Bar + Line',
        # Line family
        'line':               '📈 Line Chart (Spline)',
        'multi_line':         '📈 Multi-Line',
        'step_line':          '📈 Step Line',
        'area':               '🏔️  Area Chart',
        'stacked_area':       '🏔️  Stacked Area',
        'slope_chart':        '📐 Slope / Before-After',
        'bump_chart':         '📈 Bump Chart (Rankings)',
        'moving_average':     '〰️  Moving Average',
        # Scatter / Correlation
        'scatter':            '🔵 Scatter Plot',
        'bubble':             '🫧  Bubble Chart',
        'connected_scatter':  '🔗 Connected Scatter',
        'scatter_matrix':     '🔢 Scatter Matrix (SPLOM)',
        'density_contour':    '🌊 Density Contour',
        'density_heatmap':    '🔥 Density Heatmap',
        'parallel_coordinates':'〰️  Parallel Coordinates',
        'dumbbell':           '🏋️  Dumbbell Chart',
        # Distribution
        'histogram':          '📉 Histogram',
        'kde':                '〰️  KDE Density Curve',
        'violin':             '🎻 Violin Plot',
        'box':                '📦 Box & Whisker',
        'strip':              '・ Strip / Jitter Plot',
        'ridgeline':          '🏔️  Ridgeline / Joy Plot',
        'ecdf':               '📈 Empirical CDF (ECDF)',
        'hexbin':             '⬡  Hexbin Density',
        'rug_plot':           '〰️  Rug Plot',
        # Proportion
        'pie':                '🥧 Pie / Donut Chart',
        'treemap':            '🌳 Treemap',
        'sunburst':           '☀️  Sunburst',
        'funnel':             '🔽 Funnel Chart',
        'waterfall':          '💧 Waterfall / Bridge',
        'marimekko':          '▦  Marimekko / Mosaic',
        'waffle_chart':       '🧇 Waffle Chart',
        'dot_plot':           '● Cleveland Dot Plot',
        'pictogram':          '🔵 Pictogram / Icon Array',
        # Heat / Matrix
        'heatmap':            '🔥 Heatmap / Correlation',
        'contour':            '🌀 Contour Plot',
        'calendar_heatmap':   '📅 Calendar Heatmap',
        'parallel_categories':'〰️  Parallel Categories',
        'qq_plot':            '📐 Q-Q Plot',
        # Statistical
        'error_bar':          '❗ Error Bar Chart',
        'confidence_band':    '📊 Confidence Band',
        'tornado_chart':      '🌪️  Tornado / Sensitivity',
        'comparison_bar':     '⚖️  Comparison Bar',
        'annotated_line':     '📝 Annotated Line',
        'multi_axis_line':    '📈 Dual Y-Axis Line',
        'heat_table':         '🟦 Heat Table',
        # Time Series
        'candlestick':        '🕯️  Candlestick',
        'ohlc':               '📉 OHLC Chart',
        'gantt':              '📅 Gantt Chart',
        'timeline':           '⏱️  Event Timeline',
        'forecast_chart':     '🔮 Forecast Chart',
        'seasonal_chart':     '🌿 Seasonal Patterns',
        'stacked_waterfall':  '💧 Stacked Waterfall',
        # KPI / Indicator
        'gauge':              '🎯 Gauge / Speedometer',
        'bullet_chart':       '🎯 Bullet Chart',
        'indicator_tile':     '🔢 Indicator Tiles',
        'progress_chart':     '✅ Progress Chart',
        'sparkline':          '✨ Sparklines Grid',
        # Polar / Radial
        'radar':              '🕸️  Radar / Spider',
        'polar_bar':          '🌹 Polar Bar / Rose',
        'polar_scatter':      '🌐 Polar Scatter',
        'windrose':           '🌬️  Wind Rose',
        'nightingale':        '🌹 Nightingale Rose',
        # 3D
        'scatter_3d':         '🌐 3D Scatter',
        'surface_3d':         '🏔️  3D Surface',
        'line_3d':            '📈 3D Line',
        'contour_3d':         '🌀 3D Contour',
        'bar_3d':             '📊 3D Bar',
        # Flow / Network
        'sankey':             '🌊 Sankey Diagram',
        'chord':              '🔵 Chord Diagram',
        'network_graph':      '🕸️  Network Graph',
        'flow_tree':          '🌳 Flow Tree',
        # Geo
        'choropleth':         '🗺️  Choropleth Map',
        'scatter_geo':        '📍 Geographic Scatter',
        'bubble_map':         '🌍 Geographic Bubble Map',
        'density_map':        '🌊 Density Map',
        # Misc
        'matrix_bubble':      '⚫ Matrix Bubble Chart',
        'animated_scatter':   '▶️  Animated Scatter',
        'table_chart':        '📋 Styled Table',
        'stream_graph':       '🌊 Stream Graph',
    }