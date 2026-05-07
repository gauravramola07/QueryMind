# components/chart_generator.py
# ============================================================
# CHART GENERATOR — ENHANCED EDITION
# 12 interactive Plotly chart types + Smart Auto-Detection
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    config = None  # Graceful fallback if config not available


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

# Extended palette for charts that need many distinct colors
EXTENDED_COLORS = NEON_COLORS + [
    '#84cc16', '#eab308', '#14b8a6', '#0ea5e9',
    '#d946ef', '#fb7185', '#a3e635', '#facc15'
]

CHART_FONT = 'Inter, Arial, sans-serif'

# Supported chart types (used by get_chart_type_options & dispatch)
SUPPORTED_CHART_TYPES = [
    'bar', 'line', 'pie', 'scatter', 'histogram',
    'area', 'treemap', 'heatmap', 'funnel',
    'box', 'bubble', 'waterfall'
]


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_chart(result_df: pd.DataFrame, question: str, chart_type: str = None) -> dict:
    """
    Main function — validates data, picks a chart type, builds and styles the figure.

    Returns:
        dict with keys: success (bool), figure (go.Figure|None),
                        chart_type (str|None), error (str|None)
    """
    try:
        if result_df is None or result_df.empty:
            return _fail("No data to visualize", chart_type)

        # Resolve chart type
        if not chart_type or chart_type == 'auto':
            chart_type = detect_chart_type(result_df, question)

        print(f"📊 Generating premium '{chart_type}' chart...")

        chart_creators = {
            'bar':       create_bar_chart,
            'line':      create_line_chart,
            'pie':       create_pie_chart,
            'scatter':   create_scatter_chart,
            'histogram': create_histogram,
            'area':      create_area_chart,
            'treemap':   create_treemap,
            'heatmap':   create_heatmap,
            'funnel':    create_funnel_chart,
            'box':       create_box_plot,
            'bubble':    create_bubble_chart,
            'waterfall': create_waterfall_chart,
        }

        creator = chart_creators.get(chart_type, create_bar_chart)
        fig = creator(result_df, question)

        if fig is None:
            return _fail("Chart creator returned no figure", chart_type)

        fig = apply_premium_styling(fig, chart_type)

        return {'success': True, 'figure': fig, 'chart_type': chart_type, 'error': None}

    except Exception as exc:
        print(f"❌ Chart error: {exc}")
        return _fail(str(exc), chart_type)


def _fail(msg: str, chart_type) -> dict:
    return {'success': False, 'figure': None, 'chart_type': chart_type, 'error': msg}


# ─────────────────────────────────────────────────────────────
# CHART TYPE DETECTION
# ─────────────────────────────────────────────────────────────

def detect_chart_type(result_df: pd.DataFrame, question: str) -> str:
    """
    Intelligently selects the best chart type based on:
    1. Keywords in the question
    2. Shape and column types of the DataFrame
    """
    q = question.lower()
    num_rows = len(result_df)

    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols    = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    n_num        = len(numeric_cols)
    n_txt        = len(text_cols)

    # ── Explicit keyword overrides ────────────────────────────
    time_kws    = ['trend', 'over time', 'monthly', 'yearly', 'daily',
                   'growth', 'timeline', 'by month', 'by year', 'by date', 'week']
    area_kws    = ['area', 'cumulative', 'filled', 'stacked']
    dist_kws    = ['distribution', 'share', 'percentage', 'proportion',
                   'breakdown', 'composition', 'ratio']
    scatter_kws = ['correlation', 'relationship', 'vs', 'scatter', 'between', 'compare']
    hist_kws    = ['histogram', 'frequency', 'spread', 'range', 'density']
    treemap_kws = ['hierarchy', 'treemap', 'nested', 'drill', 'tree']
    heatmap_kws = ['heatmap', 'heat map', 'matrix', 'intensity', 'frequency map']
    funnel_kws  = ['funnel', 'conversion', 'pipeline', 'stages', 'drop-off', 'dropout']
    box_kws     = ['box', 'boxplot', 'quartile', 'outlier', 'whisker', 'iqr', 'variance']
    bubble_kws  = ['bubble', 'size', 'magnitude', 'weight', 'three variables', '3 variables']
    waterfall_kws = ['waterfall', 'bridge', 'variance', 'contribution', 'delta', 'change']

    if any(w in q for w in waterfall_kws):
        return 'waterfall'
    if any(w in q for w in heatmap_kws):
        return 'heatmap'
    if any(w in q for w in treemap_kws):
        return 'treemap'
    if any(w in q for w in funnel_kws):
        return 'funnel'
    if any(w in q for w in box_kws):
        return 'box'
    if any(w in q for w in bubble_kws) and n_num >= 3:
        return 'bubble'
    if any(w in q for w in hist_kws):
        return 'histogram'
    if any(w in q for w in scatter_kws):
        return 'scatter' if n_num >= 2 else 'bar'
    if any(w in q for w in dist_kws):
        return 'pie' if num_rows <= 8 else 'bar'
    if any(w in q for w in time_kws):
        return 'area' if any(w in q for w in area_kws) else 'line'

    # ── Structure-based fallback ───────────────────────────────
    if n_num >= 3 and n_txt >= 1:
        return 'bubble'
    if n_num >= 2 and n_txt == 0:
        return 'heatmap' if num_rows >= 5 else 'bar'
    if num_rows <= 8 and n_txt == 1 and n_num == 1:
        return 'pie'
    if num_rows <= 30:
        return 'bar'

    return 'bar'


# ─────────────────────────────────────────────────────────────
# CHART 1 — Bar Chart (horizontal gradient)
# ─────────────────────────────────────────────────────────────

def create_bar_chart(df: pd.DataFrame, question: str):
    """Horizontal gradient bar chart, auto-sorts top 20 rows."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        if text_cols:
            x_col = text_cols[0]
            y_col = numeric_cols[0]
            df_plot = df[[x_col, y_col]].dropna().sort_values(y_col, ascending=True).tail(20)
            n = len(df_plot)

            # Interpolate indigo→purple gradient
            colors = [
                f'rgb({int(102 + i/(max(n-1,1))*(118-102))},'
                f'{int(126 + i/(max(n-1,1))*(75-126))},'
                f'{int(234 + i/(max(n-1,1))*(162-234))})'
                for i in range(n)
            ]

            fig = go.Figure(go.Bar(
                x=df_plot[y_col],
                y=df_plot[x_col],
                orientation='h',
                marker=dict(color=colors, line=dict(width=0), cornerradius=6),
                text=[f'{v:,.0f}' for v in df_plot[y_col]],
                textposition='outside',
                textfont=dict(size=11, color='#cbd5e1'),
                hovertemplate=f'<b>%{{y}}</b><br>{format_col_name(y_col)}: %{{x:,.2f}}<extra></extra>'
            ))
            fig.update_layout(
                title=format_title(question),
                xaxis_title=format_col_name(y_col),
                yaxis_title='',
                showlegend=False,
            )
            return fig

        # Multiple numeric columns — grouped bar
        fig = go.Figure()
        for i, col in enumerate(numeric_cols[:6]):
            fig.add_trace(go.Bar(
                name=format_col_name(col),
                y=df[col].head(20),
                marker=dict(color=NEON_COLORS[i % len(NEON_COLORS)], cornerradius=5),
                hovertemplate=f'<b>{format_col_name(col)}</b><br>Value: %{{y:,.2f}}<extra></extra>'
            ))
        fig.update_layout(
            title=format_title(question),
            barmode='group',
            bargap=0.15,
        )
        return fig

    except Exception as exc:
        print(f"❌ Bar chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 2 — Line Chart (spline + gradient fill)
# ─────────────────────────────────────────────────────────────

def create_line_chart(df: pd.DataFrame, question: str):
    """Smooth spline line with subtle gradient fill and up to 3 series."""
    try:
        numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric   = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        x_col  = non_numeric[0] if non_numeric else None
        x_data = df[x_col] if x_col else list(range(len(df)))
        df     = df.head(100)

        fig = go.Figure()

        for i, col in enumerate(numeric_cols[:3]):
            fill_color = 'rgba(102,126,234,0.08)' if i == 0 else 'rgba(0,0,0,0)'
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df[col],
                mode='lines+markers',
                name=format_col_name(col),
                line=dict(color=NEON_COLORS[i], width=3, shape='spline'),
                marker=dict(size=7, color=NEON_COLORS[i], line=dict(width=2, color='white')),
                fill='tozeroy' if i == 0 else 'none',
                fillcolor=fill_color,
                hovertemplate=f'<b>%{{x}}</b><br>{format_col_name(col)}: %{{y:,.2f}}<extra></extra>'
            ))

        fig.update_layout(
            title=format_title(question),
            xaxis_title=format_col_name(x_col) if x_col else '',
            yaxis_title=format_col_name(numeric_cols[0]),
            hovermode='x unified',
        )
        return fig

    except Exception as exc:
        print(f"❌ Line chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 3 — Pie / Donut Chart
# ─────────────────────────────────────────────────────────────

def create_pie_chart(df: pd.DataFrame, question: str):
    """Donut chart, collapses small slices into 'Others', shows total in center."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        names_col  = text_cols[0] if text_cols else None
        values_col = numeric_cols[0]

        df = df[[names_col, values_col]].dropna() if names_col else df[[values_col]].dropna()

        if names_col is None:
            df = df.copy()
            df['Category'] = [f'Item {i+1}' for i in range(len(df))]
            names_col = 'Category'

        # Consolidate beyond top 7 → Others
        if len(df) > 8:
            df_top = df.nlargest(7, values_col).copy()
            others_val = df.nsmallest(len(df) - 7, values_col)[values_col].sum()
            df = pd.concat(
                [df_top, pd.DataFrame({names_col: ['Others'], values_col: [others_val]})],
                ignore_index=True
            )

        n = len(df)
        total = df[values_col].sum()

        fig = go.Figure(go.Pie(
            labels=df[names_col],
            values=df[values_col],
            hole=0.55,
            marker=dict(colors=NEON_COLORS[:n], line=dict(color='rgba(255,255,255,0.1)', width=2)),
            textinfo='percent+label',
            textposition='outside',
            textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>',
            pull=[0.03] * n,
            rotation=45,
        ))

        fig.add_annotation(
            text=f"<b>{format_number(total)}</b><br>"
                 f"<span style='font-size:11px'>Total {format_col_name(values_col)}</span>",
            showarrow=False,
            font=dict(size=14, color='#e2e8f0')
        )

        fig.update_layout(
            title=format_title(question),
            showlegend=True,
            legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05),
        )
        return fig

    except Exception as exc:
        print(f"❌ Pie chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 4 — Scatter Plot
# ─────────────────────────────────────────────────────────────

def create_scatter_chart(df: pd.DataFrame, question: str):
    """Scatter plot, colored by categorical column when available."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(numeric_cols) < 2:
            return create_bar_chart(df, question)

        x_col     = numeric_cols[0]
        y_col     = numeric_cols[1]
        color_col = text_cols[0] if text_cols else None

        if len(df) > 500:
            df = df.sample(500, random_state=42)

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col,
            title=format_title(question),
            color_discrete_sequence=NEON_COLORS,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,0.3)')))
        fig.update_layout(
            xaxis_title=format_col_name(x_col),
            yaxis_title=format_col_name(y_col),
        )
        return fig

    except Exception as exc:
        print(f"❌ Scatter error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 5 — Histogram
# ─────────────────────────────────────────────────────────────

def create_histogram(df: pd.DataFrame, question: str):
    """Distribution histogram with mean & median reference lines."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None

        col = numeric_cols[0]
        data = df[col].dropna()

        fig = go.Figure(go.Histogram(
            x=data,
            nbinsx=min(50, max(10, len(data) // 10)),
            marker=dict(color='rgba(99,102,241,0.7)', line=dict(color='#6366f1', width=1)),
            hovertemplate=f'{format_col_name(col)}: %{{x:,.2f}}<br>Count: %{{y}}<extra></extra>'
        ))

        mean_val   = data.mean()
        median_val = data.median()

        fig.add_vline(x=mean_val,   line_dash='dash', line_color='#ef4444', line_width=2,
                      annotation_text=f'Mean: {mean_val:,.2f}',
                      annotation_position='top right',
                      annotation_font=dict(size=12, color='#ef4444'))
        fig.add_vline(x=median_val, line_dash='dot',  line_color='#f59e0b', line_width=2,
                      annotation_text=f'Median: {median_val:,.2f}',
                      annotation_position='top left',
                      annotation_font=dict(size=12, color='#f59e0b'))

        fig.update_layout(
            title=format_title(question),
            xaxis_title=format_col_name(col),
            yaxis_title='Frequency',
            bargap=0.04,
        )
        return fig

    except Exception as exc:
        print(f"❌ Histogram error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 6 — Area Chart
# ─────────────────────────────────────────────────────────────

def create_area_chart(df: pd.DataFrame, question: str):
    """Filled area chart with gradient fill, ideal for cumulative trends."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric  = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        x_col  = non_numeric[0] if non_numeric else None
        x_data = df[x_col] if x_col else list(range(len(df)))

        fig = go.Figure()

        fill_colors = [
            'rgba(99,102,241,0.2)',
            'rgba(139,92,246,0.15)',
            'rgba(6,182,212,0.15)',
        ]

        for i, col in enumerate(numeric_cols[:3]):
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df[col],
                fill='tozeroy' if i == 0 else 'tonexty',
                fillcolor=fill_colors[i],
                line=dict(color=NEON_COLORS[i], width=2.5),
                mode='lines',
                name=format_col_name(col),
                hovertemplate=f'<b>%{{x}}</b><br>{format_col_name(col)}: %{{y:,.2f}}<extra></extra>'
            ))

        fig.update_layout(
            title=format_title(question),
            xaxis_title=format_col_name(x_col) if x_col else '',
            yaxis_title=format_col_name(numeric_cols[0]),
            hovermode='x unified',
        )
        return fig

    except Exception as exc:
        print(f"❌ Area chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 7 — Treemap (NEW)
# ─────────────────────────────────────────────────────────────

def create_treemap(df: pd.DataFrame, question: str):
    """
    Hierarchical treemap.
    Works with 1 or 2 categorical columns + 1 numeric.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols or not text_cols:
            return create_bar_chart(df, question)

        value_col = numeric_cols[0]

        if len(text_cols) >= 2:
            parent_col = text_cols[0]
            child_col  = text_cols[1]
            df_plot    = df[[parent_col, child_col, value_col]].dropna()
            df_plot    = df_plot[df_plot[value_col] > 0]

            fig = px.treemap(
                df_plot,
                path=[px.Constant('All'), parent_col, child_col],
                values=value_col,
                title=format_title(question),
                color=value_col,
                color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3], GRADIENT_COLORS[5]],
            )
        else:
            label_col = text_cols[0]
            df_plot   = df[[label_col, value_col]].dropna()
            df_plot   = df_plot[df_plot[value_col] > 0]

            fig = px.treemap(
                df_plot,
                path=[px.Constant('All'), label_col],
                values=value_col,
                title=format_title(question),
                color=value_col,
                color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3], GRADIENT_COLORS[5]],
            )

        fig.update_traces(
            textinfo='label+value+percent root',
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percentRoot:.1%}<extra></extra>',
            marker=dict(line=dict(width=2, color='rgba(0,0,0,0.3)'))
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    except Exception as exc:
        print(f"❌ Treemap error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 8 — Heatmap (NEW)
# ─────────────────────────────────────────────────────────────

def create_heatmap(df: pd.DataFrame, question: str):
    """
    Correlation heatmap for all-numeric DataFrames,
    or a pivot-based frequency heatmap for mixed data.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # ── Correlation matrix (all numeric) ─────────────────
        if len(numeric_cols) >= 2 and len(text_cols) < 2:
            corr = df[numeric_cols].corr()
            labels = [format_col_name(c) for c in corr.columns]

            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale=[
                    [0.0, '#ef4444'],
                    [0.5, '#1e1b4b'],
                    [1.0, '#6366f1'],
                ],
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=11),
                hovertemplate='%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title='r', tickfont=dict(color='#e2e8f0')),
            ))
            fig.update_layout(
                title=format_title(question) or 'Correlation Matrix',
                xaxis=dict(side='bottom'),
            )
            return fig

        # ── Pivot heatmap (2 categorical + 1 numeric) ────────
        if len(text_cols) >= 2 and numeric_cols:
            row_col = text_cols[0]
            col_col = text_cols[1]
            val_col = numeric_cols[0]

            pivot = (
                df.groupby([row_col, col_col])[val_col]
                  .sum()
                  .unstack(fill_value=0)
            )

            # Limit size for readability
            pivot = pivot.iloc[:20, :20]

            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale=[[0, '#0f0c29'], [0.5, '#302b63'], [1, '#667eea']],
                hovertemplate=f'{format_col_name(row_col)}: %{{y}}<br>'
                              f'{format_col_name(col_col)}: %{{x}}<br>'
                              f'{format_col_name(val_col)}: %{{z:,.0f}}<extra></extra>',
                colorbar=dict(tickfont=dict(color='#e2e8f0')),
            ))
            fig.update_layout(
                title=format_title(question),
                xaxis_title=format_col_name(col_col),
                yaxis_title=format_col_name(row_col),
            )
            return fig

        return create_bar_chart(df, question)

    except Exception as exc:
        print(f"❌ Heatmap error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 9 — Funnel Chart (NEW)
# ─────────────────────────────────────────────────────────────

def create_funnel_chart(df: pd.DataFrame, question: str):
    """
    Conversion/pipeline funnel chart.
    Uses first text column as stage labels and first numeric as values.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        value_col = numeric_cols[0]
        label_col = text_cols[0] if text_cols else None

        df_plot = df[[label_col, value_col]].dropna() if label_col else df[[value_col]].dropna()
        df_plot = df_plot.sort_values(value_col, ascending=False).head(12)

        if label_col is None:
            df_plot = df_plot.copy()
            df_plot['Stage'] = [f'Stage {i+1}' for i in range(len(df_plot))]
            label_col = 'Stage'

        n = len(df_plot)
        colors = NEON_COLORS[:n]

        fig = go.Figure(go.Funnel(
            y=df_plot[label_col],
            x=df_plot[value_col],
            textinfo='value+percent initial+percent previous',
            marker=dict(color=colors, line=dict(width=1.5, color='rgba(255,255,255,0.1)')),
            connector=dict(line=dict(color='rgba(255,255,255,0.05)', width=2)),
            hovertemplate=f'<b>%{{y}}</b><br>{format_col_name(value_col)}: %{{x:,.0f}}<br>'
                          f'% of Total: %{{percentInitial}}<extra></extra>',
        ))

        fig.update_layout(
            title=format_title(question),
            funnelmode='stack',
        )
        return fig

    except Exception as exc:
        print(f"❌ Funnel chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 10 — Box Plot (NEW)
# ─────────────────────────────────────────────────────────────

def create_box_plot(df: pd.DataFrame, question: str):
    """
    Box-and-whisker plot for distribution comparison.
    Groups by categorical column when available, otherwise plots each numeric column.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        fig = go.Figure()

        if text_cols and len(numeric_cols) >= 1:
            # One numeric, grouped by categorical
            group_col = text_cols[0]
            val_col   = numeric_cols[0]
            groups    = df[group_col].dropna().unique()[:12]  # limit categories

            for i, grp in enumerate(groups):
                subset = df[df[group_col] == grp][val_col].dropna()
                fig.add_trace(go.Box(
                    y=subset,
                    name=str(grp),
                    marker_color=NEON_COLORS[i % len(NEON_COLORS)],
                    boxmean='sd',  # show mean + std deviation marker
                    hovertemplate=f'<b>{grp}</b><br>'
                                  f'Median: %{{median:,.2f}}<br>'
                                  f'Q1: %{{q1:,.2f}}<br>'
                                  f'Q3: %{{q3:,.2f}}<extra></extra>'
                ))
            fig.update_layout(
                title=format_title(question),
                yaxis_title=format_col_name(val_col),
                xaxis_title=format_col_name(group_col),
                boxgap=0.3,
            )

        else:
            # Multiple numeric columns side by side
            for i, col in enumerate(numeric_cols[:8]):
                fig.add_trace(go.Box(
                    y=df[col].dropna(),
                    name=format_col_name(col),
                    marker_color=NEON_COLORS[i % len(NEON_COLORS)],
                    boxmean='sd',
                ))
            fig.update_layout(
                title=format_title(question),
                yaxis_title='Value',
                boxgap=0.3,
            )

        return fig

    except Exception as exc:
        print(f"❌ Box plot error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 11 — Bubble Chart (NEW)
# ─────────────────────────────────────────────────────────────

def create_bubble_chart(df: pd.DataFrame, question: str):
    """
    3-dimensional scatter where bubble size encodes a third numeric variable.
    Color encodes a fourth categorical dimension when available.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(numeric_cols) < 2:
            return create_scatter_chart(df, question)

        x_col     = numeric_cols[0]
        y_col     = numeric_cols[1]
        size_col  = numeric_cols[2] if len(numeric_cols) >= 3 else None
        color_col = text_cols[0] if text_cols else None

        if len(df) > 300:
            df = df.sample(300, random_state=42)

        # Normalize bubble sizes to keep them visually meaningful
        size_data = None
        if size_col:
            raw = df[size_col].fillna(0)
            min_v, max_v = raw.min(), raw.max()
            if max_v > min_v:
                size_data = 6 + 44 * (raw - min_v) / (max_v - min_v)
            else:
                size_data = [20] * len(df)

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            size=size_col if size_col else None,
            color=color_col,
            title=format_title(question),
            color_discrete_sequence=NEON_COLORS,
            size_max=55,
            opacity=0.75,
            hover_data=df.columns.tolist(),
        )
        fig.update_traces(
            marker=dict(line=dict(width=1, color='rgba(255,255,255,0.25)'))
        )
        fig.update_layout(
            xaxis_title=format_col_name(x_col),
            yaxis_title=format_col_name(y_col),
        )
        return fig

    except Exception as exc:
        print(f"❌ Bubble chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CHART 12 — Waterfall Chart (NEW)
# ─────────────────────────────────────────────────────────────

def create_waterfall_chart(df: pd.DataFrame, question: str):
    """
    Waterfall / bridge chart — ideal for showing how individual components
    contribute to a running total (e.g. revenue build-up, cost variances).
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols    = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        value_col = numeric_cols[0]
        label_col = text_cols[0] if text_cols else None

        df_plot = df.head(20).copy()

        if label_col:
            labels = df_plot[label_col].astype(str).tolist()
        else:
            labels = [f'Step {i+1}' for i in range(len(df_plot))]

        values = df_plot[value_col].fillna(0).tolist()

        # Classify bars: first and last are totals; middle are relative deltas
        measure = ['relative'] * len(values)
        if len(values) > 1:
            measure[0]  = 'absolute'
            measure[-1] = 'total'

        # Color by positive / negative
        colors = []
        for i, m in enumerate(measure):
            if m == 'total':
                colors.append('#6366f1')
            elif m == 'absolute':
                colors.append('#8b5cf6')
            elif values[i] >= 0:
                colors.append('#34d399')   # green for positive
            else:
                colors.append('#ef4444')   # red for negative

        fig = go.Figure(go.Waterfall(
            name='',
            orientation='v',
            measure=measure,
            x=labels,
            y=values,
            connector=dict(line=dict(color='rgba(255,255,255,0.15)', width=1.5, dash='dot')),
            increasing=dict(marker=dict(color='#34d399', line=dict(width=0))),
            decreasing=dict(marker=dict(color='#ef4444', line=dict(width=0))),
            totals=dict(marker=dict(color='#6366f1',   line=dict(width=0))),
            textposition='outside',
            text=[f'{v:+,.0f}' if m == 'relative' else f'{v:,.0f}' for v, m in zip(values, measure)],
            hovertemplate='<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>',
        ))

        fig.update_layout(
            title=format_title(question),
            xaxis_title=format_col_name(label_col) if label_col else '',
            yaxis_title=format_col_name(value_col),
            showlegend=False,
        )
        return fig

    except Exception as exc:
        print(f"❌ Waterfall chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# KPI DASHBOARD (unchanged API, bug-fixed)
# ─────────────────────────────────────────────────────────────

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
            r = (i // cols) + 1
            c = (i % cols) + 1
            fig.add_trace(go.Indicator(
                mode='number',
                value=float(kpi['value']),
                title={'text': kpi['label'], 'font': {'size': 14, 'color': '#e2e8f0'}},
                number={'font': {'size': 28, 'color': NEON_COLORS[i % len(NEON_COLORS)]},
                        'valueformat': ',.2f'},
            ), row=r, col=c)

        fig.update_layout(
            height=200 * rows,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family=CHART_FONT, color='#e2e8f0'),
        )
        return fig

    except Exception as exc:
        print(f"❌ KPI dashboard error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# PREMIUM STYLING (applied to every chart)
# ─────────────────────────────────────────────────────────────

def apply_premium_styling(fig: go.Figure, chart_type: str = 'bar') -> go.Figure:
    """Apply dark glassmorphic styling consistent with the QueryMind UI."""

    fig.update_layout(
        template='plotly_dark',
        height=500,
        font=dict(family=CHART_FONT, size=12, color='#e2e8f0'),
        title=dict(
            font=dict(size=18, color='#f7fafc', family=CHART_FONT),
            x=0.5, xanchor='center', y=0.97,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(
            bgcolor='rgba(10,10,26,0.95)',
            bordercolor='#667eea',
            font_size=13,
            font_family=CHART_FONT,
            font_color='#f7fafc',
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(102,126,234,0.2)',
            borderwidth=1,
            font=dict(size=11, color='#e2e8f0'),
        ),
    )

    axis_style = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.05)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.1)',
        zeroline=False,
        title_font=dict(size=13, color='#a78bfa'),
        tickfont=dict(size=11, color='#cbd5e1'),
    )

    # Axes are not present on pie / treemap / funnel — guard with try
    try:
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
    except Exception:
        pass

    return fig


# ─────────────────────────────────────────────────────────────
# AUTO BUSINESS VISUALIZATIONS (Smart EDA — enhanced)
# ─────────────────────────────────────────────────────────────

def generate_auto_business_visualizations(df: pd.DataFrame, column_categories: dict) -> list:
    """
    Generates up to 5 business-focused charts automatically based on schema.
    Returns a list of dicts: {title, desc, fig}
    """
    charts = []

    date_cols  = column_categories.get('date_columns', [])
    num_cols   = column_categories.get('numeric_columns', [])
    cat_cols   = column_categories.get('categorical_columns', [])
    id_cols    = set(column_categories.get('id_columns', []))

    valid_nums = [c for c in num_cols if c not in id_cols]
    valid_cats = [c for c in cat_cols if c not in id_cols]

    if not valid_nums:
        return charts

    # ── Identify primary & secondary business metrics ─────────
    business_kws = ['revenue', 'sales', 'profit', 'amount', 'quantity', 'margin', 'cost', 'price']
    primary_metric   = valid_nums[0]
    secondary_metric = valid_nums[1] if len(valid_nums) > 1 else None

    for col in valid_nums:
        if any(kw in col.lower() for kw in business_kws):
            primary_metric = col
            break

    for col in valid_nums:
        if col != primary_metric and any(kw in col.lower() for kw in business_kws):
            secondary_metric = col
            break

    # 1. TIME SERIES TREND ─────────────────────────────────────
    if date_cols:
        try:
            d_col    = date_cols[0]
            temp_df  = df.copy()
            temp_df[d_col] = pd.to_datetime(temp_df[d_col], errors='coerce')
            trend_df = (
                temp_df.groupby(temp_df[d_col].dt.to_period('M'))[primary_metric]
                       .sum()
                       .reset_index()
            )
            trend_df[d_col] = trend_df[d_col].astype(str)

            if len(trend_df) > 1:
                fig = px.line(
                    trend_df, x=d_col, y=primary_metric,
                    title=f'Monthly Trend: {format_col_name(primary_metric)}',
                    markers=True,
                    color_discrete_sequence=[NEON_COLORS[0]],
                )
                fig.update_traces(fill='tozeroy', fillcolor='rgba(99,102,241,0.1)')
                charts.append({
                    'title': '📈 Business Growth Trajectory',
                    'desc':  f'Tracks {format_col_name(primary_metric)} momentum over time.',
                    'fig':   apply_premium_styling(fig, 'line'),
                })
        except Exception as exc:
            print(f"Auto-viz trend error: {exc}")

    # 2. TOP PERFORMERS — Bar ──────────────────────────────────
    if valid_cats:
        try:
            cat_col = valid_cats[0]
            bar_df  = (
                df.groupby(cat_col)[primary_metric]
                  .sum()
                  .reset_index()
                  .nlargest(10, primary_metric)
            )
            if not bar_df.empty:
                fig = px.bar(
                    bar_df, x=primary_metric, y=cat_col, orientation='h',
                    title=f'Top 10 {format_col_name(cat_col)}s by {format_col_name(primary_metric)}',
                    color=primary_metric,
                    color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[3]],
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  coloraxis_showscale=False)
                charts.append({
                    'title': '🏆 Top Performers',
                    'desc':  f'Which {format_col_name(cat_col)} drives the most {format_col_name(primary_metric)}.',
                    'fig':   apply_premium_styling(fig, 'bar'),
                })
        except Exception as exc:
            print(f"Auto-viz top performers error: {exc}")

    # 3. MARKET SHARE — Donut ─────────────────────────────────
    if len(valid_cats) > 1:
        try:
            cat_col2 = valid_cats[1]
            pie_df   = df.groupby(cat_col2)[primary_metric].sum().reset_index()

            if len(pie_df) > 5:
                top5      = pie_df.nlargest(5, primary_metric)
                others_v  = pie_df.nsmallest(len(pie_df) - 5, primary_metric)[primary_metric].sum()
                pie_df    = pd.concat(
                    [top5, pd.DataFrame({cat_col2: ['Others'], primary_metric: [others_v]})],
                    ignore_index=True,
                )

            fig = px.pie(
                pie_df, names=cat_col2, values=primary_metric, hole=0.45,
                title=f'{format_col_name(primary_metric)} by {format_col_name(cat_col2)}',
                color_discrete_sequence=NEON_COLORS,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            charts.append({
                'title': '🥧 Market Share',
                'desc':  f'Proportional breakdown across {format_col_name(cat_col2)}.',
                'fig':   apply_premium_styling(fig, 'pie'),
            })
        except Exception as exc:
            print(f"Auto-viz market share error: {exc}")

    # 4. CORRELATION — Scatter ────────────────────────────────
    if primary_metric and secondary_metric:
        try:
            scatter_df = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
            color_col  = valid_cats[0] if valid_cats else None
            fig = px.scatter(
                scatter_df, x=primary_metric, y=secondary_metric,
                color=color_col,
                title=f'Correlation: {format_col_name(primary_metric)} vs {format_col_name(secondary_metric)}',
                opacity=0.7,
                color_discrete_sequence=NEON_COLORS,
            )
            charts.append({
                'title': '🔗 Metric Correlation',
                'desc':  f'How {format_col_name(primary_metric)} and {format_col_name(secondary_metric)} relate.',
                'fig':   apply_premium_styling(fig, 'scatter'),
            })
        except Exception as exc:
            print(f"Auto-viz scatter error: {exc}")

    # 5. DISTRIBUTION — Box Plot ──────────────────────────────
    if valid_cats and len(valid_nums) >= 1:
        try:
            cat_col  = valid_cats[0]
            val_col  = primary_metric
            groups   = df[cat_col].dropna().value_counts().head(8).index.tolist()
            df_box   = df[df[cat_col].isin(groups)]

            fig = go.Figure()
            for i, grp in enumerate(groups):
                subset = df_box[df_box[cat_col] == grp][val_col].dropna()
                fig.add_trace(go.Box(
                    y=subset, name=str(grp),
                    marker_color=NEON_COLORS[i % len(NEON_COLORS)],
                    boxmean='sd',
                ))
            fig.update_layout(
                title=f'Distribution of {format_col_name(val_col)} by {format_col_name(cat_col)}',
                yaxis_title=format_col_name(val_col),
                boxgap=0.3,
            )
            charts.append({
                'title': '📦 Distribution Analysis',
                'desc':  f'Spread and outliers in {format_col_name(val_col)} per {format_col_name(cat_col)}.',
                'fig':   apply_premium_styling(fig, 'box'),
            })
        except Exception as exc:
            print(f"Auto-viz box plot error: {exc}")

    return charts


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def format_title(question: str) -> str:
    """Capitalize and trim the question for use as a chart title."""
    title = (question or '').strip()
    if title:
        title = title[0].upper() + title[1:]
    return title[:77] + '...' if len(title) > 80 else title


def format_col_name(col_name) -> str:
    """Convert snake_case / snake case to Title Case."""
    if not col_name:
        return ''
    return str(col_name).replace('_', ' ').title()


def format_number(value) -> str:
    """Abbreviate large numbers: 1200000 → 1.2M."""
    try:
        value   = float(value)
        abs_val = abs(value)
        sign    = '-' if value < 0 else ''
        if abs_val >= 1_000_000_000:
            return f'{sign}{abs_val/1_000_000_000:.1f}B'
        if abs_val >= 1_000_000:
            return f'{sign}{abs_val/1_000_000:.1f}M'
        if abs_val >= 1_000:
            return f'{sign}{abs_val/1_000:.1f}K'
        return f'{value:,.0f}'
    except (TypeError, ValueError):
        return str(value)


def get_chart_type_options() -> dict:
    """Return display-ready chart type options for a Streamlit selectbox."""
    return {
        'auto':      '🤖 Auto Detect',
        'bar':       '📊 Bar Chart',
        'line':      '📈 Line Chart',
        'pie':       '🥧 Pie / Donut Chart',
        'scatter':   '🔵 Scatter Plot',
        'histogram': '📉 Histogram',
        'area':      '🏔️  Area Chart',
        'treemap':   '🌳 Treemap',
        'heatmap':   '🔥 Heatmap',
        'funnel':    '🔽 Funnel Chart',
        'box':       '📦 Box Plot',
        'bubble':    '🫧  Bubble Chart',
        'waterfall': '💧 Waterfall Chart',
    }