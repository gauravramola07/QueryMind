# components/chart_generator.py
# ============================================
# CHART GENERATOR - PREMIUM EDITION
# Beautiful, interactive Plotly charts
# ============================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# PREMIUM COLOR PALETTES
# ─────────────────────────────────────────────

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

GLASSMORPHIC_BG = 'rgba(255, 255, 255, 0.05)'
CHART_FONT = 'Inter, Arial, sans-serif'


# ─────────────────────────────────────────────
# MAIN FUNCTION: Generate Chart
# ─────────────────────────────────────────────

def generate_chart(result_df, question, chart_type=None):
    """
    Main function - generates premium chart
    """
    try:
        if result_df is None or result_df.empty:
            return {
                'success': False,
                'figure': None,
                'chart_type': None,
                'error': "No data to visualize"
            }

        if chart_type is None or chart_type == 'auto':
            chart_type = detect_chart_type(result_df, question)

        print(f"📊 Generating premium {chart_type} chart...")

        chart_creators = {
            'bar': create_bar_chart,
            'line': create_line_chart,
            'pie': create_pie_chart,
            'scatter': create_scatter_chart,
            'histogram': create_histogram,
            'area': create_area_chart,
        }

        creator = chart_creators.get(chart_type, create_bar_chart)
        fig = creator(result_df, question)

        if fig is None:
            return {
                'success': False,
                'figure': None,
                'chart_type': chart_type,
                'error': "Could not create chart"
            }

        fig = apply_premium_styling(fig, chart_type)

        return {
            'success': True,
            'figure': fig,
            'chart_type': chart_type,
            'error': None
        }

    except Exception as e:
        print(f"❌ Chart error: {e}")
        return {
            'success': False,
            'figure': None,
            'chart_type': chart_type,
            'error': str(e)
        }


# ─────────────────────────────────────────────
# CHART TYPE DETECTION
# ─────────────────────────────────────────────

def detect_chart_type(result_df, question):
    """Intelligently detect best chart type"""

    question_lower = question.lower()
    num_rows = len(result_df)

    numeric_cols = result_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    text_cols = result_df.select_dtypes(
        include=['object']
    ).columns.tolist()

    # Keyword detection
    if any(w in question_lower for w in [
        'trend', 'over time', 'monthly', 'yearly',
        'daily', 'growth', 'timeline', 'by month',
        'by year', 'by date'
    ]):
        return 'line'

    if any(w in question_lower for w in [
        'distribution', 'share', 'percentage',
        'proportion', 'breakdown', 'composition'
    ]):
        return 'pie' if num_rows <= 8 else 'bar'

    if any(w in question_lower for w in [
        'correlation', 'relationship', 'vs',
        'scatter', 'between'
    ]):
        return 'scatter' if len(numeric_cols) >= 2 else 'bar'

    if any(w in question_lower for w in [
        'histogram', 'frequency'
    ]):
        return 'histogram'

    # Structure based
    if num_rows <= 8 and len(text_cols) == 1 and len(numeric_cols) == 1:
        return 'pie'

    if num_rows <= 30:
        return 'bar'

    return 'bar'


# ─────────────────────────────────────────────
# CHART 1: Premium Bar Chart
# ─────────────────────────────────────────────

def create_bar_chart(df, question):
    """Create premium gradient bar chart"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            return None

        if text_cols and numeric_cols:
            x_col = text_cols[0]
            y_col = numeric_cols[0]

            df_sorted = df.sort_values(y_col, ascending=True).tail(20)
            n = len(df_sorted)

            # Create gradient colors
            colors = []
            for i in range(n):
                ratio = i / max(n - 1, 1)
                r = int(102 + ratio * (118 - 102))
                g = int(126 + ratio * (75 - 126))
                b = int(234 + ratio * (162 - 234))
                colors.append(f'rgb({r},{g},{b})')

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=df_sorted[y_col],
                y=df_sorted[x_col],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                    cornerradius=6
                ),
                text=[f'{v:,.0f}' for v in df_sorted[y_col]],
                textposition='outside',
                textfont=dict(size=11, color='#4a5568'),
                hovertemplate=(
                    f'<b>%{{y}}</b><br>'
                    f'{format_col_name(y_col)}: '
                    f'%{{x:,.2f}}<extra></extra>'
                )
            ))

            fig.update_layout(
                title=dict(text=format_title(question)),
                xaxis_title=format_col_name(y_col),
                yaxis_title='',
                showlegend=False,
            )

            return fig

        elif len(numeric_cols) >= 2:
            fig = go.Figure()
            for i, col in enumerate(numeric_cols[:4]):
                fig.add_trace(go.Bar(
                    name=format_col_name(col),
                    y=df[col].head(20),
                    marker=dict(
                        color=NEON_COLORS[i],
                        cornerradius=5
                    ),
                    hovertemplate=(
                        f'<b>{format_col_name(col)}</b><br>'
                        f'Value: %{{y:,.2f}}<extra></extra>'
                    )
                ))

            fig.update_layout(
                title=dict(text=format_title(question)),
                barmode='group',
                bargap=0.15,
            )
            return fig

        return None

    except Exception as e:
        print(f"❌ Bar chart error: {e}")
        return None


# ─────────────────────────────────────────────
# CHART 2: Premium Line Chart
# ─────────────────────────────────────────────

def create_line_chart(df, question):
    """Create premium line chart with gradient fill"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        x_col = non_numeric[0] if non_numeric else None
        y_col = numeric_cols[0]

        if len(df) > 100:
            df = df.head(100)

        fig = go.Figure()

        x_data = df[x_col] if x_col else list(range(len(df)))

        # Main line
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df[y_col],
            mode='lines+markers',
            name=format_col_name(y_col),
            line=dict(
                color='#667eea',
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=8,
                color='#667eea',
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.08)',
            hovertemplate=(
                f'<b>%{{x}}</b><br>'
                f'{format_col_name(y_col)}: '
                f'%{{y:,.2f}}<extra></extra>'
            )
        ))

        # Add more lines if multiple numeric cols
        for i, col in enumerate(numeric_cols[1:3], 1):
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df[col],
                mode='lines+markers',
                name=format_col_name(col),
                line=dict(
                    color=NEON_COLORS[i + 2],
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=7,
                    color=NEON_COLORS[i + 2],
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    f'<b>%{{x}}</b><br>'
                    f'{format_col_name(col)}: '
                    f'%{{y:,.2f}}<extra></extra>'
                )
            ))

        fig.update_layout(
            title=dict(text=format_title(question)),
            xaxis_title=format_col_name(x_col) if x_col else '',
            yaxis_title=format_col_name(y_col),
            hovermode='x unified',
        )

        return fig

    except Exception as e:
        print(f"❌ Line chart error: {e}")
        return None


# ─────────────────────────────────────────────
# CHART 3: Premium Pie / Donut Chart
# ─────────────────────────────────────────────

def create_pie_chart(df, question):
    """Create premium donut chart with glow effect"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        if not numeric_cols:
            return None

        names_col = text_cols[0] if text_cols else None
        values_col = numeric_cols[0]

        if names_col is None:
            df = df.copy()
            df['Category'] = [f"Item {i+1}" for i in range(len(df))]
            names_col = 'Category'

        # Limit to top 8 + Others
        if len(df) > 8:
            df_top = df.nlargest(7, values_col).copy()
            others = df.nsmallest(len(df) - 7, values_col)[values_col].sum()
            others_row = pd.DataFrame({
                names_col: ['Others'],
                values_col: [others]
            })
            df = pd.concat([df_top, others_row], ignore_index=True)

        n = len(df)
        colors = NEON_COLORS[:n]

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=df[names_col],
            values=df[values_col],
            hole=0.55,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=3)
            ),
            textinfo='percent+label',
            textposition='outside',
            textfont=dict(size=12, color='#4a5568'),
            hovertemplate=(
                '<b>%{label}</b><br>'
                'Value: %{value:,.0f}<br>'
                'Share: %{percent}<extra></extra>'
            ),
            pull=[0.03] * n,
            rotation=45
        ))

        # Center text
        total = df[values_col].sum()
        fig.add_annotation(
            text=(
                f"<b style='font-size:18px'>"
                f"{format_number(total)}</b><br>"
                f"<span style='font-size:11px; color:#718096'>"
                f"Total {format_col_name(values_col)}</span>"
            ),
            showarrow=False,
            font=dict(size=14, color='#2d3748')
        )

        fig.update_layout(
            title=dict(text=format_title(question)),
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.05,
                font=dict(size=11)
            )
        )

        return fig

    except Exception as e:
        print(f"❌ Pie chart error: {e}")
        return None


# ─────────────────────────────────────────────
# CHART 4: Premium Scatter Plot
# ─────────────────────────────────────────────

def create_scatter_chart(df, question):
    """Create premium scatter plot"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) < 2:
            return create_bar_chart(df, question)

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        color_col = text_cols[0] if text_cols else None

        if len(df) > 500:
            df = df.sample(500, random_state=42)

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col,
            title=format_title(question),
            color_discrete_sequence=NEON_COLORS,
            opacity=0.7,
            size_max=15
        )

        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1, color='white')
            )
        )

        fig.update_layout(
            xaxis_title=format_col_name(x_col),
            yaxis_title=format_col_name(y_col),
        )

        return fig

    except Exception as e:
        print(f"❌ Scatter error: {e}")
        return None


# ─────────────────────────────────────────────
# CHART 5: Premium Histogram
# ─────────────────────────────────────────────

def create_histogram(df, question):
    """Create premium histogram"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None

        col = numeric_cols[0]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=30,
            marker=dict(
                color='rgba(102, 126, 234, 0.7)',
                line=dict(color='#667eea', width=1)
            ),
            hovertemplate=(
                f'{format_col_name(col)}: '
                f'%{{x:,.2f}}<br>'
                f'Count: %{{y}}<extra></extra>'
            )
        ))

        # Mean line
        mean_val = df[col].mean()
        fig.add_vline(
            x=mean_val,
            line_dash='dash',
            line_color='#ef4444',
            line_width=2,
            annotation_text=f'Mean: {mean_val:,.2f}',
            annotation_position='top right',
            annotation_font=dict(size=12, color='#ef4444')
        )

        # Median line
        median_val = df[col].median()
        fig.add_vline(
            x=median_val,
            line_dash='dot',
            line_color='#f59e0b',
            line_width=2,
            annotation_text=f'Median: {median_val:,.2f}',
            annotation_position='top left',
            annotation_font=dict(size=12, color='#f59e0b')
        )

        fig.update_layout(
            title=dict(text=format_title(question)),
            xaxis_title=format_col_name(col),
            yaxis_title='Frequency',
            bargap=0.05,
        )

        return fig

    except Exception as e:
        print(f"❌ Histogram error: {e}")
        return None


# ─────────────────────────────────────────────
# CHART 6: Premium Area Chart
# ─────────────────────────────────────────────

def create_area_chart(df, question):
    """Create premium area chart"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_cols:
            return None

        x_col = non_numeric[0] if non_numeric else None
        y_col = numeric_cols[0]

        x_data = df[x_col] if x_col else list(range(len(df)))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_data,
            y=df[y_col],
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.15)',
            line=dict(color='#667eea', width=3),
            mode='lines',
            name=format_col_name(y_col)
        ))

        fig.update_layout(
            title=dict(text=format_title(question)),
            xaxis_title=format_col_name(x_col) if x_col else '',
            yaxis_title=format_col_name(y_col),
        )

        return fig

    except Exception as e:
        print(f"❌ Area chart error: {e}")
        return None


# ─────────────────────────────────────────────
# KPI Dashboard
# ─────────────────────────────────────────────

def create_kpi_dashboard(kpis):
    """Create KPI indicator cards"""
    try:
        if not kpis:
            return None

        kpis = kpis[:6]
        n = len(kpis)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'indicator'}] * cols for _ in range(rows)]
        )

        for i, kpi in enumerate(kpis):
            r = (i // cols) + 1
            c = (i % cols) + 1
            fig.add_trace(go.Indicator(
                mode='number',
                value=float(kpi['value']),
                title={'text': kpi['label'], 'font': {'size': 14}},
                number={
                    'font': {'size': 28, 'color': NEON_COLORS[i]},
                    'valueformat': ',.2f'
                }
            ), row=r, col=c)

        fig.update_layout(
            height=200 * rows,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    except Exception as e:
        return None


# ─────────────────────────────────────────────
# PREMIUM STYLING
# ─────────────────────────────────────────────

def apply_premium_styling(fig, chart_type='bar'):
    """Apply premium glassmorphic dark styling to blend with the UI"""

    fig.update_layout(
        template='plotly_dark', # Switch to dark template
        height=500,
        font=dict(
            family=CHART_FONT,
            size=12,
            color='#e2e8f0' # Light text for dark background
        ),
        title=dict(
            font=dict(
                size=18,
                color='#f7fafc',
                family=CHART_FONT
            ),
            x=0.5,
            xanchor='center',
            y=0.97
        ),
        # Make backgrounds completely transparent!
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 26, 0.95)',
            bordercolor='#667eea',
            font_size=13,
            font_family=CHART_FONT,
            font_color='#f7fafc'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(102,126,234,0.2)',
            borderwidth=1,
            font=dict(size=11, color='#e2e8f0')
        )
    )

    # Subtle, faint grid styling so it doesn't overpower the neon lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.05)', # Faint white
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.1)',
        zeroline=False,
        title_font=dict(size=13, color='#a78bfa'), # Purple accent
        tickfont=dict(size=11, color='#cbd5e1')
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.05)',
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.1)',
        zeroline=False,
        title_font=dict(size=13, color='#a78bfa'),
        tickfont=dict(size=11, color='#cbd5e1')
    )

    return fig

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def format_title(question):
    title = question.strip()
    title = title[0].upper() + title[1:] if title else ""
    if len(title) > 80:
        title = title[:77] + "..."
    return title


def format_col_name(col_name):
    if not col_name:
        return ""
    return col_name.replace('_', ' ').title()


def format_number(value):
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


def get_chart_type_options():
    return {
        'auto': '🤖 Auto Detect',
        'bar': '📊 Bar Chart',
        'line': '📈 Line Chart',
        'pie': '🥧 Pie Chart',
        'scatter': '🔵 Scatter Plot',
        'histogram': '📉 Histogram',
        'area': '🏔️ Area Chart',
    }

# ─────────────────────────────────────────────
# AUTO-GENERATED VISUAL ANALYTICS (Smart EDA)
# ─────────────────────────────────────────────

def generate_auto_business_visualizations(df, column_categories):
    """
    Intelligently generates crucial business charts based on schema.
    It ignores IDs and focuses on Trends, Top Performers, and Market Share.
    """
    charts = []

    # Safely extract column types
    date_cols = column_categories.get('date_columns', [])
    num_cols = column_categories.get('numeric_columns', [])
    cat_cols = column_categories.get('categorical_columns', [])
    id_cols = set(column_categories.get('id_columns', []))

    # Filter out IDs from numerics and categoricals
    valid_nums = [c for c in num_cols if c not in id_cols]
    valid_cats = [c for c in cat_cols if c not in id_cols]

    if not valid_nums:
        return charts # Cannot plot without metrics

    # ── Hunt for Primary Business Metrics ──
    business_kws = ['revenue', 'sales', 'profit', 'amount', 'quantity', 'margin']
    primary_metric = valid_nums[0]
    secondary_metric = valid_nums[1] if len(valid_nums) > 1 else None

    # Try to find a financial metric as the primary driver
    for col in valid_nums:
        if any(kw in col.lower() for kw in business_kws):
            primary_metric = col
            break

    # Try to find a secondary metric for correlations
    for col in valid_nums:
        if col != primary_metric and any(kw in col.lower() for kw in business_kws):
            secondary_metric = col
            break

    # 1. TIME SERIES TREND (Crucial for tracking growth)
    if date_cols:
        d_col = date_cols[0]
        temp_df = df.copy()
        temp_df[d_col] = pd.to_datetime(temp_df[d_col], errors='coerce')
        # Group by Month
        trend_df = temp_df.groupby(temp_df[d_col].dt.to_period('M'))[primary_metric].sum().reset_index()
        trend_df[d_col] = trend_df[d_col].astype(str) # Plotly needs strings
        
        if not trend_df.empty and len(trend_df) > 1:
            fig = px.line(
                trend_df, x=d_col, y=primary_metric, 
                title=f"Monthly Trend: Total {format_col_name(primary_metric)}", 
                markers=True, color_discrete_sequence=[NEON_COLORS[0]]
            )
            fig.update_traces(fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)')
            charts.append({
                "title": "📈 Business Growth Trajectory", 
                "desc": f"Analyzes the historical momentum of {format_col_name(primary_metric)} over time.", 
                "fig": apply_premium_styling(fig, 'line')
            })

    # 2. CATEGORICAL BREAKDOWN (Crucial for identifying Top Performers)
    if valid_cats:
        cat_col = valid_cats[0] # e.g., 'Region' or 'Product'
        bar_df = df.groupby(cat_col)[primary_metric].sum().reset_index().nlargest(10, primary_metric)
        
        if not bar_df.empty:
            fig = px.bar(
                bar_df, x=primary_metric, y=cat_col, orientation='h', 
                title=f"Top 10 {format_col_name(cat_col)}s by {format_col_name(primary_metric)}", 
                color=primary_metric, 
                # FIX: Changed COLORS to GRADIENT_COLORS to match the premium theme
                color_continuous_scale=[GRADIENT_COLORS[0], GRADIENT_COLORS[1]]
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
            charts.append({
                "title": "🏆 Top Performers Analysis", 
                "desc": f"Identifies which {format_col_name(cat_col)} drives the most {format_col_name(primary_metric)}.", 
                "fig": apply_premium_styling(fig, 'bar')
            })

        # 3. SECOND CATEGORY / MARKET SHARE (Donut Chart)
        if len(valid_cats) > 1:
            cat_col2 = valid_cats[1]
            pie_df = df.groupby(cat_col2)[primary_metric].sum().reset_index()
            
            # Limit to top 5 + 'Others' for a clean, readable pie chart
            if len(pie_df) > 5:
                top5 = pie_df.nlargest(5, primary_metric)
                others_val = pie_df.nsmallest(len(pie_df)-5, primary_metric)[primary_metric].sum()
                others_df = pd.DataFrame({cat_col2: ['Others'], primary_metric: [others_val]})
                pie_df = pd.concat([top5, others_df], ignore_index=True)
                
            fig = px.pie(
                pie_df, names=cat_col2, values=primary_metric, hole=0.45, 
                title=f"Share of {format_col_name(primary_metric)} by {format_col_name(cat_col2)}", 
                color_discrete_sequence=NEON_COLORS
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            charts.append({
                "title": "🥧 Market Share & Composition", 
                "desc": f"Shows the proportional breakdown of {format_col_name(primary_metric)} across {format_col_name(cat_col2)}.", 
                "fig": apply_premium_styling(fig, 'pie')
            })

    # 4. CORRELATION (Scatter Plot)
    if primary_metric and secondary_metric:
        # Sample data if too large to prevent UI lag
        scatter_df = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
        fig = px.scatter(
            scatter_df, x=primary_metric, y=secondary_metric, 
            title=f"Correlation: {format_col_name(primary_metric)} vs {format_col_name(secondary_metric)}", 
            opacity=0.7, color_discrete_sequence=[NEON_COLORS[2]]
        )
        charts.append({
            "title": "🔗 Metric Correlation", 
            "desc": f"Reveals how {format_col_name(primary_metric)} and {format_col_name(secondary_metric)} affect one another.", 
            "fig": apply_premium_styling(fig, 'scatter')
        })

    return charts