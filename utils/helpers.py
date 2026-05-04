# utils/helpers.py
# ============================================
# HELPER UTILITIES
# Schema detection and data profiling
# ============================================

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# FUNCTION 1: Detect Column Categories
# ─────────────────────────────────────────────

def detect_column_categories(df):
    """
    Intelligently categorize each column
    Goes beyond just dtype — understands business meaning
    
    Categories:
    - ID columns (order_id, customer_id etc.)
    - Date/Time columns
    - Numeric/Metric columns (revenue, profit etc.)
    - Categorical columns (region, product etc.)
    - Text columns (descriptions, notes)
    - Boolean columns
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict: Column categories with column names
    """
    categories = {
        'id_columns': [],
        'date_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'text_columns': [],
        'boolean_columns': [],
    }
    
    for col in df.columns:
        col_lower = col.lower()
        dtype = str(df[col].dtype)
        n_rows = len(df)
        unique_ratio = df[col].nunique() / n_rows if n_rows > 0 else 0.0
        
        # ── Boolean columns ────────────────────
        if dtype == 'bool':
            categories['boolean_columns'].append(col)
            continue
        
        # ── Date columns ───────────────────────
        if 'datetime' in dtype:
            categories['date_columns'].append(col)
            continue
        
        # ── ID columns ─────────────────────────
        # IDs usually have high uniqueness and contain 'id'
        id_keywords = ['id', 'code', 'number', 'num', 'no', 'key']
        if any(keyword in col_lower for keyword in id_keywords):
            if unique_ratio > 0.5:  # More than 50% unique = likely ID
                categories['id_columns'].append(col)
                continue
        
        # ── Numeric columns ────────────────────
        if dtype in ['int64', 'float64', 'int32', 'float32']:
            categories['numeric_columns'].append(col)
            continue
        
        # ── Categorical vs Text ────────────────
        if dtype == 'object':
            if unique_ratio < 0.1:  # Less than 10% unique = categorical
                categories['categorical_columns'].append(col)
            elif unique_ratio < 0.5:  # 10-50% unique = could be either
                # Check average string length
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:  # Long strings = text
                    categories['text_columns'].append(col)
                else:
                    categories['categorical_columns'].append(col)
            else:  # More than 50% unique = text
                categories['text_columns'].append(col)
    
    return categories


# ─────────────────────────────────────────────
# FUNCTION 2: Detect Business KPIs
# ─────────────────────────────────────────────

def detect_business_kpis(df, column_categories):
    """
    Automatically detect potential KPIs from column names
    
    Args:
        df: pandas DataFrame
        column_categories: dict from detect_column_categories()
    
    Returns:
        list: Detected KPIs with details
    """
    kpis = []
    
    # ── KPI Keywords Mapping ───────────────────
    kpi_keywords = {
        'revenue': {'label': 'Total Revenue', 'agg': 'SUM', 'format': 'currency'},
        'sales': {'label': 'Total Sales', 'agg': 'SUM', 'format': 'currency'},
        'profit': {'label': 'Total Profit', 'agg': 'SUM', 'format': 'currency'},
        'cost': {'label': 'Total Cost', 'agg': 'SUM', 'format': 'currency'},
        'price': {'label': 'Average Price', 'agg': 'AVG', 'format': 'currency'},
        'quantity': {'label': 'Total Quantity', 'agg': 'SUM', 'format': 'number'},
        'amount': {'label': 'Total Amount', 'agg': 'SUM', 'format': 'currency'},
        'count': {'label': 'Total Count', 'agg': 'COUNT', 'format': 'number'},
        'rating': {'label': 'Average Rating', 'agg': 'AVG', 'format': 'decimal'},
        'score': {'label': 'Average Score', 'agg': 'AVG', 'format': 'decimal'},
        'discount': {'label': 'Average Discount', 'agg': 'AVG', 'format': 'percentage'},
        'margin': {'label': 'Average Margin', 'agg': 'AVG', 'format': 'percentage'},
        'growth': {'label': 'Average Growth', 'agg': 'AVG', 'format': 'percentage'},
        'age': {'label': 'Average Age', 'agg': 'AVG', 'format': 'number'},
        'salary': {'label': 'Average Salary', 'agg': 'AVG', 'format': 'currency'},
        'income': {'label': 'Total Income', 'agg': 'SUM', 'format': 'currency'},
        'expense': {'label': 'Total Expenses', 'agg': 'SUM', 'format': 'currency'},
        'units': {'label': 'Total Units', 'agg': 'SUM', 'format': 'number'},
        'orders': {'label': 'Total Orders', 'agg': 'COUNT', 'format': 'number'},
        'customers': {'label': 'Total Customers', 'agg': 'COUNT', 'format': 'number'},
    }
    
    # ── Check numeric columns for KPIs ─────────
    for col in column_categories['numeric_columns']:
        col_lower = col.lower()
        
        for keyword, kpi_info in kpi_keywords.items():
            if keyword in col_lower:
                # Calculate the KPI value
                try:
                    if kpi_info['agg'] == 'SUM':
                        value = df[col].sum()
                    elif kpi_info['agg'] == 'AVG':
                        value = df[col].mean()
                    elif kpi_info['agg'] == 'COUNT':
                        value = df[col].count()
                    else:
                        value = df[col].sum()
                    
                    kpis.append({
                        'column': col,
                        'label': kpi_info['label'],
                        'value': value,
                        'aggregation': kpi_info['agg'],
                        'format': kpi_info['format'],
                        'formatted_value': format_kpi_value(value, kpi_info['format'])
                    })
                    break
                    
                except Exception as e:
                    print(f"⚠️ Could not calculate KPI for {col}: {e}")
    
    return kpis


# ─────────────────────────────────────────────
# FUNCTION 3: Format KPI Values
# ─────────────────────────────────────────────

def format_kpi_value(value, format_type):
    """
    Format KPI values for display
    
    Args:
        value: Numeric value
        format_type: 'currency', 'number', 'decimal', 'percentage'
    
    Returns:
        str: Formatted value string
    """
    try:
        if pd.isna(value):
            return "N/A"
        
        abs_val = abs(value)
        sign = "-" if value < 0 else ""

        if format_type == 'currency':
            if abs_val >= 1_000_000:
                return f"{sign}${abs_val/1_000_000:.2f}M"
            elif abs_val >= 1_000:
                return f"{sign}${abs_val/1_000:.2f}K"
            else:
                return f"${value:.2f}"
        
        elif format_type == 'number':
            if abs_val >= 1_000_000:
                return f"{sign}{abs_val/1_000_000:.2f}M"
            elif abs_val >= 1_000:
                return f"{sign}{abs_val/1_000:.2f}K"
            else:
                return f"{value:,.0f}"
        
        elif format_type == 'decimal':
            return f"{value:.2f}"
        
        elif format_type == 'percentage':
            return f"{value:.1f}%"
        
        else:
            return str(round(value, 2))
    
    except:
        return str(value)


# ─────────────────────────────────────────────
# FUNCTION 4: Generate Smart Schema for LLM
# ─────────────────────────────────────────────

def generate_smart_schema(df, file_info, column_categories):
    """
    Generate a comprehensive, smart schema description
    This is what gets sent to Gemini as context
    
    Args:
        df: pandas DataFrame
        file_info: dict from generate_file_info()
        column_categories: dict from detect_column_categories()
    
    Returns:
        str: Complete schema description for LLM
    """
    lines = []
    
    # ── Header ────────────────────────────────
    lines.append("=" * 60)
    lines.append("DATABASE SCHEMA INFORMATION")
    lines.append("=" * 60)
    lines.append(f"Table Name: {config.DB_TABLE_NAME}")
    lines.append(f"Total Rows: {file_info['num_rows']:,}")
    lines.append(f"Total Columns: {file_info['num_cols']}")
    lines.append("")
    
    # ── Column Details ────────────────────────
    lines.append("COLUMN DETAILS:")
    lines.append("-" * 60)
    
    for col_detail in file_info['column_details']:
        col_name = col_detail['name']
        col_type = col_detail['type']
        unique = col_detail['unique_count']
        nulls = col_detail['null_count']
        
        line = f"  • {col_name}"
        line += f" | Type: {col_type}"
        line += f" | Unique values: {unique:,}"
        
        if nulls > 0:
            line += f" | Missing: {nulls}"
        
        # Add type-specific info
        if col_type == 'Numeric':
            line += f" | Min: {col_detail.get('min')}"
            line += f" | Max: {col_detail.get('max')}"
            line += f" | Avg: {col_detail.get('mean')}"
        
        elif col_type == 'Text' and 'top_values' in col_detail:
            top_vals = list(col_detail['top_values'].keys())[:5]
            line += f" | Values: [{', '.join(map(str, top_vals))}]"
        
        elif col_type == 'Date/Time':
            line += f" | From: {col_detail.get('min')}"
            line += f" | To: {col_detail.get('max')}"
        
        lines.append(line)
    
    lines.append("")
    
    # ── Column Categories ─────────────────────
    lines.append("COLUMN CATEGORIES:")
    lines.append("-" * 60)
    
    if column_categories['id_columns']:
        lines.append(f"  ID Columns: {', '.join(column_categories['id_columns'])}")
    
    if column_categories['date_columns']:
        lines.append(f"  Date Columns: {', '.join(column_categories['date_columns'])}")
    
    if column_categories['numeric_columns']:
        lines.append(f"  Numeric/Metric Columns: {', '.join(column_categories['numeric_columns'])}")
    
    if column_categories['categorical_columns']:
        lines.append(f"  Categorical Columns: {', '.join(column_categories['categorical_columns'])}")
    
    if column_categories['text_columns']:
        lines.append(f"  Text Columns: {', '.join(column_categories['text_columns'])}")
    
    lines.append("")
    
    # ── Sample Data ───────────────────────────
    lines.append("SAMPLE DATA (First 5 rows):")
    lines.append("-" * 60)
    
    # Convert to string with nice formatting
    sample_df = df.head(5)
    lines.append(sample_df.to_string(index=False))
    lines.append("")
    
    # ── Important Notes for LLM ────────────────
    lines.append("IMPORTANT NOTES FOR SQL GENERATION:")
    lines.append("-" * 60)
    lines.append(f"  1. Always use table name: '{config.DB_TABLE_NAME}'")
    lines.append(f"  2. Column names are lowercase with underscores")
    lines.append(f"  3. Only generate SELECT queries (no INSERT/UPDATE/DELETE)")
    lines.append(f"  4. For date operations use SQLite date functions")
    lines.append(f"  5. Use LIMIT clause to avoid returning too many rows")
    lines.append(f"  6. Numeric columns: {', '.join(column_categories['numeric_columns'])}")
    
    if column_categories['categorical_columns']:
        lines.append(f"  7. Group by these for categories: {', '.join(column_categories['categorical_columns'])}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ─────────────────────────────────────────────
# FUNCTION 5: Generate Smart Question Suggestions
# ─────────────────────────────────────────────

def generate_smart_suggestions(df, file_info, column_categories):
    """
    Generate smart, data-aware question suggestions.
    Filters out ID columns and prioritizes real business metrics.
    """
    suggestions = []
    
    # 1. Filter out ID columns from our numeric pool
    id_cols = set(column_categories.get('id_columns', []))
    valid_numeric = [c for c in column_categories.get('numeric_columns', []) if c not in id_cols]
    cat_cols = column_categories.get('categorical_columns', [])
    date_cols = column_categories.get('date_columns', [])
    
    # Always include a summary option
    suggestions.append("Give me a complete executive summary of this dataset")
    
    if valid_numeric:
        # 2. Try to find a REAL business metric first
        business_keywords = ['revenue', 'sales', 'profit', 'amount', 'cost', 'quantity', 'price', 'total', 'margin']
        primary_metric = valid_numeric[0] # Default to first valid numeric
        
        for col in valid_numeric:
            if any(kw in col.lower() for kw in business_keywords):
                primary_metric = col
                break
                
        # Clean up the metric name for display (e.g., "unit_price" -> "Unit Price")
        metric_name = primary_metric.replace('_', ' ').title()
        
        # Add numeric-based suggestions
        suggestions.append(f"What is the total {metric_name}?")
        suggestions.append(f"Show me the top 10 records by {metric_name}")
        
        # 3. Add categorical breakdown suggestions
        if cat_cols:
            # Try to avoid ID-like categorical columns
            valid_cats = [c for c in cat_cols if c not in id_cols]
            if valid_cats:
                primary_cat = valid_cats[0].replace('_', ' ').title()
                suggestions.append(f"What is the total {metric_name} by {primary_cat}?")
                suggestions.append(f"Which {primary_cat} has the highest {metric_name}?")
                
                # If we have a second category, do a comparison
                if len(valid_cats) > 1:
                    second_cat = valid_cats[1].replace('_', ' ').title()
                    suggestions.append(f"Compare {metric_name} across different {second_cat}s")
                    
        # 4. Add time-series suggestions
        if date_cols:
            suggestions.append(f"Show me the monthly trend of {metric_name}")
            
    # 5. General fallback questions if we need more to fill out the UI
    general_questions = [
        "Show me the distribution of the data",
        "Are there any outliers in the numeric columns?",
        "What is the correlation between the numeric values?"
    ]
    
    for q in general_questions:
        if len(suggestions) < 6:
            suggestions.append(q)
            
    # Return exactly 6 unique suggestions to fit perfectly in your 2-column UI
    return list(dict.fromkeys(suggestions))[:6]


# ─────────────────────────────────────────────
# FUNCTION 6: Format DataFrame for Display
# ─────────────────────────────────────────────

def format_dataframe_for_display(df, max_rows=100):
    """
    Format DataFrame for clean display in Streamlit
    
    Args:
        df: pandas DataFrame
        max_rows: Maximum rows to display
    
    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Limit rows
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    # Round float columns to 2 decimal places
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = df[col].round(2)
    
    return df


# ─────────────────────────────────────────────
# FUNCTION 7: Detect Chart Type
# ─────────────────────────────────────────────

def detect_best_chart_type(result_df, question):
    """
    Automatically detect the best chart type for results
    
    Args:
        result_df: Query result DataFrame
        question: Original user question
    
    Returns:
        str: Chart type ('bar', 'line', 'pie', 'scatter', 'table')
    """
    question_lower = question.lower()
    num_rows = len(result_df)
    num_cols = len(result_df.columns)
    
    # ── Keyword-based detection ────────────────
    if any(word in question_lower for word in ['trend', 'over time', 'monthly', 'yearly', 'daily', 'growth']):
        return 'line'
    
    if any(word in question_lower for word in ['distribution', 'share', 'percentage', 'proportion', 'breakdown']):
        if num_rows <= 8:
            return 'pie'
        else:
            return 'bar'
    
    if any(word in question_lower for word in ['correlation', 'relationship', 'vs', 'versus', 'scatter']):
        return 'scatter'
    
    if any(word in question_lower for word in ['compare', 'comparison', 'top', 'best', 'highest', 'lowest']):
        return 'bar'
    
    # ── Structure-based detection ──────────────
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    
    if num_rows > 50:
        return 'table'  # Too many rows for chart
    
    if len(numeric_cols) >= 2 and num_cols == 2:
        return 'bar'
    
    if num_rows <= 8 and num_cols == 2:
        return 'pie'
    
    if num_rows <= 20:
        return 'bar'
    
    return 'table'  # Default to table


# ─────────────────────────────────────────────
# FUNCTION 8: Get Data Health Score
# ─────────────────────────────────────────────

def get_data_health_score(df, file_info):
    """
    Calculate a data quality/health score (0-100)
    
    Args:
        df: pandas DataFrame
        file_info: dict from generate_file_info()
    
    Returns:
        dict: Health score and breakdown
    """
    scores = {}
    
    # ── Completeness (no missing values) ───────
    missing_pct = file_info['missing_percentage']
    completeness = max(0, 100 - (missing_pct * 2))
    scores['completeness'] = round(completeness, 1)
    
    # ── Size (enough data to analyze) ──────────
    num_rows = file_info['num_rows']
    if num_rows >= 1000:
        size_score = 100
    elif num_rows >= 100:
        size_score = 70
    elif num_rows >= 10:
        size_score = 40
    else:
        size_score = 20
    scores['size'] = size_score
    
    # ── Diversity (mix of column types) ────────
    has_numeric = len(file_info['numeric_columns']) > 0
    has_categorical = len(file_info['text_columns']) > 0
    has_dates = len(file_info['date_columns']) > 0
    
    diversity = 0
    if has_numeric:
        diversity += 40
    if has_categorical:
        diversity += 40
    if has_dates:
        diversity += 20
    scores['diversity'] = diversity
    
    # ── Overall score ──────────────────────────
    overall = round(
        (scores['completeness'] * 0.4) +
        (scores['size'] * 0.3) +
        (scores['diversity'] * 0.3),
        1
    )
    
    # ── Grade ──────────────────────────────────
    if overall >= 90:
        grade = 'A'
        label = 'Excellent'
    elif overall >= 75:
        grade = 'B'
        label = 'Good'
    elif overall >= 60:
        grade = 'C'
        label = 'Fair'
    else:
        grade = 'D'
        label = 'Needs Improvement'
    
    return {
        'overall': overall,
        'grade': grade,
        'label': label,
        'breakdown': scores
    }

# ─────────────────────────────────────────────
# FUNCTION 9: Should Show Chart?
# ─────────────────────────────────────────────

def should_show_chart(result_df, question):
    """
    Intelligently decide if a chart is needed
    
    Some questions only need a text/table answer:
    - "Which product has highest X?" → Single answer
    - "What is the total revenue?" → Single number
    - "How many rows?" → Single number
    
    Returns:
        bool: True if chart should be shown
    """
    question_lower = question.lower()
    
    # ── Single answer questions (NO chart) ─────
    single_answer_keywords = [
        'which', 'what is the', 'who is', 'who has',
        'how many', 'how much', 'what was', 'when',
        'highest', 'lowest', 'maximum', 'minimum',
        'best', 'worst', 'most', 'least', 'biggest',
        'smallest', 'first', 'last', 'oldest', 'newest',
        'total number', 'count of', 'sum of',
    ]
    
    # Check if question asks for single answer
    is_single_answer = any(
        kw in question_lower for kw in single_answer_keywords
    )
    
    # ── Check result size ──────────────────────
    if result_df is None or result_df.empty:
        return False
    
    num_rows = len(result_df)
    num_cols = len(result_df.columns)
    
    # Single value → no chart
    if num_rows == 1 and num_cols <= 2:
        return False
    
    # Single row with single answer question → no chart
    if num_rows == 1 and is_single_answer:
        return False
    
    # Only 2 rows → usually not worth charting
    if num_rows <= 2 and is_single_answer:
        return False
    
    # ── Multi-row results (YES chart) ──────────
    chart_keywords = [
        'by', 'each', 'every', 'all', 'compare',
        'distribution', 'trend', 'over time',
        'group', 'breakdown', 'across', 'per',
        'top 5', 'top 10', 'top 3', 'top 20',
        'show me', 'display', 'visualize', 'chart',
        'graph', 'plot', 'monthly', 'yearly',
        'correlation', 'relationship',
    ]
    
    wants_chart = any(
        kw in question_lower for kw in chart_keywords
    )
    
    # More than 3 rows usually deserves a chart
    if num_rows >= 3 and wants_chart:
        return True
    
    # More than 5 rows → always show chart
    if num_rows >= 5:
        return True
    
    # 3-4 rows with grouping → show chart
    if num_rows >= 3 and num_cols == 2:
        return True
    
    return False