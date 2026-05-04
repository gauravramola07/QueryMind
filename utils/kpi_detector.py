# utils/kpi_detector.py
# ============================================
# KPI DETECTOR
# Auto-detects and displays business KPIs
# ============================================

import pandas as pd
import numpy as np

from .helpers import (
    detect_column_categories,
    detect_business_kpis,
    format_kpi_value
)


def get_all_kpis(df, file_info):
    """
    Get all detected KPIs for the dataset
    
    Args:
        df: pandas DataFrame
        file_info: dict from generate_file_info()
    
    Returns:
        dict: All KPI information
    """
    # Detect column categories
    column_categories = detect_column_categories(df)
    
    # Detect KPIs
    kpis = detect_business_kpis(df, column_categories)
    
    # Add record count KPI
    kpis.insert(0, {
        'column': 'total_records',
        'label': 'Total Records',
        'value': len(df),
        'aggregation': 'COUNT',
        'format': 'number',
        'formatted_value': f"{len(df):,}"
    })
    
    return {
        'kpis': kpis,
        'column_categories': column_categories,
        'total_kpis': len(kpis)
    }