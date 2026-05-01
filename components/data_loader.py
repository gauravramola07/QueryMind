# components/data_loader.py
# ============================================
# DATA LOADER COMPONENT
# Handles CSV/Excel file upload and parsing
# ============================================

import pandas as pd
import numpy as np
import io
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# MAIN FUNCTION: Load uploaded file
# ─────────────────────────────────────────────

def load_file(uploaded_file):
    """
    Main function to load CSV or Excel file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        dict: {
            'success': True/False,
            'dataframe': pd.DataFrame or None,
            'error': error message or None,
            'file_info': dict with file details
        }
    """
    try:
        # ── STEP 1: Get file details ──────────────
        file_name = uploaded_file.name
        file_size = uploaded_file.size  # in bytes
        file_extension = os.path.splitext(file_name)[1].lower()
        
        print(f"📁 Loading file: {file_name}")
        print(f"📊 File size: {file_size / 1024:.2f} KB")
        print(f"📌 File type: {file_extension}")
        
        # ── STEP 2: Validate file format ─────────
        if file_extension not in config.SUPPORTED_FORMATS:
            return {
                'success': False,
                'dataframe': None,
                'error': f"❌ Unsupported format: {file_extension}. Please upload CSV or Excel files only.",
                'file_info': None
            }
        
        # ── STEP 3: Validate file size ───────────
        max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            return {
                'success': False,
                'dataframe': None,
                'error': f"❌ File too large. Maximum size is {config.MAX_FILE_SIZE_MB}MB.",
                'file_info': None
            }
        
        # ── STEP 4: Read file into DataFrame ─────
        df = read_file(uploaded_file, file_extension)
        
        # ── STEP 5: Validate DataFrame ───────────
        if df is None:
            return {
                'success': False,
                'dataframe': None,
                'error': "❌ Could not read file. Please check if file is corrupted.",
                'file_info': None
            }
        
        if df.empty:
            return {
                'success': False,
                'dataframe': None,
                'error': "❌ File is empty. Please upload a file with data.",
                'file_info': None
            }
        
        # ── STEP 6: Clean the DataFrame ──────────
        df = clean_dataframe(df)
        
        # ── STEP 7: Generate file info ────────────
        file_info = generate_file_info(df, file_name, file_size, file_extension)
        
        print(f"✅ File loaded successfully!")
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return {
            'success': True,
            'dataframe': df,
            'error': None,
            'file_info': file_info
        }
    
    except Exception as e:
        return {
            'success': False,
            'dataframe': None,
            'error': f"❌ Unexpected error: {str(e)}",
            'file_info': None
        }


# ─────────────────────────────────────────────
# HELPER FUNCTION 1: Read file based on extension
# ─────────────────────────────────────────────

def read_file(uploaded_file, file_extension):
    """
    Read uploaded file into pandas DataFrame
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        file_extension: '.csv', '.xlsx', or '.xls'
    
    Returns:
        pd.DataFrame or None
    """
    try:
        if file_extension == '.csv':
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(
                        uploaded_file,
                        encoding=encoding,
                        on_bad_lines='skip',    # Skip problematic rows
                        low_memory=False        # Better type detection
                    )
                    print(f"✅ CSV read with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"⚠️ Error with {encoding}: {e}")
                    continue
            
            return None
        
        elif file_extension in ['.xlsx', '.xls']:
            uploaded_file.seek(0)
            
            # Read Excel - get all sheet names first
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            print(f"📋 Excel sheets found: {sheet_names}")
            
            # Read the first sheet by default
            uploaded_file.seek(0)
            df = pd.read_excel(
                uploaded_file,
                sheet_name=0,       # First sheet
                engine='openpyxl' if file_extension == '.xlsx' else 'xlrd'
            )
            
            print(f"✅ Excel read - Sheet: {sheet_names[0]}")
            return df
        
        return None
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


# ─────────────────────────────────────────────
# HELPER FUNCTION 2: Clean the DataFrame
# ─────────────────────────────────────────────

def clean_dataframe(df):
    """
    Clean and prepare the DataFrame
    
    - Strips whitespace from column names
    - Removes completely empty rows/columns
    - Fixes column name formatting
    - Converts obvious date columns
    
    Args:
        df: Raw pandas DataFrame
    
    Returns:
        Cleaned pandas DataFrame
    """
    # ── Clean column names ────────────────────
    # Remove spaces, special characters
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
    df.columns = df.columns.str.lower()
    
    # Handle duplicate column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i > 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols
    
    # ── Remove empty rows and columns ─────────
    df = df.dropna(how='all')       # Remove rows where ALL values are NaN
    df = df.dropna(axis=1, how='all')  # Remove columns where ALL values are NaN
    
    # ── Reset index ───────────────────────────
    df = df.reset_index(drop=True)
    
    # ── Try to convert date columns ───────────
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # ── Limit rows for performance ─────────────
    if len(df) > config.MAX_ROWS_QUERY:
        print(f"⚠️ Large file detected. Using first {config.MAX_ROWS_QUERY} rows.")
        df = df.head(config.MAX_ROWS_QUERY)
    
    return df


# ─────────────────────────────────────────────
# HELPER FUNCTION 3: Generate File Info
# ─────────────────────────────────────────────

def generate_file_info(df, file_name, file_size, file_extension):
    """
    Generate comprehensive information about the loaded file
    
    Args:
        df: Cleaned pandas DataFrame
        file_name: Original file name
        file_size: File size in bytes
        file_extension: File extension
    
    Returns:
        dict: Comprehensive file information
    """
    # ── Basic info ────────────────────────────
    num_rows, num_cols = df.shape
    
    # ── Column type analysis ──────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    # ── Missing value analysis ────────────────
    missing_info = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_info[col] = {
                'count': int(missing_count),
                'percentage': round((missing_count / num_rows) * 100, 2)
            }
    
    # ── Column details ────────────────────────
    column_details = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].count()),
            'null_count': int(df[col].isna().sum()),
            'unique_count': int(df[col].nunique()),
        }
        
        # Add stats for numeric columns
        if col in numeric_cols:
            col_info['type'] = 'Numeric'
            col_info['min'] = round(float(df[col].min()), 2) if not pd.isna(df[col].min()) else None
            col_info['max'] = round(float(df[col].max()), 2) if not pd.isna(df[col].max()) else None
            col_info['mean'] = round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else None
        elif col in date_cols:
            col_info['type'] = 'Date/Time'
            col_info['min'] = str(df[col].min())
            col_info['max'] = str(df[col].max())
        elif col in text_cols:
            col_info['type'] = 'Text'
            # Top 5 most common values
            top_values = df[col].value_counts().head(5).to_dict()
            col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        elif col in bool_cols:
            col_info['type'] = 'Boolean'
        else:
            col_info['type'] = 'Other'
        
        column_details.append(col_info)
    
    # ── Numeric summary ───────────────────────
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        numeric_summary = desc.to_dict()
    
    # ── File info dict ────────────────────────
    file_info = {
        # Basic
        'file_name': file_name,
        'file_size_kb': round(file_size / 1024, 2),
        'file_size_mb': round(file_size / (1024 * 1024), 2),
        'file_type': file_extension,
        
        # Shape
        'num_rows': num_rows,
        'num_cols': num_cols,
        
        # Column types
        'numeric_columns': numeric_cols,
        'text_columns': text_cols,
        'date_columns': date_cols,
        'bool_columns': bool_cols,
        'all_columns': df.columns.tolist(),
        
        # Quality
        'missing_info': missing_info,
        'has_missing_values': len(missing_info) > 0,
        'total_missing': int(df.isna().sum().sum()),
        'missing_percentage': round((df.isna().sum().sum() / (num_rows * num_cols)) * 100, 2),
        
        # Details
        'column_details': column_details,
        'numeric_summary': numeric_summary,
    }
    
    return file_info


# ─────────────────────────────────────────────
# HELPER FUNCTION 4: Get Schema for LLM
# ─────────────────────────────────────────────

def get_schema_for_llm(df, file_info):
    """
    Generate a text description of the dataset schema
    This is what we send to Gemini so it understands the data
    
    Args:
        df: pandas DataFrame
        file_info: dict from generate_file_info()
    
    Returns:
        str: Schema description for LLM prompt
    """
    schema_parts = []
    
    # ── Table info ────────────────────────────
    schema_parts.append(f"Table Name: {config.DB_TABLE_NAME}")
    schema_parts.append(f"Total Rows: {file_info['num_rows']:,}")
    schema_parts.append(f"Total Columns: {file_info['num_cols']}")
    schema_parts.append("")
    
    # ── Column descriptions ───────────────────
    schema_parts.append("COLUMNS:")
    schema_parts.append("-" * 50)
    
    for col_detail in file_info['column_details']:
        col_name = col_detail['name']
        col_type = col_detail['type']
        unique = col_detail['unique_count']
        null_pct = round((col_detail['null_count'] / file_info['num_rows']) * 100, 1)
        
        col_desc = f"• {col_name} ({col_type})"
        col_desc += f" | Unique: {unique}"
        col_desc += f" | Nulls: {null_pct}%"
        
        if col_type == 'Numeric':
            col_desc += f" | Range: {col_detail.get('min')} to {col_detail.get('max')}"
            col_desc += f" | Mean: {col_detail.get('mean')}"
        elif col_type == 'Text' and 'top_values' in col_detail:
            top_vals = list(col_detail['top_values'].keys())[:3]
            col_desc += f" | Sample values: {', '.join(top_vals)}"
        elif col_type == 'Date/Time':
            col_desc += f" | Range: {col_detail.get('min')} to {col_detail.get('max')}"
        
        schema_parts.append(col_desc)
    
    # ── Sample data ───────────────────────────
    schema_parts.append("")
    schema_parts.append("SAMPLE DATA (First 3 rows):")
    schema_parts.append("-" * 50)
    
    sample = df.head(3).to_string(index=False)
    schema_parts.append(sample)
    
    return "\n".join(schema_parts)


# ─────────────────────────────────────────────
# HELPER FUNCTION 5: Get Quick Stats
# ─────────────────────────────────────────────

def get_quick_stats(df):
    """
    Get quick statistics for display in UI
    
    Args:
        df: pandas DataFrame
    
    Returns:
        list of dicts with stat name, value, icon
    """
    stats = []
    
    # Total rows
    stats.append({
        'label': 'Total Rows',
        'value': f"{len(df):,}",
        'icon': '📊'
    })
    
    # Total columns
    stats.append({
        'label': 'Total Columns',
        'value': str(len(df.columns)),
        'icon': '📋'
    })
    
    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats.append({
        'label': 'Numeric Columns',
        'value': str(len(num_cols)),
        'icon': '🔢'
    })
    
    # Missing values
    missing = df.isna().sum().sum()
    stats.append({
        'label': 'Missing Values',
        'value': f"{missing:,}",
        'icon': '⚠️' if missing > 0 else '✅'
    })
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    stats.append({
        'label': 'Memory Usage',
        'value': f"{memory_mb:.2f} MB",
        'icon': '💾'
    })
    
    return stats