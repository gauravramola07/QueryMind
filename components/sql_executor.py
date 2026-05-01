# components/sql_executor.py
# ============================================
# SQL EXECUTOR COMPONENT
# Fixed version - handles Streamlit reruns
# ============================================

import pandas as pd
import numpy as np
import sqlite3
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# GLOBAL CONNECTION (persists in memory)
# ─────────────────────────────────────────────
_conn = None
_current_df = None


# ─────────────────────────────────────────────
# MAIN FUNCTION 1: Load DataFrame into SQLite
# ─────────────────────────────────────────────

def load_dataframe_to_db(df):
    """
    Load pandas DataFrame into SQLite in-memory database
    Uses a persistent connection stored globally
    """
    global _conn, _current_df

    try:
        print(f"🗄️ Loading DataFrame into SQLite...")
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── Close existing connection ──────────────
        if _conn is not None:
            try:
                _conn.close()
            except:
                pass

        # ── Create new persistent connection ──────
        _conn = sqlite3.connect(':memory:', check_same_thread=False)

        # ── Load DataFrame ─────────────────────────
        df.to_sql(
            name=config.DB_TABLE_NAME,
            con=_conn,
            index=False,
            if_exists='replace'
        )

        # ── Store reference ────────────────────────
        _current_df = df.copy()

        # ── Verify ────────────────────────────────
        cursor = _conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {config.DB_TABLE_NAME}")
        row_count = cursor.fetchone()[0]

        print(f"✅ Data loaded into SQLite!")
        print(f"📊 Rows in DB: {row_count:,}")

        return {
            'success': True,
            'message': f"✅ {row_count:,} rows loaded successfully!",
            'table_name': config.DB_TABLE_NAME,
            'row_count': row_count
        }

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}",
            'table_name': None,
            'row_count': 0
        }


# ─────────────────────────────────────────────
# MAIN FUNCTION 2: Execute SQL Query
# ─────────────────────────────────────────────

def execute_sql_query(sql_query):
    """
    Execute SQL query - auto-reloads DB if needed
    """
    global _conn, _current_df

    # ── Auto-reload if connection lost ─────────
    if _conn is None:
        if _current_df is not None:
            print("🔄 Reconnecting to database...")
            load_dataframe_to_db(_current_df)
        else:
            return {
                'success': False,
                'dataframe': None,
                'error': "❌ No data loaded. Please upload a file first.",
                'row_count': 0,
                'query_used': sql_query
            }

    try:
        # ── Verify table exists ────────────────────
        cursor = _conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (config.DB_TABLE_NAME,)
        )
        table_exists = cursor.fetchone()

        # ── Reload if table missing ────────────────
        if not table_exists:
            if _current_df is not None:
                print("🔄 Table missing - reloading data...")
                _current_df.to_sql(
                    name=config.DB_TABLE_NAME,
                    con=_conn,
                    index=False,
                    if_exists='replace'
                )
            else:
                return {
                    'success': False,
                    'dataframe': None,
                    'error': "❌ No data found. Please re-upload your file.",
                    'row_count': 0,
                    'query_used': sql_query
                }

        # ── Clean SQL ──────────────────────────────
        clean_query = clean_sql_query(sql_query)
        print(f"🔍 Executing: {clean_query[:100]}...")

        # ── Safety check ───────────────────────────
        safety = is_query_safe(clean_query)
        if not safety['safe']:
            return {
                'success': False,
                'dataframe': None,
                'error': f"❌ Unsafe query: {safety['reason']}",
                'row_count': 0,
                'query_used': clean_query
            }

        # ── Execute query ──────────────────────────
        result_df = pd.read_sql_query(clean_query, _conn)

        print(f"✅ Query returned {len(result_df)} rows")

        return {
            'success': True,
            'dataframe': result_df,
            'error': None,
            'row_count': len(result_df),
            'query_used': clean_query
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Query error: {error_msg}")
        return {
            'success': False,
            'dataframe': None,
            'error': get_friendly_error(error_msg),
            'row_count': 0,
            'query_used': sql_query
        }


# ─────────────────────────────────────────────
# HELPER: Ensure DB is loaded
# ─────────────────────────────────────────────

def ensure_db_loaded(df):
    """
    Call this before every query to make sure
    database is loaded with current DataFrame
    """
    global _conn, _current_df

    needs_reload = False

    # Check if connection exists
    if _conn is None:
        needs_reload = True

    # Check if table exists
    if not needs_reload:
        try:
            cursor = _conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (config.DB_TABLE_NAME,)
            )
            if not cursor.fetchone():
                needs_reload = True
        except:
            needs_reload = True

    # Reload if needed
    if needs_reload and df is not None:
        print("🔄 Reloading database...")
        return load_dataframe_to_db(df)

    return {'success': True, 'message': 'DB already loaded'}


# ─────────────────────────────────────────────
# HELPER: Clean SQL Query
# ─────────────────────────────────────────────

def clean_sql_query(sql_query):
    """Clean SQL query from LLM output"""

    # Remove markdown
    sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'```\s*', '', sql_query)
    sql_query = sql_query.strip().rstrip(';')

    # Fix wrong table names
    wrong_names = [
        'your_table', 'mytable', 'data_table',
        'the_table', 'csv_data', 'excel_data',
        'my_table', 'table_name', 'dataset'
    ]
    for wrong in wrong_names:
        sql_query = re.sub(
            rf'\b{wrong}\b',
            config.DB_TABLE_NAME,
            sql_query,
            flags=re.IGNORECASE
        )

    return sql_query


# ─────────────────────────────────────────────
# HELPER: Safety Check
# ─────────────────────────────────────────────

def is_query_safe(sql_query):
    """Only allow SELECT queries"""

    query_upper = sql_query.upper().strip()

    if not query_upper.startswith('SELECT'):
        return {
            'safe': False,
            'reason': "Only SELECT queries allowed."
        }

    dangerous = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE',
        'CREATE', 'ALTER', 'TRUNCATE', 'EXEC'
    ]

    for kw in dangerous:
        if re.search(rf'\b{kw}\b', query_upper):
            return {
                'safe': False,
                'reason': f"Keyword '{kw}' not allowed."
            }

    return {'safe': True, 'reason': 'Safe'}


# ─────────────────────────────────────────────
# HELPER: Friendly Errors
# ─────────────────────────────────────────────

def get_friendly_error(error_msg):
    """Convert SQL errors to friendly messages"""

    error_lower = error_msg.lower()

    if 'no such table' in error_lower:
        return "❌ Table not found. Please re-upload your file."
    elif 'no such column' in error_lower:
        col = re.search(r'no such column: (\w+)', error_lower)
        col_name = col.group(1) if col else 'unknown'
        return f"❌ Column '{col_name}' not found in your data."
    elif 'syntax error' in error_lower:
        return "❌ SQL syntax error. Please rephrase your question."
    else:
        return f"❌ Query error: {error_msg}"


# ─────────────────────────────────────────────
# HELPER: Get Table Info
# ─────────────────────────────────────────────

def get_table_info():
    """Get current database table info"""
    global _conn

    if _conn is None:
        return None

    try:
        cursor = _conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {config.DB_TABLE_NAME}")
        row_count = cursor.fetchone()[0]

        cursor.execute(f"PRAGMA table_info({config.DB_TABLE_NAME})")
        columns = cursor.fetchall()

        return {
            'table_name': config.DB_TABLE_NAME,
            'row_count': row_count,
            'columns': columns,
            'num_columns': len(columns)
        }
    except Exception as e:
        print(f"❌ Error getting table info: {e}")
        return None


# ─────────────────────────────────────────────
# HELPER: Reset Database
# ─────────────────────────────────────────────

def reset_database():
    """Reset the database"""
    global _conn, _current_df

    if _conn:
        try:
            _conn.close()
        except:
            pass

    _conn = None
    _current_df = None
    print("🔄 Database reset successfully")
    return True


# ─────────────────────────────────────────────
# HELPER: Get Sample Results
# ─────────────────────────────────────────────

def get_sample_query_results():
    """Get sample data preview"""

    result = execute_sql_query(
        f"SELECT * FROM {config.DB_TABLE_NAME} LIMIT 5"
    )

    if result['success']:
        return {'preview': result['dataframe']}
    return None