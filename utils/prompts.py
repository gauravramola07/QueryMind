# utils/prompts.py
# ============================================
# ALL PROMPT TEMPLATES
# Centralized prompt management
# ============================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── System Prompt ─────────────────────────────
SYSTEM_PROMPT = f"""
You are QueryMind, an expert Business Intelligence analyst AI.
You help business executives and analysts query data using natural language.
You are professional, concise, and focused on business insights.
Always refer to the table as '{config.DB_TABLE_NAME}'.
"""

# ── SQL Generation Prompt Template ────────────
SQL_GENERATION_TEMPLATE = """
Convert this business question into a SQL query:
Question: {question}

Schema: {schema}

Rules:
- Only SELECT queries
- Table name: {table_name}
- SQLite syntax only
- Add ROUND() for decimals
- Add meaningful aliases
"""

# ── Insight Generation Prompt ─────────────────
INSIGHT_TEMPLATE = """
As a Business Intelligence analyst, provide insights for:
Question: {question}
Data: {data}

Format:
- Key Finding (1 sentence)
- 2-3 bullet points
- Business recommendation
"""

# ── Summary Prompt ────────────────────────────
SUMMARY_TEMPLATE = """
Provide an executive summary for this dataset:
Schema: {schema}
KPIs: {kpis}

Include: Overview, highlights, recommendations.
Max 200 words. Professional tone.
"""