# -*- coding: utf-8 -*-
# app.py

import os
import sys

# 1. CRITICAL: Setup the path BEFORE any local imports to fix ImportError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as _st_components
import pandas as pd
import numpy as np
from datetime import datetime

# 2. Now import local components
from components.multi_file_joiner import (
    detect_joinable_columns, merge_dataframes, score_badge,
    ai_analyze_join_strategy, execute_all_join_types,
    ai_join_type_insights, ai_plan_multi_join,
)
from components.data_cleaner import auto_clean_data, generate_cleaning_report
from components.report_generator import generate_pdf_report
from utils.kpi_detector import get_all_kpis

import config

from components.data_loader import (
    load_file, get_quick_stats, get_schema_for_llm
)
from components.sql_executor import (
    load_dataframe_to_db, execute_sql_query,
    reset_database, ensure_db_loaded
)
from components.llm_engine import (
    setup_llm, generate_sql_query,
    generate_text_response, generate_data_summary
)
from components.chart_generator import (
    generate_chart, create_kpi_dashboard,
    get_chart_type_options
)

from utils.helpers import (
    detect_column_categories, generate_smart_schema,
    generate_smart_suggestions, get_data_health_score,
    format_dataframe_for_display, should_show_chart
)

# ─────────────────────────────────────────────
# PAGE CONFIG 
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="QueryMind - Talk to Your Data",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# ALL CSS STYLES
# ─────────────────────────────────────────────

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d0d2b 70%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f7fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Main Container Padding */
    /* Change from 2rem to 0rem for the first value */
    .main .block-container {
        max-width: 1200px;
        padding: 0.5rem 2rem 2rem; /* Reduced top padding */
        position: relative;
        z-index: 1;
    }

    /* ── Animated Background layer ── */
    .bg-layer {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }
    .bg-grid {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-image:
            linear-gradient(rgba(102,126,234,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(102,126,234,0.04) 1px, transparent 1px);
        background-size: 80px 80px;
    }

    /* ── Hero Section (Logo & Title) ── */
    /* Change from 2.5rem to 0.5rem */
    .hero-section {
        text-align: center;
        padding: 0.5rem 0 1rem; /* Tightened top space */
        position: relative;
    }
    .hero-logo {
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 30%, #f093fb 60%, #4facfe 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        color: transparent !important;
        margin: 0 !important;
        letter-spacing: -1px;
        line-height: 1.2 !important;
        display: inline-block; /* Forces gradient to apply to text bounds */
        filter: drop-shadow(0 0 20px rgba(102,126,234,0.3));
    }
    .hero-subtitle {
        font-size: 1.15rem !important;
        color: rgba(255,255,255,0.5) !important;
        margin: 0.5rem 0 0 !important;
        font-weight: 300 !important;
        letter-spacing: 1px;
    }
    .hero-tagline {
        font-size: 0.85rem !important;
        color: rgba(255,255,255,0.25) !important;
        margin: 0.3rem 0 0 !important;
        font-style: italic;
    }

    /* ── Navbar ── */
    .nav-bar {
        display: flex; justify-content: center; align-items: center;
        gap: 0.5rem; padding: 0.8rem 1.5rem; margin: 1.5rem auto;
        max-width: 750px; background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 50px; box-shadow: 0 4px 30px rgba(0,0,0,0.3); flex-wrap: wrap;
    }
    .nav-item {
        padding: 0.5rem 1.2rem; border-radius: 25px; font-size: 0.82rem;
        font-weight: 600; color: rgba(255,255,255,0.5) !important;
        display: inline-flex; align-items: center; gap: 0.4rem; letter-spacing: 0.5px;
    }
    .nav-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .nav-dot.green { background: #48bb78; box-shadow: 0 0 8px #48bb78; }
    .nav-dot.red { background: #fc8181; box-shadow: 0 0 8px #fc8181; }

    /* ── Glass & Features ── */
    .glass-card {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
        padding: 1.8rem; margin-bottom: 1.2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .feature-premium {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 22px;
        padding: 2.5rem 1.5rem; text-align: center; transition: all 0.4s;
    }
    .feature-premium:hover { transform: translateY(-8px); border-color: rgba(102,126,234,0.3); }
    .feat-icon { font-size: 3rem; display: block; margin-bottom: 1rem; }
    .feat-title { font-size: 1.1rem; font-weight: 700; color: #f7fafc !important; margin: 0.5rem 0; }
    .feat-desc { font-size: 0.85rem; color: rgba(255,255,255,0.4) !important; line-height: 1.6; }

    /* ── File Uploader ── */
    /* Target the main container of the uploader */
    [data-testid="stFileUploader"] {
        max-width: 600px;
        margin: 0 auto; /* Center it to match the upload zone above */
    }
    
    /* Target the dropzone area */
    [data-testid="stFileUploader"] > section {
        background: rgba(255,255,255,0.03) !important;
        backdrop-filter: blur(15px);
        border: 2px dashed rgba(102,126,234,0.4) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    
    /* Hover effect for the dropzone */
    [data-testid="stFileUploader"] > section:hover {
        border-color: #a78bfa !important;
        background: rgba(102, 126, 234, 0.08) !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Style the text inside the uploader */
    [data-testid="stFileUploader"] .css-1b0udgb, 
    [data-testid="stFileUploader"] .css-1wrcr25 {
        color: #e2e8f0 !important;
    }
    
    /* Style the small text (limits/extensions) */
    [data-testid="stFileUploader"] small {
        color: rgba(255,255,255,0.5) !important;
    }

    /* Style the "Browse files" button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Hover effect for the button */
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
    }

    /* ── KPIs ── */
    .kpi-premium {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 18px;
        padding: 1.3rem 1rem; text-align: center; position: relative; overflow: hidden;
    }
    .kpi-premium::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 3px; border-radius: 0 0 18px 18px;
    }
    .kpi-premium.c1::after { background: linear-gradient(90deg, #667eea, #764ba2); }
    .kpi-premium.c2::after { background: linear-gradient(90deg, #43e97b, #38f9d7); }
    .kpi-premium.c3::after { background: linear-gradient(90deg, #fa709a, #fee140); }
    .kpi-premium.c4::after { background: linear-gradient(90deg, #4facfe, #00f2fe); }
    .kpi-premium.c5::after { background: linear-gradient(90deg, #a18cd1, #fbc2eb); }
    .kpi-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .kpi-val {
        font-size: 1.5rem; font-weight: 800; margin: 0;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .kpi-lbl {
        font-size: 0.7rem; color: rgba(255,255,255,0.4) !important;
        margin: 0.3rem 0 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
    }

    /* ── Chat ── */
    .chat-user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2); color: white !important;
        padding: 0.9rem 1.3rem; border-radius: 18px 18px 4px 18px;
        margin: 0.6rem 0; max-width: 70%; margin-left: auto; font-size: 0.9rem; font-weight: 500;
    }
    .chat-user-msg * { color: white !important; }
    .chat-bot-msg {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08); padding: 1rem 1.3rem;
        border-radius: 18px 18px 18px 4px; margin: 0.6rem 0; max-width: 85%; font-size: 0.9rem;
    }
    .chat-bot-msg * { color: #e2e8f0 !important; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #a78bfa !important;
        margin: 1.5rem 0 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(167,139,250,0.15);
    }
    .example-q {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.85rem; color: rgba(255,255,255,0.5) !important;
    }

    /* ── Interactive Elements ── */
    .stButton > button {
        background: rgba(255,255,255,0.06) !important; backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1) !important; color: #e2e8f0 !important;
        border-radius: 12px !important; font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important; color: #f7fafc !important;
    }
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important; border: 2px dashed rgba(102,126,234,0.2) !important;
        border-radius: 18px !important;
    }
/* ── Tabs (Nuclear Override) ── */
    
    /* 1. The container holding the tabs */
    .stTabs [data-baseweb="tab-list"] { 
        background: rgba(255, 255, 255, 0.04) !important; 
        border-radius: 16px !important; 
        padding: 6px !important; 
        gap: 8px !important; 
        display: flex !important;
        border-bottom: none !important; /* Removes the default gray line */
    }
    
    /* 2. Every single tab button */
    .stTabs [data-baseweb="tab"] { 
        background: transparent !important; 
        border-radius: 12px !important; 
        color: rgba(255, 255, 255, 0.5) !important; 
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important; 
        font-size: 1.05rem !important; 
        letter-spacing: 0.5px !important; 
        flex: 1 1 0px !important; /* Forces perfect equal stretching */
        display: flex !important;
        justify-content: center !important; 
        padding: 0.8rem 1rem !important; 
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    /* 3. Hover effect for unselected tabs */
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    /* 4. The Active Tab (Make it pop!) */
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #667eea, #764ba2) !important; 
        color: #ffffff !important; 
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important; 
    } 

    /* 5. KILL THE DEFAULT RED LINE */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
        background-color: transparent !important;
    }
    /* ── Non-Sticky Footer ── */
    .premium-footer {
        background: rgba(10,10,26,0.5); backdrop-filter: blur(15px);
        border-top: 1px solid rgba(255,255,255,0.06); padding: 1.5rem 2rem;
        text-align: center; margin-top: 4rem; border-radius: 16px 16px 0 0;
        position: relative; /* Fixed so it scrolls naturally with page */
    }
    .footer-text { font-size: 0.75rem; color: rgba(255,255,255,0.3) !important; margin: 0; letter-spacing: 0.5px; }
    .footer-name { color: #a78bfa !important; font-weight: 600; }
    .footer-div { color: rgba(255,255,255,0.15) !important; margin: 0 0.5rem; }

    /* ── Hide ALL Streamlit chrome: menus, header bar, footer, deploy badge ── */
    /* FIX: use display:none (not visibility:hidden) so the elements are    */
    /* fully removed from layout — visibility:hidden kept the "anonymous"   */
    /* user-chip visible on Streamlit Cloud because it still occupies space. */
    #MainMenu                          { display: none !important; }
    footer                             { display: none !important; }
    header                             { display: none !important; }
    [data-testid="stHeader"]           { display: none !important; }
    [data-testid="stToolbar"]          { display: none !important; }
    [data-testid="stDecoration"]       { display: none !important; }
    [data-testid="stStatusWidget"]     { display: none !important; }
    /* Hides the "Deploy" button / viewer badge on Streamlit Community Cloud */
    [data-testid="manage-app-button"]  { display: none !important; }
    /* ... (previous CSS code) ... */

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* ── Custom Glassmorphic HTML Tables ── */
    .glass-table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    .glass-table thead {
        background: rgba(102, 126, 234, 0.12);
    }
    .glass-table th {
        color: #a78bfa !important;
        font-weight: 700 !important;
        padding: 14px 16px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-align: left;
    }
    .glass-table td {
        padding: 12px 16px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    .glass-table tbody tr {
        transition: background 0.3s ease;
    }
    .glass-table tbody tr:hover {
        background: rgba(102, 126, 234, 0.15) !important;
    }

    /* Soften the edges of Streamlit's native large dataframes */
    [data-testid="stDataFrame"] > div {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }

    /* ... (rest of the CSS) ... */
    
    /* ── Hide "Press Enter to submit form" Text ── */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* ── Premium Form Container ── */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        margin-top: 1rem;
    }

    /* ── Premium Search Bar (Input) ── */
    /* Target the wrapper to remove the native red focus ring */
    .stTextInput > div > div {
        border-color: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: transparent !important;
        box-shadow: none !important;
    }
    
    /* Target the actual input field */
    .stTextInput > div > div > input {
        background: rgba(10, 10, 26, 0.5) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 20px !important;
        color: #f7fafc !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.05rem !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }

    /* Add Neon Purple Glow on Focus */
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 20px rgba(167, 139, 250, 0.3), inset 0 2px 5px rgba(0,0,0,0.3) !important;
        background: rgba(10, 10, 26, 0.8) !important;
    }

    /* ── Premium Ask Button ── */
    [data-testid="stForm"] .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 20px !important;
        min-height: 54px !important; /* Matches the new input height */
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stForm"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
    }
    
     /* ── Premium Enhanced Toggle ── */
     
     /* 1. Premium container with glassmorphism - ULTRA SPECIFIC */
     div[data-testid="stToggle"], .stToggle, [data-testid="stToggle"] {
         padding: 14px 18px !important;
         border-radius: 18px !important;
         background: rgba(255, 255, 255, 0.05) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255, 255, 255, 0.1) !important;
         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
         margin-bottom: 8px !important;
         transition: all 0.3s ease !important;
     }
     
     div[data-testid="stToggle"]:hover, .stToggle:hover, [data-testid="stToggle"]:hover {
         background: rgba(255, 255, 255, 0.08) !important;
         border-color: rgba(102, 126, 234, 0.3) !important;
         box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15) !important;
     }

     /* 2. Premium label with gradient text */
     div[data-testid="stToggle"] p, .stToggle p, [data-testid="stToggle"] p,
     div[data-testid="stToggle"] label, .stToggle label, [data-testid="stToggle"] label {
         font-weight: 800 !important;
         background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%) !important;
         -webkit-background-clip: text !important;
         -webkit-text-fill-color: transparent !important;
         background-clip: text !important;
         font-size: 1.15rem !important;
         letter-spacing: 0.3px !important;
         margin-bottom: 8px !important;
         color: transparent !important;
     }

     /* 3. Enhanced switch container - larger with glow */
     div[data-testid="stToggle"] div[role="switch"],
     .stToggle div[role="switch"],
     [data-testid="stToggle"] div[role="switch"] {
         height: 2rem !important;
         width: 3.8rem !important;
         background: rgba(255, 255, 255, 0.12) !important;
         border: 1px solid rgba(255, 255, 255, 0.25) !important;
         border-radius: 100px !important;
         transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
         position: relative !important;
         overflow: hidden !important;
     }
     
     /* Inner glow effect */
     div[data-testid="stToggle"] div[role="switch"]::before,
     .stToggle div[role="switch"]::before,
     [data-testid="stToggle"] div[role="switch"]::before {
         content: '' !important;
         position: absolute !important;
         top: 0 !important;
         left: 0 !important;
         right: 0 !important;
         bottom: 0 !important;
         background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, transparent 70%) !important;
         opacity: 0 !important;
         transition: opacity 0.3s ease !important;
         z-index: 1 !important;
     }

     /* 4. ACTIVE state - vibrant gradient with strong glow */
     div[data-testid="stToggle"] div[aria-checked="true"],
     div[data-testid="stToggle"] div[role="switch"][aria-checked="true"],
     .stToggle div[aria-checked="true"],
     .stToggle div[role="switch"][aria-checked="true"],
     [data-testid="stToggle"] div[aria-checked="true"],
     [data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #a78bfa 100%) !important;
         border-color: rgba(255, 255, 255, 0.4) !important;
         box-shadow: 0 0 25px rgba(102, 126, 234, 0.8), inset 0 1px 0 rgba(255,255,255,0.2) !important;
     }
     
     div[data-testid="stToggle"] div[aria-checked="true"]::before,
     div[data-testid="stToggle"] div[role="switch"][aria-checked="true"]::before,
     .stToggle div[aria-checked="true"]::before,
     .stToggle div[role="switch"][aria-checked="true"]::before,
     [data-testid="stToggle"] div[aria-checked="true"]::before,
     [data-testid="stToggle"] div[role="switch"][aria-checked="true"]::before {
         opacity: 1 !important;
     }

     /* 5. Enhanced moving knob - larger with shadow and animation */
     div[data-testid="stToggle"] div[role="switch"] > div,
     .stToggle div[role="switch"] > div,
     [data-testid="stToggle"] div[role="switch"] > div {
         background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%) !important;
         border: none !important;
         width: 1.8rem !important;
         height: 1.8rem !important;
         border-radius: 50% !important;
         transform: translateX(0) scale(1) !important;
         transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 0 8px rgba(255,255,255,0.2) !important;
         position: relative !important;
         left: 2px !important;
         z-index: 2 !important;
     }
     
     /* Knob position when ON */
     div[data-testid="stToggle"] div[aria-checked="true"] > div,
     div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] > div,
     .stToggle div[aria-checked="true"] > div,
     .stToggle div[role="switch"][aria-checked="true"] > div,
     [data-testid="stToggle"] div[aria-checked="true"] > div,
     [data-testid="stToggle"] div[role="switch"][aria-checked="true"] > div {
         transform: translateX(1.8rem) scale(1) !important;
         box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 12px rgba(167, 139, 250, 0.5) !important;
     }
     
     /* Knob hover effect */
     div[data-testid="stToggle"] div[role="switch"]:hover > div,
     .stToggle div[role="switch"]:hover > div,
     [data-testid="stToggle"] div[role="switch"]:hover > div {
         transform: scale(1.05) !important;
     }
     
     div[data-testid="stToggle"] div[aria-checked="true"]:hover > div,
     div[data-testid="stToggle"] div[role="switch"][aria-checked="true"]:hover > div,
     .stToggle div[aria-checked="true"]:hover > div,
     .stToggle div[role="switch"][aria-checked="true"]:hover > div,
     [data-testid="stToggle"] div[aria-checked="true"]:hover > div,
     [data-testid="stToggle"] div[role="switch"][aria-checked="true"]:hover > div {
         transform: translateX(1.8rem) scale(1.05) !important;
     }
                 
     /* ── Premium Action Buttons (Generate PDF, Download PDF, Download CSV, Download Cleaned) ── */
      div[data-testid="stButton"] button[kind="primary"],
      div[data-testid="stButton"] button[kind="secondary"],
      div[data-testid="stDownloadButton"] button {
          background: linear-gradient(135deg, #667eea, #764ba2) !important;
          border: none !important;
          border-radius: 14px !important;
          color: white !important;
          font-weight: 600 !important;
          padding: 0.8rem 1.5rem !important;
          transition: all 0.3s ease !important;
          box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
      }
      
      div[data-testid="stButton"] button[kind="primary"]:hover,
      div[data-testid="stButton"] button[kind="secondary"]:hover,
      div[data-testid="stDownloadButton"] button:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
          background: linear-gradient(135deg, #764ba2, #667eea) !important;
      }

     div[data-testid="stButton"] button[type="secondary"] {
         background: rgba(255,255,255,0.06) !important;
         backdrop-filter: blur(10px);
         border: 1px solid rgba(255,255,255,0.1) !important;
         color: #e2e8f0 !important;
         border-radius: 12px !important;
         font-family: 'Inter', sans-serif !important;
         font-size: 0.85rem !important;
         transition: all 0.3s ease !important;
     }
     
     div[data-testid="stButton"] button[type="secondary"]:hover {
         background: rgba(102, 126, 234, 0.15) !important;
         border-color: rgba(102, 126, 234, 0.4) !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
     }

     /* ── Premium Stats Card ── */
     .stat-card {
         background: rgba(255,255,255,0.04);
         backdrop-filter: blur(15px);
         border: 1px solid rgba(255,255,255,0.08);
         border-radius: 16px;
         padding: 1.2rem;
         text-align: center;
         transition: all 0.3s ease;
     }
     .stat-card:hover {
         transform: translateY(-4px);
         border-color: rgba(102, 126, 234, 0.3);
         box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
     }
     .stat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
     .stat-value { font-size: 1.6rem; font-weight: 800; color: #f7fafc !important; margin: 0; }
     .stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.5) !important; margin-top: 0.3rem; }

     /* ── Settings Buttons with Hover ── */
     .stButton > button[key="clear_chat_btn"],
     .stButton > button[key="upload_new_btn"],
     .stButton > button[key="regen_summary_btn"],
     .stButton > button[key="reconnect_ai"] {
         background: rgba(255,255,255,0.06) !important;
         backdrop-filter: blur(10px);
         border: 1px solid rgba(255,255,255,0.1) !important;
         color: #e2e8f0 !important;
         border-radius: 12px !important;
         font-family: 'Inter', sans-serif !important;
         font-size: 0.85rem !important;
         transition: all 0.3s ease !important;
     }
     
     .stButton > button[key="clear_chat_btn"]:hover,
     .stButton > button[key="upload_new_btn"]:hover,
     .stButton > button[key="regen_summary_btn"]:hover,
     .stButton > button[key="reconnect_ai"]:hover {
         background: rgba(102, 126, 234, 0.15) !important;
         border-color: rgba(102, 126, 234, 0.4) !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
     }

     /* ── Chat Suggestion Buttons ── */
     .stButton > button[key^="quick_"],
     .stButton > button[key^="sug_"] {
         background: rgba(255,255,255,0.04) !important;
         backdrop-filter: blur(10px);
         border: 1px solid rgba(255,255,255,0.08) !important;
         color: #e2e8f0 !important;
         border-radius: 12px !important;
         font-size: 0.85rem !important;
         transition: all 0.3s ease !important;
     }
     
     .stButton > button[key^="quick_"]:hover,
     .stButton > button[key^="sug_"]:hover {
         background: rgba(102, 126, 234, 0.12) !important;
         border-color: rgba(102, 126, 234, 0.4) !important;
         transform: translateY(-1px) !important;
     }

     /* ── Premium Selectbox Styling ── */
     div[data-testid="stSelectbox"] > div > div,
     [data-testid="stSelectbox"] > div > div,
     .stSelectbox > div > div {
         background: rgba(255,255,255,0.05) !important;
         backdrop-filter: blur(20px) !important;
         border: 1px solid rgba(255,255,255,0.15) !important;
         border-radius: 14px !important;
         transition: all 0.3s ease !important;
         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
     }
     
     div[data-testid="stSelectbox"] > div > div:hover,
     [data-testid="stSelectbox"] > div > div:hover,
     .stSelectbox > div > div:hover {
         border-color: rgba(102, 126, 234, 0.6) !important;
         box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
         background: rgba(255,255,255,0.08) !important;
     }
     
     div[data-testid="stSelectbox"] .stSelectbox-label,
     [data-testid="stSelectbox"] .stSelectbox-label,
     .stSelectbox .stSelectbox-label,
     div[data-testid="stSelectbox"] label,
     [data-testid="stSelectbox"] label,
     .stSelectbox label {
         color: #f7fafc !important;
         font-weight: 600 !important;
         font-size: 1rem !important;
         letter-spacing: 0.3px !important;
     }
     
     /* ── Dropdown Menu (the list that appears) ── */
     div[data-testid="stSelectbox"] div[role="listbox"],
     [data-testid="stSelectbox"] div[role="listbox"],
     .stSelectbox div[role="listbox"],
     div[data-testid="stSelectbox"] [role="listbox"],
     [data-testid="stSelectbox"] [role="listbox"],
     .stSelectbox [role="listbox"] {
         background: rgba(20, 20, 40, 0.95) !important;
         backdrop-filter: blur(30px) !important;
         border: 1px solid rgba(102, 126, 234, 0.3) !important;
         border-radius: 12px !important;
         box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
         margin-top: 8px !important;
         overflow: hidden !important;
         padding: 8px 0 !important;
         max-height: 300px !important;
         overflow-y: auto !important;
         z-index: 9999 !important;
     }
     
     /* Individual dropdown options */
     div[data-testid="stSelectbox"] div[role="option"],
     [data-testid="stSelectbox"] div[role="option"],
     .stSelectbox div[role="option"],
     div[data-testid="stSelectbox"] [role="option"],
     [data-testid="stSelectbox"] [role="option"],
     .stSelectbox [role="option"] {
         color: rgba(255,255,255,0.8) !important;
         padding: 12px 16px !important;
         font-family: 'Inter', sans-serif !important;
         font-size: 0.95rem !important;
         font-weight: 500 !important;
         transition: all 0.2s ease !important;
         border-left: 3px solid transparent !important;
         cursor: pointer !important;
     }
     
     /* Hover state for options */
     div[data-testid="stSelectbox"] div[role="option"]:hover,
     [data-testid="stSelectbox"] div[role="option"]:hover,
     .stSelectbox div[role="option"]:hover,
     div[data-testid="stSelectbox"] [role="option"]:hover,
     [data-testid="stSelectbox"] [role="option"]:hover,
     .stSelectbox [role="option"]:hover {
         background: rgba(102, 126, 234, 0.2) !important;
         color: #ffffff !important;
         border-left-color: #667eea !important;
         padding-left: 20px !important;
     }
     
     /* Selected option */
     div[data-testid="stSelectbox"] div[role="option"][aria-selected="true"],
     [data-testid="stSelectbox"] div[role="option"][aria-selected="true"],
     .stSelectbox div[role="option"][aria-selected="true"],
     div[data-testid="stSelectbox"] [role="option"][aria-selected="true"],
     [data-testid="stSelectbox"] [role="option"][aria-selected="true"],
     .stSelectbox [role="option"][aria-selected="true"] {
         background: linear-gradient(90deg, rgba(102, 126, 234, 0.25), rgba(167, 139, 250, 0.15)) !important;
         color: #ffffff !important;
         font-weight: 700 !important;
         border-left-color: #a78bfa !important;
     }
     
     /* Scrollbar styling for dropdown */
     div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar,
     [data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar,
     .stSelectbox div[role="listbox"]::-webkit-scrollbar {
         width: 8px !important;
     }
     
     div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-track,
     [data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-track,
     .stSelectbox div[role="listbox"]::-webkit-scrollbar-track {
         background: rgba(255,255,255,0.05) !important;
         border-radius: 4px !important;
     }
     
     div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-thumb,
     [data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-thumb,
     .stSelectbox div[role="listbox"]::-webkit-scrollbar-thumb {
         background: linear-gradient(135deg, #667eea, #a78bfa) !important;
         border-radius: 4px !important;
     }
     
     div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-thumb:hover,
     [data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-thumb:hover,
     .stSelectbox div[role="listbox"]::-webkit-scrollbar-thumb:hover {
         background: linear-gradient(135deg, #764ba2, #667eea) !important;
     }

     /* ── Premium AI Activation Button ── */
    .stButton > button[key="activate_ai"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 1.2rem !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        letter-spacing: 2px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.4s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
    }

    .stButton > button[key="activate_ai"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def load_animated_bg():
    """Subtle animated background - doesn't block content"""
    st.markdown("""
    <style>
    /* ── Animation Keyframes ── */
    @keyframes floatUp1 {
        0% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
        50% { transform: translateY(-20px) translateX(10px); opacity: 0.6; }
        100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
    }
    @keyframes floatUp2 {
        0% { transform: translateY(0px) translateX(0px); opacity: 0.2; }
        50% { transform: translateY(-15px) translateX(-8px); opacity: 0.5; }
        100% { transform: translateY(0px) translateX(0px); opacity: 0.2; }
    }
    @keyframes scanDown {
        0% { top: -2px; }
        100% { top: 100%; }
    }
    @keyframes orbMove {
        0% { transform: translate(0,0) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.1); }
        100% { transform: translate(0,0) scale(1); }
    }
    @keyframes streamFlow {
        0% { transform: translateX(-100%); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateX(200%); opacity: 0; }
    }

    /* ── Background Layer (behind everything) ── */
    .bg-layer {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    /* ── Grid Pattern ── */
    .bg-grid {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            linear-gradient(rgba(102,126,234,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(102,126,234,0.04) 1px, transparent 1px);
        background-size: 80px 80px;
        animation: floatUp2 10s ease-in-out infinite;
    }

    /* ── Glowing Orbs ── */
    .bg-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
    }
    .bg-orb-1 {
        width: 350px; height: 350px;
        background: rgba(102, 126, 234, 0.08);
        top: 5%; right: -50px;
        animation: orbMove 20s ease-in-out infinite;
    }
    .bg-orb-2 {
        width: 250px; height: 250px;
        background: rgba(167, 139, 250, 0.06);
        bottom: 15%; left: -30px;
        animation: orbMove 25s ease-in-out infinite reverse;
    }
    .bg-orb-3 {
        width: 200px; height: 200px;
        background: rgba(240, 147, 251, 0.05);
        top: 50%; left: 40%;
        animation: orbMove 18s ease-in-out infinite;
    }

    /* ── Floating Dots ── */
    .bg-dot { position: absolute; border-radius: 50%; }
    .bg-dot-1 { width: 4px; height: 4px; background: rgba(102,126,234,0.5); top: 15%; left: 10%; animation: floatUp1 8s ease-in-out infinite; }
    .bg-dot-2 { width: 6px; height: 6px; background: rgba(167,139,250,0.4); top: 30%; left: 80%; animation: floatUp2 10s ease-in-out infinite; }
    .bg-dot-3 { width: 3px; height: 3px; background: rgba(240,147,251,0.5); top: 60%; left: 20%; animation: floatUp1 12s ease-in-out infinite; }
    .bg-dot-4 { width: 5px; height: 5px; background: rgba(79,172,254,0.4); top: 45%; left: 65%; animation: floatUp2 9s ease-in-out infinite; }
    .bg-dot-5 { width: 4px; height: 4px; background: rgba(67,233,123,0.4); top: 75%; left: 45%; animation: floatUp1 11s ease-in-out infinite; }
    .bg-dot-6 { width: 3px; height: 3px; background: rgba(102,126,234,0.5); top: 20%; left: 55%; animation: floatUp2 7s ease-in-out infinite; }
    .bg-dot-7 { width: 5px; height: 5px; background: rgba(250,112,154,0.3); top: 85%; left: 75%; animation: floatUp1 13s ease-in-out infinite; }
    .bg-dot-8 { width: 4px; height: 4px; background: rgba(167,139,250,0.4); top: 10%; left: 35%; animation: floatUp2 8s ease-in-out infinite; }
    .bg-dot-9 { width: 6px; height: 6px; background: rgba(79,172,254,0.3); top: 55%; left: 90%; animation: floatUp1 14s ease-in-out infinite; }
    .bg-dot-10 { width: 3px; height: 3px; background: rgba(67,233,123,0.5); top: 40%; left: 5%; animation: floatUp2 10s ease-in-out infinite; }

    /* ── Scan Line ── */
    .bg-scan {
        position: absolute; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(102,126,234,0.15) 20%, rgba(167,139,250,0.3) 50%, rgba(102,126,234,0.15) 80%, transparent 100%);
        animation: scanDown 12s linear infinite;
    }

    /* ── Data Streams ── */
    .bg-stream {
        position: absolute; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.2), transparent);
        animation: streamFlow 8s linear infinite;
    }
    .bg-stream-1 { top: 20%; width: 200px; animation-delay: 0s; }
    .bg-stream-2 { top: 40%; width: 300px; animation-delay: 3s; }
    .bg-stream-3 { top: 65%; width: 150px; animation-delay: 6s; }
    .bg-stream-4 { top: 80%; width: 250px; animation-delay: 2s; }
    </style>

    <!-- HTML Layout for the Animations -->
    <div class="bg-layer">
        <div class="bg-grid"></div>
        <div class="bg-orb bg-orb-1"></div>
        <div class="bg-orb bg-orb-2"></div>
        <div class="bg-orb bg-orb-3"></div>
        <div class="bg-dot bg-dot-1"></div>
        <div class="bg-dot bg-dot-2"></div>
        <div class="bg-dot bg-dot-3"></div>
        <div class="bg-dot bg-dot-4"></div>
        <div class="bg-dot bg-dot-5"></div>
        <div class="bg-dot bg-dot-6"></div>
        <div class="bg-dot bg-dot-7"></div>
        <div class="bg-dot bg-dot-8"></div>
        <div class="bg-dot bg-dot-9"></div>
        <div class="bg-dot bg-dot-10"></div>
        <div class="bg-scan"></div>
        <div class="bg-stream bg-stream-1"></div>
        <div class="bg-stream bg-stream-2"></div>
        <div class="bg-stream bg-stream-3"></div>
        <div class="bg-stream bg-stream-4"></div>
    </div>
    """, unsafe_allow_html=True)


def load_footer():
    st.markdown("""
    <div class='premium-footer'>
        <p class='footer-text'>
            Copyright 2026 <span class='footer-name'>QueryMind</span>
            <span class='footer-div'>|</span>
            Developed by <span class='footer-name'>Gaurav Ramola</span>
            <span class='footer-div'>|</span>
            MBA - AI & Data Science
            <span class='footer-div'>|</span>
            Graphic Era University
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        'is_cleaning': False,
        'cleaning_applied': False,   # from our previous fix
        'df': None,                  # ← only once
        'file_info': None,
        'schema': None,
        'column_categories': None,
        'kpis': None,
        'suggestions': None,
        'chat_history': [],
        'llm_model': None,
        'llm_ready': False,
        'db_loaded': False,
        'file_uploaded': False,
        'current_file_name': None,
        'show_sql': True,
        'chart_type': 'auto',
        'data_summary': None,
        'query_count': 0,
        'show_chat': False,
        'cleaning_report': None,   # before/after diff from generate_cleaning_report()
        'pdf_report': None,        # raw bytes of last generated PDF
        'pdf_report_name': None,   # filename for the download button
        # ── Join feature (N-file pool) ────────────────────────────────────
        'extra_files':        [],     # list of {name, df, fi, size_mb}
        'join_candidates':    None,   # ranked list from detect_joinable_columns
        'join_right_name':    None,   # currently selected right-file name
        'merged_df':          None,   # result of merge_dataframes
        'df_original':        None,   # backup of original primary df
        'using_merged':       False,  # True → chat/SQL use merged_df
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# HEADER + NAVBAR
# ─────────────────────────────────────────────

def render_header():
    # Includes the Futuristic Glowing Core AI Brain SVG
    st.markdown("""
    <div class='hero-section'>
        <div class='hero-logo'>
            <svg width="140" height="140" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#00ffff" stop-opacity="1" />
                        <stop offset="40%" stop-color="#00aaff" stop-opacity="0.8" />
                        <stop offset="100%" stop-color="#0000ff" stop-opacity="0" />
                    </radialGradient>
                    <linearGradient id="brainGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#667eea"/>
                        <stop offset="100%" stop-color="#a78bfa"/>
                    </linearGradient>
                    <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <circle cx="50" cy="55" r="18" fill="url(#coreGlow)">
                    <animate attributeName="r" values="15;19;15" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.6;1;0.6" dur="3s" repeatCount="indefinite" />
                </circle>
                <circle cx="50" cy="55" r="6" fill="#ffffff" filter="url(#neonGlow)" />
                <path d="M50 20 C30 20, 15 35, 15 55 C15 70, 25 85, 40 90 L45 90 L45 75" fill="none" stroke="url(#brainGlow)" stroke-width="2.5" filter="url(#neonGlow)"/>
                <path d="M45 75 L35 65 L35 45 L45 35" fill="none" stroke="#4facfe" stroke-width="1.5" />
                <circle cx="35" cy="65" r="2.5" fill="#00ffff" />
                <circle cx="35" cy="45" r="2.5" fill="#f093fb" />
                <path d="M50 20 C70 20, 85 35, 85 55 C85 70, 75 85, 60 90 L55 90 L55 75" fill="none" stroke="url(#brainGlow)" stroke-width="2.5" filter="url(#neonGlow)"/>
                <path d="M55 75 L65 65 L65 45 L55 35" fill="none" stroke="#4facfe" stroke-width="1.5" />
                <circle cx="65" cy="65" r="2.5" fill="#00ffff" />
                <circle cx="65" cy="45" r="2.5" fill="#f093fb" />
                <line x1="50" y1="20" x2="50" y2="40" stroke="url(#brainGlow)" stroke-width="2" />
                <line x1="45" y1="35" x2="50" y2="40" stroke="#4facfe" stroke-width="1.5" />
                <line x1="55" y1="35" x2="50" y2="40" stroke="#4facfe" stroke-width="1.5" />
                <line x1="45" y1="75" x2="50" y2="70" stroke="#4facfe" stroke-width="1.5" />
                <line x1="55" y1="75" x2="50" y2="70" stroke="#4facfe" stroke-width="1.5" />
            </svg>
        </div>
        <div class='hero-title'>QueryMind</div>
        <p class='hero-subtitle'>Talk to Your Data - Powered by Artificial Intelligence</p>
        <p class='hero-tagline' style='min-height:1.4em;'>
            "<span id='typewriter-text'></span><span id='typewriter-cursor' style='display:inline-block;width:2px;height:0.85em;background:rgba(255,255,255,0.4);margin-left:2px;vertical-align:middle;animation:tw-blink 0.75s step-end infinite;'></span>"
        </p>
    </div>
    <style>@keyframes tw-blink { 0%,100%{opacity:1} 50%{opacity:0} }</style>
    """, unsafe_allow_html=True)

    _st_components.html("""
    <script>
    (function() {
        var quotes = [
            "Transforming raw data into actionable business intelligence",
            "Your data speaks \u2014 QueryMind listens and translates",
            "From spreadsheets to strategy, powered by AI",
            "Ask anything. Discover everything. Decide smarter.",
            "Where numbers end, intelligence begins"
        ];
        var qIdx = 0, charIdx = 0, deleting = false;
        var TYPING = 45, DELETING = 22, PAUSE_AFTER = 2200, PAUSE_BEFORE = 350;
        function getEl() {
            try { return window.parent.document.getElementById('typewriter-text'); }
            catch(e) { return null; }
        }
        function tick() {
            var el = getEl();
            if (!el) { setTimeout(tick, 300); return; }
            var q = quotes[qIdx];
            if (!deleting) {
                el.textContent = q.slice(0, charIdx + 1);
                charIdx++;
                if (charIdx === q.length) { deleting = true; setTimeout(tick, PAUSE_AFTER); return; }
                setTimeout(tick, TYPING);
            } else {
                el.textContent = q.slice(0, charIdx - 1);
                charIdx--;
                if (charIdx === 0) {
                    deleting = false;
                    qIdx = (qIdx + 1) % quotes.length;
                    setTimeout(tick, PAUSE_BEFORE);
                    return;
                }
                setTimeout(tick, DELETING);
            }
        }
        setTimeout(tick, 500);
    })();
    </script>
    """, height=0)


def render_navbar():
    ai_dot = "green" if st.session_state.llm_ready else "red"
    ai_text = "Connected" if st.session_state.llm_ready else "Offline"

    if st.session_state.file_uploaded:
        base_name = st.session_state.file_info['file_name']
        file_text = f"{base_name} ⋈ Merged" if st.session_state.get('using_merged') else base_name
    else:
        file_text = "No file"

    st.markdown(f"""
    <div class='nav-bar'>
        <span class='nav-item'><span class='nav-dot {ai_dot}'></span> AI: {ai_text}</span>
        <span class='nav-item'>📄 {file_text}</span>
        <span class='nav-item'>💬 Queries: {st.session_state.query_count}</span>
        <span class='nav-item'>📊 {config.LLM_MODEL}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE UPLOAD + LLM
# ─────────────────────────────────────────────

def handle_file_upload(uploaded_file):
    with st.spinner(f"📊 Analyzing {uploaded_file.name}..."):
        reset_database()
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.session_state.data_summary = None

        r = load_file(uploaded_file)
        if not r['success']:
            st.error(r['error']); return

        df = r['dataframe']
        fi = r['file_info']

        db = load_dataframe_to_db(df)
        if not db['success']:
            st.error(db['message']); return

        cats = detect_column_categories(df)
        schema = generate_smart_schema(df, fi, cats)
        kpis = get_all_kpis(df, fi)
        sugs = generate_smart_suggestions(df, fi, cats)

        st.session_state.update({
            'df': df, 'file_info': fi, 'schema': schema,
            'column_categories': cats, 'kpis': kpis['kpis'],
            'suggestions': sugs, 'file_uploaded': True,
            'current_file_name': uploaded_file.name,
            'db_loaded': True,
        })

        if not st.session_state.llm_ready:
            initialize_llm()
        if st.session_state.llm_ready:
            generate_ai_summary()

        st.success(f"✅ Loaded {fi['num_rows']:,} rows × {fi['num_cols']} columns")


def initialize_llm():
    with st.spinner("🤖 Connecting AI..."):
        r = setup_llm()
        if r['success']:
            st.session_state.llm_model = r['model']
            st.session_state.llm_ready = True
        else:
            st.error(r['error'])


def generate_ai_summary():
    # Guard: nothing to do if LLM isn't ready
    if not st.session_state.llm_ready or not st.session_state.schema:
        return

    try:
        r = generate_data_summary(
            st.session_state.schema,
            st.session_state.kpis,
            st.session_state.llm_model
        )
        if r['success']:
            st.session_state.data_summary = r['summary']
        else:
            # LLM returned a failure — clear summary so Overview shows
            # the "Generate Summary" button instead of stale text
            st.session_state.data_summary = None
    except Exception as e:
        st.session_state.data_summary = None   # never leave stale text behind


# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────

def render_welcome():
    col1, col2, col3 = st.columns(3)
    features = [
        ("💬", "Natural Language Queries", "Ask questions in plain English. No SQL or coding required."),
        ("📊", "Auto Visualization", "Beautiful interactive charts generated automatically from results."),
        ("🎯", "Smart Business Insights", "AI-powered analysis and executive recommendations."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class='feature-premium'>
                <span class='feat-icon'>{icon}</span>
                <div class='feat-title'>{title}</div>
                <div class='feat-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload Zone Text & Icon ──
    st.markdown("""
    <div style='text-align: center; margin: 2rem auto 1rem auto; max-width: 600px;'>
        <div style='font-size: 3.5rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(102,126,234,0.4));'>📁</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #f7fafc !important;'>
            Upload your dataset for AI based analytics
        </div>
        <div style='font-size: 0.9rem; color: rgba(255,255,255,0.5) !important; margin-top: 0.5rem;'>
            Supports CSV and Excel files (up to 200MB)
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
    if uploaded:
        if uploaded.name != st.session_state.current_file_name:
            handle_file_upload(uploaded)
            st.rerun()


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

def render_dashboard():
    # Render the top KPI row first
    render_kpi_row()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Define all 8 tabs — new "Join & Merge" tab added at position 6
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Data Overview",
        "📈 Visual Analytics",
        "🔍 Data Preview",
        "📋 Schema Info",
        "✨ AI Refinement",
        "🔗 Join & Merge",      # ← NEW: multi-file join feature
        "💬 Chat & Analysis",
        "⚙️ Settings",
    ])

    # Route each tab to its correct function
    with tab1:
        render_overview_tab()
    with tab2:
        render_visual_analytics_tab()
    with tab3:
        render_preview_tab()
    with tab4:
        render_schema_tab()
    with tab5:
        render_refinement_tab()
    with tab6:
        render_join_tab()           # ← NEW
    with tab7:
        render_chat_section()
    with tab8:
        render_settings_tab()


def render_kpi_row():
    kpis = st.session_state.kpis
    if not kpis: return

    display = kpis[:5]
    cols = st.columns(len(display))
    icons = ['💰', '📈', '🏷️', '⭐', '📦']
    for i, (col, kpi) in enumerate(zip(cols, display)):
        with col:
            st.markdown(f"""
            <div class='kpi-premium c{i+1}'>
                <div class='kpi-icon'>{icons[i % len(icons)]}</div>
                <p class='kpi-val'>{kpi['formatted_value']}</p>
                <p class='kpi-lbl'>{kpi['label']}</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT SECTION
# ─────────────────────────────────────────────
def render_chat_section():
    st.markdown("<p class='section-title'>💬 Ask Questions About Your Data</p>", unsafe_allow_html=True)

    # ── CASE 1: AI ENGINE IS OFFLINE (Show Activation Screen) ──
    if not st.session_state.show_chat:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem;'>
            <div class='glass-card' style='padding: 3rem; border: 1px dashed rgba(102,126,234,0.3); max-width: 500px; margin: 0 auto;'>
                <div style='font-size: 5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(102,126,234,0.5));'>🤖</div>
                <h2 style='margin-bottom: 0.5rem; background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem;'>
                    Neural Engine Offline
                </h2>
                <p style='color:rgba(255,255,255,0.6) !important; margin-bottom: 2rem; font-size: 1.05rem; line-height: 1.6;'>
                    Activate the QueryMind AI Assistant to start natural language discovery.<br>Ask questions in plain English and get instant insights.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # This button uses the special 'activate_ai' key for our premium CSS styling
        if st.button("🚀 ACTIVATE AI NEURAL ENGINE", key="activate_ai", use_container_width=True):
            st.session_state.show_chat = True
            st.rerun()

        # Show quick suggestions even when offline to encourage the user
        sugs = st.session_state.suggestions
        if sugs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p class='section-title' style='font-size: 1rem;'>💡 Quick Discovery Questions</p>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, s in enumerate(sugs[:4]):
                with cols[i % 2]:
                    if st.button(f"💬 {s}", key=f"quick_{i}", use_container_width=True):
                        st.session_state.show_chat = True
                        process_question(s)
                        st.rerun()
        return

    # ── CASE 2: AI ENGINE IS ONLINE (Show Chat Interface) ──
    
    # Small Deactivate button at the top to go back
    c1, c2 = st.columns([5, 1])
    with c2:
        if st.button("❌ CLOSE CHAT", use_container_width=True):
            st.session_state.show_chat = False
            st.rerun()

    if not st.session_state.chat_history:
        st.markdown("""
        <div class='chat-bot-msg'>
            <strong>🤖 QueryMind:</strong><br><br>
            Neural Engine is online. I've analyzed your dataset and I'm ready to help. 
            Ask me any question in plain English!
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class='chat-user-msg'>
                <strong>You:</strong> {msg['question']}
            </div>
            """, unsafe_allow_html=True)

            with st.container():
                if st.session_state.show_sql and msg.get('sql_query'):
                    with st.expander("🔍 SQL Generated", expanded=False):
                        st.code(msg['sql_query'], language='sql')

                if msg.get('explanation'):
                    st.markdown(f"""
                    <div class='chat-bot-msg'>
                        <strong>🤖 QueryMind:</strong><br>{msg['explanation']}
                    </div>
                    """, unsafe_allow_html=True)

                if msg.get('result_df') is not None and not msg['result_df'].empty:
                    st.caption(f"📊 Query Result: {len(msg['result_df'])} rows")
                    
                    df_to_show = format_dataframe_for_display(msg['result_df'], 10)
                    html_table = df_to_show.to_html(index=False, classes="glass-table")
                    st.markdown(f'<div style="overflow-x: auto; padding-bottom: 10px;">{html_table}</div>', unsafe_allow_html=True)

                if msg.get('figure') is not None:
                    st.plotly_chart(msg['figure'], use_container_width=True, key=f"chart_{i}")

                if msg.get('insight'):
                    with st.expander("💡 AI Strategic Insight", expanded=True):
                        st.markdown(msg['insight'])

            st.divider()

    # Suggestions (Internal)
    sugs = st.session_state.suggestions
    if sugs:
        st.markdown("<p class='section-title'>💡 Discovery Suggestions</p>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, s in enumerate(sugs[:6]):
            with cols[i % 2]:
                if st.button(f"💬 {s}", key=f"sug_{i}", use_container_width=True):
                    process_question(s)
                    st.rerun()

    # Chat Input Form
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Ask", placeholder="e.g., What are the total sales by region?",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("Ask 🚀", use_container_width=True)

        if submit and user_input:
            process_question(user_input)
            st.rerun()


# ─────────────────────────────────────────────
# QUESTION PROCESSING
# ─────────────────────────────────────────────

def process_question(question):
    if not st.session_state.llm_ready:
        st.error("❌ AI not connected."); return
    if not st.session_state.db_loaded:
        st.error("❌ No data loaded."); return
    # NOTE: ensure_db_loaded() is intentionally NOT called here.
    # db_loaded=True (checked above) proves the DB was populated at upload/clean
    # time. Calling ensure_db_loaded on every question re-loads the entire
    # DataFrame into SQLite unconditionally — unnecessary I/O on every turn.
    # Recovery is handled below: if execute_sql_query fails with a connection
    # error, we reload once and retry rather than pre-loading every time.

    with st.spinner("🤖 Analyzing..."):
        llm_r = generate_sql_query(
            question, st.session_state.schema,
            st.session_state.llm_model, get_chat_context()
        )

        if llm_r['response_type'] == 'text':
            st.session_state.chat_history.append({
                'question': question, 'response_type': 'text',
                'sql_query': None, 'explanation': llm_r['explanation'],
                'result_df': None, 'figure': None, 'insight': None,
                'timestamp': datetime.now().strftime("%H:%M")
            })
            st.session_state.query_count += 1
            return

        if not llm_r['success']:
            st.session_state.chat_history.append({
                'question': question, 'response_type': 'error',
                'sql_query': None, 'explanation': llm_r.get('error', 'Error'),
                'result_df': None, 'figure': None, 'insight': None,
                'timestamp': datetime.now().strftime("%H:%M")
            })
            return

        sql = llm_r['sql_query']
        exec_r = execute_sql_query(sql)

        # ── Recovery path: if the in-memory SQLite connection was lost
        # (e.g. after a Streamlit rerun that reset the module-level connection),
        # reload the DB once and retry the query rather than failing immediately.
        _DB_ERRORS = ('no such table', 'unable to open', 'disk I/O error',
                      'database is closed', 'OperationalError')
        if not exec_r['success'] and any(
            e.lower() in str(exec_r.get('error', '')).lower()
            for e in _DB_ERRORS
        ):
            ensure_db_loaded(st.session_state.df)
            exec_r = execute_sql_query(sql)   # one retry

        if not exec_r['success']:
            st.session_state.chat_history.append({
                'question': question, 'response_type': 'error',
                'sql_query': sql, 'explanation': exec_r['error'],
                'result_df': None, 'figure': None, 'insight': None,
                'timestamp': datetime.now().strftime("%H:%M")
            })
            return

        result_df = exec_r['dataframe']

        figure = None
        if should_show_chart(result_df, question):
            cr = generate_chart(result_df, question, st.session_state.chart_type)
            if cr['success']:
                figure = cr['figure']

        insight = None
        if not result_df.empty:
            try:
                ir = generate_text_response(
                    question, st.session_state.schema,
                    result_df.head(20).to_string(index=False),
                    st.session_state.llm_model
                )
                if ir['success']:
                    insight = ir['response']
            except:
                pass

        st.session_state.chat_history.append({
            'question': question, 'response_type': 'sql',
            'sql_query': sql, 'explanation': llm_r.get('explanation', ''),
            'result_df': result_df, 'figure': figure, 'insight': insight,
            'answer_summary': f"Returned {len(result_df)} rows",
            'timestamp': datetime.now().strftime("%H:%M")
        })
        st.session_state.query_count += 1

        if len(st.session_state.chat_history) > config.MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-config.MAX_CHAT_HISTORY:]


def get_chat_context():
    # Build standard recent Q&A history
    history = [
        {
            'question': m.get('question', ''),
            'answer_summary': m.get('answer_summary', '')
        }
        for m in st.session_state.chat_history[-5:]
    ]

    # ── FIX: Prepend a system note when cleaning has been applied ──
    # Without this, the LLM has no idea cleaning happened and answers
    # based solely on the schema — which may still look ambiguous.
    if st.session_state.get('cleaning_applied', False):
        df = st.session_state.df
        remaining_nulls = int(df.isnull().sum().sum())
        null_msg = (
            "0 null values remain — data is fully clean."
            if remaining_nulls == 0
            else f"{remaining_nulls} null value(s) remain (e.g. in date/time columns that could not be inferred)."
        )
        history.insert(0, {
            'question': '[DATA STATE]',
            'answer_summary': (
                f"Smart Cleaning was applied to this dataset. "
                f"Missing values filled, duplicates removed, string columns standardised. "
                f"{null_msg}"
            )
        })

    return history


# ─────────────────────────────────────────────
# TABS (Overview, Preview, Schema, Settings)
# ─────────────────────────────────────────────

def render_overview_tab():
    df = st.session_state.df
    fi = st.session_state.file_info
    if df is None: return

    st.markdown("<p class='section-title'>📊 Dataset Statistics</p>", unsafe_allow_html=True)
    stats = get_quick_stats(df)
    cols = st.columns(len(stats))
    for col, s in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>{s['icon']}</div>
                <p class='stat-value'>{s['value']}</p>
                <p class='stat-label'>{s['label']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<p class='section-title'>📋 Column Types</p>", unsafe_allow_html=True)
        cats = st.session_state.column_categories
        if cats:
            data = [{'Type': k.replace('_', ' ').title(), 'Count': len(v), 'Columns': ', '.join(v)}
                    for k, v in cats.items() if v]
            
            # FIX 1: Removed the duplicate 'if data:' and fixed indentation
            if data:
                # Convert to elegant HTML table
                html_table = pd.DataFrame(data).to_html(index=False, classes="glass-table")
                st.markdown(html_table, unsafe_allow_html=True)

    with c2:
        st.markdown("<p class='section-title'>🏥 Data Quality</p>", unsafe_allow_html=True)
        h = get_data_health_score(df, fi)

        # ── SYNC with AI Refinement tab: override grade when cleaning was applied ──
        cleaning_applied = st.session_state.get('cleaning_applied', False)
        if cleaning_applied:
            adjustable = {k: v for k, v in h['breakdown'].items() if k != 'size'}
            adj_avg = sum(adjustable.values()) / len(adjustable) if adjustable else h['score']
            display_grade   = 'A' if adj_avg >= 90 else 'B' if adj_avg >= 75 else 'C' if adj_avg >= 60 else 'D'
            display_overall = round(adj_avg, 1)
            display_label   = {'A': 'Excellent', 'B': 'Good', 'C': 'Fair', 'D': 'Poor'}.get(display_grade, 'Fair')
        else:
            display_grade, display_overall, display_label = h['grade'], h['overall'], h['label']

        clr = {'A': '#48bb78', 'B': '#4299e1', 'C': '#ed8936', 'D': '#fc8181'}.get(display_grade, '#667eea')
        st.markdown(f"""
        <div style='text-align:center;padding:1rem;'>
            <div style='font-size:3rem;font-weight:900;color:{clr};'>{display_grade}</div>
            <div style='color:rgba(255,255,255,0.5) !important;'>{display_label} ({display_overall}/100)</div>
            {'<div style="font-size:0.75rem;color:#48bb78;margin-top:0.3rem;">✅ Smart Cleaning Applied</div>' if cleaning_applied else ''}
        </div>
        """, unsafe_allow_html=True)

        for m, s in h['breakdown'].items():
            suffix = " (excluded post-clean)" if m == 'size' and cleaning_applied else ""
            st.progress(s / 100, text=f"{m.title()}: {s}{suffix}")

    st.divider()
    st.markdown("<p class='section-title'>🤖 AI Executive Summary</p>", unsafe_allow_html=True)
    if st.session_state.data_summary:
        st.markdown(st.session_state.data_summary)
    else:
        if st.button("🤖 Generate Summary", type="primary"):
            with st.spinner("Generating..."):
                generate_ai_summary()
                st.rerun()

    st.divider()
    st.markdown("<p class='section-title'>🔢 Numeric Summary</p>", unsafe_allow_html=True)
    
    # FIX 2: Standardized the variable name to 'numeric_df' to avoid NameErrors
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        # Convert to elegant HTML table (index=True keeps Mean, Min, Max labels)
        html_table = numeric_df.describe().round(2).to_html(classes="glass-table")
        st.markdown(html_table, unsafe_allow_html=True)

    # ── PDF Export ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("<p class='section-title'>📄 Export Full Report</p>",
                unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        generate_clicked = st.button(
            "📄 Generate PDF Report",
            use_container_width=True,
            key="gen_pdf_btn",
        )
    with col_info:
        st.caption(
            "Exports a branded PDF containing the data quality score, "
            "key metrics, AI executive summary, charts, and cleaning report "
            "(if Smart Cleaning was applied)."
        )

    if generate_clicked:
        with st.spinner("Building your PDF report…"):
            try:
                # Collect Plotly figures if the visual analytics tab generated them
                figures = []
                try:
                    from components.chart_generator import generate_auto_business_visualizations
                    chart_objects = generate_auto_business_visualizations(
                        df, st.session_state.column_categories
                    )
                    figures = [c['fig'] for c in chart_objects if 'fig' in c]
                except Exception:
                    figures = []   # graceful — matplotlib fallback kicks in

                health_for_pdf = get_data_health_score(df, fi)

                pdf_bytes = generate_pdf_report(
                    df              = df,
                    fi              = fi,
                    health          = health_for_pdf,
                    kpis            = st.session_state.kpis or [],
                    data_summary    = st.session_state.data_summary,
                    cleaning_report = st.session_state.get('cleaning_report'),
                    figures         = figures,
                    file_name       = st.session_state.get('current_file_name', 'dataset'),
                )

                base = st.session_state.get('current_file_name', 'report').rsplit('.', 1)[0]
                st.session_state.pdf_report      = pdf_bytes
                st.session_state.pdf_report_name = f"{base}_querymind_report.pdf"
                st.success("✅ PDF ready — click Download below.")

            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    # Persists across reruns once generated
    if st.session_state.get('pdf_report'):
        st.download_button(
            label               = "📥 Download PDF Report",
            data                = st.session_state.pdf_report,
            file_name           = st.session_state.get('pdf_report_name', 'report.pdf'),
            mime                = "application/pdf",
            use_container_width = True,
            key                 = "download_pdf_btn",
        )
        size_kb = len(st.session_state.pdf_report) // 1024
        st.caption(
            f"Ready: **{st.session_state.get('pdf_report_name', 'report.pdf')}** "
            f"({size_kb} KB)"
        )


# ─────────────────────────────────────────────
# TAB 3: VISUAL ANALYTICS
# ─────────────────────────────────────────────

def render_visual_analytics_tab():
    st.markdown("<p class='section-title'>📈 Automated Visual Analytics</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:rgba(255,255,255,0.6); margin-bottom: 2rem; max-width: 800px;'>
        These smart visualizations were automatically generated based on your data schema. 
        The AI engine identified your primary business metrics and generated the most crucial insights instantly.
    </p>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    if df is None: return

    with st.spinner("🧠 AI is generating smart visualizations..."):
        # Import the new function we just made
        from components.chart_generator import generate_auto_business_visualizations
        
        charts = generate_auto_business_visualizations(df, st.session_state.column_categories)

        if not charts:
            st.warning("⚠️ Not enough numeric or categorical data to generate business visualizations.")
            return

        # Display charts in a beautiful 2-column layout
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            
            # Left Column Chart
            with cols[0]:
                st.markdown(f"""
                <div class='glass-card' style='padding: 1rem; margin-bottom: 0;'>
                    <h4 style='color: #f7fafc !important; margin: 0 0 0.2rem 0;'>{charts[i]['title']}</h4>
                    <p style='color: #a78bfa !important; font-size: 0.85rem; margin: 0 0 1rem 0;'>{charts[i]['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(charts[i]['fig'], use_container_width=True)
            
            # Right Column Chart (if it exists)
            if i + 1 < len(charts):
                with cols[1]:
                    st.markdown(f"""
                    <div class='glass-card' style='padding: 1rem; margin-bottom: 0;'>
                        <h4 style='color: #f7fafc !important; margin: 0 0 0.2rem 0;'>{charts[i+1]['title']}</h4>
                        <p style='color: #a78bfa !important; font-size: 0.85rem; margin: 0 0 1rem 0;'>{charts[i+1]['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(charts[i+1]['fig'], use_container_width=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)


def render_preview_tab():
    df = st.session_state.df
    if df is None: return

    c1, c2 = st.columns([3, 1])
    with c1:
        n = st.slider("Rows", 5, min(100, len(df)), 10, 5)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_t = st.checkbox("Show types")

    # FIX 1: Convert the main data preview table to the glassmorphic HTML style
    # We wrap it in a div with overflow-x: auto so wide tables scroll smoothly left/right!
    html_table = df.head(n).to_html(index=False, classes="glass-table")
    st.markdown(f'<div style="overflow-x: auto; padding-bottom: 10px;">{html_table}</div>', unsafe_allow_html=True)
    
    st.caption(f"Showing {n} of {len(df):,}")

    if show_t:
        dt = pd.DataFrame({
            'Column': df.dtypes.index, 'Type': df.dtypes.values.astype(str),
            'Non-Null': df.count().values, 'Null': df.isna().sum().values
        })
        # FIX 2: Convert the 'Show types' table to the glassmorphic HTML style too
        dt_html = dt.to_html(index=False, classes="glass-table")
        st.markdown(f'<div style="overflow-x: auto;">{dt_html}</div>', unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv,
                        f"querymind_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv", use_container_width=True, key="download_csv_btn")

def render_schema_tab():
    fi = st.session_state.file_info
    if not fi: return

    st.markdown("<p class='section-title'>📋 Column Details</p>", unsafe_allow_html=True)
    
    data = []
    for c in fi['column_details']:
        row = {'Column': c['name'], 'Type': c['type'],
               'Non-Null': c['non_null_count'], 'Nulls': c['null_count'],
               'Unique': c['unique_count']}
        if c['type'] == 'Numeric':
            row.update({'Min': c.get('min', '-'), 'Max': c.get('max', '-'), 'Mean': c.get('mean', '-')})
        data.append(row)

    # FIX 1: Changed 'col_data' to 'data' to match the list above
    html_table = pd.DataFrame(data).to_html(index=False, classes="glass-table")
    st.markdown(html_table, unsafe_allow_html=True)

    with st.expander("🔍 Full Schema"):
        st.text(st.session_state.schema)

    if fi.get('has_missing_values'):
        st.markdown("<p class='section-title'>⚠️ Missing Values</p>", unsafe_allow_html=True)
        md = [{'Column': c, 'Count': i['count'], '%': f"{i['percentage']}%"}
              for c, i in fi['missing_info'].items()]
        
        # FIX 2: Upgraded the Missing Values table to the glassmorphic HTML style too!
        missing_html = pd.DataFrame(md).to_html(index=False, classes="glass-table")
        st.markdown(missing_html, unsafe_allow_html=True)
    else:
        st.success("✅ No missing values!")


def render_settings_tab():
    st.markdown("<p class='section-title'>⚙️ Application Settings</p>", unsafe_allow_html=True)

    # ── Dark-theme CSS — matching the app's glass/purple palette ─────────────
    st.markdown("""
    <style>

    /* ── Selectbox: dark glass card, purple accent ── */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(26, 26, 62, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(102, 126, 234, 0.35) !important;
        border-radius: 12px !important;
        color: #f7fafc !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
        padding: 2px 8px !important;
    }
    div[data-testid="stSelectbox"] > div > div:hover,
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: rgba(102, 126, 234, 0.75) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15),
                    0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }
    /* Selectbox inner text */
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div[class*="singleValue"] {
        color: #f7fafc !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    /* Selectbox label */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] > label {
        color: #a78bfa !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.4px !important;
        text-transform: uppercase !important;
        text-decoration: none !important;
    }
    /* Dropdown arrow */
    div[data-testid="stSelectbox"] svg {
        fill: #a78bfa !important;
    }
    /* Dropdown list panel */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background: rgba(22, 22, 50, 0.97) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(20px) !important;
        padding: 6px !important;
    }
    /* Each dropdown option */
    div[data-baseweb="menu"] li,
    div[data-baseweb="popover"] li {
        border-radius: 8px !important;
        color: #f7fafc !important;
        font-size: 0.9rem !important;
        padding: 8px 12px !important;
        transition: background 0.15s ease !important;
    }
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] li:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #a78bfa !important;
    }
    /* Selected option highlight */
    div[data-baseweb="menu"] li[aria-selected="true"],
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background: rgba(102, 126, 234, 0.3) !important;
        color: #a78bfa !important;
        font-weight: 600 !important;
    }

    /* ── Toggle: glass card, purple track when ON ── */
    div[data-testid="stToggle"] {
        background: rgba(26, 26, 62, 0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.25) !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.2s ease !important;
        text-decoration: none !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stToggle"]:hover {
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    /* Toggle label text */
    div[data-testid="stToggle"] label p,
    div[data-testid="stToggle"] p {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        letter-spacing: 0.2px !important;
    }
    /* Toggle track (the pill shape) */
    div[data-testid="stToggle"] div[role="switch"] {
        background: rgba(45, 55, 72, 0.9) !important;
        border: 1.5px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 100px !important;
        height: 1.5rem !important;
        width: 3rem !important;
        transition: background 0.25s ease, border-color 0.25s ease !important;
    }
    /* Track when checked/ON */
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.4) !important;
    }
    /* Toggle knob */
    div[data-testid="stToggle"] div[role="switch"] > div {
        background: #ffffff !important;
        border: none !important;
        width: 1.15rem !important;
        height: 1.15rem !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        # AI status card
        ai_status  = '✅ Connected' if st.session_state.llm_ready else '❌ Offline'
        status_clr = '#48bb78'      if st.session_state.llm_ready else '#fc8181'
        st.markdown(f"""
        <div style='background:rgba(26,26,62,0.6);border:1px solid rgba(102,126,234,0.25);
                    border-radius:16px;padding:1.2rem 1.4rem;margin-bottom:1rem;
                    backdrop-filter:blur(12px);'>
            <p style='color:#a78bfa;font-size:0.75rem;font-weight:700;
                      letter-spacing:0.8px;text-transform:uppercase;margin:0 0 0.8rem 0;'>
                🤖 AI Configuration
            </p>
            <div style='display:flex;flex-direction:column;gap:0.45rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#a0aec0;font-size:0.88rem;'>Status</span>
                    <span style='color:{status_clr};font-weight:600;font-size:0.88rem;'>
                        {ai_status}
                    </span>
                </div>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#a0aec0;font-size:0.88rem;'>Model</span>
                    <span style='color:#f7fafc;font-weight:500;font-size:0.88rem;'>
                        {config.LLM_MODEL}
                    </span>
                </div>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:#a0aec0;font-size:0.88rem;'>Provider</span>
                    <span style='color:#f7fafc;font-weight:500;font-size:0.88rem;'>
                        Groq AI
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.llm_ready:
            if st.button("🔄 Reconnect AI", use_container_width=True, key="reconnect_ai"):
                initialize_llm()
                st.rerun()

    with c2:
        st.markdown("""
        <p style='color:#a78bfa;font-size:0.75rem;font-weight:700;
                  letter-spacing:0.8px;text-transform:uppercase;margin:0 0 0.6rem 0;'>
            📊 Display Settings
        </p>
        """, unsafe_allow_html=True)

        chart_opts = get_chart_type_options()
        st.session_state.chart_type = st.selectbox(
            "Default Chart Type",
            list(chart_opts.keys()),
            format_func=lambda x: chart_opts[x],
            index=0,
        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # ── CSS injected separately (plain string, no f-string, no {} escaping issues) ──
        st.markdown("""
        <style>
        .qm-toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(26,26,62,0.7);
            border: 1px solid rgba(102,126,234,0.25);
            border-radius: 14px;
            padding: 13px 16px;
            margin-bottom: 4px;
        }
        .qm-toggle-info { display:flex; flex-direction:column; gap:3px; }
        .qm-toggle-title { color:#e2e8f0; font-size:0.92rem; font-weight:500; }
        .qm-toggle-sub   { color:#718096; font-size:0.74rem; }
        .qm-pill-on  { display:inline-block; font-size:0.68rem; font-weight:600;
                        padding:2px 8px; border-radius:20px; margin-left:6px;
                        background:rgba(72,187,120,0.15); color:#48bb78; }
        .qm-pill-off { display:inline-block; font-size:0.68rem; font-weight:600;
                        padding:2px 8px; border-radius:20px; margin-left:6px;
                        background:rgba(160,174,192,0.1); color:#718096; }
        .qm-switch-on  { display:flex; align-items:center; justify-content:flex-end;
                          width:52px; height:28px; border-radius:100px; padding:0 4px;
                          background:linear-gradient(135deg,#667eea,#764ba2);
                          box-shadow:0 0 12px rgba(102,126,234,0.45); }
        .qm-switch-off { display:flex; align-items:center; justify-content:flex-start;
                          width:52px; height:28px; border-radius:100px; padding:0 4px;
                          background:rgba(45,55,72,0.9);
                          border:1.5px solid rgba(100,100,140,0.4); }
        .qm-knob { width:20px; height:20px; border-radius:50%; background:#fff;
                   box-shadow:0 2px 6px rgba(0,0,0,0.35); flex-shrink:0; }
        /* Hide Streamlit button borders for the toggle button */
        div[data-testid="stButton"].qm-toggle-btn > button {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Toggle state lives entirely in session_state.show_sql ──
        # A native st.button triggers the flip + rerun. The visual is pure HTML
        # rendered from the current state — no JS cross-frame hacks needed.
        _sql_on = st.session_state.get('show_sql', True)

        _track_cls  = "qm-switch-on"  if _sql_on else "qm-switch-off"
        _pill_cls   = "qm-pill-on"    if _sql_on else "qm-pill-off"
        _pill_txt   = "ON"            if _sql_on else "OFF"
        _sub_txt    = "SQL shown in chat results" if _sql_on else "SQL hidden from results"
        _icon       = "🔍"            if _sql_on else "🔇"

        # Render the visual card (display only — not interactive itself)
        st.markdown(
            '<div class="qm-toggle-row">'
              '<div class="qm-toggle-info">'
                f'<span class="qm-toggle-title">{_icon} Show SQL Queries'
                  f'<span class="{_pill_cls}">{_pill_txt}</span>'
                '</span>'
                f'<span class="qm-toggle-sub">{_sub_txt}</span>'
              '</div>'
              f'<div class="{_track_cls}"><div class="qm-knob"></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # The real clickable control — a Streamlit button that flips state
        if st.button(
            "Toggle SQL display",
            key="toggle_sql_btn",
            use_container_width=True,
            help="Click to toggle SQL query visibility in chat results",
        ):
            st.session_state.show_sql = not _sql_on
            st.rerun()

        # Style the toggle button to look minimal under the card
        st.markdown("""
        <style>
        div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"][title*="Toggle SQL"]) button,
        div[data-testid="stButton"]:has(button[kind="secondary"]) button {
            margin-top: -4px !important;
            font-size: 0.78rem !important;
            color: rgba(102,126,234,0.7) !important;
            background: transparent !important;
            border: 1px dashed rgba(102,126,234,0.25) !important;
            border-radius: 8px !important;
            padding: 4px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.rerun()
    with c2:
        if st.button("🔄 Upload New File", use_container_width=True, key="upload_new_btn"):
            reset_all()
            st.rerun()
    with c3:
        if st.button("🤖 Regenerate Summary", use_container_width=True, key="regen_summary_btn"):
            with st.spinner("Generating..."):
                generate_ai_summary()
                st.rerun()

# ─────────────────────────────────────────────
# JOIN TAB HELPERS
# ─────────────────────────────────────────────

_MAX_TOTAL_MB = 200.0

def _build_merged_fi(merged_df, name: str) -> dict:
    """
    Build a file_info dict for a merged DataFrame that is fully compatible
    with ALL helpers including get_data_health_score.

    Includes every key that load_file() would normally populate so that
    render_overview_tab / get_data_health_score / get_all_kpis never hit a
    KeyError after a merge is applied.

    Keys added vs the previous version:
        numeric_columns, categorical_columns, datetime_columns, boolean_columns
    """
    if merged_df is None or merged_df.empty:
        return {
            "file_name": name, "file_size": "0 KB",
            "num_rows": 0, "num_cols": 0,
            "has_missing_values": False, "missing_percentage": 0.0,
            "missing_info": {}, "column_details": [],
            "numeric_columns": [], "text_columns": [],
            "date_columns": [], "boolean_columns": [],
            "categorical_columns": [], "datetime_columns": [],
        }

    total_cells = len(merged_df) * len(merged_df.columns)
    null_total  = int(merged_df.isnull().sum().sum())
    missing_pct = round(null_total / total_cells * 100, 2) if total_cells > 0 else 0.0

    col_details  = []
    missing_info = {}
    for col in merged_df.columns:
        nc  = int(merged_df[col].isnull().sum())
        pct = round(nc / len(merged_df) * 100, 2) if len(merged_df) else 0.0
        col_details.append({
            "name":           col,
            "type":           str(merged_df[col].dtype),
            "non_null_count": int(merged_df[col].count()),
            "null_count":     nc,
            "unique_count":   int(merged_df[col].nunique()),
            "percentage":     pct,
        })
        if nc > 0:
            missing_info[col] = {"count": nc, "percentage": pct}

    # ── Column-type groups ────────────────────────────────────────────────────
    # Key names must match EXACTLY what helpers.py/get_data_health_score reads:
    #   file_info['numeric_columns']  (line 512)
    #   file_info['text_columns']     (line 513)  ← NOT 'categorical_columns'
    #   file_info['date_columns']     (line 514)  ← NOT 'datetime_columns'
    numeric_cols  = merged_df.select_dtypes(include="number").columns.tolist()
    text_cols     = merged_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    date_cols     = merged_df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()
    boolean_cols  = merged_df.select_dtypes(include="bool").columns.tolist()

    return {
        "file_name":           name,
        "file_size":           f"{merged_df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        "num_rows":            len(merged_df),
        "num_cols":            len(merged_df.columns),
        "has_missing_values":  null_total > 0,
        "missing_percentage":  missing_pct,
        "missing_info":        missing_info,
        "column_details":      col_details,
        # Exact key names required by helpers.py → get_data_health_score
        "numeric_columns":     numeric_cols,
        "text_columns":        text_cols,       # helpers.py line 513
        "date_columns":        date_cols,       # helpers.py line 514
        "boolean_columns":     boolean_cols,
        # Aliases kept for any other helper that uses the longer names
        "categorical_columns": text_cols,
        "datetime_columns":    date_cols,
    }


def _pool_total_mb() -> float:
    """Total MB across primary df + all extra files."""
    primary_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024) \
                 if st.session_state.df is not None else 0.0
    extra_mb   = sum(f["size_mb"] for f in st.session_state.get("extra_files", []))
    return primary_mb + extra_mb


# ─────────────────────────────────────────────
# JOIN TAB
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# JOIN TAB
# ─────────────────────────────────────────────

def render_join_tab():
    """
    Multi-File Upload + AI Auto Join Detection (N files, up to 200 MB total).
    """
    st.markdown("<p class='section-title'>🔗 Multi-File Join & Merge</p>", unsafe_allow_html=True)

    df1 = st.session_state.df
    fi1 = st.session_state.file_info
    if df1 is None:
        st.info("Upload a primary file first.")
        return

    extra_files: list = st.session_state.get("extra_files", [])
    total_mb = _pool_total_mb()

    # ── LLM Wrapper for multi_file_joiner functions ──
    def _llm_caller(prompt: str) -> str:
        if not st.session_state.llm_ready:
            return "{}"
        try:
            res = generate_text_response(
                question=prompt, 
                schema="Multi-File Schema Analysis", 
                context="Generating AI Join Plan", 
                llm_model=st.session_state.llm_model
            )
            return res.get('response', '{}') if isinstance(res, dict) else str(res)
        except Exception:
            return "{}"

    # ── Active-merge banner ──
    if st.session_state.get("using_merged"):
        merged = st.session_state.merged_df
        st.markdown(f"""
        <div style='background:rgba(72,187,120,0.12);border:1px solid rgba(72,187,120,0.35);
                    border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:1rem;'>
            <span style='font-size:1.6rem;'>✅</span>
            <div>
                <p style='color:#48bb78;font-weight:700;margin:0;font-size:0.95rem;'>
                    Merged Dataset Active
                </p>
                <p style='color:rgba(255,255,255,0.55);font-size:0.82rem;margin:0.2rem 0 0;'>
                    Chat &amp; SQL running on combined dataset —
                    {merged.shape[0]:,} rows × {merged.shape[1]} columns
                    {'· ' + str(len(extra_files)) + ' file(s) still in pool' if extra_files else ''}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("↩️ Revert to Original Dataset", key="revert_merge_btn"):
            orig = st.session_state.get("df_original")
            if orig is not None:
                cats   = detect_column_categories(orig)
                schema = generate_smart_schema(orig, fi1, cats)
                kpis   = get_all_kpis(orig, fi1)
                sugs   = generate_smart_suggestions(orig, fi1, cats)
                reset_database()
                load_dataframe_to_db(orig)
                st.session_state.update({
                    "df": orig, "file_info": fi1,
                    "schema": schema, "column_categories": cats,
                    "kpis": kpis["kpis"], "suggestions": sugs,
                    "using_merged": False, "merged_df": None,
                    "extra_files": [], "ai_join_plan": None,
                    "db_loaded": True, "chat_history": [],
                    "query_count": 0, "data_summary": None,
                })
                st.success("✅ Reverted to original dataset.")
                st.rerun()
        st.divider()

    # ── File pool cards ──
    st.markdown("<p class='section-title' style='font-size:0.9rem;'>📂 Dataset Pool</p>", unsafe_allow_html=True)

    all_cards = [{"label": "📄 PRIMARY", "name": fi1.get("file_name", "primary"),
                  "rows": df1.shape[0], "cols": df1.shape[1], "is_primary": True}]
    for ef in extra_files:
        all_cards.append({"label": "📄 EXTRA", "name": ef["name"],
                          "rows": ef["df"].shape[0], "cols": ef["df"].shape[1],
                          "is_primary": False, "size_mb": ef["size_mb"]})

    for row_start in range(0, len(all_cards), 3):
        cols = st.columns(min(3, len(all_cards) - row_start))
        for ci, card in enumerate(all_cards[row_start:row_start+3]):
            border = "rgba(102,126,234,0.5)" if card["is_primary"] else "rgba(72,187,120,0.3)"
            lbl_color = "#a78bfa" if card["is_primary"] else "#48bb78"
            with cols[ci]:
                st.markdown(f"""
                <div class='glass-card' style='padding:0.9rem;border:1px solid {border};'>
                    <p style='color:{lbl_color};font-size:0.7rem;font-weight:700;
                              letter-spacing:0.8px;text-transform:uppercase;margin:0 0 0.3rem;'>
                        {card["label"]}
                    </p>
                    <p style='color:#f7fafc;font-weight:700;margin:0;font-size:0.88rem;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                        {card["name"]}
                    </p>
                    <p style='color:rgba(255,255,255,0.45);font-size:0.78rem;margin:0.25rem 0 0;'>
                        {card["rows"]:,} rows × {card["cols"]} cols {f"· {card.get('size_mb',0):.1f} MB" if not card["is_primary"] else ""}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    pct_used = min(total_mb / _MAX_TOTAL_MB * 100, 100)
    bar_color = "#fc8181" if pct_used > 90 else "#ecc94b" if pct_used > 70 else "#48bb78"
    st.markdown(f"""
    <div style='margin:0.8rem 0;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
            <span style='color:rgba(255,255,255,0.5);font-size:0.78rem;'>Total pool size</span>
            <span style='color:{bar_color};font-size:0.78rem;font-weight:700;'>
                {total_mb:.1f} MB / {_MAX_TOTAL_MB:.0f} MB
            </span>
        </div>
        <div style='background:rgba(255,255,255,0.08);border-radius:6px;height:6px;'>
            <div style='background:{bar_color};width:{pct_used:.1f}%;height:6px;
                        border-radius:6px;transition:width 0.4s ease;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Multi-file uploader ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-title' style='font-size:0.9rem;'>📤 Add Files to Pool</p>", unsafe_allow_html=True)

    uploaded_list = st.file_uploader(
        "Add files", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
        label_visibility="collapsed", key="multi_file_uploader",
    )

    if uploaded_list:
        existing_names = {ef["name"] for ef in extra_files}
        new_files_added = 0
        size_errors = []

        for uf in uploaded_list:
            if uf.name in existing_names:
                continue

            size_mb = uf.size / (1024 * 1024)
            projected = _pool_total_mb() + size_mb

            if projected > _MAX_TOTAL_MB:
                size_errors.append(f"'{uf.name}' skipped — exceeds {_MAX_TOTAL_MB:.0f} MB pool limit.")
                continue

            with st.spinner(f"Loading {uf.name}..."):
                r = load_file(uf)

            if not r["success"]:
                st.error(f"Could not load '{uf.name}': {r['error']}")
                continue

            extra_files.append({
                "name": uf.name, "df": r["dataframe"],
                "fi": r["file_info"], "size_mb": size_mb,
            })
            existing_names.add(uf.name)
            new_files_added += 1

        if size_errors:
            for msg in size_errors: st.warning(f"⚠️ {msg}")

        if new_files_added:
            st.session_state["extra_files"] = extra_files
            st.session_state["ai_join_plan"] = None
            st.rerun()

    if extra_files:
        with st.expander("🗑️ Remove a file from pool"):
            remove_name = st.selectbox("Select file to remove", [ef["name"] for ef in extra_files], key="remove_file_sel")
            if st.button("Remove selected file", key="remove_file_btn"):
                st.session_state["extra_files"] = [ef for ef in extra_files if ef["name"] != remove_name]
                st.session_state["ai_join_plan"] = None
                st.rerun()

    if not extra_files:
        st.info("⬆️ Upload one or more files above to start joining.")
        return

    st.divider()

    # ── 🤖 AI AUTO-JOIN ORCHESTRATOR ──
    st.markdown("<p class='section-title'>🧠 AI Auto-Join Orchestrator</p>", unsafe_allow_html=True)
    st.markdown("""<p style='color:rgba(255,255,255,0.6); font-size:0.9rem;'>Let the Neural Engine analyze the schemas of all files in your pool to determine the optimal sequence, column keys, and join types to combine your data seamlessly.</p>""", unsafe_allow_html=True)

    if st.button("✨ Ask AI to Analyze Pool & Plan Joins", use_container_width=True):
        if not st.session_state.llm_ready:
            st.error("Please connect the AI first in the Settings tab.")
        else:
            with st.spinner("AI is analyzing all dataset schemas and mapping relationships..."):
                plan = ai_plan_multi_join(df1, fi1.get("file_name", "primary"), extra_files, _llm_caller)
                st.session_state["ai_join_plan"] = plan

    if st.session_state.get("ai_join_plan"):
        plan = st.session_state["ai_join_plan"]
        st.success("✅ Neural Engine has mapped the optimal join strategy.")
        
        for p in plan:
            st.markdown(f"""
            <div class='glass-card' style='padding:1.2rem; border-left: 4px solid #a78bfa; margin-bottom: 0.8rem;'>
                <h4 style='margin-top:0; color:#f7fafc;'>Step {p.get('step', 1)}: Join <code>{p.get('right_file', 'unknown')}</code></h4>
                <p style='margin:0.5rem 0;'><b>Strategy:</b> <code>{p.get('left_col')}</code> ↔ <code>{p.get('right_col')}</code> via <span style='color:#4facfe; font-weight:bold;'>{p.get('join_type', 'left').upper()}</span> join.</p>
                <p style='margin:0; font-size: 0.9rem; color:rgba(255,255,255,0.7);'><b>AI Reasoning:</b> {p.get('reason', 'Optimal match based on schema overlap.')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        if st.button("🚀 Execute Full AI Plan Automatically", type="primary", use_container_width=True):
            current_df = df1
            current_fi = fi1
            files_to_remove = []
            
            with st.spinner("Executing sequence and rebuilding database for chat..."):
                for step in plan:
                    r_name = step.get("right_file")
                    r_file = next((f for f in extra_files if f["name"] == r_name), None)
                    if not r_file: continue
                    
                    res = merge_dataframes(current_df, r_file["df"], step.get("left_col"), step.get("right_col"), step.get("join_type", "left"))
                    if res["success"]:
                        current_df = res["dataframe"]
                        current_name = f"{current_fi.get('file_name', 'primary')} ⋈ {r_name}"
                        current_fi = _build_merged_fi(current_df, current_name)
                        files_to_remove.append(r_name)
                    else:
                        st.error(f"Failed to join {r_name}: {res['error']}")
                        break
                        
                if files_to_remove:
                    cats = detect_column_categories(current_df)
                    schema = generate_smart_schema(current_df, current_fi, cats)
                    try:
                        kpis_raw = get_all_kpis(current_df, current_fi)
                        kpis_list = kpis_raw["kpis"] if isinstance(kpis_raw, dict) else kpis_raw
                    except Exception:
                        kpis_list = []
                        
                    sugs = generate_smart_suggestions(current_df, current_fi, cats)
                    reset_database()
                    load_dataframe_to_db(current_df)
                    
                    if st.session_state.get("df_original") is None: # <--- FIXED HERE
                        st.session_state["df_original"] = df1.copy()
                        
                    new_pool = [f for f in extra_files if f["name"] not in files_to_remove]
                    
                    st.session_state.update({
                        "df": current_df, "file_info": current_fi, "schema": schema,
                        "column_categories": cats, "kpis": kpis_list, "suggestions": sugs,
                        "db_loaded": True, "using_merged": True, "merged_df": current_df,
                        "extra_files": new_pool, "ai_join_plan": None, "chat_history": [],
                        "query_count": 0, "data_summary": None, "cleaning_applied": False
                    })
                    generate_ai_summary()
                    st.rerun()

    st.divider()

    # ── 🔍 DEEP DIVE / MANUAL JOIN OVERRIDE ──
    st.markdown("<p class='section-title'>🔍 Deep Dive & Manual Join Settings</p>", unsafe_allow_html=True)
    
    extra_names = [ef["name"] for ef in extra_files]
    right_name = st.selectbox("Inspect individual file for manual join →", extra_names, key="right_file_selectbox")
    right_entry = next((ef for ef in extra_files if ef["name"] == right_name), None)
    
    if right_entry is None: return
    df2 = right_entry["df"]

    # Generate insights on-demand
    if st.button("💡 Generate AI Strategic Insights For This File", key="gen_insights"):
        with st.spinner("Simulating all join types and writing executive summaries..."):
            candidates = detect_joinable_columns(df1, df2)
            strategy = ai_analyze_join_strategy(df1, df2, fi1.get("file_name"), right_name, candidates, _llm_caller)
            results_all = execute_all_join_types(df1, df2, strategy['best_left'], strategy['best_right'])
            insights = ai_join_type_insights(df1, df2, results_all, fi1.get("file_name"), right_name, strategy['best_left'], strategy['best_right'], _llm_caller)
            
            st.markdown(f"**🧠 Top AI Recommendation:** {strategy.get('reasoning', 'Best columns based on schema evaluation.')}")
            
            m1, m2 = st.columns(2)
            cols = [m1, m2, m1, m2]
            for i, jtype in enumerate(["inner", "left", "right", "outer"]):
                with cols[i]:
                    st.markdown(f"""
                    <div class='glass-card' style='padding:1rem; margin-bottom: 1rem;'>
                        <h5 style='margin:0; color:#a78bfa; text-transform:capitalize;'>{jtype} Join</h5>
                        <p style='margin:0.5rem 0 0; font-size:0.85rem; color:rgba(255,255,255,0.8);'>{insights.get(jtype, 'Insight unavailable.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
    # Manual overriding inputs
    candidates = detect_joinable_columns(df1, df2)
    best = candidates[0] if candidates else {}
    
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1: left_col = st.selectbox("Primary column", list(df1.columns), index=list(df1.columns).index(best.get("col_df1", list(df1.columns)[0])) if best.get("col_df1") in df1.columns else 0, key="join_left_col")
    with cfg2: right_col = st.selectbox(f"Column from '{right_name}'", list(df2.columns), index=list(df2.columns).index(best.get("col_df2", list(df2.columns)[0])) if best.get("col_df2") in df2.columns else 0, key="join_right_col")
    with cfg3: join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=1, key="join_type_sel")

    if st.button("✅ Apply Manual Join", key="apply_manual_btn", use_container_width=True):
        with st.spinner("Merging and reloading database..."):
            result = merge_dataframes(df1, df2, left_on=left_col, right_on=right_col, how=join_type)

        if not result["success"]:
            st.error(f"❌ Merge failed: {result['error']}")
            return

        merged_df = result["dataframe"]
        merged_name = f"{fi1.get('file_name', 'primary')} ⋈ {right_name}"
        merged_fi = _build_merged_fi(merged_df, merged_name)
        cats = detect_column_categories(merged_df)
        schema = generate_smart_schema(merged_df, merged_fi, cats)

        try:
            kpis_raw = get_all_kpis(merged_df, merged_fi)
            kpis_list = kpis_raw["kpis"] if isinstance(kpis_raw, dict) else kpis_raw
        except Exception:
            kpis_list = []

        sugs = generate_smart_suggestions(merged_df, merged_fi, cats)
        reset_database()
        load_dataframe_to_db(merged_df)

        if st.session_state.get("df_original") is None: # <--- FIXED HERE
            st.session_state["df_original"] = df1.copy()

        new_pool = [ef for ef in extra_files if ef["name"] != right_name]

        st.session_state.update({
            "df": merged_df, "file_info": merged_fi, "schema": schema,
            "column_categories": cats, "kpis": kpis_list, "suggestions": sugs,
            "db_loaded": True, "using_merged": True, "merged_df": merged_df,
            "extra_files": new_pool, "ai_join_plan": None, "chat_history": [],
            "query_count": 0, "data_summary": None, "cleaning_applied": False
        })
        generate_ai_summary()
        st.rerun()

def render_refinement_tab():
    st.markdown("<p class='section-title'>✨ AI Data Refinement & Healing</p>", unsafe_allow_html=True)

    df  = st.session_state.df
    fi  = st.session_state.file_info
    health           = get_data_health_score(df, fi)
    cleaning_applied = st.session_state.get('cleaning_applied', False)

    # ── Left column: health grade display ──
    c1, c2 = st.columns([2, 1])
    with c1:
        if cleaning_applied:
            # Recompute grade excluding 'size' — cleaning cannot add rows
            adjustable = {k: v for k, v in health['breakdown'].items() if k != 'size'}
            adj_avg    = sum(adjustable.values()) / len(adjustable) if adjustable else health['score']
            adj_grade  = 'A' if adj_avg >= 90 else 'B' if adj_avg >= 75 else 'C' if adj_avg >= 60 else 'D'

            st.markdown(f"### Current Data Health: {adj_grade}")
            st.success("✅ Smart Cleaning Applied — data quality improved.")

            for issue, score in health['breakdown'].items():
                if issue == 'size':
                    st.write(f"- **{issue.title()}**: {score}/100 *(row count unchanged — not penalised post-clean)*")
                else:
                    st.write(f"- **{issue.title()}**: {score}/100")
        else:
            st.markdown(f"### Current Data Health: {health['grade']}")
            if health['breakdown'].get('size', 100) < 30:
                st.info("💡 Note: Grade is limited by the small 'Size' of this test file.")
            for issue, score in health['breakdown'].items():
                st.write(f"- **{issue.title()}**: {score}/100")

    # ── Right column: action buttons ──
    with c2:
        st.markdown("#### 🪄 AI Quick Fix")

        # ── Apply Smart Cleaning button ──
        if st.button("✨ Apply Smart Cleaning", key="clean_btn", use_container_width=True):

            # ── PRE-FLIGHT: skip if data is already fully clean ────────────
            _nulls     = int(df.isnull().sum().sum())
            _dupes     = int(df.duplicated().sum())
            _neg_cols  = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if any(x in c.lower() for x in ['price','qty','quantity','sales','amount'])
                and bool((df[c].dropna() < 0).any())
            ]
            _already_clean = (_nulls == 0 and _dupes == 0 and len(_neg_cols) == 0)

            if _already_clean:
                st.info(
                    "✅ Your data is already fully clean — "
                    "no nulls, no duplicates, no logical errors detected. "
                    "Smart Cleaning has nothing to fix."
                )
            else:
                _rerun = False
                with st.spinner("Neural Engine healing your dataset..."):
                    try:
                        # 1. Snapshot BEFORE state so the report can diff it
                        df_snapshot = st.session_state.df.copy()

                        # 2. Run the cleaner
                        # Create a quick wrapper function to pass to the cleaner
                        def clean_llm_caller(prompt):
                            from components.llm_engine import generate_text_response
                            res = generate_text_response(
                                question=prompt, 
                                schema="Data Cleaning Context", 
                                context="Infer missing value", 
                                llm_model=st.session_state.llm_model
                            )
                            return res.get('response', 'Unknown') if isinstance(res, dict) else str(res)

                        # 2. Run the AI-enhanced cleaner
                        cleaned_df = auto_clean_data(st.session_state.df, llm_fn=clean_llm_caller)

                        # 3. Generate before/after report immediately after cleaning
                        st.session_state.cleaning_report = generate_cleaning_report(
                            df_snapshot, cleaned_df
                        )

                        # 4. Rebuild file_info from actual cleaned data
                        new_fi = fi.copy()
                        new_fi['num_rows']           = len(cleaned_df)
                        new_fi['has_missing_values']  = bool(cleaned_df.isnull().any().any())
                        new_fi['missing_info']        = {}

                        new_fi['column_details'] = []
                        for col in cleaned_df.columns:
                            null_count = int(cleaned_df[col].isnull().sum())
                            new_fi['column_details'].append({
                                'name':           col,
                                'type':           str(cleaned_df[col].dtype),
                                'non_null_count':  int(cleaned_df[col].count()),
                                'null_count':     null_count,
                                'unique_count':   int(cleaned_df[col].nunique()),
                                'percentage':     round(null_count / len(cleaned_df) * 100, 2) if len(cleaned_df) else 0.0
                            })

                        # 5. Sync all session state
                        st.session_state.df                = cleaned_df
                        st.session_state.file_info         = new_fi
                        st.session_state.column_categories  = detect_column_categories(cleaned_df)
                        st.session_state.schema            = generate_smart_schema(
                            cleaned_df, new_fi, st.session_state.column_categories
                        )
                        st.session_state.kpis              = get_all_kpis(cleaned_df, new_fi)['kpis']
                        st.session_state.cleaning_applied   = True
                        st.session_state.data_summary       = None
                        generate_ai_summary()

                        # 6. Sync the SQL database
                        reset_database()
                        db_result = load_dataframe_to_db(cleaned_df)
                        if db_result['success']:
                            st.session_state.db_loaded = True
                        else:
                            st.session_state.db_loaded = False
                            st.error(f"⚠️ Database reload failed: {db_result['message']}")

                        st.success("🎉 Data Healed! Refreshing...")
                        _rerun = True

                    except Exception as e:
                        st.error(f"❌ Cleaning failed: {e}")

                # st.rerun() MUST sit outside the spinner — calling it inside
                # causes RerunException to be swallowed by the spinner's __exit__
                if _rerun:
                    st.rerun()

        # ── Download button — only shown after cleaning is applied ──
        if cleaning_applied:
            st.markdown("<br>", unsafe_allow_html=True)   # breathing room between buttons

            original_name = st.session_state.get('current_file_name', 'data')
            # Strip the original extension and append _cleaned.csv
            base_name     = original_name.rsplit('.', 1)[0]
            download_name = f"{base_name}_cleaned.csv"

            csv_bytes = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
                use_container_width=True,
                key="download_cleaned_csv",
                help=f"Downloads the cleaned dataset as '{download_name}'"
            )

            # Show a compact stat line so user knows what they're downloading
            rows, cols = df.shape
            nulls      = int(df.isnull().sum().sum())
            null_label = f"{nulls} null(s) remain" if nulls > 0 else "0 nulls"
            st.caption(f"{rows:,} rows · {cols} columns · {null_label}")

        # ── Before/After Cleaning Report ──
    report = st.session_state.get('cleaning_report')
    if cleaning_applied and report:

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>📋 Before / After Cleaning Report</p>",
                    unsafe_allow_html=True)

        # ── Summary metric strip ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='kpi-premium c1' style='padding:1rem;'>
                <p class='kpi-val'>{report['total_nulls_filled']}</p>
                <p class='kpi-lbl'>Nulls Filled</p>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='kpi-premium c2' style='padding:1rem;'>
                <p class='kpi-val'>{report['duplicates_removed']}</p>
                <p class='kpi-lbl'>Duplicates Removed</p>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class='kpi-premium c3' style='padding:1rem;'>
                <p class='kpi-val'>{report['columns_changed']}</p>
                <p class='kpi-lbl'>Columns Changed</p>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class='kpi-premium c4' style='padding:1rem;'>
                <p class='kpi-val'>{report['rows_after']:,}</p>
                <p class='kpi-lbl'>Rows After Clean</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Per-column detail table ──
        # Uses pd.DataFrame.to_html() — same pattern as render_preview_tab /
        # render_schema_tab — avoids deeply-nested f-string HTML that confuses
        # Streamlit's markdown renderer and causes raw text display.
        report_rows = []
        for col in report['columns']:
            nulls_str = (
                f"{col['nulls_before']} -> {col['nulls_after']} (-{col['nulls_filled']} fixed)"
                if col['nulls_filled'] > 0
                else str(col['nulls_before'])
            )
            type_str = (
                f"{col['dtype_before']} -> {col['dtype_after']}"
                if col['dtype_before'] != col['dtype_after']
                else col['dtype_before']
            )
            unique_str = (
                f"{col['unique_before']} -> {col['unique_after']}"
                if col['unique_before'] != col['unique_after']
                else str(col['unique_before'])
            )
            action_str  = "  |  ".join(col['actions']) if col['actions'] else "No changes"
            status_str  = "Yes" if col['changed'] else "-"

            report_rows.append({
                'Column':         col['name'],
                'Type':           type_str,
                'Nulls':          nulls_str,
                'Unique Values':  unique_str,
                'Actions Taken':  action_str,
                'Changed':        status_str,
            })

        report_df  = pd.DataFrame(report_rows)
        html_table = report_df.to_html(index=False, classes="glass-table")
        st.markdown(
            f'<div style="overflow-x: auto; padding-bottom: 10px;">{html_table}</div>',
            unsafe_allow_html=True
        )

        # ── Row-level summary footnote ──
        st.caption(
            f"Dataset: {report['rows_before']:,} rows before → "
            f"{report['rows_after']:,} rows after "
            f"({report['duplicates_removed']} duplicate row(s) removed)."
        )
# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────
def reset_all():
    reset_database()
    for k in [
        # original keys
        'df', 'file_info', 'schema', 'column_categories',
        'kpis', 'suggestions', 'chat_history',
        'file_uploaded', 'current_file_name', 'db_loaded',
        'data_summary', 'query_count', 'show_chat',
        'cleaning_applied', 'cleaning_report',
        'pdf_report', 'pdf_report_name',
        # join-feature keys (N-file pool)
        'extra_files', 'join_candidates', 'join_right_name',
        'merged_df', 'df_original', 'using_merged',
    ]:
        if k in st.session_state:
            del st.session_state[k]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_css()
    load_animated_bg()
    init_session_state()

    if not st.session_state.llm_ready:
        initialize_llm()

    render_header()
    render_navbar()

    if st.session_state.file_uploaded:
        render_dashboard()
    else:
        render_welcome()

    load_footer()


if __name__ == "__main__":
    main()