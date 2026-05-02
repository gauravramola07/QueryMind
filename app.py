# -*- coding: utf-8 -*-
# app.py

import os
import sys

# 1. CRITICAL: Setup the path BEFORE any local imports to fix ImportError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 2. Now import local components
from components.data_cleaner import auto_clean_data
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
    setup_gemini, generate_sql_query,
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
    page_title="QueryMind — Talk to Your Data",
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
    .css-1d391kg { display: none !important; }

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

    /* Hide Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    
    /* ── The Ironclad AI Toggle ── */
    
    /* 1. Target the overall container to add a premium border */
    [data-testid="stToggle"] {
        padding: 10px !important;
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* 2. Target the label text specifically */
    [data-testid="stToggle"] p {
        font-weight: 700 !important;
        color: white !important;
        font-size: 1.1rem !important;
    }

    /* 3. The "Switch" container - forcing the background to NEVER be red */
    [data-testid="stToggle"] div[role="switch"] {
        height: 1.5rem !important;
        width: 2.8rem !important;
        background-color: rgba(255, 255, 255, 0.1) !important; /* Default OFF color */
    }

    /* 4. Target the ACTIVE state - Using multiple selectors to "win" the CSS war */
    [data-testid="stToggle"] div[aria-checked="true"],
    [data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        background-color: transparent !important; /* Kills the red layer underneath */
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.6) !important;
    }

    /* 5. Target the moving knob - making it larger and cleaner */
    [data-testid="stToggle"] div[role="switch"] > div {
        background-color: white !important;
        border: none !important;
        transform: scale(0.9) !important;
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
        'cleaning_applied': False,   # ← ADD THIS
        'df': None,
        'file_info': None, 'schema': None,
        'column_categories': None, 'kpis': None,
        'suggestions': None, 'chat_history': [],
        'llm_model': None, 'llm_ready': False,
        'db_loaded': False, 'file_uploaded': False,
        'current_file_name': None, 'show_sql': True,
        'chart_type': 'auto', 'data_summary': None,
        'query_count': 0, 'show_chat': False,
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
        <p class='hero-tagline'>"Transforming raw data into actionable business intelligence"</p>
    </div>
    """, unsafe_allow_html=True)


def render_navbar():
    ai_dot = "green" if st.session_state.llm_ready else "red"
    ai_text = "Connected" if st.session_state.llm_ready else "Offline"
    file_text = st.session_state.file_info['file_name'] if st.session_state.file_uploaded else "No file"

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
        r = setup_gemini()
        if r['success']:
            st.session_state.llm_model = r['model']
            st.session_state.llm_ready = True
        else:
            st.error(r['error'])


def generate_ai_summary():
    try:
        r = generate_data_summary(
            st.session_state.schema,
            st.session_state.kpis,
            st.session_state.llm_model
        )
        if r['success']:
            st.session_state.data_summary = r['summary']
    except:
        pass


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
    
    # Define all 6 tabs in the new requested order
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Data Overview",
        "📈 Visual Analytics",
        "🔍 Data Preview",
        "📋 Schema Info",
        "✨ AI Refinement",
        "💬 Chat & Analysis",  # <--- Moved to second-to-last
        "⚙️ Settings"
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
        render_chat_section()   # <--- Renders the chat here now
    with tab7:
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
        <div class='glass-card' style='text-align:center; padding: 3rem; border: 1px dashed rgba(102,126,234,0.3);'>
            <div style='font-size: 4.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102,126,234,0.4));'>🤖</div>
            <h2 style='margin-bottom: 0.5rem; background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Neural Engine Offline
            </h2>
            <p style='color:rgba(255,255,255,0.5) !important; margin-bottom: 2rem; font-size: 1rem;'>
                Activate the QueryMind AI Assistant to start natural language discovery and automated insights.
            </p>
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

    with st.spinner("🤖 Analyzing..."):
        ensure_db_loaded(st.session_state.df)

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
    return [
        {'question': m.get('question', ''), 'answer_summary': m.get('answer_summary', '')}
        for m in st.session_state.chat_history[-5:]
    ]


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
            st.metric(f"{s['icon']} {s['label']}", s['value'])

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
        clr = {'A': '#48bb78', 'B': '#4299e1', 'C': '#ed8936', 'D': '#fc8181'}.get(h['grade'], '#667eea')
        st.markdown(f"""
        <div style='text-align:center;padding:1rem;'>
            <div style='font-size:3rem;font-weight:900;color:{clr};'>{h['grade']}</div>
            <div style='color:rgba(255,255,255,0.5) !important;'>{h['label']} ({h['overall']}/100)</div>
        </div>
        """, unsafe_allow_html=True)
        for m, s in h['breakdown'].items():
            st.progress(s / 100, text=f"{m.title()}: {s}")

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


# ─────────────────────────────────────────────
# TAB 3: VISUAL ANALYTICS
# ─────────────────────────────────────────────

def render_visual_analytics_tab():
    st.markdown("<p class='section-glass'>📈 Automated Visual Analytics</p>", unsafe_allow_html=True)
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
                        "text/csv", use_container_width=True)

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

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🤖 AI Configuration")
        st.write(f"**Status:** {'✅ Connected' if st.session_state.llm_ready else '❌ Offline'}")
        st.write(f"**Model:** {config.LLM_MODEL}")
        st.write(f"**Provider:** Groq AI")
        if not st.session_state.llm_ready:
            if st.button("🔄 Reconnect AI", use_container_width=True):
                initialize_llm()
                st.rerun()

    with c2:
        st.markdown("#### 📊 Display Settings")
        chart_opts = get_chart_type_options()
        st.session_state.chart_type = st.selectbox(
            "Default Chart Type", list(chart_opts.keys()),
            format_func=lambda x: chart_opts[x], index=0
        )
        st.session_state.show_sql = st.toggle("Show SQL Queries", value=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.rerun()
    with c2:
        if st.button("🔄 Upload New File", use_container_width=True):
            reset_all()
            st.rerun()
    with c3:
        if st.button("🤖 Regenerate Summary", use_container_width=True):
            with st.spinner("Generating..."):
                generate_ai_summary()
                st.rerun()

def render_refinement_tab():
    st.markdown("<p class='section-title'>✨ AI Data Refinement & Healing</p>", unsafe_allow_html=True)

    df = st.session_state.df
    fi = st.session_state.file_info
    health = get_data_health_score(df, fi)
    cleaning_applied = st.session_state.get('cleaning_applied', False)

    c1, c2 = st.columns([2, 1])
    with c1:
        if cleaning_applied:
            # ── FIX 2: Recompute grade EXCLUDING 'size' after cleaning ──
            # Size cannot be improved by cleaning (no rows are added),
            # so it is unfair to penalise data quality grade for it.
            adjustable = {k: v for k, v in health['breakdown'].items() if k != 'size'}
            adj_avg = sum(adjustable.values()) / len(adjustable) if adjustable else health['score']

            if adj_avg >= 90:
                adj_grade = 'A'
            elif adj_avg >= 75:
                adj_grade = 'B'
            elif adj_avg >= 60:
                adj_grade = 'C'
            else:
                adj_grade = 'D'

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
                st.info("💡 Note: Grade is limited by the small 'Size' of this test file (9 records).")
            for issue, score in health['breakdown'].items():
                st.write(f"- **{issue.title()}**: {score}/100")

    with c2:
        st.markdown("#### 🪄 AI Quick Fix")
        if st.button("Apply Smart Cleaning", key="clean_btn", use_container_width=True):

            _rerun = False  # ← Flag controls rerun OUTSIDE the spinner

            with st.spinner("Neural Engine healing your dataset..."):
                try:
                    # 1. Run the cleaner
                    cleaned_df = auto_clean_data(st.session_state.df)

                    # 2. Rebuild file_info from actual cleaned data (not hardcoded)
                    new_fi = fi.copy()
                    new_fi['num_rows'] = len(cleaned_df)
                    new_fi['has_missing_values'] = bool(cleaned_df.isnull().any().any())
                    new_fi['missing_info'] = {}  # Wipe stale missing logs

                    new_fi['column_details'] = []
                    for col in cleaned_df.columns:
                        null_count = int(cleaned_df[col].isnull().sum())
                        new_fi['column_details'].append({
                            'name': col,
                            'type': str(cleaned_df[col].dtype),
                            'non_null_count': int(cleaned_df[col].count()),
                            'null_count': null_count,
                            'unique_count': int(cleaned_df[col].nunique()),
                            'percentage': round(null_count / len(cleaned_df) * 100, 2) if len(cleaned_df) else 0.0
                        })

                    # 3. Sync all session state
                    st.session_state.df = cleaned_df
                    st.session_state.file_info = new_fi
                    st.session_state.column_categories = detect_column_categories(cleaned_df)
                    st.session_state.schema = generate_smart_schema(
                        cleaned_df, new_fi, st.session_state.column_categories
                    )
                    # ── FIX 3: Refresh KPIs so top cards reflect cleaned data ──
                    st.session_state.kpis = get_all_kpis(cleaned_df, new_fi)['kpis']
                    st.session_state.cleaning_applied = True  # ← Triggers grade override on rerun

                    # 4. Sync the SQL database
                    reset_database()
                    load_dataframe_to_db(cleaned_df)

                    st.success("🎉 Data Healed! Refreshing...")
                    _rerun = True  # ← Signal rerun, don't call it yet

                except Exception as e:
                    st.error(f"❌ Cleaning failed: {e}")

            # ── FIX 1: st.rerun() OUTSIDE the spinner context ──
            # Calling st.rerun() inside `with st.spinner()` raises RerunException
            # which the spinner's __exit__ silently swallows, killing the rerun.
            if _rerun:
                st.rerun()
# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────
def reset_all():
    reset_database()
    for k in ['df', 'file_info', 'schema', 'column_categories',
              'kpis', 'suggestions', 'chat_history',
              'file_uploaded', 'current_file_name', 'db_loaded',
              'data_summary', 'query_count', 'show_chat',
              'cleaning_applied']:   # ← ADD 'cleaning_applied'
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
