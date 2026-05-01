# config.py
# ============================================
# PROJECT CONFIGURATION FILE
# QueryMind - Business Intelligence Chatbot
# ============================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── PROJECT INFO ──────────────────────────────
PROJECT_NAME = "QueryMind"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "AI-Powered Business Intelligence Chatbot"
AUTHOR = "Your Name"  # ← Change this to your name

# ── API CONFIGURATION ─────────────────────────
# ── API CONFIGURATION ─────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── LLM PROVIDER ──────────────────────────────
LLM_PROVIDER = "groq"          # "groq" or "gemini"

# ── LLM SETTINGS ──────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq model (free)
MAX_TOKENS = 2048
TEMPERATURE = 0.1                  # Low = more precise/deterministic

# ── DATA SETTINGS ─────────────────────────────
MAX_FILE_SIZE_MB = 50              # Maximum upload file size
MAX_ROWS_PREVIEW = 5               # Rows to show in data preview
MAX_ROWS_QUERY = 10000             # Max rows to process in query
SUPPORTED_FORMATS = [".csv", ".xlsx", ".xls"]

# ── DATABASE SETTINGS ─────────────────────────
DB_TABLE_NAME = "uploaded_data"    # SQLite table name
DB_CONNECTION = "sqlite:///:memory:"  # In-memory database

# ── UI SETTINGS ───────────────────────────────
PAGE_TITLE = "QueryMind - Talk to Your Data"
PAGE_ICON = "🧠"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# ── CHART SETTINGS ────────────────────────────
DEFAULT_CHART_HEIGHT = 500
DEFAULT_CHART_THEME = "plotly_white"
MAX_CHART_CATEGORIES = 20          # Max categories in bar chart

# ── CHAT SETTINGS ─────────────────────────────
MAX_CHAT_HISTORY = 20              # Remember last 20 messages
SYSTEM_ROLE = "Business Intelligence Analyst"

# ── PATHS ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

# ── SAMPLE QUESTIONS ──────────────────────────
# These appear as suggestions in the chat
SAMPLE_QUESTIONS = [
    "What is the total revenue by category?",
    "Show me the top 10 records by value",
    "What are the monthly trends?",
    "Give me a summary of this dataset",
    "Which region has the highest sales?",
    "Show me a correlation between columns",
    "What is the average value by group?",
    "Identify any outliers in the data",
]

# ── COLOR SCHEME ──────────────────────────────
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#48bb78",
    "warning": "#ed8936",
    "error": "#fc8181",
    "background": "#f7fafc",
}

print(f"✅ Config loaded for {PROJECT_NAME} v{PROJECT_VERSION}")
