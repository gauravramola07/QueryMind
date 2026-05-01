# test_schema.py

import sys
sys.path.append('.')

import pandas as pd
from components.data_loader import clean_dataframe, generate_file_info
from utils.helpers import (
    detect_column_categories,
    detect_business_kpis,
    generate_smart_schema,
    generate_smart_suggestions,
    detect_best_chart_type,
    get_data_health_score
)
from utils.kpi_detector import get_all_kpis

print("=" * 60)
print("🧪 TESTING SCHEMA DETECTION")
print("=" * 60)

# ── Load data ─────────────────────────────────
df = pd.read_csv('data/sample_sales_data.csv')
df_clean = clean_dataframe(df)
file_info = generate_file_info(df_clean, 'sample_sales_data.csv', 50000, '.csv')

# ── Test 1: Column Categories ─────────────────
print("\n📋 Test 1: Column Categories")
print("-" * 40)
categories = detect_column_categories(df_clean)
for cat_name, cols in categories.items():
    if cols:
        print(f"  {cat_name}: {cols}")

# ── Test 2: KPI Detection ─────────────────────
print("\n💰 Test 2: KPI Detection")
print("-" * 40)
kpi_result = get_all_kpis(df_clean, file_info)
for kpi in kpi_result['kpis']:
    print(f"  {kpi['label']}: {kpi['formatted_value']}")

# ── Test 3: Smart Schema ──────────────────────
print("\n📊 Test 3: Smart Schema (for LLM)")
print("-" * 40)
schema = generate_smart_schema(df_clean, file_info, categories)
print(schema)

# ── Test 4: Smart Suggestions ─────────────────
print("\n💡 Test 4: Smart Question Suggestions")
print("-" * 40)
suggestions = generate_smart_suggestions(df_clean, file_info, categories)
for i, suggestion in enumerate(suggestions, 1):
    print(f"  {i}. {suggestion}")

# ── Test 5: Chart Type Detection ─────────────
print("\n📈 Test 5: Chart Type Detection")
print("-" * 40)
test_cases = [
    ("What is the revenue trend over time?", df_clean[['date', 'revenue']]),
    ("Show distribution by region", df_clean[['region', 'revenue']]),
    ("Top 5 products by sales", df_clean[['product', 'revenue']].head(5)),
    ("Correlation between price and revenue", df_clean[['unit_price', 'revenue']]),
]
for question, result_df in test_cases:
    chart_type = detect_best_chart_type(result_df, question)
    print(f"  Q: '{question}'")
    print(f"  → Chart: {chart_type}\n")

# ── Test 6: Data Health Score ─────────────────
print("🏥 Test 6: Data Health Score")
print("-" * 40)
health = get_data_health_score(df_clean, file_info)
print(f"  Overall Score: {health['overall']}/100")
print(f"  Grade: {health['grade']} ({health['label']})")
print(f"  Breakdown:")
for metric, score in health['breakdown'].items():
    print(f"    {metric}: {score}")

print("\n" + "=" * 60)
print("✅ ALL SCHEMA TESTS COMPLETE!")
print("=" * 60)