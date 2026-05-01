# test_charts.py

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from components.chart_generator import (
    generate_chart,
    create_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_scatter_chart,
    create_histogram,
    create_kpi_dashboard,
    detect_chart_type,
    get_chart_type_options
)

print("=" * 60)
print("🧪 TESTING CHART GENERATOR")
print("=" * 60)

# ── Load test data ────────────────────────────
df = pd.read_csv('data/sample_sales_data.csv')
print(f"✅ Data loaded: {df.shape}")

# ── Test 1: Chart Type Detection ──────────────
print("\n📋 Test 1: Chart Type Detection")
print("-" * 40)

test_cases = [
    ("What is revenue by region?",
     df[['region', 'revenue']].groupby('region').sum().reset_index()),
    
    ("Show monthly revenue trend",
     df[['date', 'revenue']].head(12)),
    
    ("What is the distribution by category?",
     df[['category', 'revenue']].groupby('category').sum().reset_index()),
    
    ("Correlation between price and revenue",
     df[['unit_price', 'revenue']]),
    
    ("Show revenue histogram",
     df[['revenue']]),
]

for question, data in test_cases:
    chart_type = detect_chart_type(data, question)
    print(f"  Q: '{question[:45]}...'")
    print(f"  → Detected: {chart_type}\n")

# ── Test 2: Bar Chart ─────────────────────────
print("\n📊 Test 2: Bar Chart")
print("-" * 40)
region_data = df.groupby('region')['revenue'].sum().reset_index()
result = generate_chart(
    region_data,
    "Total revenue by region",
    'bar'
)
print(f"  Success: {result['success']}")
print(f"  Chart type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 3: Line Chart ────────────────────────
print("\n📈 Test 3: Line Chart")
print("-" * 40)
monthly_data = df.groupby('date')['revenue'].sum().reset_index().head(30)
result = generate_chart(
    monthly_data,
    "Revenue trend over time",
    'line'
)
print(f"  Success: {result['success']}")
print(f"  Chart type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 4: Pie Chart ─────────────────────────
print("\n🥧 Test 4: Pie Chart")
print("-" * 40)
category_data = df.groupby('category')['revenue'].sum().reset_index()
result = generate_chart(
    category_data,
    "Revenue distribution by category",
    'pie'
)
print(f"  Success: {result['success']}")
print(f"  Chart type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 5: Scatter Chart ─────────────────────
print("\n🔵 Test 5: Scatter Chart")
print("-" * 40)
scatter_data = df[['unit_price', 'revenue', 'profit']].head(100)
result = generate_chart(
    scatter_data,
    "Correlation between price and revenue",
    'scatter'
)
print(f"  Success: {result['success']}")
print(f"  Chart type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 6: Histogram ─────────────────────────
print("\n📉 Test 6: Histogram")
print("-" * 40)
revenue_data = df[['revenue']]
result = generate_chart(
    revenue_data,
    "Revenue distribution histogram",
    'histogram'
)
print(f"  Success: {result['success']}")
print(f"  Chart type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 7: Auto Detection ────────────────────
print("\n🤖 Test 7: Auto Chart Detection")
print("-" * 40)
product_data = df.groupby('product')['revenue'].sum().reset_index()
result = generate_chart(
    product_data,
    "Show revenue breakdown by product",
    'auto'
)
print(f"  Success: {result['success']}")
print(f"  Auto-detected type: {result['chart_type']}")
print(f"  Figure created: {result['figure'] is not None}")

# ── Test 8: KPI Dashboard ─────────────────────
print("\n💰 Test 8: KPI Dashboard")
print("-" * 40)
kpis = [
    {'label': 'Total Revenue', 'value': 25310000, 'formatted_value': '$25.31M'},
    {'label': 'Total Profit', 'value': 7250000, 'formatted_value': '$7.25M'},
    {'label': 'Total Orders', 'value': 1000, 'formatted_value': '1,000'},
    {'label': 'Avg Rating', 'value': 3.98, 'formatted_value': '3.98'},
]
fig = create_kpi_dashboard(kpis)
print(f"  KPI Dashboard created: {fig is not None}")

# ── Test 9: Available Chart Types ─────────────
print("\n📋 Test 9: Available Chart Types")
print("-" * 40)
options = get_chart_type_options()
for key, label in options.items():
    print(f"  {key}: {label}")

# ── Test 10: Save one chart as HTML ───────────
print("\n💾 Test 10: Save Chart as HTML")
print("-" * 40)
region_data = df.groupby('region')['revenue'].sum().reset_index()
result = generate_chart(
    region_data,
    "Total Revenue by Region",
    'bar'
)
if result['success']:
    result['figure'].write_html('exports/test_chart.html')
    print("  ✅ Chart saved to exports/test_chart.html")
    print("  Open this file in browser to see the chart!")

print("\n" + "=" * 60)
print("✅ ALL CHART TESTS COMPLETE!")
print("=" * 60)