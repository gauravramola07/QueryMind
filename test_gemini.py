# test_gemini.py

import sys
sys.path.append('.')

import pandas as pd
from components.data_loader import clean_dataframe, generate_file_info
from components.llm_engine import (
    setup_gemini,
    generate_sql_query,
    generate_data_summary,
    test_gemini_connection
)
from utils.helpers import (
    detect_column_categories,
    generate_smart_schema,
    detect_business_kpis
)

print("=" * 60)
print("🧪 TESTING GEMINI AI CONNECTION")
print("=" * 60)

# ── Setup Gemini ──────────────────────────────
print("\n🤖 Step 1: Setting up Gemini...")
gemini_result = setup_gemini()

if not gemini_result['success']:
    print(f"❌ Setup failed: {gemini_result['error']}")
    exit()

model = gemini_result['model']
print("✅ Gemini setup successful!")

# ── Test Connection ───────────────────────────
print("\n📡 Step 2: Testing connection...")
test_result = test_gemini_connection(model)
print(test_result['message'])

# ── Load Data ─────────────────────────────────
print("\n📁 Step 3: Loading sample data...")
df = pd.read_csv('data/sample_sales_data.csv')
df_clean = clean_dataframe(df)
file_info = generate_file_info(df_clean, 'sample_sales_data.csv', 50000, '.csv')
categories = detect_column_categories(df_clean)
schema = generate_smart_schema(df_clean, file_info, categories)
print("✅ Data loaded!")

# ── Test SQL Generation ───────────────────────
print("\n💬 Step 4: Testing Text-to-SQL...")

test_questions = [
    "What is the total revenue by region?",
    "Show me the top 5 products by profit",
    "What is the average customer rating by product?",
]

for question in test_questions:
    print(f"\n❓ Question: {question}")
    result = generate_sql_query(question, schema, model)
    
    if result['success']:
        print(f"✅ Type: {result['response_type']}")
        if result['response_type'] == 'sql':
            print(f"📝 SQL: {result['sql_query']}")
            print(f"💡 Explanation: {result['explanation']}")
    else:
        print(f"❌ Error: {result['error']}")

# ── Test Data Summary ─────────────────────────
print("\n📊 Step 5: Testing Data Summary Generation...")
kpis = detect_business_kpis(df_clean, categories)
summary_result = generate_data_summary(schema, kpis, model)

if summary_result['success']:
    print("✅ Summary generated!")
    print("\n" + summary_result['summary'])
else:
    print(f"❌ Error: {summary_result['error']}")

print("\n" + "=" * 60)
print("✅ ALL GEMINI TESTS COMPLETE!")
print("=" * 60)