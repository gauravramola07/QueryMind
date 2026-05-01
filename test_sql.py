# test_sql.py

import sys
sys.path.append('.')

import pandas as pd
from components.data_loader import clean_dataframe, generate_file_info
from components.sql_executor import (
    load_dataframe_to_db,
    execute_sql_query,
    get_table_info,
    get_sample_query_results,
    reset_database
)

print("=" * 50)
print("🧪 TESTING SQL EXECUTOR")
print("=" * 50)

# ── Load sample data ──────────────────────────
print("\n📁 Step 1: Loading sample data...")
df = pd.read_csv('data/sample_sales_data.csv')
df_clean = clean_dataframe(df)
print(f"✅ Data loaded: {df_clean.shape}")

# ── Load into SQLite ──────────────────────────
print("\n🗄️ Step 2: Loading into SQLite...")
result = load_dataframe_to_db(df_clean)
print(f"Result: {result['message']}")

# ── Get table info ────────────────────────────
print("\n📋 Step 3: Getting table info...")
table_info = get_table_info()
print(f"Table: {table_info['table_name']}")
print(f"Rows: {table_info['row_count']}")
print(f"Columns: {table_info['num_columns']}")

# ── Test SQL queries ──────────────────────────
print("\n🔍 Step 4: Testing SQL queries...")

# Test 1: Basic SELECT
print("\n--- Test 1: Basic SELECT ---")
r1 = execute_sql_query("SELECT * FROM uploaded_data LIMIT 3")
print(f"Success: {r1['success']}")
print(r1['dataframe'])

# Test 2: Aggregation
print("\n--- Test 2: Aggregation ---")
r2 = execute_sql_query("""
    SELECT region, SUM(revenue) as total_revenue 
    FROM uploaded_data 
    GROUP BY region 
    ORDER BY total_revenue DESC
""")
print(f"Success: {r2['success']}")
print(r2['dataframe'])

# Test 3: Filter
print("\n--- Test 3: Filter ---")
r3 = execute_sql_query("""
    SELECT product, AVG(customer_rating) as avg_rating
    FROM uploaded_data 
    GROUP BY product
    ORDER BY avg_rating DESC
""")
print(f"Success: {r3['success']}")
print(r3['dataframe'])

# Test 4: Safety Check (should FAIL)
print("\n--- Test 4: Safety Check (DROP should fail) ---")
r4 = execute_sql_query("DROP TABLE uploaded_data")
print(f"Success: {r4['success']}")
print(f"Error: {r4['error']}")

# ── Reset database ────────────────────────────
print("\n🔄 Step 5: Reset database...")
reset_database()
print("✅ Database reset!")

print("\n" + "=" * 50)
print("✅ ALL TESTS COMPLETE!")
print("=" * 50)