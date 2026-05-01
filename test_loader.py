# test_loader.py

import sys
sys.path.append('.')

import pandas as pd
from components.data_loader import (
    clean_dataframe,
    generate_file_info,
    get_schema_for_llm,
    get_quick_stats
)

# Load sample data
df = pd.read_csv('data/sample_sales_data.csv')
print('✅ File loaded with pandas')
print(f'Shape: {df.shape}')

# Test clean_dataframe
df_clean = clean_dataframe(df)
print(f'✅ DataFrame cleaned')
print(f'Columns: {df_clean.columns.tolist()}')

# Test generate_file_info
file_info = generate_file_info(
    df_clean,
    'sample_sales_data.csv',
    50000,
    '.csv'
)
print(f'✅ File info generated')
print(f'Numeric columns: {file_info["numeric_columns"]}')
print(f'Text columns: {file_info["text_columns"]}')

# Test get_schema_for_llm
schema = get_schema_for_llm(df_clean, file_info)
print(f'✅ Schema generated for LLM')
print(schema[:500])

# Test get_quick_stats
stats = get_quick_stats(df_clean)
print(f'✅ Quick stats generated')
for stat in stats:
    print(f'  {stat["icon"]} {stat["label"]}: {stat["value"]}')