import pandas as pd
import numpy as np

def auto_clean_data(df):
    new_df = df.copy()
    
    # 1. Standardize: Remove whitespace from column names and string data
    new_df.columns = [str(c).strip() for c in new_df.columns]
    
    # 2. Convert "Hidden Nulls" (whitespace or empty strings) to actual NaN[cite: 5]
    new_df = new_df.replace(r'^\s*$', np.nan, regex=True)
    
    # 3. Remove completely empty rows and duplicates[cite: 5]
    new_df = new_df.dropna(how='all')
    new_df = new_df.drop_duplicates()
    
    # 4. Fix Numeric Columns[cite: 5]
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Fill nulls with median[cite: 5]
        new_df[col] = new_df[col].fillna(new_df[col].median())
        # Fix logical errors (e.g., negative quantities become absolute)[cite: 5]
        if 'quantity' in col.lower() or 'price' in col.lower():
            new_df[col] = new_df[col].abs()
            
    # 5. Fix Text/Object Columns[cite: 5]
    char_cols = new_df.select_dtypes(include=['object']).columns
    for col in char_cols:
        new_df[col] = new_df[col].fillna("Unknown").astype(str).str.strip()
    
    return new_df