import pandas as pd
import numpy as np

def auto_clean_data(df):
    new_df = df.copy()
    
    # 1. Strip whitespace from column names and string data
    new_df.columns = [str(c).strip() for c in new_df.columns]
    
    # 2. Convert all types of "Empty" to actual NaN
    new_df = new_df.replace(r'^\s*$', np.nan, regex=True)
    new_df = new_df.replace(['None', 'null', 'nan', 'N/A'], np.nan)
    
    # 3. Drop fully empty rows and duplicates
    new_df = new_df.dropna(how='all')
    new_df = new_df.drop_duplicates()
    
    # 4. Deep Clean Numeric Columns
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        new_df[col] = new_df[col].fillna(new_df[col].median())
        if any(x in col.lower() for x in ['qty', 'quantity', 'price', 'sales']):
            new_df[col] = new_df[col].abs() # Fix negative logical errors
            
    # 5. Deep Clean Categorical/Object Columns
    char_cols = new_df.select_dtypes(include=['object', 'category']).columns
    for col in char_cols:
        new_df[col] = new_df[col].fillna("Unknown").astype(str).str.strip()
    
    return new_df