import pandas as pd
import numpy as np

def auto_clean_data(df):
    new_df = df.copy()
    
    # 1. Remove completely empty rows and duplicates
    new_df = new_df.dropna(how='all')
    new_df = new_df.drop_duplicates()
    
    # 2. Fill missing numeric values with Median
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        new_df[col] = new_df[col].fillna(new_df[col].median())
            
    # 3. Fill missing text with 'Unknown'
    char_cols = new_df.select_dtypes(include=['object']).columns
    for col in char_cols:
        new_df[col] = new_df[col].fillna("Unknown")
    
    return new_df