import pandas as pd
import numpy as np

def generate_cleaning_plan(health_report, schema, model):
    """Asks the AI to suggest specific cleaning steps based on health issues."""
    prompt = f"""
    Act as a Senior Data Engineer. Analyze this dataset health report:
    {health_report}
    
    Current Schema:
    {schema}
    
    List the top 3-5 critical cleaning steps needed (e.g., handling nulls, fixing types, removing outliers).
    For each step, provide a brief explanation of WHY it is necessary for accurate BI analysis.
    """
    # Use your existing generate_text_response logic here
    # return AI-generated steps
    pass

def auto_clean_data(df):
    """Applies standard AI-recommended best practices for cleaning."""
    new_df = df.copy()
    
    # 1. Automatic Type Inference
    for col in new_df.columns:
        if new_df[col].dtype == 'object':
            try:
                new_df[col] = pd.to_datetime(new_df[col])
            except:
                pass
                
    # 2. Simple Null Handling (Based on industry standards)
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns
    new_df[numeric_cols] = new_df[numeric_cols].fillna(new_df[numeric_cols].median())
    
    char_cols = new_df.select_dtypes(include=['object']).columns
    new_df[char_cols] = new_df[char_cols].fillna("Unknown")
    
    return new_df