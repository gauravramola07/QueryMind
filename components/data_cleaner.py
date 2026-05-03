import pandas as pd
import numpy as np

# Change the definition to accept the LLM function
def auto_clean_data(df, llm_fn=None):
    new_df = df.copy()
    
    # ... (Keep Steps 1 through 4 exactly as they are) ...

    # 5. AI-Powered Deep Clean for Categorical/Object Columns
    if llm_fn:
        # If AI is connected, use the smart imputer first
        new_df = ai_smart_impute(new_df, llm_fn)
        
    # Catch-all for remaining text columns (or if AI wasn't passed)
    char_cols = new_df.select_dtypes(include=['object', 'category']).columns
    for col in char_cols:
        new_df[col] = new_df[col].fillna("Unknown").astype(str).str.strip()
        
        # Fix inconsistent price/currency formatting (keep your existing logic here)
        if any(x in col.lower() for x in ['price', 'cost', 'amount', 'fee', 'revenue']):
            cleaned_num = (
                new_df[col]
                .str.replace(r'[^\d.\-]', '', regex=True)
                .replace('', np.nan)
            )
            if cleaned_num.notna().sum() / len(new_df) > 0.7:
                new_df[col] = pd.to_numeric(cleaned_num, errors='coerce').fillna(0)

    # ... (Keep Steps 6 and 7 exactly as they are) ...
    
    return new_df

def generate_cleaning_report(df_before, df_after):
    """
    Compare df_before and df_after column by column and return a
    structured dict describing exactly what changed during cleaning.
    """
    rows_before = len(df_before)
    rows_after  = len(df_after)

    total_nulls_before = int(df_before.isnull().sum().sum())
    total_nulls_after  = int(df_after.isnull().sum().sum())

    columns = []
    for col in df_before.columns:
        if col not in df_after.columns:
            continue

        before_series = df_before[col]
        after_series  = df_after[col]

        nulls_before  = int(before_series.isnull().sum())
        nulls_after   = int(after_series.isnull().sum())
        nulls_filled  = max(0, nulls_before - nulls_after)

        dtype_before  = str(before_series.dtype)
        dtype_after   = str(after_series.dtype)

        unique_before = int(before_series.nunique(dropna=True))
        unique_after  = int(after_series.nunique(dropna=True))

        # Build a plain-English list of actions that were taken on this column
        actions = []

        if nulls_filled > 0:
            if np.issubdtype(after_series.dtype, np.number):
                median_val = round(after_series.median(), 2)
                actions.append(f"Filled {nulls_filled} null(s) with median ({median_val})")
            else:
                actions.append(f"Filled {nulls_filled} null(s) with 'Unknown'")

        if dtype_before != dtype_after:
            actions.append(f"Type converted: {dtype_before} → {dtype_after}")

        # Detect if negatives were corrected (abs applied)
        if np.issubdtype(before_series.dtype, np.number):
            had_negatives = bool((before_series.dropna() < 0).any())
            has_negatives = bool((after_series.dropna() < 0).any())
            if had_negatives and not has_negatives:
                neg_count = int((before_series.dropna() < 0).sum())
                actions.append(f"Fixed {neg_count} negative value(s)")

        # Detect string standardisation (whitespace stripped)
        if dtype_before == 'object' and dtype_after == 'object':
            stripped = before_series.dropna().astype(str).str.strip()
            if not stripped.equals(before_series.dropna().astype(str)):
                actions.append("Stripped whitespace")

        # Detect duplicate rows removed — reported at row level, shown per first column
        changed = bool(actions) or (dtype_before != dtype_after)

        columns.append({
            'name':          col,
            'dtype_before':  dtype_before,
            'dtype_after':   dtype_after,
            'nulls_before':  nulls_before,
            'nulls_after':   nulls_after,
            'nulls_filled':  nulls_filled,
            'unique_before': unique_before,
            'unique_after':  unique_after,
            'actions':       actions,
            'changed':       changed,
        })

    return {
        'rows_before':        rows_before,
        'rows_after':         rows_after,
        'duplicates_removed': max(0, rows_before - rows_after),
        'total_nulls_before': total_nulls_before,
        'total_nulls_after':  total_nulls_after,
        'total_nulls_filled': max(0, total_nulls_before - total_nulls_after),
        'columns_changed':    sum(1 for c in columns if c['changed']),
        'columns_unchanged':  sum(1 for c in columns if not c['changed']),
        'columns':            columns,
    }

import json

def ai_smart_impute(df, llm_fn):
    """
    Uses AI to infer and fill missing categorical/text values 
    based on the context of the other columns in the same row.
    """
    new_df = df.copy()
    
    # Find columns that are text/categorical and have missing values
    text_cols_with_nulls = [
        col for col in new_df.select_dtypes(include=['object', 'category']).columns 
        if new_df[col].isnull().sum() > 0
    ]
    
    if not text_cols_with_nulls:
        return new_df # Nothing to impute

    # Limit to a maximum number of rows to avoid massive API delays/costs
    # We will only AI-impute if there are fewer than 50 missing rows total
    total_nulls = sum(new_df[col].isnull().sum() for col in text_cols_with_nulls)
    if total_nulls > 50:
        # Fallback to rule-based if there's too much missing data
        for col in text_cols_with_nulls:
            new_df[col] = new_df[col].fillna("Unknown")
        return new_df

    # Process each column that has missing values
    for target_col in text_cols_with_nulls:
        # Get the rows where this specific column is missing
        missing_mask = new_df[target_col].isnull()
        rows_to_fix = new_df[missing_mask]
        
        for index, row in rows_to_fix.iterrows():
            # Convert the row (excluding the missing target) to a JSON string for context
            context_data = row.drop(target_col).dropna().to_dict()
            
            prompt = f"""
            You are a data cleaning assistant. 
            I have a row of dataset where the column '{target_col}' is missing.
            
            Here is the rest of the data in that row:
            {json.dumps(context_data, indent=2)}
            
            Based on this context, what is the most logical value for '{target_col}'?
            Reply ONLY with the best guess value. Do not include quotes, explanations, or periods.
            If you absolutely cannot guess, reply with "Unknown".
            """
            
            try:
                # Call your LLM
                ai_guess = llm_fn(prompt).strip()
                # Clean up the response just in case the LLM added quotes
                ai_guess = ai_guess.strip("'\"") 
                
                # Apply the fix
                new_df.at[index, target_col] = ai_guess
            except Exception:
                new_df.at[index, target_col] = "Unknown"
                
    return new_df