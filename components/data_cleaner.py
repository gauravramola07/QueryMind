import pandas as pd
import numpy as np

def auto_clean_data(df):
    new_df = df.copy()

    # 1. Strip whitespace from column names
    new_df.columns = [str(c).strip() for c in new_df.columns]

    # 2. Convert all representations of empty to actual NaN
    new_df = new_df.replace(r'^\s*$', np.nan, regex=True)
    new_df = new_df.replace(['None', 'null', 'nan', 'N/A', 'n/a', 'NA', '#N/A'], np.nan)

    # 3. Drop fully empty rows and exact duplicates
    new_df = new_df.dropna(how='all')
    new_df = new_df.drop_duplicates()

    # 4. Deep Clean Numeric Columns
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        new_df[col] = new_df[col].fillna(new_df[col].median())
        if any(x in col.lower() for x in ['qty', 'quantity', 'price', 'sales', 'amount']):
            new_df[col] = new_df[col].abs()

    # 5. Deep Clean Categorical/Object Columns
    #    Also handles price columns stored as strings (e.g. "$10.00", "10.5 USD")
    char_cols = new_df.select_dtypes(include=['object', 'category']).columns
    for col in char_cols:
        new_df[col] = new_df[col].fillna("Unknown").astype(str).str.strip()

        # Fix inconsistent price/currency formatting → convert to clean numeric if possible
        if any(x in col.lower() for x in ['price', 'cost', 'amount', 'fee', 'revenue']):
            cleaned_num = (
                new_df[col]
                .str.replace(r'[^\d.\-]', '', regex=True)   # strip $, commas, spaces, etc.
                .replace('', np.nan)
            )
            if cleaned_num.notna().sum() / len(new_df) > 0.7:  # >70% parseable → convert
                new_df[col] = pd.to_numeric(cleaned_num, errors='coerce').fillna(0)

    # 6. ── FIX: Handle Datetime Columns (previously skipped entirely) ──
    date_cols = new_df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    for col in date_cols:
        if new_df[col].isnull().any():
            # Forward-fill then backward-fill to preserve temporal continuity
            new_df[col] = new_df[col].ffill().bfill()

    # 7. Try to parse object columns that look like dates but weren't auto-detected
    for col in new_df.select_dtypes(include=['object']).columns:
        if any(x in col.lower() for x in ['date', 'time', 'created', 'updated', 'timestamp']):
            try:
                parsed = pd.to_datetime(new_df[col], errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() / len(new_df) > 0.7:   # >70% parseable → convert
                    new_df[col] = parsed.ffill().bfill()
            except Exception:
                pass

    return new_df