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