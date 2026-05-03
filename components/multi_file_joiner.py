# -*- coding: utf-8 -*-
# components/multi_file_joiner.py
"""
Multi-File Join Detection & Merge Engine
=========================================
Compares two DataFrames and auto-detects joinable column pairs by scoring:
  • Name similarity  (0–40 pts)
  • Type compatibility (0–20 pts)
  • Value overlap / Jaccard similarity (0–40 pts)

Public API
----------
detect_joinable_columns(df1, df2)       → list[dict]   # ranked candidates
merge_dataframes(df1, df2, ...)         → dict          # {success, dataframe, ...}
get_join_summary_html(candidates)       → str           # pretty HTML badge row
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase + strip punctuation for fuzzy name comparison."""
    return name.lower().strip().replace("_", " ").replace("-", " ").replace(".", " ")


def _type_category(dtype) -> str:
    """Bucket a dtype into broad category for compatibility check."""
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "string"


def _jaccard_overlap(s1: pd.Series, s2: pd.Series, sample: int = 300) -> float:
    """
    Jaccard similarity of *string* value sets between two Series.
    Samples up to `sample` non-null values from each side for speed.
    Returns a float in [0, 1].
    """
    try:
        v1 = s1.dropna()
        v2 = s2.dropna()
        if v1.empty or v2.empty:
            return 0.0

        # Sample for performance on large frames
        v1 = v1.sample(min(sample, len(v1)), random_state=0).astype(str)
        v2 = v2.sample(min(sample, len(v2)), random_state=0).astype(str)

        set1, set2 = set(v1), set(v2)
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: detect_joinable_columns
# ─────────────────────────────────────────────────────────────────────────────

def detect_joinable_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Compare every column in df1 against every column in df2.
    Returns the top-k candidates sorted by descending score.

    Each candidate dict contains:
      col_df1        : str   column name in df1
      col_df2        : str   column name in df2
      name_match     : str   human-readable match reason
      type_df1       : str   dtype string
      type_df2       : str   dtype string
      type_compatible: bool
      value_overlap  : float  0-100 (%)
      score          : float  0-100
      recommended    : bool   score >= 55
      join_hint      : str   suggested join type
    """
    candidates: list[dict] = []

    for col1 in df1.columns:
        n1 = _norm(col1)
        cat1 = _type_category(df1[col1].dtype)

        for col2 in df2.columns:
            n2 = _norm(col2)
            cat2 = _type_category(df2[col2].dtype)

            # ── 1. Name score (0–40) ─────────────────────────────────────────
            if n1 == n2:
                name_score = 40
                name_match = "Exact name match"
            elif n1 in n2 or n2 in n1:
                name_score = 28
                name_match = "Substring match"
            else:
                words1 = set(n1.split())
                words2 = set(n2.split())
                common_words = words1 & words2
                # ignore very short noise words
                common_words = {w for w in common_words if len(w) > 2}
                if common_words:
                    name_score = 15
                    name_match = f"Shared words: {', '.join(sorted(common_words))}"
                else:
                    continue  # No name relation → skip pair entirely

            # ── 2. Type score (0–20) ─────────────────────────────────────────
            type_compat = cat1 == cat2
            type_score = 20 if type_compat else 0

            # ── 3. Value overlap score (0–40) ────────────────────────────────
            overlap_ratio = _jaccard_overlap(df1[col1], df2[col2])
            value_score = round(overlap_ratio * 40, 1)

            total = round(name_score + type_score + value_score, 1)

            # ── Join hint ────────────────────────────────────────────────────
            if overlap_ratio > 0.8:
                hint = "inner (high overlap → few rows lost)"
            elif overlap_ratio > 0.4:
                hint = "left (keep all of primary file)"
            else:
                hint = "left or outer (low overlap)"

            candidates.append(
                {
                    "col_df1": col1,
                    "col_df2": col2,
                    "name_match": name_match,
                    "type_df1": str(df1[col1].dtype),
                    "type_df2": str(df2[col2].dtype),
                    "type_compatible": type_compat,
                    "value_overlap": round(overlap_ratio * 100, 1),
                    "score": total,
                    "recommended": total >= 55,
                    "join_hint": hint,
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: merge_dataframes
# ─────────────────────────────────────────────────────────────────────────────

def merge_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner",
    df1_suffix: str = "_file1",
    df2_suffix: str = "_file2",
) -> dict[str, Any]:
    """
    Merge df1 and df2 on the chosen columns.

    Returns:
      success     : bool
      dataframe   : merged DataFrame (or None on error)
      rows_before : (int, int) rows in df1 and df2
      rows_merged : int
      cols_merged : int
      dropped_rows: int   rows lost (inner/left join difference)
      error       : str   (only if success=False)
    """
    # ── Pre-merge validation ──────────────────────────────────────────────────
    if left_on not in df1.columns:
        return {
            "success": False, "dataframe": None,
            "error": f"Column '{left_on}' does not exist in the primary dataset. "
                     f"Available: {list(df1.columns)}"
        }
    if right_on not in df2.columns:
        return {
            "success": False, "dataframe": None,
            "error": f"Column '{right_on}' does not exist in the right dataset. "
                     f"Available: {list(df2.columns)}"
        }
    if how not in ("inner", "left", "right", "outer"):
        return {
            "success": False, "dataframe": None,
            "error": f"Invalid join type '{how}'. Must be one of: inner, left, right, outer."
        }

    try:
        # Coerce key column types to match when one side is object and the other
        # is numeric/datetime — a common user mistake that produces 0 join rows.
        try:
            if df1[left_on].dtype != df2[right_on].dtype:
                df2 = df2.copy()
                df2[right_on] = df2[right_on].astype(df1[left_on].dtype)
        except (ValueError, TypeError):
            pass   # leave as-is; pandas will still attempt the join

        merged = pd.merge(
            df1,
            df2,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffixes=(df1_suffix, df2_suffix),
        )

        # If join keys have different names, drop the redundant right key
        if left_on != right_on and right_on in merged.columns:
            merged.drop(columns=[right_on], inplace=True, errors="ignore")

        dropped = len(df1) - len(merged) if how in ("inner", "left") else 0

        return {
            "success": True,
            "dataframe": merged,
            "rows_before": (len(df1), len(df2)),
            "rows_merged": len(merged),
            "cols_merged": len(merged.columns),
            "dropped_rows": max(0, dropped),
        }

    except Exception as exc:
        return {"success": False, "dataframe": None, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: score badge HTML (used by render_join_tab in app.py)
# ─────────────────────────────────────────────────────────────────────────────

def score_badge(score: float) -> str:
    """Return an HTML span with colour-coded score badge."""
    if score >= 70:
        color, label = "#48bb78", "Excellent"
    elif score >= 50:
        color, label = "#a78bfa", "Good"
    elif score >= 30:
        color, label = "#ecc94b", "Weak"
    else:
        color, label = "#fc8181", "Poor"

    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}66;"
        f"border-radius:8px;padding:2px 10px;font-size:0.78rem;font-weight:700;'>"
        f"{label} ({score})</span>"
    )