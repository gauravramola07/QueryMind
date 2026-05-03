# -*- coding: utf-8 -*-
# components/multi_file_joiner.py
"""
AI-Powered Multi-File Join Engine  (v2 — Full AI Auto-Join)
=============================================================
Scores join candidates (name + type + value overlap) exactly as before,
then adds three AI layers:

  ai_analyze_join_strategy(df1, df2, df1_name, df2_name, candidates, llm_fn)
      → dict  {best_left, best_right, join_type, reasoning, file1_role, file2_role}

  execute_all_join_types(df1, df2, left_on, right_on)
      → dict  {inner|left|right|outer: merge_result_dict}

  ai_join_type_insights(df1, df2, merged_results, df1_name, df2_name,
                         left_on, right_on, llm_fn)
      → dict  {inner|left|right|outer: str}

  ai_plan_multi_join(primary_df, primary_name, extra_files, llm_fn)
      → list[dict]  ordered join plan across N extra files

Public API (unchanged from v1)
-------------------------------
  detect_joinable_columns(df1, df2)  → list[dict]
  merge_dataframes(df1, df2, ...)    → dict
  score_badge(score)                 → str (HTML)
"""

from __future__ import annotations

import json
import re
import pandas as pd
import numpy as np
from typing import Any, Callable


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.lower().strip().replace("_", " ").replace("-", " ").replace(".", " ")


def _type_category(dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):   return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype): return "datetime"
    if pd.api.types.is_bool_dtype(dtype):      return "bool"
    return "string"


def _jaccard_overlap(s1: pd.Series, s2: pd.Series, sample: int = 300) -> float:
    try:
        v1 = s1.dropna()
        v2 = s2.dropna()
        if v1.empty or v2.empty:
            return 0.0
        v1 = v1.sample(min(sample, len(v1)), random_state=0).astype(str)
        v2 = v2.sample(min(sample, len(v2)), random_state=0).astype(str)
        set1, set2 = set(v1), set(v2)
        union = set1 | set2
        return 0.0 if not union else len(set1 & set2) / len(union)
    except Exception:
        return 0.0


def _schema_summary(df: pd.DataFrame, name: str, max_rows: int = 5) -> str:
    """Compact schema text for LLM prompts."""
    lines = [f"File: {name}  |  {len(df):,} rows × {len(df.columns)} columns"]
    lines.append("Columns:")
    for col in df.columns:
        cat  = _type_category(df[col].dtype)
        uniq = df[col].nunique()
        sample_vals = df[col].dropna().astype(str).unique()[:max_rows].tolist()
        lines.append(f"  {col} [{cat}, {uniq} unique, e.g. {sample_vals}]")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from LLM reply (handles ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Try to find the first {...} or [...] block
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL PUBLIC API — detect_joinable_columns
# ─────────────────────────────────────────────────────────────────────────────

def detect_joinable_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Compare every column in df1 vs df2 and return top-k join candidates
    scored on name similarity + type compatibility + value overlap.
    """
    candidates: list[dict] = []

    for col1 in df1.columns:
        n1   = _norm(col1)
        cat1 = _type_category(df1[col1].dtype)

        for col2 in df2.columns:
            n2   = _norm(col2)
            cat2 = _type_category(df2[col2].dtype)

            # ── 1. Name score (0–40) ──────────────────────────────────────
            if n1 == n2:
                name_score, name_match = 40, "Exact name match"
            elif n1 in n2 or n2 in n1:
                name_score, name_match = 28, "Substring match"
            else:
                common = {w for w in (set(n1.split()) & set(n2.split())) if len(w) > 2}
                if common:
                    name_score = 15
                    name_match = f"Shared words: {', '.join(sorted(common))}"
                else:
                    continue   # no name relation → skip

            # ── 2. Type score (0–20) ──────────────────────────────────────
            type_compat = cat1 == cat2
            type_score  = 20 if type_compat else 0

            # ── 3. Value overlap (0–40) ───────────────────────────────────
            overlap_ratio = _jaccard_overlap(df1[col1], df2[col2])
            value_score   = round(overlap_ratio * 40, 1)

            total = round(name_score + type_score + value_score, 1)

            if overlap_ratio > 0.8:
                hint = "inner (high overlap → few rows lost)"
            elif overlap_ratio > 0.4:
                hint = "left (keep all primary rows)"
            else:
                hint = "left or outer (low overlap)"

            candidates.append({
                "col_df1":         col1,
                "col_df2":         col2,
                "name_match":      name_match,
                "type_df1":        str(df1[col1].dtype),
                "type_df2":        str(df2[col2].dtype),
                "type_compatible": type_compat,
                "value_overlap":   round(overlap_ratio * 100, 1),
                "score":           total,
                "recommended":     total >= 55,
                "join_hint":       hint,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL PUBLIC API — merge_dataframes
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
    """Merge df1 and df2 on the chosen columns."""
    if left_on not in df1.columns:
        return {"success": False, "dataframe": None,
                "error": f"Column '{left_on}' not in primary dataset."}
    if right_on not in df2.columns:
        return {"success": False, "dataframe": None,
                "error": f"Column '{right_on}' not in right dataset."}
    if how not in ("inner", "left", "right", "outer"):
        return {"success": False, "dataframe": None,
                "error": f"Invalid join type '{how}'."}

    try:
        if df1[left_on].dtype != df2[right_on].dtype:
            try:
                df2 = df2.copy()
                df2[right_on] = df2[right_on].astype(df1[left_on].dtype)
            except (ValueError, TypeError):
                pass

        merged = pd.merge(df1, df2, left_on=left_on, right_on=right_on,
                          how=how, suffixes=(df1_suffix, df2_suffix))

        if left_on != right_on and right_on in merged.columns:
            merged.drop(columns=[right_on], inplace=True, errors="ignore")

        dropped = len(df1) - len(merged) if how in ("inner", "left") else 0
        return {
            "success":      True,
            "dataframe":    merged,
            "rows_before":  (len(df1), len(df2)),
            "rows_merged":  len(merged),
            "cols_merged":  len(merged.columns),
            "dropped_rows": max(0, dropped),
        }
    except Exception as exc:
        return {"success": False, "dataframe": None, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL PUBLIC API — score_badge
# ─────────────────────────────────────────────────────────────────────────────

def score_badge(score: float) -> str:
    if score >= 70:   color, label = "#48bb78", "Excellent"
    elif score >= 50: color, label = "#a78bfa", "Good"
    elif score >= 30: color, label = "#ecc94b", "Weak"
    else:             color, label = "#fc8181", "Poor"
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}66;"
        f"border-radius:8px;padding:2px 10px;font-size:0.78rem;font-weight:700;'>"
        f"{label} ({score})</span>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEW: AI STRATEGY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def ai_analyze_join_strategy(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str,
    df2_name: str,
    candidates: list[dict],
    llm_fn: Callable[[str], str],
) -> dict[str, Any]:
    """
    Ask the LLM to choose the best join column pair, join type, and explain
    the relationship between the two files.

    Returns:
      best_left    : str   recommended left column
      best_right   : str   recommended right column
      join_type    : str   inner|left|right|outer
      reasoning    : str   why this join strategy
      file1_role   : str   what df1 represents
      file2_role   : str   what df2 represents
      enrichment   : str   what new info df2 adds
    """
    # Fallback defaults from pure algorithmic detection
    best = candidates[0] if candidates else {}
    fallback = {
        "best_left":  best.get("col_df1", df1.columns[0] if len(df1.columns) else ""),
        "best_right": best.get("col_df2", df2.columns[0] if len(df2.columns) else ""),
        "join_type":  "left",
        "reasoning":  "Selected highest-scoring candidate by name+type+value overlap.",
        "file1_role": f"Primary dataset ({df1_name})",
        "file2_role": f"Supplementary dataset ({df2_name})",
        "enrichment": "Additional columns and attributes.",
    }

    schema1 = _schema_summary(df1, df1_name)
    schema2 = _schema_summary(df2, df2_name)

    top_cands = []
    for c in candidates[:5]:
        top_cands.append(
            f"  {c['col_df1']} ↔ {c['col_df2']}  "
            f"score={c['score']}  overlap={c['value_overlap']}%  "
            f"type_compat={'yes' if c['type_compatible'] else 'no'}  "
            f"hint={c['join_hint']}"
        )
    cands_text = "\n".join(top_cands) if top_cands else "  No strong candidates found."

    prompt = f"""You are a senior data engineer analyzing two datasets for joining.

=== FILE 1 ===
{schema1}

=== FILE 2 ===
{schema2}

=== TOP JOIN CANDIDATES (scored by name+type+overlap) ===
{cands_text}

Task: Recommend the best join strategy.

Respond ONLY with valid JSON, no extra text, no markdown fences:
{{
  "best_left":  "<column name from File 1>",
  "best_right": "<column name from File 2>",
  "join_type":  "<inner|left|right|outer>",
  "reasoning":  "<2-3 sentences explaining why this join + column pair is best>",
  "file1_role": "<what File 1 represents in 1 sentence>",
  "file2_role": "<what File 2 represents in 1 sentence>",
  "enrichment": "<what new information or value File 2 adds to File 1 in 1-2 sentences>"
}}"""

    try:
        raw = llm_fn(prompt)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and "best_left" in parsed:
            # Validate the columns exist
            if parsed["best_left"] not in df1.columns:
                parsed["best_left"] = fallback["best_left"]
            if parsed["best_right"] not in df2.columns:
                parsed["best_right"] = fallback["best_right"]
            if parsed.get("join_type") not in ("inner", "left", "right", "outer"):
                parsed["join_type"] = "left"
            return {**fallback, **parsed}
    except Exception:
        pass

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# NEW: EXECUTE ALL 4 JOIN TYPES
# ─────────────────────────────────────────────────────────────────────────────

def execute_all_join_types(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_on: str,
    right_on: str,
) -> dict[str, dict[str, Any]]:
    """
    Run inner / left / right / outer joins and return a dict of results.
    Each value is the dict returned by merge_dataframes().
    """
    results = {}
    for jtype in ("inner", "left", "right", "outer"):
        results[jtype] = merge_dataframes(df1, df2, left_on=left_on,
                                          right_on=right_on, how=jtype)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# NEW: AI INSIGHTS PER JOIN TYPE
# ─────────────────────────────────────────────────────────────────────────────

def ai_join_type_insights(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    merged_results: dict[str, dict],
    df1_name: str,
    df2_name: str,
    left_on: str,
    right_on: str,
    llm_fn: Callable[[str], str],
) -> dict[str, str]:
    """
    For each join type, generate a 2–3 sentence AI insight about what
    the result reveals and when you would use it.

    Returns: {"inner": "...", "left": "...", "right": "...", "outer": "..."}
    """
    def _stats(jtype):
        r = merged_results.get(jtype, {})
        if not r.get("success") or r.get("dataframe") is None:
            return f"  {jtype}: failed"
        mdf = r["dataframe"]
        nulls = int(mdf.isnull().sum().sum())
        return (
            f"  {jtype}: {r['rows_merged']:,} rows × {r['cols_merged']} cols  "
            f"dropped={r.get('dropped_rows', 0):,}  nulls_in_result={nulls:,}"
        )

    stats_block = "\n".join(_stats(j) for j in ("inner", "left", "right", "outer"))

    new_cols = [c for c in (merged_results.get("left", {}).get("dataframe") or
                             pd.DataFrame()).columns
                if c not in df1.columns][:10]

    prompt = f"""You are a data analyst explaining dataset joins to a business user.

Join key: {df1_name}.{left_on}  ↔  {df2_name}.{right_on}

Join results:
{stats_block}

New columns added from {df2_name}: {new_cols}

For each join type write a short, plain-English insight (2-3 sentences) explaining:
- what rows the result contains
- what is lost or gained vs other joins
- the ideal use-case

Respond ONLY with valid JSON (no fences, no preamble):
{{
  "inner":  "...",
  "left":   "...",
  "right":  "...",
  "outer":  "..."
}}"""

    defaults = {
        "inner": (
            "Inner join keeps only rows where both files share the same key value. "
            f"You lose {merged_results.get('inner',{}).get('dropped_rows',0):,} unmatched rows from {df1_name}. "
            "Best when you only care about perfectly matched records."
        ),
        "left": (
            f"Left join retains every row in {df1_name} regardless of a match. "
            f"Unmatched rows from {df2_name} appear as NaN. "
            "Ideal when the primary dataset must stay complete."
        ),
        "right": (
            f"Right join keeps all rows from {df2_name} and matches what it can from {df1_name}. "
            "Use this when the supplementary file drives your analysis."
        ),
        "outer": (
            "Full outer join keeps every row from both files, filling gaps with NaN. "
            "Produces the largest dataset and shows exactly where the two files diverge. "
            "Best for gap analysis or auditing."
        ),
    }

    try:
        raw = llm_fn(prompt)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and "inner" in parsed:
            return {k: parsed.get(k, defaults[k]) for k in defaults}
    except Exception:
        pass

    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# NEW: MULTI-FILE AI JOIN PLAN
# ─────────────────────────────────────────────────────────────────────────────

def ai_plan_multi_join(
    primary_df: pd.DataFrame,
    primary_name: str,
    extra_files: list[dict],   # each: {"name": str, "df": DataFrame}
    llm_fn: Callable[[str], str],
) -> list[dict[str, Any]]:
    """
    When N > 1 extra files exist, ask the LLM to produce an ordered join plan.

    Returns list of step dicts:
      {"step": 1, "right_file": name, "left_col": col, "right_col": col,
       "join_type": type, "reason": str}
    """
    schemas = [_schema_summary(primary_df, f"PRIMARY: {primary_name}")]
    for ef in extra_files:
        schemas.append(_schema_summary(ef["df"], ef["name"]))

    # Pre-compute candidates for each pair to give the LLM concrete options
    pair_info = []
    for ef in extra_files:
        cands = detect_joinable_columns(primary_df, ef["df"], top_k=3)
        best  = cands[0] if cands else {}
        pair_info.append(
            f"  {primary_name} ↔ {ef['name']}: best pair = "
            f"{best.get('col_df1','?')} ↔ {best.get('col_df2','?')}  "
            f"score={best.get('score',0)}  overlap={best.get('value_overlap',0)}%"
        )

    schema_block = "\n\n".join(schemas)
    pair_block   = "\n".join(pair_info)

    prompt = f"""You are a senior data engineer. The user wants to join multiple files
into a single dataset. Plan the optimal join order and column selections.

=== FILE SCHEMAS ===
{schema_block}

=== PRE-COMPUTED BEST COLUMN PAIRS ===
{pair_block}

Rules:
- Each extra file joins onto the CURRENT accumulated result (chain pattern).
- Pick join types and column pairs that maximise data quality.
- Order files from highest relevance / overlap to lowest.

Respond ONLY with valid JSON array (no fences):
[
  {{
    "step":       1,
    "right_file": "<exact file name>",
    "left_col":   "<column in left/primary>",
    "right_col":  "<column in right file>",
    "join_type":  "<inner|left|right|outer>",
    "reason":     "<1 sentence why>"
  }},
  ...
]"""

    # Algorithmic fallback plan
    fallback = []
    for i, ef in enumerate(extra_files, 1):
        cands = detect_joinable_columns(primary_df, ef["df"], top_k=1)
        best  = cands[0] if cands else {}
        fallback.append({
            "step":       i,
            "right_file": ef["name"],
            "left_col":   best.get("col_df1", primary_df.columns[0]),
            "right_col":  best.get("col_df2", ef["df"].columns[0]),
            "join_type":  "left",
            "reason":     "Highest-scoring column pair detected automatically.",
        })

    try:
        raw = llm_fn(prompt)
        parsed = _extract_json(raw)
        if isinstance(parsed, list) and parsed and "step" in parsed[0]:
            # Validate file names
            valid_names = {ef["name"] for ef in extra_files}
            validated = [
                s for s in parsed
                if s.get("right_file") in valid_names
            ]
            if validated:
                return validated
    except Exception:
        pass

    return fallback