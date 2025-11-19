"""
Logic for generating preset call lists for sales teams.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd

from icp.schema import COL_CUSTOMER_ID, COL_COMPANY_NAME, COL_INDUSTRY, canonicalize_customer_id


def _pick_column(df: pd.DataFrame, candidates: Sequence[str], default=None):
    for name in candidates:
        if name in df.columns:
            return df[name]
    return default


def normalize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the scored accounts dataframe for call list generation."""
    data = df.copy()
    if COL_CUSTOMER_ID not in data.columns:
        raise ValueError(f"Missing required column '{COL_CUSTOMER_ID}' in scored accounts.")
    data[COL_CUSTOMER_ID] = canonicalize_customer_id(data[COL_CUSTOMER_ID])

    if COL_COMPANY_NAME not in data.columns:
        company_fallback = _pick_column(data, ["company_name", "Account Name"])
        if company_fallback is not None:
            data[COL_COMPANY_NAME] = company_fallback
    data[COL_COMPANY_NAME] = (
        data.get(COL_COMPANY_NAME, pd.Series(index=data.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    grade_candidates = ["ICP_grade_hardware", "ICP_grade", "grade"]
    grade_series = _pick_column(data, grade_candidates, pd.Series("C", index=data.index))
    data["grade"] = grade_series.astype(str).str.upper().str.strip()

    score_candidates = ["ICP_score_hardware", "ICP_score", "score"]
    score_series = _pick_column(data, score_candidates, pd.Series(50.0, index=data.index))
    data["score"] = pd.to_numeric(score_series, errors="coerce").fillna(50.0).clip(0, 100)

    adoption_candidates = ["adoption_score", "Hardware_score", "adoption_assets"]
    adopt_series = _pick_column(data, adoption_candidates, pd.Series(0.5, index=data.index))
    data["adoption_score"] = pd.to_numeric(adopt_series, errors="coerce").fillna(0.5).clip(0, 1)

    relationship_candidates = ["relationship_score", "Software_score", "relationship_profit"]
    rel_series = _pick_column(data, relationship_candidates, pd.Series(0.5, index=data.index))
    data["relationship_score"] = pd.to_numeric(rel_series, errors="coerce").fillna(0.5).clip(0, 1)

    profit_candidates = [
        "profit_since_2023",
        "GP_Since_2023_Total",
        "Profit_Since_2023_Total",
    ]
    profit_series = _pick_column(data, profit_candidates, pd.Series(0.0, index=data.index))
    data["profit_since_2023"] = pd.to_numeric(profit_series, errors="coerce").fillna(0.0)

    printer_candidates = ["printer_count", "Printers", "Qty_Printers"]
    printer_series = _pick_column(data, printer_candidates, pd.Series(0.0, index=data.index))
    data["printer_count"] = pd.to_numeric(printer_series, errors="coerce").fillna(0.0)
    data["revenue_only_flag"] = (data["printer_count"] <= 0.5).astype(bool)
    data["heavy_fleet_flag"] = (data["printer_count"] >= 10).astype(bool)

    seg_candidates = ["customer_segment", "activity_segment"]
    segment_series = _pick_column(data, seg_candidates)
    if segment_series is None or segment_series.isna().all():
        quantiles = data["profit_since_2023"].quantile([0.5, 0.85]).tolist()
        # Ensure bins are unique and increasing
        bins = sorted(list(set([-np.inf, quantiles[0], quantiles[1], np.inf])))
        # If bins collapsed (e.g. all profit same), fallback to Core
        if len(bins) < 4:
             data["customer_segment"] = "Core"
        else:
             labels = ["Core", "Growth", "Strategic"]
             data["customer_segment"] = pd.cut(data["profit_since_2023"], bins=bins, labels=labels).astype(str)
    else:
        data["customer_segment"] = segment_series.fillna("Unassigned").astype(str)

    activity_candidates = ["activity_segment"]
    activity_series = _pick_column(data, activity_candidates)
    if activity_series is None:
        data["activity_segment"] = np.where(data["profit_since_2023"] > 0, "Warm", "Dormant")
    else:
        data["activity_segment"] = activity_series.fillna("Unknown").astype(str)

    data["territory"] = (
        _pick_column(data, ["territory", "AM_Territory"], pd.Series("Unassigned", index=data.index))
        .fillna("Unassigned")
        .astype(str)
    )
    data["sales_rep"] = (
        _pick_column(data, ["sales_rep", "am_sales_rep"], pd.Series("Unassigned", index=data.index))
        .fillna("Unassigned")
        .astype(str)
    )
    if "call_to_action" not in data.columns:
        data["call_to_action"] = np.where(
            data["adoption_score"] < 0.4,
            "Launch expansion sprint",
            "Protect install base",
        )
    else:
        data["call_to_action"] = data["call_to_action"].fillna("Maintain success cadence").astype(str)

    data["whitespace_score"] = (1 - data["adoption_score"]).clip(0, 1)
    data["relationship_band"] = pd.cut(
        data["relationship_score"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    data["adoption_band"] = pd.cut(
        data["adoption_score"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    data[COL_INDUSTRY] = data.get(COL_INDUSTRY, pd.Series("Unknown", index=data.index)).fillna("Unknown").astype(str)
    return data


def format_table(df: pd.DataFrame, profit_label: str) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Customer ID": df[COL_CUSTOMER_ID],
            "Company": df[COL_COMPANY_NAME],
            "Segment": df["customer_segment"],
            "Territory": df["territory"],
            "Owner": df["sales_rep"],
            "Grade": df["grade"],
            "Score": df["score"].round(1),
            "Adoption Score": df["adoption_score"].round(3),
            "Relationship Score": df["relationship_score"].round(3),
            profit_label: df["profit_since_2023"].round(0),
            "Printer Count": df["printer_count"].astype(int),
            "Suggested Playbook": df["call_to_action"],
        }
    )
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table


def preset_top_ab_by_segment(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    subset = df[df["grade"].isin(["A", "B"])]
    frames = []
    for segment in sorted(subset["customer_segment"].dropna().unique().tolist()):
        seg_df = subset[subset["customer_segment"] == segment].copy()
        seg_df = seg_df.sort_values(["score", "profit_since_2023"], ascending=[False, False]).head(75)
        frames.append(seg_df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)
    table = format_table(combined, "GP Since Jan 2023")
    meta = {
        "preset": "top_ab_by_segment",
        "description": "Top A/B accounts within each segment ranked by ICP score.",
        "filters": {"grades": ["A", "B"], "per_segment_limit": 75},
        "rows": len(table),
    }
    return table, meta


def preset_revenue_only_high_relationship(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    mask = (df["printer_count"] <= 0.5) & (df["relationship_score"] >= 0.6) & (df["profit_since_2023"] > 0)
    subset = df[mask].copy()
    subset = subset.sort_values(["relationship_score", "profit_since_2023"], ascending=[False, False]).head(150)
    table = format_table(subset, "GP Since Jan 2023")
    meta = {
        "preset": "revenue_only_high_relationship",
        "description": "Accounts with strong relationship signals but no active printers.",
        "filters": {
            "printer_count": "<=0",
            "relationship_score": ">=0.6",
            "profit_since_2023": ">0",
        },
        "rows": len(table),
    }
    return table, meta


def preset_heavy_fleet_expansion(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    mask = (df["printer_count"] >= 10) & (df["whitespace_score"] >= 0.3)
    subset = df[mask].copy()
    subset = subset.sort_values(["whitespace_score", "score"], ascending=[False, False]).head(150)
    table = format_table(subset, "GP Since Jan 2023")
    meta = {
        "preset": "heavy_fleet_expansion",
        "description": "High-value fleets with sizable whitespace.",
        "filters": {
            "printer_count": ">=10",
            "whitespace_score": ">=0.30",
        },
        "rows": len(table),
    }
    return table, meta
