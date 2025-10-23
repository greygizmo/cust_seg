# features/pov_tags.py
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Thresholds centralized for easy tuning
THRESH = dict(
    momentum_top_pct=0.80,
    yoy_rebound=0.10,  # >= +10% YoY
    price_sensitive=0.15,  # >= 15% discount
    hhi_bursty=0.20,
    sw_dom=0.60,
    hw_low_share=0.25,
)

# Precedence order for primary tag
PRIMARY_ORDER = [
    "SW-Dominant → HW Whitespace",
    "Momentum Buyer (Hardware)",
    "Momentum Buyer (Software)",
    "Momentum Buyer",
    "New Hardware Buyer (≤90d)",
    "Reorder Window Approaching",
    "Seasonal Rebound",
    "Cross-Division Expansion",
    "Price Sensitive",
    "Concentrated Buyer",
    "Dormancy Risk",
    "Bundle Candidate",
]


def make_pov_tags(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Input: per-account features already merged (one row per account).
    Output: adds 'pov_primary' and 'pov_tags_all' columns.
    """
    df = accounts.copy()

    # Individual booleans (12 tags)
    df["tag_sw_hw_whitespace"] = (
        (df["sw_dominance_score"] >= THRESH["sw_dom"])
        & (df["hw_share_12m"].fillna(0) <= THRESH["hw_low_share"])
    )

    # Momentum general (top 20% momentum)
    p80 = (
        df["momentum_score"].quantile(THRESH["momentum_top_pct"])
        if df["momentum_score"].notna().any()
        else np.inf
    )
    df["tag_momentum_general"] = df["momentum_score"] >= p80

    # Momentum scoped by SuperDivision via slope filters computed upstream
    # For simplicity, reuse general momentum + dominance of HW/SW share as hint
    df["tag_momentum_hw"] = df["tag_momentum_general"] & (df["hw_share_12m"].fillna(0) >= 0.5)
    df["tag_momentum_sw"] = df["tag_momentum_general"] & (df["sw_share_12m"].fillna(0) >= 0.5)

    df["tag_dormancy"] = (
        (df["days_since_last_order"] > 2 * df["median_interpurchase_days"])
        | ((df["delta_13w"] < 0) & (df["yoy_13w_pct"] < -0.20))
    )

    df["tag_seasonal_rebound"] = df["seasonality_factor_13w"] >= (1 + THRESH["yoy_rebound"])

    # Reorder window (consumables): within ±20% of median interval
    df["tag_reorder_window"] = np.where(
        df["median_interpurchase_days"].notna()
        & df["median_interpurchase_days"].gt(0)
        & df["days_since_last_order"].notna(),
        np.abs(df["days_since_last_order"] - df["median_interpurchase_days"])
        <= (0.2 * df["median_interpurchase_days"]),
        False,
    )

    # Bundle candidate: small breadth plus >=2 sub-divisions in LTM
    df["tag_bundle"] = (df["breadth_score_hw"].fillna(0) < 0.40) & (
        df["breadth_hw_subdiv_12m"].fillna(0) >= 2
    )

    df["tag_price_sensitive"] = df["discount_pct"].fillna(0) >= THRESH["price_sensitive"]

    df["tag_concentrated"] = df["month_conc_hhi_12m"].fillna(0) >= THRESH["hhi_bursty"]

    # New HW buyer: first HW purchase in last 90 days (approximated by low days_since_last_hw_order and HW share > 0)
    df["tag_new_hw_buyer"] = (df["days_since_last_hw_order"] <= 90) & (
        df["hw_share_12m"].fillna(0) > 0
    )

    # Cross-Division Expansion: diversified across sub-divisions but imbalanced HW vs SW
    df["tag_cross_div_expansion"] = (
        df["top_subdivision_share_12m"].fillna(1) < 0.60
    ) & (
        np.abs(df["hw_share_12m"].fillna(0) - df["sw_share_12m"].fillna(0)) >= 0.50
    )

    # Assemble all-tags list
    TAG_MAP = {
        "SW-Dominant → HW Whitespace": "tag_sw_hw_whitespace",
        "Momentum Buyer (Hardware)": "tag_momentum_hw",
        "Momentum Buyer (Software)": "tag_momentum_sw",
        "Momentum Buyer": "tag_momentum_general",
        "New Hardware Buyer (≤90d)": "tag_new_hw_buyer",
        "Reorder Window Approaching": "tag_reorder_window",
        "Seasonal Rebound": "tag_seasonal_rebound",
        "Cross-Division Expansion": "tag_cross_div_expansion",
        "Price Sensitive": "tag_price_sensitive",
        "Concentrated Buyer": "tag_concentrated",
        "Dormancy Risk": "tag_dormancy",
        "Bundle Candidate": "tag_bundle",
    }

    def list_tags(row) -> List[str]:
        return [name for name, flag_col in TAG_MAP.items() if bool(row.get(flag_col, False))]

    df["pov_tags_all"] = df.apply(list_tags, axis=1).apply(
        lambda lst: ", ".join(lst) if lst else ""
    )

    def primary_tag(row) -> str:
        available = row["pov_tags_all"].split(", ") if row["pov_tags_all"] else []
        for name in PRIMARY_ORDER:
            if name in available:
                return name
        return "General ICP Match"

    df["pov_primary"] = df.apply(primary_tag, axis=1)
    return df[["account_id", "pov_primary", "pov_tags_all"]]
