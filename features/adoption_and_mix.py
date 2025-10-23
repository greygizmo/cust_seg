# features/adoption_and_mix.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["date"] > start) & (df["date"] <= end)]


def compute_adoption_and_mix(
    tx: pd.DataFrame,
    products_hw_subdiv: list[str],
    as_of: pd.Timestamp,
    months_ltm: int = 12,
    weeks_short: int = 13,
) -> pd.DataFrame:
    """
    Computes per-account HW/SW LTM spend, shares, breadth across HW sub-divisions,
    recency of HW, and a no-printer Hardware Adoption Score:
      0.5 * HW share (12M) + 0.3 * breadth (HW sub-div) + 0.2 * recency(HW)
    Also computes 'consumables_to_hw_ratio' if 'Consumables' exists under Hardware.
    """
    as_of = pd.to_datetime(as_of)
    start_12m = as_of - pd.DateOffset(months=months_ltm)

    # Filter LTM
    ltm = _window(tx, start_12m, as_of)

    # HW/SW totals
    tot = ltm.groupby(["account_id", "super_division"])["net_revenue"].sum().unstack(fill_value=0)
    tot = tot.rename(columns=str.lower)
    for col in ("hardware", "software"):
        if col not in tot.columns:
            tot[col] = 0.0
    tot["spend_12m"] = tot["hardware"] + tot["software"]

    # shares
    tot["hw_share_12m"] = np.where(tot["spend_12m"] > 0, tot["hardware"] / tot["spend_12m"], np.nan)
    tot["sw_share_12m"] = np.where(tot["spend_12m"] > 0, tot["software"] / tot["spend_12m"], np.nan)

    # breadth across HW sub-divisions
    hw_sub = ltm.loc[ltm["super_division"].str.lower() == "hardware"]
    breadth = (hw_sub.groupby(["account_id", "sub_division"])["net_revenue"].sum() > 0).reset_index()
    breadth = breadth.groupby("account_id")["sub_division"].nunique().rename("breadth_hw_subdiv_12m")

    max_hw_subdiv = int(len(set(products_hw_subdiv))) if products_hw_subdiv else int(hw_sub["sub_division"].nunique())
    # recency of HW
    last_hw = (
        tx.loc[tx["super_division"].str.lower() == "hardware"]
        .groupby("account_id")["date"].max()
        .rename("last_hw_date")
    )
    # assemble
    out = (
        tot.join(breadth, how="outer")
        .join(last_hw, how="left")
    ).reset_index().rename(columns={"hardware": "hw_spend_12m", "software": "sw_spend_12m"})

    # derived
    out["breadth_hw_subdiv_12m"] = out["breadth_hw_subdiv_12m"].fillna(0).astype(int)
    out["max_hw_subdiv"] = max_hw_subdiv if max_hw_subdiv > 0 else np.nan
    out["breadth_score_hw"] = np.where(out["max_hw_subdiv"] > 0, out["breadth_hw_subdiv_12m"] / out["max_hw_subdiv"], 0)
    out["days_since_last_hw_order"] = (as_of - out["last_hw_date"]).dt.days
    out["recency_score_hw"] = 1.0 / (1.0 + (out["days_since_last_hw_order"] / 30.0))
    out.loc[out["last_hw_date"].isna(), "recency_score_hw"] = 0.0

    # adoption (no printers, no Big/Small box)
    out["hardware_adoption_score"] = (
        0.5 * out["hw_share_12m"].fillna(0)
        + 0.3 * out["breadth_score_hw"].fillna(0)
        + 0.2 * out["recency_score_hw"].fillna(0)
    )

    # top subdivision (by LTM spend) and share
    subspend = hw_sub.groupby(["account_id", "sub_division"])["net_revenue"].sum().reset_index()
    top = (
        subspend.sort_values(["account_id", "net_revenue"], ascending=[True, False])
        .groupby("account_id")
        .head(1)
        .rename(columns={"sub_division": "top_subdivision_12m", "net_revenue": "top_subdivision_spend_12m"})
    )
    out = out.merge(top[["account_id", "top_subdivision_12m", "top_subdivision_spend_12m"]], how="left", on="account_id")
    out["top_subdivision_share_12m"] = np.where(
        out["hw_spend_12m"] > 0, out["top_subdivision_spend_12m"] / out["hw_spend_12m"], np.nan
    )

    # Consumables/HW ratio (if exists)
    if "consumables" in set(map(str.lower, ltm["sub_division"].unique())):
        cons = (
            ltm.loc[
                (ltm["super_division"].str.lower() == "hardware")
                & (ltm["sub_division"].str.lower() == "consumables")
            ]
            .groupby("account_id")["net_revenue"].sum()
            .rename("consumables_12m")
        )
        out = out.merge(cons, left_on="account_id", right_index=True, how="left")
        out["consumables_to_hw_ratio"] = np.where(
            out["hw_spend_12m"] > 0, out["consumables_12m"] / out["hw_spend_12m"], np.nan
        )
    else:
        out["consumables_to_hw_ratio"] = np.nan

    # select final cols
    final = out[
        [
            "account_id",
            "hw_spend_12m",
            "sw_spend_12m",
            "spend_12m",
            "hw_share_12m",
            "sw_share_12m",
            "breadth_hw_subdiv_12m",
            "max_hw_subdiv",
            "breadth_score_hw",
            "days_since_last_hw_order",
            "recency_score_hw",
            "hardware_adoption_score",
            "top_subdivision_12m",
            "top_subdivision_share_12m",
            "consumables_to_hw_ratio",
        ]
    ]
    return final
