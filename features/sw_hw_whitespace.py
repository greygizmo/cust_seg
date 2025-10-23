# features/sw_hw_whitespace.py
from __future__ import annotations

import numpy as np
import pandas as pd


def sw_dominance_and_whitespace(
    tx: pd.DataFrame, as_of: pd.Timestamp, weeks_short: int = 13
) -> pd.DataFrame:
    """
    Computes two per-account scores:
      SW Dominance: 0.6*SW share(12M) + 0.2*SW recency + 0.2*SW cadence (13W)
      SW→HW Whitespace: 0.6*SW Dominance + 0.4*(1 - HW share(12M))
    Assumes tx already has 'super_division' joined.
    """
    as_of = pd.to_datetime(as_of)
    start_12m = as_of - pd.DateOffset(months=12)
    ltm = tx.loc[(tx["date"] > start_12m) & (tx["date"] <= as_of)]

    # shares
    tot = ltm.groupby(["account_id", "super_division"])["net_revenue"].sum().unstack(fill_value=0)
    for col in ("Hardware", "Software"):
        if col not in tot.columns:
            tot[col] = 0.0
    tot.columns = [c.lower() for c in tot.columns]
    tot["spend_12m"] = tot["hardware"] + tot["software"]
    tot["sw_share_12m"] = np.where(tot["spend_12m"] > 0, tot["software"] / tot["spend_12m"], np.nan)
    tot["hw_share_12m"] = np.where(tot["spend_12m"] > 0, tot["hardware"] / tot["spend_12m"], np.nan)

    # recency SW
    last_sw = (
        tx.loc[tx["super_division"].str.lower() == "software"]
        .groupby("account_id")["date"].max()
        .rename("last_sw_date")
    )
    rec_sw = 1.0 / (1.0 + ((as_of - last_sw).dt.days / 30.0))
    rec_sw = rec_sw.fillna(0.0).rename("rec_sw")

    # cadence SW: active weeks in 13W
    sw_13 = tx.loc[
        (tx["super_division"].str.lower() == "software")
        & (tx["date"] > (as_of - pd.Timedelta(days=7 * weeks_short)))
        & (tx["date"] <= as_of)
    ]
    wk = (
        sw_13.set_index("date")
        .groupby("account_id")["net_revenue"]
        .resample("W-MON")
        .sum()
        .reset_index()
    )
    wk["active"] = (wk["net_revenue"] > 0).astype(int)
    cad_sw = wk.groupby("account_id")["active"].sum() / weeks_short
    cad_sw = cad_sw.rename("cad_sw")

    z = (
        tot[["sw_share_12m", "hw_share_12m"]]
        .join(rec_sw, how="left")
        .join(cad_sw, how="left")
    ).reset_index()

    z["sw_dominance_score"] = (
        0.6 * z["sw_share_12m"].fillna(0)
        + 0.2 * z["rec_sw"].fillna(0)
        + 0.2 * z["cad_sw"].fillna(0)
    )
    z["sw_to_hw_whitespace_score"] = (
        0.6 * z["sw_dominance_score"]
        + 0.4 * (1 - z["hw_share_12m"].fillna(0))
    )

    return z[["account_id", "sw_dominance_score", "sw_to_hw_whitespace_score"]]
