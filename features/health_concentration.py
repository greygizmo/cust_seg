# features/health_concentration.py
from __future__ import annotations

import numpy as np
import pandas as pd


def month_hhi_12m(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    LTM monthly concentration per account (Herfindahl index 0..1; higher = bursty).
    """
    as_of = pd.to_datetime(as_of)
    start = as_of - pd.DateOffset(months=12)
    ltm = tx.loc[(tx["date"] > start) & (tx["date"] <= as_of)].copy()
    ltm["month"] = ltm["date"].values.astype("datetime64[M]")

    m = ltm.groupby(["account_id", "month"])["net_revenue"].sum().reset_index()
    out = []
    for acc, g in m.groupby("account_id"):
        total = g["net_revenue"].sum()
        if total <= 0:
            out.append((acc, np.nan))
            continue
        share2 = ((g["net_revenue"] / total) ** 2).sum()
        out.append((acc, float(share2)))
    return pd.DataFrame(out, columns=["account_id", "month_conc_hhi_12m"])


def discount_pct(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Effective discount% = 1 - net/list, if list_price_revenue column exists.
    """
    if "list_price_revenue" not in tx.columns:
        return pd.DataFrame({"account_id": tx["account_id"].unique(), "discount_pct": np.nan})
    g = tx.groupby("account_id")[["net_revenue", "list_price_revenue"]].sum()
    g["discount_pct"] = 1 - (g["net_revenue"] / g["list_price_revenue"]).replace({0: np.nan})
    return g[["discount_pct"]].reset_index()
