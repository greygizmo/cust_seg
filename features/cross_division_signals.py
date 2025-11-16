"""Cross-division signals that surface list-building knobs for sellers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(series: pd.Series, *, lowercase: bool = True) -> pd.Series:
    values = series.astype(str).str.strip()
    if lowercase:
        values = values.str.lower()
    return values.replace({"nan": "", "None": ""})


def _pivot_sum(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot = (
        df.groupby(["account_id", "super_division_norm"])["net_revenue"].sum().unstack(fill_value=0.0)
    )
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def _calc_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    ratio = numer / denom
    return ratio


def compute_cross_division_signals(
    tx: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    weeks_short: int = 13,
    months_ltm: int = 12,
) -> pd.DataFrame:
    """Return per-account divisional and cross-divisional list-builder signals."""

    tx = tx.rename(columns={c: c.lower() for c in tx.columns})

    required = {"account_id", "date", "net_revenue"}
    missing = required - set(tx.columns)
    if missing:
        raise KeyError(f"Transactions missing required columns: {', '.join(sorted(missing))}")

    tx = tx.copy()
    tx["account_id"] = tx["account_id"].astype(str)
    tx["date"] = pd.to_datetime(tx["date"]).dt.tz_localize(None)
    tx["super_division_norm"] = _normalize(tx.get("super_division", ""))
    div_col = "division" if "division" in tx.columns else "goal" if "goal" in tx.columns else None
    tx["division_norm"] = _normalize(tx[div_col]) if div_col else ""

    as_of = pd.to_datetime(as_of).tz_localize(None).normalize()
    start_13w = as_of - pd.Timedelta(days=7 * weeks_short)
    start_13w_prior = start_13w - pd.Timedelta(days=7 * weeks_short)
    start_12m = as_of - pd.DateOffset(months=months_ltm)

    ltm = tx[(tx["date"] > start_12m) & (tx["date"] <= as_of)]
    recent = tx[(tx["date"] > start_13w) & (tx["date"] <= as_of)]
    prior = tx[(tx["date"] > start_13w_prior) & (tx["date"] <= start_13w)]

    if tx.empty:
        columns = [
            "account_id",
            "hw_spend_13w",
            "hw_spend_13w_prior",
            "hw_delta_13w",
            "hw_delta_13w_pct",
            "sw_spend_13w",
            "sw_spend_13w_prior",
            "sw_delta_13w",
            "sw_delta_13w_pct",
            "super_division_breadth_12m",
            "division_breadth_12m",
            "software_division_breadth_12m",
            "cross_division_balance_score",
            "hw_to_sw_cross_sell_score",
            "sw_to_hw_cross_sell_score",
            "training_to_hw_ratio",
            "training_to_cre_ratio",
        ]
        return pd.DataFrame({col: [] for col in columns})

    account_index = pd.Index(sorted(tx["account_id"].unique()), name="account_id")
    out = pd.DataFrame(index=account_index)

    recent_pivot = _pivot_sum(recent)
    prior_pivot = _pivot_sum(prior)
    ltm_pivot = _pivot_sum(ltm)

    for norm, prefix in (("hardware", "hw"), ("software", "sw")):
        recent_series = recent_pivot.get(norm, pd.Series(dtype=float))
        prior_series = prior_pivot.get(norm, pd.Series(dtype=float))
        out[f"{prefix}_spend_13w"] = recent_series.reindex(account_index, fill_value=0.0)
        out[f"{prefix}_spend_13w_prior"] = prior_series.reindex(account_index, fill_value=0.0)
        delta = out[f"{prefix}_spend_13w"] - out[f"{prefix}_spend_13w_prior"]
        out[f"{prefix}_delta_13w"] = delta
        denom = out[f"{prefix}_spend_13w_prior"].replace(0.0, np.nan)
        out[f"{prefix}_delta_13w_pct"] = delta / denom

    hw_ltm = ltm_pivot.get("hardware", pd.Series(dtype=float)).reindex(account_index, fill_value=0.0)
    sw_ltm = ltm_pivot.get("software", pd.Series(dtype=float)).reindex(account_index, fill_value=0.0)
    total_ltm = hw_ltm + sw_ltm
    with np.errstate(divide="ignore", invalid="ignore"):
        hw_share = hw_ltm / total_ltm.replace(0.0, np.nan)
        sw_share = sw_ltm / total_ltm.replace(0.0, np.nan)
    balance = 1.0 - (hw_share - sw_share).abs()
    out["cross_division_balance_score"] = balance
    out["hw_to_sw_cross_sell_score"] = np.maximum(hw_share - sw_share, 0.0)
    out["sw_to_hw_cross_sell_score"] = np.maximum(sw_share - hw_share, 0.0)

    super_breadth = (
        ltm.loc[ltm["super_division_norm"] != ""]
        .groupby("account_id")["super_division_norm"]
        .nunique()
        .rename("super_division_breadth_12m")
    )
    division_breadth = (
        ltm.loc[ltm["division_norm"] != ""]
        .groupby("account_id")["division_norm"]
        .nunique()
        .rename("division_breadth_12m")
    )
    software_division_breadth = (
        ltm.loc[ltm["super_division_norm"] == "software"]
        .groupby("account_id")["division_norm"]
        .nunique()
        .rename("software_division_breadth_12m")
    )

    out = out.join(super_breadth, how="left")
    out = out.join(division_breadth, how="left")
    out = out.join(software_division_breadth, how="left")

    training_aliases = {"training/services", "training", "success plan"}
    training_spend = (
        ltm.loc[ltm["division_norm"].isin(training_aliases)]
        .groupby("account_id")["net_revenue"]
        .sum()
    )
    out["training_to_hw_ratio"] = _calc_ratio(
        training_spend.reindex(account_index, fill_value=0.0),
        hw_ltm.replace(0.0, np.nan),
    )
    out["training_to_cre_ratio"] = _calc_ratio(
        training_spend.reindex(account_index, fill_value=0.0),
        sw_ltm.replace(0.0, np.nan),
    )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.reset_index()

