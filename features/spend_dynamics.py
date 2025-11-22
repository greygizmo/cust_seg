# features/spend_dynamics.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _endpoints(
    as_of: pd.Timestamp,
    weeks_short: int,
    months_ltm: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return anchor dates for primary 13W/12M windows."""

    end = as_of.normalize()
    start_13w = end - pd.Timedelta(days=7 * weeks_short)
    start_13w_prior = start_13w - pd.Timedelta(days=7 * weeks_short)
    start_12m = end - pd.DateOffset(months=months_ltm)
    return start_13w, start_13w_prior, start_12m, end


def _weekly(df: pd.DataFrame, date_col="date", value_col="net_revenue"):
    wk = df.set_index(date_col).resample("W-MON")[value_col].sum().rename("y")
    wk.index = wk.index.date
    return wk


def _slope_last_n(weekly: pd.Series, n_weeks: int) -> float:
    """OLS slope over last n points; x = 0..n-1; returns slope in revenue per week."""

    if weekly.empty:
        return np.nan
    y = weekly.tail(n_weeks).to_numpy(dtype=float)
    if len(y) < 2 or np.all(~np.isfinite(y)):
        return np.nan
    x = np.arange(len(y), dtype=float)
    # slope = cov(x,y)/var(x) ; numerically stable
    x_mean, y_mean = x.mean(), np.nanmean(y)
    num = np.nansum((x - x_mean) * (y - y_mean))
    den = np.nansum((x - x_mean) ** 2)
    return float(num / den) if den else np.nan


def _month_hhi(monthly: pd.Series) -> float:
    """Herfindahl index across months in window; p_i^2 summed."""

    total = monthly.sum()
    if total <= 0:
        return np.nan
    shares = monthly / total
    return float((shares**2).sum())


def compute_spend_dynamics(
    tx: pd.DataFrame,
    as_of: pd.Timestamp,
    weeks_short: int = 13,
    months_ltm: int = 12,
    month_windows: Tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """
    Returns one row per account_id with dynamic features:
    levels, deltas, YoY, slopes, acceleration, volatility, seasonality,
    recency/cadence, magnitude and momentum components.

    month_windows allows callers to request multiple long-horizon spend windows
    (e.g., 12M/24M/36M). Each window also includes prior-window growth deltas.
    """

    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.tz_localize(None)
    as_of = pd.to_datetime(as_of).tz_localize(None)

    if month_windows is None:
        month_windows = (months_ltm,)
    month_windows = tuple(sorted({int(m) for m in month_windows if m and m > 0}))
    primary_month_window = months_ltm if months_ltm in month_windows else month_windows[0]

    s13, s13p, s12m, end = _endpoints(as_of, weeks_short, months_ltm)

    # Pre-aggregate helpers
    def window(df, start, end):
        return df.loc[(df["date"] > start) & (df["date"] <= end)]

    # group worker to compute per-account metrics
    out = []
    for acc, g in tx.groupby("account_id"):
        g = g.sort_values("date")

        # Restrict to activity on or before the analysis date
        g = g[g["date"] <= as_of]
        if g.empty:
            continue

        # Level windows
        g_13 = window(g, s13, end)
        g_13p = window(g, s13p, s13)
        g_12 = window(g, s12m, end)

        spend_13w = g_13["net_revenue"].sum()
        spend_13w_prior = g_13p["net_revenue"].sum()
        delta_13w = spend_13w - spend_13w_prior
        delta_13w_pct = (delta_13w / spend_13w_prior) if spend_13w_prior else np.nan

        # Rolling month windows (e.g., 12M/24M/36M) with prior-window growth
        spend_by_window: dict[int, float] = {}
        spend_by_window_prior: dict[int, float] = {}
        for months in month_windows:
            start_ltm = end - pd.DateOffset(months=months)
            g_win = window(g, start_ltm, end)
            spend_by_window[months] = g_win["net_revenue"].sum()

            start_prior = start_ltm - pd.DateOffset(months=months)
            g_prior_win = window(g, start_prior, start_ltm)
            spend_by_window_prior[months] = g_prior_win["net_revenue"].sum()

        # YoY 13W: compare to 13W shifted -1y
        g_lastyear = g.copy()
        g_lastyear["date_shift"] = g_lastyear["date"] + pd.DateOffset(years=1)
        g_13_ly = g_lastyear.loc[
            (g_lastyear["date_shift"] > s13) & (g_lastyear["date_shift"] <= end)
        ]
        yoy_13w_pct = (
            (spend_13w - g_13_ly["net_revenue"].sum())
            / g_13_ly["net_revenue"].sum()
        ) if g_13_ly["net_revenue"].sum() else np.nan

        # Weekly time series for slope/volatility
        wk = _weekly(window(g, end - pd.Timedelta(days=7 * (weeks_short * 2)), end))
        slope_13w = _slope_last_n(wk, weeks_short)

        # Prior slope = previous window of same length
        slope_13w_prior = (
            _slope_last_n(wk.iloc[:-weeks_short], weeks_short)
            if len(wk) > weeks_short
            else np.nan
        )
        acceleration_13w = (
            slope_13w - slope_13w_prior
            if np.isfinite(slope_13w) and np.isfinite(slope_13w_prior)
            else np.nan
        )

        # Volatility over recent window
        vol_13w = float(np.nanstd(wk.tail(weeks_short))) if len(wk) else np.nan

        # Seasonality vs last year window
        spend_13w_ly = g_13_ly["net_revenue"].sum()
        seasonality_factor_13w = (spend_13w / spend_13w_ly) if spend_13w_ly else np.nan

        # Recency & cadence
        last_order_dt = g["date"].max()
        days_since_last_order = (
            (as_of - last_order_dt).days if pd.notnull(last_order_dt) else np.nan
        )

        # Active weeks in last 13W (weekly bins with revenue>0)
        active_weeks_13w = int((wk.tail(weeks_short) > 0).sum()) if len(wk) else 0

        # Purchase streak (months) - consecutive months with any spend up to as_of
        max_month_window = max(month_windows) if month_windows else months_ltm
        g_m = g.set_index("date").resample("MS")["net_revenue"].sum()
        if not g_m.empty:
            rev_flag = (g_m > 0).astype(int).sort_index()
            # compute streak ending at current month
            streak = 0
            for v in reversed(rev_flag.tail(max_month_window).tolist()):
                if v == 1:
                    streak += 1
                else:
                    break
            purchase_streak_months = streak
        else:
            purchase_streak_months = 0

        # Interpurchase median days (based on distinct invoice dates)
        inv_dates = (
            g.loc[g["date"].notna()]
            .drop_duplicates(subset=["account_id", "invoice_id", "date"])
            ["date"]
            .dt.normalize()
            .sort_values()
            .unique()
            .tolist()
        )
        if len(inv_dates) >= 2:
            inv_arr = np.array(inv_dates, dtype="datetime64[ns]")
            diffs = np.diff(inv_arr) / np.timedelta64(1, "D")
            median_interpurchase_days = float(np.median(diffs.astype(float)))
        else:
            median_interpurchase_days = np.nan

        # Momentum components (no ML)
        recency_score = (
            1.0 / (1.0 + (days_since_last_order / 30.0))
            if days_since_last_order == days_since_last_order
            else np.nan
        )
        baseline_spend = spend_by_window.get(primary_month_window, 0.0)
        magnitude_score = (spend_13w / baseline_spend) if baseline_spend else np.nan
        cadence_score = active_weeks_13w / weeks_short if weeks_short else np.nan

        row = dict(
            account_id=acc,
            spend_13w=spend_13w,
            spend_13w_prior=spend_13w_prior,
            delta_13w=delta_13w,
            delta_13w_pct=delta_13w_pct,
            yoy_13w_pct=yoy_13w_pct,
            slope_13w=slope_13w,
            slope_13w_prior=slope_13w_prior,
            acceleration_13w=acceleration_13w,
            volatility_13w=vol_13w,
            seasonality_factor_13w=seasonality_factor_13w,
            days_since_last_order=days_since_last_order,
            active_weeks_13w=active_weeks_13w,
            purchase_streak_months=purchase_streak_months,
            median_interpurchase_days=median_interpurchase_days,
            recency_score=recency_score,
            magnitude_score=magnitude_score,
            cadence_score=cadence_score,
        )

        # Attach dynamic month-window spend + prior/growth columns
        for months in month_windows:
            spend_val = spend_by_window.get(months, 0.0)
            prior_val = spend_by_window_prior.get(months, 0.0)
            delta_val = spend_val - prior_val
            delta_pct_val = (delta_val / prior_val) if prior_val else np.nan
            row[f"spend_{months}m"] = spend_val
            row[f"spend_{months}m_prior"] = prior_val
            row[f"delta_{months}m"] = delta_val
            row[f"delta_{months}m_pct"] = delta_pct_val

        out.append(row)

    base = pd.DataFrame(out)

    # Trend score as percentile of slope across accounts (higher slope => higher score)
    if "slope_13w" in base.columns:
        base["trend_score"] = base["slope_13w"].rank(pct=True, ascending=True)
    else:
        base["trend_score"] = pd.Series(dtype=float)

    return base
