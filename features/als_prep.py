from __future__ import annotations

import numpy as np
import pandas as pd


def _days_since(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    now = pd.Timestamp.utcnow().normalize()
    return (now - d).dt.days


def _exp_decay(days: pd.Series, half_life_days: float) -> pd.Series:
    days = pd.to_numeric(days, errors="coerce").fillna(days.max() if len(days) else 0)
    lam = np.log(2.0) / max(1.0, float(half_life_days))
    return np.exp(-lam * days)


def build_als_input_from_signals(
    profit_rollup: pd.DataFrame,
    assets_rollup: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Compose a rich implicit strength signal per (account_id, product_id=item_rollup) using
    multiple observed metrics we already collect:
      - Profit_Since_2023 (transactions)
      - asset_count, seats_sum, active_assets (assets)
      - recency (days since last_purchase_date / expiration at the rollup level)

    Returns DataFrame with columns: account_id, product_id, net_revenue (strength)
    suitable for ALS model input.
    """
    cfg = cfg or {}
    w_rev = float(cfg.get("w_rev", 1.0))
    w_seats = float(cfg.get("w_seats", 0.3))
    w_assets = float(cfg.get("w_assets", 0.2))
    w_active = float(cfg.get("w_active", 0.1))
    w_recency = float(cfg.get("w_recency", 0.3))
    recency_half_life = float(cfg.get("recency_half_life_days", 180))

    # Normalize column names for merge
    prof = (profit_rollup or pd.DataFrame()).copy()
    if isinstance(prof, pd.DataFrame) and not prof.empty:
        if "Customer ID" in prof.columns:
            prof["Customer ID"] = prof["Customer ID"].astype(str)
        prof = prof.rename(columns={
            "Customer ID": "account_id",
            "item_rollup": "product_id",
            "Profit_Since_2023": "profit_since_2023",
        })
        prof = prof[[c for c in ["account_id", "product_id", "profit_since_2023"] if c in prof.columns]]
        prof["profit_since_2023"] = pd.to_numeric(prof["profit_since_2023"], errors="coerce").fillna(0.0)
    else:
        prof = pd.DataFrame(columns=["account_id", "product_id", "profit_since_2023"])

    assets = (assets_rollup or pd.DataFrame()).copy()
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        if "Customer ID" in assets.columns:
            assets["Customer ID"] = assets["Customer ID"].astype(str)
        assets = assets.rename(columns={
            "Customer ID": "account_id",
            "item_rollup": "product_id",
        })
        # Expected columns: asset_count, seats_sum, active_assets, first_purchase_date, last_purchase_date, last_expiration_date
        for col in ["asset_count", "seats_sum", "active_assets"]:
            if col in assets.columns:
                assets[col] = pd.to_numeric(assets[col], errors="coerce").fillna(0.0)
        # Recency at rollup level
        if "last_purchase_date" in assets.columns:
            assets["days_since_last_rollup_purchase"] = _days_since(assets["last_purchase_date"])
        elif "first_purchase_date" in assets.columns:
            assets["days_since_last_rollup_purchase"] = _days_since(assets["first_purchase_date"])
        else:
            assets["days_since_last_rollup_purchase"] = np.nan
        if "last_expiration_date" in assets.columns:
            assets["days_since_last_rollup_expiry"] = _days_since(assets["last_expiration_date"])
        else:
            assets["days_since_last_rollup_expiry"] = np.nan
        # Recency weight (max of purchase/expiry recency signal)
        rec_purchase = _exp_decay(assets["days_since_last_rollup_purchase"], recency_half_life)
        rec_expiry = _exp_decay(assets["days_since_last_rollup_expiry"], recency_half_life)
        assets["recency_weight"] = np.nanmax(np.vstack([rec_purchase.to_numpy(), rec_expiry.to_numpy()]), axis=0)
        assets = assets[[
            c for c in [
                "account_id","product_id","asset_count","seats_sum","active_assets","recency_weight"
            ] if c in assets.columns
        ]]
    else:
        assets = pd.DataFrame(columns=["account_id","product_id","asset_count","seats_sum","active_assets","recency_weight"])

    combined = pd.merge(prof, assets, on=["account_id","product_id"], how="outer")
    combined[["profit_since_2023","asset_count","seats_sum","active_assets","recency_weight"]] = (
        combined[["profit_since_2023","asset_count","seats_sum","active_assets","recency_weight"]]
        .astype(float)
        .fillna(0.0)
    )
    # Compose strength with log1p profit & seats, additive assets/active, and multiplicative recency boost
    strength = (
        w_rev * np.log1p(combined["profit_since_2023"]) +
        w_seats * np.log1p(combined.get("seats_sum", 0.0)) +
        w_assets * combined.get("asset_count", 0.0) +
        w_active * combined.get("active_assets", 0.0)
    )
    # Apply recency as a (1 + w_recency*recency_weight) multiplier
    rec = combined.get("recency_weight", 0.0)
    strength = strength * (1.0 + w_recency * rec)

    out = pd.DataFrame({
        "account_id": combined["account_id"].astype(str),
        "product_id": combined["product_id"].astype(str),
        "net_revenue": strength.astype("float32"),
    })
    # Drop zero-strength rows
    out = out[out["net_revenue"] > 0]
    return out

