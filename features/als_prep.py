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
    prof = profit_rollup.copy() if isinstance(profit_rollup, pd.DataFrame) else pd.DataFrame()
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

    assets = assets_rollup.copy() if isinstance(assets_rollup, pd.DataFrame) else pd.DataFrame()
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
    # Clip negatives to avoid log1p warnings/NaNs and ensure non-negative implicit signals
    p = np.clip(combined["profit_since_2023"].to_numpy(dtype=float), 0.0, None)
    s = np.clip(combined.get("seats_sum", 0.0), 0.0, None)
    a = np.clip(combined.get("asset_count", 0.0), 0.0, None)
    act = np.clip(combined.get("active_assets", 0.0), 0.0, None)
    strength = (
        w_rev * np.log1p(p) +
        w_seats * np.log1p(s) +
        w_assets * a +
        w_active * act
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


def build_multi_als_inputs(
    profit_rollup: pd.DataFrame,
    assets_rollup: pd.DataFrame,
    cfg: dict | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Build multiple ALS component inputs to capture broader collaborative signals:
      - rollup: account x item_rollup composite strength
      - goal:   account x Goal composite strength

    Returns a list of (label, DataFrame[account_id, item_id, value])
    where 'item_id' is the dimension id (rollup or goal), and 'value' is the implicit strength.
    """
    cfg = cfg or {}
    # Reuse rollup builder
    rollup_df = build_als_input_from_signals(profit_rollup, assets_rollup, cfg)
    rollup_df = rollup_df.rename(columns={"product_id": "item_id", "net_revenue": "value"})

    # Build goal-level inputs using similar composite logic
    # Profit by Goal
    prof_goal = profit_rollup.copy() if isinstance(profit_rollup, pd.DataFrame) else pd.DataFrame()
    if isinstance(prof_goal, pd.DataFrame) and not prof_goal.empty:
        if "Customer ID" in prof_goal.columns:
            prof_goal["Customer ID"] = prof_goal["Customer ID"].astype(str)
        prof_goal = prof_goal.rename(columns={
            "Customer ID": "account_id",
            "Goal": "item_id",
            "Profit_Since_2023": "profit_since_2023",
        })
        prof_goal = prof_goal[[c for c in ["account_id", "item_id", "profit_since_2023"] if c in prof_goal.columns]]
        prof_goal["profit_since_2023"] = pd.to_numeric(prof_goal["profit_since_2023"], errors="coerce").fillna(0.0)
    else:
        prof_goal = pd.DataFrame(columns=["account_id", "item_id", "profit_since_2023"])

    # Assets by Goal (aggregate rollup assets to Goal)
    assets_goal = assets_rollup.copy() if isinstance(assets_rollup, pd.DataFrame) else pd.DataFrame()
    if isinstance(assets_goal, pd.DataFrame) and not assets_goal.empty:
        if "Customer ID" in assets_goal.columns:
            assets_goal["Customer ID"] = assets_goal["Customer ID"].astype(str)
        # Expect a Goal column on assets_rollup (from analytics_product_tags join in data_access)
        assets_goal = assets_goal.rename(columns={
            "Customer ID": "account_id",
        })
        if "Goal" in assets_goal.columns:
            g = assets_goal.copy()
            # Coerce numerics
            for col in ["asset_count", "seats_sum", "active_assets"]:
                if col in g.columns:
                    g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            # Recency proxy at goal-level: max of last_purchase/expiry in the group if present
            if "last_purchase_date" in g.columns:
                g["days_since_last_rollup_purchase"] = _days_since(g["last_purchase_date"])
            elif "first_purchase_date" in g.columns:
                g["days_since_last_rollup_purchase"] = _days_since(g["first_purchase_date"])
            else:
                g["days_since_last_rollup_purchase"] = np.nan
            if "last_expiration_date" in g.columns:
                g["days_since_last_rollup_expiry"] = _days_since(g["last_expiration_date"])
            else:
                g["days_since_last_rollup_expiry"] = np.nan
            rec_purchase = _exp_decay(g["days_since_last_rollup_purchase"], float(cfg.get("recency_half_life_days", 180)))
            rec_expiry = _exp_decay(g["days_since_last_rollup_expiry"], float(cfg.get("recency_half_life_days", 180)))
            g["recency_weight"] = np.nanmax(np.vstack([rec_purchase.to_numpy(), rec_expiry.to_numpy()]), axis=0)

            agg = g.groupby(["account_id", "Goal"], as_index=False).agg({
                "asset_count": "sum",
                "seats_sum": "sum",
                "active_assets": "sum",
                "recency_weight": "max",
            }).rename(columns={"Goal": "item_id"})
        else:
            agg = pd.DataFrame(columns=["account_id", "item_id", "asset_count", "seats_sum", "active_assets", "recency_weight"])
    else:
        agg = pd.DataFrame(columns=["account_id", "item_id", "asset_count", "seats_sum", "active_assets", "recency_weight"])

    # Join profit goal and assets goal
    goal_comb = pd.merge(prof_goal, agg, on=["account_id", "item_id"], how="outer")
    goal_comb[["profit_since_2023", "asset_count", "seats_sum", "active_assets", "recency_weight"]] = (
        goal_comb[["profit_since_2023", "asset_count", "seats_sum", "active_assets", "recency_weight"]]
        .astype(float)
        .fillna(0.0)
    )
    # Compose goal strength with same parameters as rollup
    w_rev = float(cfg.get("w_rev", 1.0))
    w_seats = float(cfg.get("w_seats", 0.3))
    w_assets = float(cfg.get("w_assets", 0.2))
    w_active = float(cfg.get("w_active", 0.1))
    w_recency = float(cfg.get("w_recency", 0.3))

    # Clip negatives at goal level as well
    gp = np.clip(goal_comb["profit_since_2023"].to_numpy(dtype=float), 0.0, None)
    gs = np.clip(goal_comb.get("seats_sum", 0.0), 0.0, None)
    ga = np.clip(goal_comb.get("asset_count", 0.0), 0.0, None)
    gact = np.clip(goal_comb.get("active_assets", 0.0), 0.0, None)
    goal_strength = (
        w_rev * np.log1p(gp) +
        w_seats * np.log1p(gs) +
        w_assets * ga +
        w_active * gact
    )
    goal_strength = goal_strength * (1.0 + w_recency * goal_comb.get("recency_weight", 0.0))

    goal_df = pd.DataFrame({
        "account_id": goal_comb["account_id"].astype(str),
        "item_id": goal_comb["item_id"].astype(str),
        "value": goal_strength.astype("float32"),
    })
    goal_df = goal_df[goal_df["value"] > 0]

    out = [("rollup", rollup_df), ("goal", goal_df)]
    return out
