"""Feature engineering functions."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings

from icp.divisions import DivisionConfig, get_division_config
from icp.scoring import calculate_scores
from icp.schema import (
    COL_CUSTOMER_ID,
    LICENSE_COL,
)

# Import existing feature modules - assuming these are in the python path or relative
# Based on previous file listing, 'features' is a top-level directory.
# We might need to adjust imports if 'features' is not a package or not in path.
# Assuming the environment is set up such that 'features' is importable.
from features.spend_dynamics import compute_spend_dynamics
from features.pov_tags import make_pov_tags

FOCUS_GOALS = {"Printers", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"}

PRINTER_SUBDIVISIONS = [
    "AM Software",
    "AM Support",
    "Consumables",
    "FDM",
    "FormLabs",
    "Metals",
    "P3",
    "Polyjet",
    "Post Processing",
    "SAF",
    "SLA",
    "Spare Parts/Repair Parts/Time & Materials",
]

FEATURE_COLUMN_ORDER = [
    "account_id",
    "spend_13w",
    "spend_13w_prior",
    "delta_13w",
    "delta_13w_pct",
    "spend_12m",
    "spend_12m_prior",
    "delta_12m",
    "delta_12m_pct",
    "spend_24m",
    "spend_24m_prior",
    "delta_24m",
    "delta_24m_pct",
    "spend_36m",
    "spend_36m_prior",
    "delta_36m",
    "delta_36m_pct",
    "yoy_13w_pct",
    "days_since_last_order",
    "active_weeks_13w",
    "purchase_streak_months",
    "median_interpurchase_days",
    "slope_13w",
    "slope_13w_prior",
    "acceleration_13w",
    "volatility_13w",
    "seasonality_factor_13w",
    "trend_score",
    "recency_score",
    "magnitude_score",
    "cadence_score",
    "momentum_score",
    "w_trend",
    "w_recency",
    "w_magnitude",
    "w_cadence",
    "hw_spend_12m",
    "sw_spend_12m",
    "spend_12m_hw",
    "spend_24m_hw",
    "spend_36m_hw",
    "spend_12m_cre",
    "spend_24m_cre",
    "spend_36m_cre",
    "spend_12m_cpe",
    "spend_24m_cpe",
    "spend_36m_cpe",
    "hw_share_12m",
    "sw_share_12m",
    "breadth_cre_rollup_12m",
    "max_cre_rollup",
    "breadth_score_cre",
    "days_since_last_cre_order",
    "recency_score_cre",
    "days_since_last_cpe_order",
    "recency_score_cpe",
    "breadth_hw_subdiv_12m",
    "max_hw_subdiv",
    "breadth_score_hw",
    "days_since_last_hw_order",
    "recency_score_hw",
    "hardware_adoption_score",
    "consumables_to_hw_ratio",
    "top_subdivision_12m",
    "top_subdivision_share_12m",
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
    "discount_pct",
    "month_conc_hhi_12m",
    "sw_dominance_score",
    "sw_to_hw_whitespace_score",
    "pov_primary",
    "pov_tags_all",
    "as_of_date",
    "run_ts_utc",
]

def _printer_rollup_slug(label: str) -> str:
    return str(label).strip().replace('/', '_').replace(' ', '_').replace('&', 'and')


def _normalize_goal_name(x: str) -> str:
    return str(x).strip().lower()


def _attach_helper_frame(df: pd.DataFrame, name: str, value: pd.DataFrame) -> None:
    """Store helper frames on attrs and legacy attributes for compatibility."""
    if not isinstance(df, pd.DataFrame):
        return
    df.attrs[name] = value
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            setattr(df, name, value)
    except Exception:
        # Avoid breaking if pandas changes attribute semantics again
        pass


def _get_attached_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Retrieve an attached helper DataFrame from attrs or fallback attributes."""
    if isinstance(df, pd.DataFrame):
        if name in df.attrs:
            return df.attrs.get(name, pd.DataFrame())
        if hasattr(df, name):
            return getattr(df, name, pd.DataFrame())
    return pd.DataFrame()

def canonicalize_customer_id(series: pd.Series) -> pd.Series:
    """Standardize customer IDs to string format."""
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def engineer_features(
    df: pd.DataFrame,
    asset_weights: dict,
    weights: dict,
    division: str | DivisionConfig | None = None,
) -> pd.DataFrame:
    """
    Engineers features for scoring using assets/seats and profit.
    """
    # Preserve access to any attached raw attributes before copying
    _base_df = df
    df = df.copy()

    division_config = division if isinstance(division, DivisionConfig) else get_division_config(division)

    assets = _get_attached_frame(_base_df, "_assets_raw")
    profit_roll = _get_attached_frame(_base_df, "_profit_rollup_raw")

    if COL_CUSTOMER_ID in df.columns:
        df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])

    # Default zeros
    df["adoption_assets"] = 0.0
    df["adoption_profit"] = 0.0
    df["relationship_profit"] = 0.0
    df["cre_adoption_assets"] = 0.0
    df["cre_adoption_profit"] = 0.0
    df["cre_relationship_profit"] = 0.0
    df["cpe_adoption_assets"] = 0.0
    df["cpe_adoption_profit"] = 0.0
    df["cpe_relationship_profit"] = 0.0
    df["printer_count"] = 0.0

    # Build adoption_assets from assets table with weights
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        # Normalize weights config keys to lower case for consistent lookups
        _raw_weights_cfg = asset_weights.get("weights", {}) or {}
        weights_cfg = {
            _normalize_goal_name(g): { (str(k).lower() if isinstance(k, str) else k): v for k, v in (m or {}).items() }
            for g, m in _raw_weights_cfg.items()
        }
        focus_goals = set(asset_weights.get("focus_goals", list(FOCUS_GOALS)))
        focus_goals = {_normalize_goal_name(g) for g in focus_goals}
        # Add common synonyms
        if 'printer' in focus_goals or 'printers' in focus_goals:
            focus_goals.update({'printer', 'printers'})

        def weighted_measure(row) -> float:
            goal = _normalize_goal_name(row.get("Goal"))
            item_rollup = str(row.get("item_rollup"))
            seats_sum = row.get("seats_sum", 0) or 0
            asset_count = row.get("asset_count", 0) or 0
            base = seats_sum if seats_sum and seats_sum > 0 else asset_count
            goal_weights = weights_cfg.get(goal, {})
            w = goal_weights.get(item_rollup, goal_weights.get("default", 1.0))
            return float(base) * float(w)

        a = assets.copy()
        if COL_CUSTOMER_ID in a.columns:
            a[COL_CUSTOMER_ID] = canonicalize_customer_id(a[COL_CUSTOMER_ID])
        a["Goal"] = a["Goal"].map(_normalize_goal_name)
        # Keep only focus goals; special handling for Training/Services: only 3DP Training rollup counts
        a_focus = a[a["Goal"].isin(focus_goals)].copy()
        a_focus.loc[:, "weighted_value"] = a_focus.apply(weighted_measure, axis=1)

        # Compute printer_count specifically from Printer assets (use asset_count)
        printer_assets = a_focus[a_focus["Goal"].isin({_normalize_goal_name("Printer"), _normalize_goal_name("Printers")})]
        printer_counts = (
            printer_assets.groupby(COL_CUSTOMER_ID)["asset_count"].sum().rename("printer_count")
        )

        adoption_assets = (
            a_focus.groupby(COL_CUSTOMER_ID)["weighted_value"].sum().rename("adoption_assets")
        )

        df = df.merge(adoption_assets, on=COL_CUSTOMER_ID, how="left")
        df = df.merge(printer_counts, on=COL_CUSTOMER_ID, how="left")

        # CRE (software) adoption assets from CAD/CPE/Draftsight/Misc/Training goals
        cre_goals = {
            _normalize_goal_name("CAD"),
            _normalize_goal_name("CPE"),
            _normalize_goal_name("Draftsight"),
            _normalize_goal_name("Miscellaneous"),
            _normalize_goal_name("Training"),
        }
        a_cre = a[a["Goal"].isin(cre_goals)].copy()
        if not a_cre.empty:
            def _cre_measure(row) -> float:
                seats_sum = row.get("seats_sum", 0) or 0
                active_assets = row.get("active_assets", 0) or 0
                base = seats_sum if seats_sum and seats_sum > 0 else active_assets
                if not base:
                    base = row.get("asset_count", 0) or 0
                return float(base)

            a_cre.loc[:, "cre_weighted"] = a_cre.apply(_cre_measure, axis=1)
            cre_assets = (
                a_cre.groupby(COL_CUSTOMER_ID)["cre_weighted"].sum().rename("cre_adoption_assets")
            )
            df = df.merge(cre_assets, on=COL_CUSTOMER_ID, how="left")

        # Resolve potential suffixes from merge: always prefer merged values when present
        if 'adoption_assets_y' in df.columns:
            if 'adoption_assets' in df.columns:
                df['adoption_assets'] = df['adoption_assets_y'].combine_first(df['adoption_assets'])
            else:
                df['adoption_assets'] = df['adoption_assets_y']
            drop_cols = [c for c in ['adoption_assets_x','adoption_assets_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        if 'printer_count_y' in df.columns:
            if 'printer_count' in df.columns:
                df['printer_count'] = df['printer_count_y'].combine_first(df['printer_count'])
            else:
                df['printer_count'] = df['printer_count_y']
            drop_cols = [c for c in ['printer_count_x','printer_count_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["adoption_assets"] = df.get("adoption_assets", 0).fillna(0.0)
        df["printer_count"] = df.get("printer_count", 0).fillna(0.0)
        if 'cre_adoption_assets_y' in df.columns:
            if 'cre_adoption_assets' in df.columns:
                df['cre_adoption_assets'] = df['cre_adoption_assets_y'].combine_first(df['cre_adoption_assets'])
            else:
                df['cre_adoption_assets'] = df['cre_adoption_assets_y']
            drop_cols = [c for c in ['cre_adoption_assets_x','cre_adoption_assets_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["cre_adoption_assets"] = df.get("cre_adoption_assets", 0).fillna(0.0)

        # CPE adoption assets from CPE goal
        cpe_goal = {_normalize_goal_name("CPE")}
        a_cpe = a[a["Goal"].isin(cpe_goal)].copy()
        if not a_cpe.empty:
            def _cpe_measure(row) -> float:
                seats_sum = row.get("seats_sum", 0) or 0
                active_assets = row.get("active_assets", 0) or 0
                base = seats_sum if seats_sum and seats_sum > 0 else active_assets
                if not base:
                    base = row.get("asset_count", 0) or 0
                return float(base)

            a_cpe.loc[:, "cpe_weighted"] = a_cpe.apply(_cpe_measure, axis=1)
            cpe_assets = (
                a_cpe.groupby(COL_CUSTOMER_ID)["cpe_weighted"].sum().rename("cpe_adoption_assets")
            )
            df = df.merge(cpe_assets, on=COL_CUSTOMER_ID, how="left")
        if 'cpe_adoption_assets_y' in df.columns:
            if 'cpe_adoption_assets' in df.columns:
                df['cpe_adoption_assets'] = df['cpe_adoption_assets_y'].combine_first(df['cpe_adoption_assets'])
            else:
                df['cpe_adoption_assets'] = df['cpe_adoption_assets_y']
            drop_cols = [c for c in ['cpe_adoption_assets_x','cpe_adoption_assets_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["cpe_adoption_assets"] = df.get("cpe_adoption_assets", 0).fillna(0.0)

    # Build adoption_profit from profit_rollup: focus goals + 3DP Training rollup
    if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
        pr = profit_roll.copy()
        if COL_CUSTOMER_ID in pr.columns:
            pr[COL_CUSTOMER_ID] = canonicalize_customer_id(pr[COL_CUSTOMER_ID])
        pr["Goal"] = pr["Goal"].map(_normalize_goal_name)
        # Sum for focus goals
        focus_goals_norm = {_normalize_goal_name(g) for g in FOCUS_GOALS}
        mask_focus_goals = pr["Goal"].isin(focus_goals_norm)
        # Include only 3DP Training within Training/Services
        mask_3dp_training = (pr["Goal"] == _normalize_goal_name("Training/Services")) & (pr["item_rollup"].astype(str).str.strip().str.lower() == "3dp training")
        mask_focus = mask_focus_goals & (~(pr["Goal"] == _normalize_goal_name("Training/Services")) | mask_3dp_training)
        adoption_profit = (
            pr[mask_focus]
            .drop_duplicates(subset=[COL_CUSTOMER_ID, "item_rollup"])
            .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"]
            .sum()
            .rename("adoption_profit")
        )
        df = df.merge(adoption_profit, on=COL_CUSTOMER_ID, how="left")

        # CRE adoption profit from CAD, Specialty Software, Draftsight, Miscellaneous, Training goals
        mask_cre = pr["Goal"].isin({
            _normalize_goal_name("CAD"),
            _normalize_goal_name("Specialty Software"),
            _normalize_goal_name("Draftsight"),
            _normalize_goal_name("Miscellaneous"),
            _normalize_goal_name("Training"),
        })
        cre_profit = (
            pr[mask_cre]
            .drop_duplicates(subset=[COL_CUSTOMER_ID, "item_rollup"])
            .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"]
            .sum()
            .rename("cre_adoption_profit")
        )
        if not cre_profit.empty:
            df = df.merge(cre_profit, on=COL_CUSTOMER_ID, how="left")
        if 'adoption_profit_y' in df.columns:
            if 'adoption_profit' in df.columns:
                df['adoption_profit'] = df['adoption_profit_y'].combine_first(df['adoption_profit'])
            else:
                df['adoption_profit'] = df['adoption_profit_y']
            drop_cols = [c for c in ['adoption_profit_x','adoption_profit_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["adoption_profit"] = df.get("adoption_profit", 0).fillna(0.0)
        if 'cre_adoption_profit_y' in df.columns:
            if 'cre_adoption_profit' in df.columns:
                df['cre_adoption_profit'] = df['cre_adoption_profit_y'].combine_first(df['cre_adoption_profit'])
            else:
                df['cre_adoption_profit'] = df['cre_adoption_profit_y']
            drop_cols = [c for c in ['cre_adoption_profit_x','cre_adoption_profit_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["cre_adoption_profit"] = df.get("cre_adoption_profit", 0).fillna(0.0)

        # CPE adoption profit from CPE goal
        mask_cpe = pr["Goal"] == _normalize_goal_name("CPE")
        cpe_profit = (
            pr[mask_cpe]
            .drop_duplicates(subset=[COL_CUSTOMER_ID, "item_rollup"])
            .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"]
            .sum()
            .rename("cpe_adoption_profit")
        )
        if not cpe_profit.empty:
            df = df.merge(cpe_profit, on=COL_CUSTOMER_ID, how="left")
        if 'cpe_adoption_profit_y' in df.columns:
            if 'cpe_adoption_profit' in df.columns:
                df['cpe_adoption_profit'] = df['cpe_adoption_profit_y'].combine_first(df['cpe_adoption_profit'])
            else:
                df['cpe_adoption_profit'] = df['cpe_adoption_profit_y']
            drop_cols = [c for c in ['cpe_adoption_profit_x','cpe_adoption_profit_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["cpe_adoption_profit"] = df.get("cpe_adoption_profit", 0).fillna(0.0)

    # Relationship signals recalculated explicitly per division (cross-division fit)
    if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
        pr_rel = profit_roll.copy()
        if COL_CUSTOMER_ID in pr_rel.columns:
            pr_rel[COL_CUSTOMER_ID] = canonicalize_customer_id(pr_rel[COL_CUSTOMER_ID])
        pr_rel['Goal'] = pr_rel['Goal'].map(_normalize_goal_name)

        # Hardware relationship: fit to CRE + CPE (CAD, Specialty Software, Draftsight, Misc, CPE)
        hw_mask = pr_rel["Goal"].isin({
            _normalize_goal_name("CAD"),
            _normalize_goal_name("Specialty Software"),
            _normalize_goal_name("Draftsight"),
            _normalize_goal_name("Miscellaneous"),
            _normalize_goal_name("CPE"),
        })
        rel_hw = (
            pr_rel[hw_mask]
            .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"]
            .sum()
            .rename("relationship_profit")
        )
        df = df.merge(rel_hw, on=COL_CUSTOMER_ID, how="left")
        if "relationship_profit" not in df.columns:
            df["relationship_profit"] = 0.0
        df["relationship_profit"] = pd.to_numeric(df["relationship_profit"], errors="coerce").fillna(0.0)

        # Fallback: if rollup-derived relationship profit is zero, reuse CRE/CPE GP columns
        rel_gp_cols = [c for c in ["GP_CAD", "GP_Specialty Software", "GP_Draftsight", "GP_Miscellaneous", "GP_CPE"] if c in df.columns]
        if rel_gp_cols:
            rel_gp_alt = df[rel_gp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            df["relationship_profit"] = df["relationship_profit"].where(df["relationship_profit"] > 0, rel_gp_alt)

    # CRE relationship: blend cross-division signals
    if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
        pr_cre = profit_roll.copy()
        # Ensure Customer ID dtype matches master (canonical string) to avoid merge type issues
        if COL_CUSTOMER_ID in pr_cre.columns:
            pr_cre[COL_CUSTOMER_ID] = canonicalize_customer_id(pr_cre[COL_CUSTOMER_ID])
        pr_cre["Goal"] = pr_cre["Goal"].map(_normalize_goal_name)
        specialty_mask = pr_cre["Goal"] == _normalize_goal_name("Specialty Software")
        training_mask = pr_cre["Goal"] == _normalize_goal_name("Training/Services")
        if "item_rollup" in pr_cre.columns:
            # Exact allowlist per taxonomy
            ir = pr_cre["item_rollup"].astype(str).str.strip().str.lower()
            allowed_train = {"success plan", "training"}
            train_allowed_mask = ir.isin(allowed_train)
        else:
            train_allowed_mask = pd.Series(False, index=pr_cre.index)
        include_mask = specialty_mask | (training_mask & train_allowed_mask)
        rel_cre_gp = (
            pr_cre[include_mask]
            .drop_duplicates(subset=[COL_CUSTOMER_ID, "item_rollup"])
            .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"]
            .sum()
            .rename("_tmp_cre_specialty_train_gp")
        )
        if not rel_cre_gp.empty:
            df = df.merge(rel_cre_gp, on=COL_CUSTOMER_ID, how="left")

    # Build composite relationship signal using cross-division assets + GP
    # Helpers: safe percentile rank (0..1)
    def _pctl(series):
        s = pd.to_numeric(series, errors="coerce")
        if isinstance(s, (int, float)):
            s = pd.Series(float(s), index=df.index)
        s = s.fillna(0)
        if len(s) == 0:
            return s
        if not (s > 0).any():
            return pd.Series(0.0, index=s.index, dtype=float)
        return s.rank(pct=True)

    def _series(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=df.index, dtype=float)

    # Pre-compute key GP/seat signals for relationship blends if not already present
    rel_roll = _get_attached_frame(_base_df, "_profit_rollup_raw")
    if isinstance(rel_roll, pd.DataFrame) and not rel_roll.empty:
        pr_tmp = rel_roll.copy()
        if COL_CUSTOMER_ID in pr_tmp.columns:
            pr_tmp[COL_CUSTOMER_ID] = canonicalize_customer_id(pr_tmp[COL_CUSTOMER_ID])
        pr_tmp["Goal"] = pr_tmp["Goal"].map(_normalize_goal_name)
        pr_tmp["item_rollup"] = pr_tmp.get("item_rollup", "").astype(str)

        goal_map = {
            _normalize_goal_name("Printers"): "GP_Printers",
            _normalize_goal_name("CPE"): "GP_CPE",
            _normalize_goal_name("Specialty Software"): "GP_Specialty Software",
            _normalize_goal_name("CAD"): "GP_CAD",
        }
        for goal, col in goal_map.items():
            series = pr_tmp.loc[pr_tmp["Goal"] == goal].groupby(COL_CUSTOMER_ID)["Profit_Since_2023"].sum()
            if col not in df.columns:
                df[col] = df[COL_CUSTOMER_ID].map(series).fillna(0.0)

        # Training/Services subset
        allowed_train = {"success plan", "training"}
        mask_train = pr_tmp["Goal"] == _normalize_goal_name("Training/Services")
        if mask_train.any():
            tr = pr_tmp[mask_train].copy()
            tr["rollup_norm"] = tr["item_rollup"].str.strip().str.lower()
            for roll, col in {
                "success plan": "GP_Training/Services_Success_Plan",
                "training": "GP_Training/Services_Training",
            }.items():
                series = tr.loc[tr["rollup_norm"] == roll].groupby(COL_CUSTOMER_ID)["Profit_Since_2023"].sum()
                if col not in df.columns:
                    df[col] = df[COL_CUSTOMER_ID].map(series).fillna(0.0)

    assets_tmp = _get_attached_frame(_base_df, "_assets_raw")
    if isinstance(assets_tmp, pd.DataFrame) and not assets_tmp.empty:
        a_tmp = assets_tmp.copy()
        if COL_CUSTOMER_ID in a_tmp.columns:
            a_tmp[COL_CUSTOMER_ID] = canonicalize_customer_id(a_tmp[COL_CUSTOMER_ID])
        a_tmp["Goal"] = a_tmp["Goal"].map(_normalize_goal_name)
        a_tmp["item_rollup"] = a_tmp.get("item_rollup", "").astype(str)
        seats_cpe_series = (
            a_tmp.loc[a_tmp["Goal"] == _normalize_goal_name("CPE")]
            .groupby(COL_CUSTOMER_ID)["seats_sum"]
            .sum()
        )
        if "Seats_CPE" not in df.columns:
            df["Seats_CPE"] = df[COL_CUSTOMER_ID].map(seats_cpe_series).fillna(0.0)

    # Training subset GP (sum of two stable columns)
    # Note: These columns come from loader assembly which might use Goal pivot.
    # If we want to be safe, we should use the _tmp_cre_specialty_train_gp we just calculated for the GP part.
    # But _tmp_cre_specialty_train_gp includes Specialty Software too.
    
    # Let's trust the component columns for now as they are specific (e.g. GP_Training/Services_Success_Plan)
    # which are likely from rollup pivot in loader, which is fine if rollup is unique.
    
    t1 = _series("GP_Training/Services_Success_Plan")
    t2 = _series("GP_Training/Services_Training")
    printer_count_series = _series("printer_count")
    gp_printers_series = _series("GP_Printers")
    gp_printer_accessories_series = _series("GP_Printer Accessories")
    seats_cpe = _series("Seats_CPE")
    gp_cpe = _series("GP_CPE")
    gp_cad = _series("GP_CAD")
    specialty_gp = _series("GP_Specialty Software")
    gp_misc = _series("GP_Miscellaneous")
    gp_draftsight = _series("GP_Draftsight")
    specialty_profit = _series("Specialty Software")

    # Cross-division relationship scoring per division
    # Hardware relationship already set above as relationship_profit (uses CRE+CPE GP)

    # CRE relationship: Hardware + CPE GP
    rel_hw_for_cre = _pctl(gp_printers_series + gp_printer_accessories_series)
    rel_cpe_for_cre = _pctl(gp_cpe)
    cre_rel_blend = 0.5 * rel_hw_for_cre + 0.5 * rel_cpe_for_cre
    df["cre_relationship_profit"] = pd.to_numeric(cre_rel_blend, errors="coerce").fillna(0.0)

    # CPE relationship: Hardware + CRE GP (CAD/Specialty/Draftsight/Misc)
    rel_hw_for_cpe = _pctl(gp_printers_series + gp_printer_accessories_series)
    rel_cre_for_cpe = _pctl(gp_cad + specialty_gp + gp_misc + gp_draftsight)
    cpe_rel_blend = 0.5 * rel_hw_for_cpe + 0.5 * rel_cre_for_cpe
    df["cpe_relationship_profit"] = pd.to_numeric(cpe_rel_blend, errors="coerce").fillna(0.0)

    # Flag for scaling
    df["scaling_flag"] = (df["printer_count"] >= 4).astype(int)

    # Compatibility column for dashboard tiering
    if "relationship_profit" in df.columns:
        df[LICENSE_COL] = df["relationship_profit"].fillna(0.0)
    else:
        df[LICENSE_COL] = 0.0

    # Calculate all component and final scores via scoring_logic
    df = calculate_scores(df, weights, division=division_config)

    # --- Per-goal quantity (assets) and GP (transactions) totals ---
    def safe_label(s: str) -> str:
        return _printer_rollup_slug(s)

    # Normalize goal labels for output
    goal_label_map = {
        'printer accessorials': 'Printer Accessories',
        'printers': 'Printers',
        'printer': 'Printers',
        'scanners': 'Scanners',
        'geomagic': 'Geomagic',
        'miscellaneous': 'Miscellaneous',
        'draftsight': 'Draftsight',
        'services': 'Services',
        'training/services': 'Training',
        'cad': 'CAD',
        'cpe': 'CPE',
        'specialty software': 'Specialty Software',
    }

    # Quantities from assets
    assets_df = _get_attached_frame(_base_df, "_assets_raw")
    if isinstance(assets_df, pd.DataFrame) and not assets_df.empty:
        a2 = assets_df.copy()
        if COL_CUSTOMER_ID in a2.columns:
            a2[COL_CUSTOMER_ID] = canonicalize_customer_id(a2[COL_CUSTOMER_ID])
        a2['Goal'] = a2['Goal'].map(_normalize_goal_name)
        # Totals per goal by asset_count
        qty_goal = (
            a2.groupby([COL_CUSTOMER_ID,'Goal'])['asset_count'].sum().unstack(fill_value=0)
        )
        if not qty_goal.empty:
            qty_goal.columns = [f"Qty_{goal_label_map.get(c, c.title())}" for c in qty_goal.columns]
            qty_goal = qty_goal.reset_index()
            df = df.merge(qty_goal, on=COL_CUSTOMER_ID, how='left')
        # Per rollup totals (asset_count), always including printer subdivisions
        weights_cfg = (asset_weights.get('weights', {}) or {})
        keep_rollups = set()
        for g, m in weights_cfg.items():
            if isinstance(m, dict):
                for r in m.keys():
                    keep_rollups.add((_normalize_goal_name(g), str(r)))
        ar = a2.copy()
        ar['item_rollup'] = ar['item_rollup'].astype(str)
        printer_goal = _normalize_goal_name("Printers")
        printer_rollups = {
            (printer_goal, str(r))
            for r in ar.loc[ar['Goal'] == printer_goal, 'item_rollup'].dropna().unique()
            if str(r).strip() and str(r).strip().lower() != 'default'
        }
        rollup_filters = keep_rollups.union(printer_rollups)
        if rollup_filters:
            ar['_combo'] = list(zip(ar['Goal'], ar['item_rollup']))
            ar = ar[ar['_combo'].isin(rollup_filters)]
            ar = ar.drop(columns=['_combo'])
        if not ar.empty:
            grp = ar.groupby([COL_CUSTOMER_ID,'Goal','item_rollup'])['asset_count'].sum().reset_index()
            for (g, r), sub in grp.groupby(['Goal','item_rollup']):
                label = f"Qty_{goal_label_map.get(g, g.title())}_{safe_label(r)}"
                m = sub.set_index(COL_CUSTOMER_ID)['asset_count']
                df[label] = df[COL_CUSTOMER_ID].map(m).fillna(0)

        for roll in PRINTER_SUBDIVISIONS:
            col = f"Qty_Printers_{safe_label(roll)}"
            if col not in df.columns:
                df[col] = 0

        # Seats per goal and rollup for CAD and Specialty Software
        if 'seats_sum' in a2.columns:
            a2['item_rollup'] = a2['item_rollup'].astype(str)
            seats_grp = (
                a2.groupby([COL_CUSTOMER_ID,'Goal','item_rollup'])['seats_sum'].sum().reset_index()
            )
            if not seats_grp.empty:
                for (g, r), sub in seats_grp.groupby(['Goal','item_rollup']):
                    # Only expose for CAD / Specialty Software
                    if g in {_normalize_goal_name('CAD'), _normalize_goal_name('Specialty Software')}:
                        label = f"Seats_{goal_label_map.get(g, g.title())}_{safe_label(r)}"
                        m = sub.set_index(COL_CUSTOMER_ID)['seats_sum']
                        df[label] = df[COL_CUSTOMER_ID].map(m).fillna(0)

    # GP per goal and per rollup from transactions
    pr_df = _get_attached_frame(_base_df, "_profit_rollup_raw")
    if isinstance(pr_df, pd.DataFrame) and not pr_df.empty:
        pr2 = pr_df.copy()
        if COL_CUSTOMER_ID in pr2.columns:
            pr2[COL_CUSTOMER_ID] = canonicalize_customer_id(pr2[COL_CUSTOMER_ID])
        pr2['Goal'] = pr2['Goal'].map(_normalize_goal_name)
        # Restrict Training/Services to CRE-allowed rollups only
        pr2['item_rollup'] = pr2['item_rollup'].astype(str)
        allowed_train = {"success plan", "training", "service", "services", "training/services"}
        is_train = pr2['Goal'] == _normalize_goal_name('Training/Services')
        pr2 = pr2[~is_train | pr2['item_rollup'].str.strip().str.lower().isin(allowed_train)]

        gp_goal = pr2.groupby([COL_CUSTOMER_ID,'Goal'])['Profit_Since_2023'].sum().unstack(fill_value=0)
        if not gp_goal.empty:
            gp_goal.columns = [f"GP_{goal_label_map.get(c, c.title())}" for c in gp_goal.columns]
            gp_goal = gp_goal.reset_index()
            overlap = [c for c in gp_goal.columns if c != COL_CUSTOMER_ID and c in df.columns]
            gp_goal = gp_goal.drop(columns=overlap) if overlap else gp_goal
            if len(gp_goal.columns) > 1:
                df = df.merge(gp_goal, on=COL_CUSTOMER_ID, how='left')
        grp = pr2.groupby([COL_CUSTOMER_ID,'Goal','item_rollup'])['Profit_Since_2023'].sum().reset_index()
        for (g, r), sub in grp.groupby(['Goal','item_rollup']):
            label = f"GP_{goal_label_map.get(g, g.title())}_{safe_label(r)}"
            m = sub.set_index(COL_CUSTOMER_ID)['Profit_Since_2023']
            df[label] = df[COL_CUSTOMER_ID].map(m).fillna(0.0)

        for roll in PRINTER_SUBDIVISIONS:
            col = f"GP_Printers_{safe_label(roll)}"
            if col not in df.columns:
                df[col] = 0.0

    # Days since metrics
    # Use timezone-naive 'now' to align with tz-naive datetimes
    # Normalize 'now' and target datetimes to tz-naive before diff
    now = pd.Timestamp.now(tz=None).normalize()
    for col, out in [
        ('EarliestPurchaseDate','Days_Since_First_Purchase'),
        ('LatestPurchaseDate','Days_Since_Last_Purchase'),
        ('LatestExpirationDate','Days_Since_Last_Expiration')
    ]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors='coerce')
            # strip tz info if present
            try:
                dt = dt.dt.tz_convert(None)
            except Exception:
                try:
                    dt = dt.dt.tz_localize(None)
                except Exception:
                    pass
            df[out] = (now - dt).dt.days

    # Phase 1: Momentum & Recency features
    # GP_Last_90D
    gp90 = _get_attached_frame(_base_df, "_gp_last90")
    if isinstance(gp90, pd.DataFrame) and not gp90.empty:
        m90 = gp90.groupby(COL_CUSTOMER_ID)['GP_Last_ND'].sum()
        df['GP_Last_90D'] = df[COL_CUSTOMER_ID].map(m90).fillna(0.0)
    else:
        df['GP_Last_90D'] = 0.0

    # Months_Active_12M & GP_Trend_Slope_12M
    monthly = _get_attached_frame(_base_df, "_gp_monthly12")
    if isinstance(monthly, pd.DataFrame) and not monthly.empty:
        # Normalize ID and build a proper YearMonth key
        mth = monthly.copy()
        mth[COL_CUSTOMER_ID] = canonicalize_customer_id(mth[COL_CUSTOMER_ID])
        mth['YM'] = mth['Year'] * 100 + mth['Month']
        # Months active
        active_counts = (
            mth.assign(Active=(mth['Profit'] > 0).astype(int))
               .groupby(COL_CUSTOMER_ID)['Active'].sum()
        )
        df['Months_Active_12M'] = df[COL_CUSTOMER_ID].map(active_counts).fillna(0).astype(int)

        # Trend slope using polyfit over last 12 months (fill missing months with 0)
        def slope_for_customer(g):
            # Build 12-length series aligned to last 12 distinct YMs
            yms_all = sorted(mth['YM'].unique())[-12:]
            if len(yms_all) == 0:
                return 0.0
            s = g.set_index('YM')['Profit']
            vals = [float(s.get(ym, 0.0)) for ym in yms_all]
            x = np.arange(1, len(vals)+1)
            if len(x) >= 2 and np.any(vals):
                try:
                    return float(np.polyfit(x, vals, 1)[0])
                except Exception:
                    return 0.0
            return 0.0
        slopes = (
            mth.groupby(COL_CUSTOMER_ID, group_keys=False, sort=False)[["YM", "Profit"]]
            .apply(slope_for_customer)
        )
        df['GP_Trend_Slope_12M'] = df[COL_CUSTOMER_ID].map(slopes).fillna(0.0)
    else:
        df['Months_Active_12M'] = 0
        df['GP_Trend_Slope_12M'] = 0.0

    return df

def enrich_with_list_builder_features(
    df_accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of_date: datetime | None = None,
    weeks_short: int = 13,
    months_ltm: int = 12,
    weeks_year: int = 52,
    w_trend: float = 0.4,
    w_recency: float = 0.3,
    w_magnitude: float = 0.2,
    w_cadence: float = 0.1,
) -> pd.DataFrame:
    """
    Appends time-series and cross-division features (spend dynamics, mix, whitespace)
    to the scored accounts dataframe. This is used for the "List Builder" and "Look-alike"
    experiences in the dashboard.
    """
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc)
    
    # Ensure account_id is string
    df_accounts = df_accounts.copy()
    df_accounts["account_id"] = df_accounts[COL_CUSTOMER_ID].astype(str)
    
    if transactions.empty:
        print("[WARN] No transactions provided for list builder enrichment. Skipping.")
        return df_accounts

    # Prepare transactions
    tx = transactions.copy()
    # Map to standard columns expected by feature modules
    if COL_CUSTOMER_ID in tx.columns:
        tx = tx.rename(columns={COL_CUSTOMER_ID: "account_id"})
    
    # Ensure account_id is string
    if "account_id" in tx.columns:
        tx["account_id"] = tx["account_id"].astype(str)

    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None) # Ensure naive for comparison if needed, or handle tz consistently

    # Initialize features_df with unique account_ids from input accounts
    features_df = pd.DataFrame({"account_id": df_accounts["account_id"].unique()})

    month_windows = tuple(sorted({months_ltm, 24, 36}))

    # --- 1. Global Spend Dynamics ---
    try:
        dyn_global = compute_spend_dynamics(
            tx,
            as_of=as_of_ts,
            weeks_short=weeks_short,
            months_ltm=months_ltm,
            month_windows=month_windows,
        )
        features_df = features_df.merge(dyn_global, on="account_id", how="left")
    except Exception as e:
        print(f"[WARN] Global spend dynamics unavailable: {e}")

    # Normalize division labels for downstream filtering
    div = tx.get("division", pd.Series(index=tx.index, dtype=str)).astype(str).str.strip().str.lower()
    sub = tx.get("sub_division", pd.Series(index=tx.index, dtype=str)).astype(str).str.strip().str.lower()
    super_div = tx.get("super_division", pd.Series(index=tx.index, dtype=str)).astype(str).str.strip().str.lower()
    start_12m = as_of_ts - pd.DateOffset(months=months_ltm)

    # --- 2. Hardware Dynamics (super_division = Hardware) ---
    try:
        tx_hw = tx[super_div == "hardware"].copy()
        if not tx_hw.empty:
            dyn_hw = compute_spend_dynamics(
                tx_hw,
                as_of=as_of_ts,
                weeks_short=weeks_short,
                months_ltm=months_ltm,
                month_windows=month_windows,
            )
            hw_rename = {c: f"{c}_hw" for c in dyn_hw.columns if c != "account_id"}
            dyn_hw = dyn_hw.rename(columns=hw_rename)
            features_df = features_df.merge(
                dyn_hw[[c for c in dyn_hw.columns if c == "account_id" or c in hw_rename.values()]],
                on="account_id",
                how="left",
            )
    except Exception as e:
        print(f"[WARN] Hardware dynamics unavailable: {e}")

    # --- 3. CRE Dynamics (CAD + Specialty Software + CRE Training) ---
    try:
        allowed_cre_goals = {"cad", "specialty software"}
        allowed_cre_train = {"success plan", "training"}
        is_cre = div.isin(allowed_cre_goals) | ((div == "training/services") & (sub.isin(allowed_cre_train)))
        tx_cre = tx[is_cre].copy()

        if not tx_cre.empty:
            dyn_cre = compute_spend_dynamics(
                tx_cre,
                as_of=as_of_ts,
                weeks_short=weeks_short,
                months_ltm=months_ltm,
                month_windows=month_windows,
            )
            cre_rename = {c: f"{c}_cre" for c in dyn_cre.columns if c != "account_id"}
            dyn_cre = dyn_cre.rename(columns=cre_rename)
            features_df = features_df.merge(
                dyn_cre[[c for c in dyn_cre.columns if c == "account_id" or c in cre_rename.values()]],
                on="account_id",
                how="left",
            )

            # CRE breadth and recency
            tx_cre_12m = tx_cre.loc[(pd.to_datetime(tx_cre["date"]) > start_12m) & (pd.to_datetime(tx_cre["date"]) <= as_of_ts)].copy()
            if not tx_cre_12m.empty:
                breadth_cre = (
                    tx_cre_12m.groupby(["account_id", "sub_division"])["net_revenue"].sum() > 0
                ).reset_index()
                breadth_counts = breadth_cre.groupby("account_id")["sub_division"].nunique().rename("breadth_cre_rollup_12m")
                products_cre_rollup = (
                    tx_cre_12m["sub_division"].dropna().astype(str).str.strip().unique().tolist()
                )
                max_cre_rollup = int(len(set(products_cre_rollup))) if products_cre_rollup else 0
                cre_breadth_df = breadth_counts.to_frame().reset_index()
                cre_breadth_df["max_cre_rollup"] = max_cre_rollup if max_cre_rollup > 0 else np.nan
                cre_breadth_df["breadth_score_cre"] = np.where(
                    cre_breadth_df["max_cre_rollup"].fillna(0) > 0,
                    cre_breadth_df["breadth_cre_rollup_12m"].astype(float) / cre_breadth_df["max_cre_rollup"],
                    0.0,
                )
                features_df = features_df.merge(cre_breadth_df, on="account_id", how="left")

            last_cre = (
                tx_cre.groupby("account_id")["date"].max().rename("last_cre_date")
            )
            last_cre = pd.to_datetime(last_cre)
            rec_df = last_cre.to_frame().reset_index()
            rec_df["days_since_last_cre_order"] = (as_of_ts - rec_df["last_cre_date"]).dt.days
            rec_df["recency_score_cre"] = 1.0 / (1.0 + (rec_df["days_since_last_cre_order"] / 30.0))
            rec_df.loc[rec_df["last_cre_date"].isna(), "recency_score_cre"] = 0.0
            features_df = features_df.merge(
                rec_df[["account_id", "days_since_last_cre_order", "recency_score_cre"]],
                on="account_id",
                how="left",
            )

    except Exception as e:
        print(f"[WARN] CRE dynamics/breadth/recency unavailable: {e}")

    # --- 4. CPE Dynamics (Goal = CPE) ---
    try:
        is_cpe = div == "cpe"
        tx_cpe = tx[is_cpe].copy()
        if not tx_cpe.empty:
            dyn_cpe = compute_spend_dynamics(
                tx_cpe,
                as_of=as_of_ts,
                weeks_short=weeks_short,
                months_ltm=months_ltm,
                month_windows=month_windows,
            )
            cpe_rename = {c: f"{c}_cpe" for c in dyn_cpe.columns if c != "account_id"}
            dyn_cpe = dyn_cpe.rename(columns=cpe_rename)
            features_df = features_df.merge(
                dyn_cpe[[c for c in dyn_cpe.columns if c == "account_id" or c in cpe_rename.values()]],
                on="account_id",
                how="left",
            )

            last_cpe = (
                tx_cpe.groupby("account_id")["date"].max().rename("last_cpe_date")
            )
            last_cpe = pd.to_datetime(last_cpe)
            rec_cpe = last_cpe.to_frame().reset_index()
            rec_cpe["days_since_last_cpe_order"] = (as_of_ts - rec_cpe["last_cpe_date"]).dt.days
            rec_cpe["recency_score_cpe"] = 1.0 / (1.0 + (rec_cpe["days_since_last_cpe_order"] / 30.0))
            rec_cpe.loc[rec_cpe["last_cpe_date"].isna(), "recency_score_cpe"] = 0.0
            features_df = features_df.merge(
                rec_cpe[["account_id", "days_since_last_cpe_order", "recency_score_cpe"]],
                on="account_id",
                how="left",
            )
    except Exception as e:
        print(f"[WARN] CPE dynamics/recency unavailable: {e}")

    # --- Additional List Builder Features for POV Tags ---
    try:
        # 1. Hardware Breadth & Recency
        tx_hw_12m = tx_hw.loc[(pd.to_datetime(tx_hw["date"]) > start_12m) & (pd.to_datetime(tx_hw["date"]) <= as_of_ts)].copy() if 'tx_hw' in locals() else tx.iloc[0:0]

        hw_breadth_df = pd.DataFrame({"account_id": features_df["account_id"].unique()})
        if not tx_hw_12m.empty:
            breadth_hw = (
                tx_hw_12m.groupby(["account_id", "sub_division"])["net_revenue"].sum() > 0
            ).reset_index()
            breadth_counts_hw = breadth_hw.groupby("account_id")["sub_division"].nunique().rename("breadth_hw_subdiv_12m")

            # Max possible HW subdivisions (observed in data)
            products_hw_rollup = tx_hw["sub_division"].dropna().astype(str).str.strip().unique().tolist() if not tx_hw.empty else []
            max_hw_rollup = int(len(set(products_hw_rollup))) if products_hw_rollup else 1

            hw_metrics = breadth_counts_hw.to_frame().reset_index()
            hw_metrics["breadth_score_hw"] = hw_metrics["breadth_hw_subdiv_12m"] / max_hw_rollup
            hw_breadth_df = hw_breadth_df.merge(hw_metrics, on="account_id", how="left")

        if "breadth_hw_subdiv_12m" not in hw_breadth_df.columns:
            hw_breadth_df["breadth_hw_subdiv_12m"] = 0
        else:
            hw_breadth_df["breadth_hw_subdiv_12m"] = hw_breadth_df["breadth_hw_subdiv_12m"].fillna(0)

        if "breadth_score_hw" not in hw_breadth_df.columns:
            hw_breadth_df["breadth_score_hw"] = 0.0
        else:
            hw_breadth_df["breadth_score_hw"] = hw_breadth_df["breadth_score_hw"].fillna(0.0)

        features_df = features_df.merge(hw_breadth_df, on="account_id", how="left")

        if 'tx_hw' in locals() and not tx_hw.empty:
            last_hw = tx_hw.groupby("account_id")["date"].max().rename("last_hw_date")
            last_hw = pd.to_datetime(last_hw)
            hw_rec_df = last_hw.to_frame().reset_index()
            hw_rec_df["days_since_last_hw_order"] = (as_of_ts - hw_rec_df["last_hw_date"]).dt.days
            features_df = features_df.merge(hw_rec_df[["account_id", "days_since_last_hw_order"]], on="account_id", how="left")

        if "days_since_last_hw_order" not in features_df.columns:
            features_df["days_since_last_hw_order"] = 9999

        # 2. SW Dominance & Shares
        sw_spend = (
            features_df.get("spend_12m_cre", 0).fillna(0.0)
            + features_df.get("spend_12m_cpe", 0).fillna(0.0)
        )
        hw_spend = features_df.get("spend_12m_hw", 0).fillna(0.0)
        total_spend = sw_spend + hw_spend

        features_df["sw_dominance_score"] = np.where(total_spend > 0, sw_spend / total_spend, 0.0)
        features_df["hw_share_12m"] = np.where(total_spend > 0, hw_spend / total_spend, 0.0)
        features_df["sw_share_12m"] = np.where(total_spend > 0, sw_spend / total_spend, 0.0)

        # 3. Concentration (HHI) & Top Subdivision Share
        # Calculate on global tx 12m
        hhi = pd.DataFrame(columns=["account_id", "month_conc_hhi_12m"])
        top_share = pd.DataFrame(columns=["account_id", "top_subdivision_share_12m"])
        tx_12m = tx.loc[(pd.to_datetime(tx["date"]) > start_12m) & (pd.to_datetime(tx["date"]) <= as_of_ts)].copy()
        if not tx_12m.empty:
            # Monthly HHI
            tx_12m["month"] = pd.to_datetime(tx_12m["date"]).dt.to_period("M")
            monthly_spend = tx_12m.groupby(["account_id", "month"])["net_revenue"].sum().reset_index()
            # Normalize by total 12m spend per account (avoid div-by-zero)
            account_total = monthly_spend.groupby("account_id")["net_revenue"].transform("sum").replace(0, np.nan)
            monthly_spend["share"] = monthly_spend["net_revenue"] / account_total
            monthly_spend["share_sq"] = monthly_spend["share"].fillna(0) ** 2
            hhi = monthly_spend.groupby("account_id")["share_sq"].sum().rename("month_conc_hhi_12m")

            # Top Subdivision Share
            sub_spend = tx_12m.groupby(["account_id", "sub_division"])["net_revenue"].sum().reset_index()
            sub_account_total = sub_spend.groupby("account_id")["net_revenue"].transform("sum").replace(0, np.nan)
            sub_spend["share"] = sub_spend["net_revenue"] / sub_account_total
            top_share = sub_spend.groupby("account_id")["share"].max().rename("top_subdivision_share_12m")

        features_df = features_df.merge(hhi, on="account_id", how="left")
        features_df = features_df.merge(top_share, on="account_id", how="left")

        if "month_conc_hhi_12m" not in features_df.columns:
            features_df["month_conc_hhi_12m"] = 0.0
        else:
            features_df["month_conc_hhi_12m"] = features_df["month_conc_hhi_12m"].fillna(0.0)

        if "top_subdivision_share_12m" not in features_df.columns:
            features_df["top_subdivision_share_12m"] = 0.0
        else:
            features_df["top_subdivision_share_12m"] = features_df["top_subdivision_share_12m"].fillna(0.0)

        # 4. Discount (Stub - no list price available)
        features_df["discount_pct"] = 0.0

    except Exception as e:
        print(f"[WARN] Error calculating additional list builder features: {e}")
        # Ensure columns exist to prevent crash
        for col in ["sw_dominance_score", "hw_share_12m", "sw_share_12m", "breadth_score_hw",
                    "breadth_hw_subdiv_12m", "days_since_last_hw_order", "month_conc_hhi_12m",
                    "top_subdivision_share_12m", "discount_pct"]:
            if col not in features_df.columns:
                features_df[col] = 0.0 if col != "days_since_last_hw_order" else 9999


    # Momentum Score
    # Ensure momentum component columns exist and are filled
    for col in ["trend_score", "recency_score", "magnitude_score", "cadence_score"]:
        if col not in features_df.columns:
            features_df[col] = 0.0
        features_df[col] = features_df[col].fillna(0.0)

    features_df["momentum_score"] = (
        w_trend * features_df["trend_score"]
        + w_recency * features_df["recency_score"]
        + w_magnitude * features_df["magnitude_score"]
        + w_cadence * features_df["cadence_score"]
    )
    features_df["w_trend"] = w_trend
    features_df["w_recency"] = w_recency
    features_df["w_magnitude"] = w_magnitude
    features_df["w_cadence"] = w_cadence

    # Percentile ranks
    def pct_rank(s: pd.Series) -> pd.Series:
        try:
            return (s.rank(pct=True, ascending=True) * 100.0).round(2)
        except Exception:
            return pd.Series(np.nan, index=s.index)

    rank_cols = [
        "spend_12m","spend_24m","spend_36m","spend_13w","delta_13w_pct","yoy_13w_pct","slope_13w","acceleration_13w",
        "spend_12m_hw","spend_13w_hw","delta_13w_pct_hw","yoy_13w_pct_hw","slope_13w_hw","acceleration_13w_hw",
        "spend_12m_cre","spend_13w_cre","delta_13w_pct_cre","yoy_13w_pct_cre","slope_13w_cre","acceleration_13w_cre",
        "spend_12m_cpe","spend_13w_cpe","delta_13w_pct_cpe","yoy_13w_pct_cpe","slope_13w_cpe","acceleration_13w_cpe",
        "breadth_score_cre","recency_score_cre","recency_score_cpe",
    ]
    for rc in rank_cols:
        if rc in features_df.columns:
            features_df[f"{rc}_pctl"] = pct_rank(pd.to_numeric(features_df[rc], errors="coerce"))

    # POV Tags
    # Ensure required columns for POV tags and other downstream logic are available
    # This covers columns that should have come from compute_spend_dynamics
    missing_cols = [
        "days_since_last_order", "days_since_last_hw_order", "days_since_last_cre_order", "days_since_last_cpe_order",
        "median_interpurchase_days", "active_weeks_13w", "purchase_streak_months",
        "spend_13w", "spend_13w_prior", "delta_13w", "delta_13w_pct",
        "spend_12m", "spend_12m_prior", "delta_12m", "delta_12m_pct",
        "spend_24m", "spend_24m_prior", "delta_24m", "delta_24m_pct",
        "spend_36m", "spend_36m_prior", "delta_36m", "delta_36m_pct",
        "spend_12m_hw", "spend_12m_cre", "spend_12m_cpe",
        "yoy_13w_pct",
        "slope_13w", "slope_13w_prior", "acceleration_13w", "volatility_13w", "seasonality_factor_13w",
        "trend_score", "recency_score", "magnitude_score", "cadence_score"
    ]
    for col in missing_cols:
        if col not in features_df.columns:
            features_df[col] = 9999 if "days" in col else 0.0
    
    # Ensure sw_dominance_score is available for POV tags (now guaranteed by block above)
    tags = make_pov_tags(features_df)
    features_df = features_df.merge(tags, on="account_id", how="left")

    features_df["as_of_date"] = as_of_date.date().isoformat()
    features_df["run_ts_utc"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    features_df["account_id"] = features_df["account_id"].astype(str)

    # Merge back to accounts
    overlapping = set(df_accounts.columns) & (set(features_df.columns) - {"account_id"})
    if overlapping:
        features_df = features_df.drop(columns=list(overlapping))
    
    scored = df_accounts.merge(features_df, left_on=COL_CUSTOMER_ID, right_on="account_id", how="left")
    
    return scored
