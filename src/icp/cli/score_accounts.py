"""Main CLI entry point for ICP scoring."""
import os
import argparse
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

import icp.data_access as da
from icp.divisions import get_division_config, available_divisions
from icp.optimization import load_optimized_weights
from icp.schema import (
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
    get_icp_scored_accounts_base_order,
)
from icp.etl.loader import (
    check_env,
    load_asset_weights,
    assemble_master_from_db,
)
from icp.features.engineering import (
    engineer_features,
    enrich_with_list_builder_features,
    FEATURE_COLUMN_ORDER,
    PRINTER_SUBDIVISIONS,
    _printer_rollup_slug,
)
from icp.validation import validate_master
from icp.reporting.visuals import build_visuals
from features.similarity_build import build_neighbors
from icp.quality import validate_outputs
from icp.scoring import calculate_scores

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
ROOT = Path(__file__).resolve().parents[3]
CLI_OPTS: dict[str, Any] = {}
WEIGHTS = {}

def main():
    """Main function to execute the scoring pipeline using Azure SQL inputs."""
    check_env()

    # Default to hardware for the initial pass (feature engineering is division-agnostic for raw features)
    division_key = "hardware" 
    division_config = get_division_config(division_key)
    
    # Load weights for Hardware
    global WEIGHTS
    WEIGHTS = load_optimized_weights(division_config)
    asset_weights = load_asset_weights()

    print(f"--- Starting ICP Scoring Pipeline ---")
    print(f"Calculating scores for Hardware and CRE divisions.")

    # 1. Data Assembly
    print("Assembling master data from Azure SQL...")
    master = assemble_master_from_db()
    print(f"Master table assembled: {len(master)} rows.")
    
    # Validate master data
    print("Validating master data...")
    master = validate_master(master)

    # 2. Feature Engineering (Calculates raw features + Hardware scores)
    print("Engineering features...")
    scored = engineer_features(master, asset_weights, WEIGHTS, division=division_config)
    
    # Rename Hardware scores to avoid overwrite
    hw_rename = {
        "ICP_score": "Hardware_ICP_Score",
        "ICP_grade": "Hardware_ICP_Grade",
        "adoption_score": "Hardware_Adoption_Score",
        "relationship_score": "Hardware_Relationship_Score",
        "vertical_score": "Hardware_Vertical_Score",
        "ICP_score_raw": "Hardware_ICP_Score_Raw",
    }
    scored = scored.rename(columns=hw_rename)
    
    # Calculate CRE Scores
    print("Calculating CRE scores...")
    cre_config = get_division_config("cre")
    cre_weights = load_optimized_weights(cre_config)
    # Note: calculate_scores expects the raw feature columns (cre_adoption_assets, etc.) which engineer_features created
    scored = calculate_scores(scored, cre_weights, division=cre_config)
    
    # Rename CRE scores
    cre_rename = {
        "ICP_score": "CRE_ICP_Score",
        "ICP_grade": "CRE_ICP_Grade",
        "adoption_score": "CRE_Adoption_Score",
        "relationship_score": "CRE_Relationship_Score",
        "vertical_score": "CRE_Vertical_Score",
        "ICP_score_raw": "CRE_ICP_Score_Raw",
    }
    scored = scored.rename(columns=cre_rename)

    # Calculate CPE Scores (Future-proofing)
    print("Calculating CPE scores...")
    try:
        cpe_config = get_division_config("cpe")
        # Use default weights if specific file missing, or handle gracefully
        try:
            cpe_weights = load_optimized_weights(cpe_config)
        except Exception:
            print("  - CPE weights not found, using default component weights.")
            cpe_weights = cpe_config.weight_dict()
            
        scored = calculate_scores(scored, cpe_weights, division=cpe_config)
        
        cpe_rename = {
            "ICP_score": "CPE_ICP_Score",
            "ICP_grade": "CPE_ICP_Grade",
            "adoption_score": "CPE_Adoption_Score",
            "relationship_score": "CPE_Relationship_Score",
            "vertical_score": "CPE_Vertical_Score",
            "ICP_score_raw": "CPE_ICP_Score_Raw",
        }
        scored = scored.rename(columns=cpe_rename)
    except Exception as e:
        print(f"[WARN] Could not calculate CPE scores: {e}")

    # 3. List Builder Enrichment
    print("Enriching with List Builder features...")
    engine = da.get_engine()
    transactions = da.get_sales_detail_since_2022(engine)
    scored = enrich_with_list_builder_features(scored, transactions)

    # 4. Final Column Ordering & Output
    print("Finalizing output schema...")
    
    # Define the columns for the final output CSV
    printer_rollup_cols = []
    for roll in PRINTER_SUBDIVISIONS:
        slug = _printer_rollup_slug(roll)
        printer_rollup_cols.extend([
            f"Qty_Printers_{slug}",
            f"GP_Printers_{slug}",
        ])

    # Ensure Training/Services subset columns exist for schema stability
    for r in ("Success_Plan", "Training"):
        col = f"GP_Training/Services_{r}"
        if col not in scored.columns:
            scored[col] = 0.0

    # Build dynamic CRE rollup columns for ordering (CAD / Specialty rollups)
    cre_rollup_cols = []
    try:
        # Seats and GP for CAD and Specialty Software rollups
        for prefix in ("Seats_CAD_", "Seats_Specialty Software_", "GP_CAD_", "GP_Specialty Software_"):
            cre_rollup_cols.extend(sorted([c for c in scored.columns if c.startswith(prefix)]))
    except Exception:
        cre_rollup_cols = []

    # Start from the canonical, grouped base schema and insert dynamic rollups.
    desired_order = list(get_icp_scored_accounts_base_order())

    # Insert printer subdivision rollups immediately after GP_Printers.
    try:
        idx_gp_printers = desired_order.index("GP_Printers") + 1
        desired_order[idx_gp_printers:idx_gp_printers] = printer_rollup_cols
    except ValueError:
        desired_order.extend(printer_rollup_cols)

    # Insert CRE rollups immediately after GP_Specialty Software.
    try:
        idx_gp_specialty = desired_order.index("GP_Specialty Software") + 1
        desired_order[idx_gp_specialty:idx_gp_specialty] = cre_rollup_cols
    except ValueError:
        desired_order.extend(cre_rollup_cols)
    
    # Backup existing CSV then save the new scored data
    out_env = os.environ.get("ICP_OUT_PATH")
    out_path = Path(out_env) if out_env else (ROOT / "data" / "processed" / "icp_scored_accounts.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        backup_dir = ROOT / "archive" / "outputs"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"icp_scored_accounts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2(out_path, backup_path)
        print(f"Backed up previous CSV to {backup_path}")

    # Remove legacy aliasing that caused confusion
    # if 'adoption_score' in scored.columns:
    #     scored['Hardware_score'] = scored['adoption_score']
    # if 'relationship_score' in scored.columns:
    #     scored['Software_score'] = scored['relationship_score']

    # Surface GP_* aliases and ensure fields are not NaN
    alias_pairs = {
        'GP_Since_2023_Total': 'Profit_Since_2023_Total',
        'GP_T4Q_Total': 'Profit_T4Q_Total',
        'GP_LastQ_Total': 'Profit_LastQ_Total',
        'GP_PrevQ_Total': 'Profit_PrevQ_Total',
        'GP_QoQ_Growth': 'Profit_QoQ_Growth',
    }
    for gp, prof in alias_pairs.items():
        if prof in scored.columns:
            scored[gp] = pd.to_numeric(scored[prof], errors='coerce').fillna(0.0)

    # Build final column order by intersecting with available columns
    feature_cols = [
        c for c in FEATURE_COLUMN_ORDER if c in scored.columns and c not in desired_order
    ]
    out_cols = [c for c in desired_order if c in scored.columns]
    out_cols.extend([c for c in feature_cols if c not in out_cols])

    # Include dynamically generated enrichment columns (printer-only and percentiles)
    dynamic_suffixes = (
        "_printers",
        "_cre",
        "_pctl",
    )
    dynamic_includes = {
        "sw_dominance_score",
        "sw_to_hw_whitespace_score",
        "pov_primary",
        "pov_tags_all",
        # Add new score columns to output
        "Hardware_ICP_Score", "Hardware_ICP_Grade", "Hardware_Adoption_Score", "Hardware_Relationship_Score", "Hardware_Vertical_Score",
        "CRE_ICP_Score", "CRE_ICP_Grade", "CRE_Adoption_Score", "CRE_Relationship_Score", "CRE_Vertical_Score",
        "CPE_ICP_Score", "CPE_ICP_Grade", "CPE_Adoption_Score", "CPE_Relationship_Score", "CPE_Vertical_Score",
    }
    dyn_cols = [
        c for c in scored.columns
        if (
            any(c.endswith(suf) for suf in dynamic_suffixes)
            or c in dynamic_includes
        )
    ]
    for c in dyn_cols:
        if c not in out_cols:
            out_cols.append(c)

    missing_features = [c for c in FEATURE_COLUMN_ORDER if c not in scored.columns]
    if missing_features:
        print(
            "[WARN] The following enrichment columns were not present in the final dataset: "
            + ", ".join(missing_features)
        )

    # Coerce newly added feature columns to numeric where applicable and round for BI friendliness
    numeric_like = set([
        "spend_13w","spend_13w_prior","delta_13w","delta_13w_pct","spend_12m","spend_52w","yoy_13w_pct",
        "days_since_last_order","active_weeks_13w","purchase_streak_months","median_interpurchase_days",
        "slope_13w","slope_13w_prior","acceleration_13w","volatility_13w","seasonality_factor_13w",
        "trend_score","recency_score","magnitude_score","cadence_score","momentum_score",
        "w_trend","w_recency","w_magnitude","w_cadence",
        "spend_13w_cre","spend_13w_prior_cre","delta_13w_cre","delta_13w_pct_cre","spend_12m_cre","spend_52w_cre","yoy_13w_pct_cre",
        "days_since_last_order_cre","active_weeks_13w_cre",
        "slope_13w_cre","slope_13w_prior_cre","acceleration_13w_cre","volatility_13w_cre","seasonality_factor_13w_cre",
        "breadth_cre_rollup_12m","max_cre_rollup","breadth_score_cre","days_since_last_cre_order","recency_score_cre",
        "hw_spend_12m","sw_spend_12m","hw_share_12m","sw_share_12m","breadth_hw_subdiv_12m","max_hw_subdiv",
        "breadth_score_hw","days_since_last_hw_order","recency_score_hw","hardware_adoption_score",
        "consumables_to_hw_ratio","top_subdivision_share_12m",
        "hw_spend_13w","hw_spend_13w_prior","hw_delta_13w","hw_delta_13w_pct",
        "sw_spend_13w","sw_spend_13w_prior","sw_delta_13w","sw_delta_13w_pct",
        "super_division_breadth_12m","division_breadth_12m","software_division_breadth_12m",
        "cross_division_balance_score","hw_to_sw_cross_sell_score","sw_to_hw_cross_sell_score","training_to_hw_ratio","training_to_cre_ratio",
        "discount_pct","month_conc_hhi_12m","sw_dominance_score","sw_to_hw_whitespace_score",
        "vertical_score","ICP_score_raw",
        "GP_PrevQ_Total","GP_QoQ_Growth","GP_T4Q_Total","GP_Since_2023_Total",
        "Seats_CAD","GP_CAD","Seats_CPE","GP_CPE","Seats_Specialty Software","GP_Specialty Software",
        "active_assets_total","seats_sum_total","Portfolio_Breadth","scaling_flag",
        "Days_Since_First_Purchase","Days_Since_Last_Purchase","Days_Since_Last_Expiration",
    ])
    for c in list(scored.columns):
        if c.endswith("_pctl") or c.endswith("_printers") or c.endswith("_cre"):
            numeric_like.add(c)
    for col in numeric_like:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored.replace([np.inf, -np.inf], np.nan, inplace=True)
    round_cols = [c for c in numeric_like if c in scored.columns and scored[c].dtype.kind in "fc"]
    if round_cols:
        scored[round_cols] = scored[round_cols].round(4)

    # Normalize EDU assets
    if "edu_assets" in scored.columns:
        def _yes_no(v):
            try:
                if pd.isna(v):
                    return "No"
            except Exception:
                pass
            s = str(v).strip().lower()
            if s in ("1","true","yes","y","t"):
                return "Yes"
            try:
                if float(s) != 0.0:
                    return "Yes"
            except Exception:
                pass
            return "No"
        scored["edu_assets"] = scored["edu_assets"].apply(_yes_no)

    # Ensure no null Customer IDs before writing to CSV
    scored = scored.dropna(subset=[COL_CUSTOMER_ID])
    scored = scored[scored[COL_CUSTOMER_ID].astype(str).str.strip() != ""]
    scored = scored[scored[COL_CUSTOMER_ID].astype(str).str.strip().str.lower() != "nan"]

    scored[out_cols].to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved {out_path}")

    # 5. Database Persistence (Optional)
    try:
        icp_db = os.getenv("ICP_AZSQL_DB", "").strip()
        if icp_db:
            print(f"[INFO] Writing scored accounts to {icp_db}.dbo.customer_icp")
            engine = da.get_engine(database=icp_db)
            
            # Ensure no null IDs before writing
            scored_db = scored[out_cols].copy()
            scored_db = scored_db.dropna(subset=[COL_CUSTOMER_ID])
            
            # Rename timestamp column if present to avoid SQL reserved word conflicts
            if "run_timestamp_utc" in scored_db.columns:
                scored_db = scored_db.rename(columns={"run_timestamp_utc": "run_ts_utc"})
            
            # Explicitly drop the table first to avoid schema conflicts (e.g. timestamp columns)
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS dbo.customer_icp"))
                conn.commit()

            scored_db.to_sql(
                "customer_icp",
                engine,
                schema="dbo",
                if_exists="replace",
                index=False,
                chunksize=5000,
            )
            print("[INFO] Wrote scored accounts to dbo.customer_icp")
        else:
            print("[INFO] Skipping DB write: ICP_AZSQL_DB is not set")
    except Exception as e:
        print(f"[WARN] Failed to write scored accounts to database: {e}")

    # 6. Neighbors Artifact
    if not CLI_OPTS.get("skip_neighbors", False):
        try:
            # Load configuration
            from icp.config import load_config
            app_config = load_config()
            
            # Convert dataclass to dict for compatibility with existing functions
            # (Assuming build_neighbors expects a dict)
            sim_cfg = {
                "k_neighbors": app_config.similarity.k_neighbors,
                "use_text": app_config.similarity.use_text,
                "use_als": app_config.similarity.use_als,
                "max_dense_accounts": app_config.similarity.max_dense_accounts,
                "row_block_size": app_config.similarity.row_block_size,
            }

            if CLI_OPTS.get("no_als", False):
                sim_cfg["use_als"] = False

            acc = scored.copy()
            acc["account_id"] = acc[COL_CUSTOMER_ID].astype(str)
            acc["account_name"] = acc.get(COL_COMPANY_NAME, acc["account_id"]).astype(str)
            if COL_INDUSTRY in acc.columns:
                acc["industry"] = acc[COL_INDUSTRY].astype(str)
            if "activity_segment" in acc.columns:
                acc["segment"] = acc["activity_segment"].astype(str)
            if "AM_Territory" in acc.columns:
                acc["territory"] = acc["AM_Territory"].astype(str)
            if "Industry_Reasoning" in acc.columns:
                acc["industry_reasoning"] = acc["Industry_Reasoning"].astype(str)

            als_df = None
            try:
                # Re-fetch raw frames for ALS if needed, or pass from master if attached
                # master has _profit_rollup_raw attached
                prof = getattr(master, "_profit_rollup_raw", pd.DataFrame())
                assets_src = getattr(master, "_assets_raw", pd.DataFrame())
                
                # Build ALS components
                from features.als_prep import build_multi_als_inputs
                from features.als_embed import als_concat_account_vectors
                
                # Use ALS config from app_config
                als_cfg_dict = {
                    "factors_rollup": app_config.als.factors_rollup,
                    "factors_goal": app_config.als.factors_goal,
                    "w_rollup_vec": app_config.als.w_rollup_vec,
                    "w_goal_vec": app_config.als.w_goal_vec,
                    "alpha": app_config.als.alpha,
                    "reg": app_config.als.reg,
                    "iterations": app_config.als.iterations,
                    "use_bm25": app_config.als.use_bm25,
                }
                
                als_components = build_multi_als_inputs(prof, assets_src, als_cfg_dict)
                
                comp_factors = {
                    'rollup': app_config.als.factors_rollup,
                    'goal': app_config.als.factors_goal,
                }
                comp_weights = {
                    'rollup': app_config.als.w_rollup_vec,
                    'goal': app_config.als.w_goal_vec,
                }
                account_list = acc['account_id'].astype(str).tolist()
                
                als_df = als_concat_account_vectors(
                    als_components,
                    accounts=account_list,
                    factors=comp_factors,
                    alpha=app_config.als.alpha,
                    reg=app_config.als.reg,
                    iterations=app_config.als.iterations,
                    use_bm25=app_config.als.use_bm25,
                    component_weights=comp_weights,
                )
            except Exception as e:
                print(f"[WARN] ALS vector build skipped: {e}")

            neighbors = build_neighbors(acc, pd.DataFrame(), sim_cfg, als_df=als_df)
            neighbors_dir = ROOT / "artifacts"
            neighbors_dir.mkdir(parents=True, exist_ok=True)
            neighbors_path = neighbors_dir / "account_neighbors.csv"
            neighbors.to_csv(neighbors_path, index=False)
            print(f"Saved neighbors artifact: {neighbors_path}")
        except Exception as e:
            print(f"[WARN] Could not generate neighbors artifact: {e}")
    else:
        print("[INFO] Skipping neighbors artifact (requested by CLI)")

    # 7. Visualizations
    if CLI_OPTS.get("skip_visuals"):
        print("[INFO] Skipping visualization rendering (--skip-visuals)")
    else:
        print("Creating visualisations...")
        build_visuals(scored, ROOT)
        print("Saved 10 PNG charts.")
    
    # 8. Final Validation
    print("Validating outputs...")
    validate_outputs(
        scored_path=str(out_path),
        neighbors_path=str(neighbors_path) if not CLI_OPTS.get("skip_neighbors", False) else None,
        raise_error=CLI_OPTS.get("strict_validation", False)
    )

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICP scoring pipeline")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--division", type=str, default="hardware", choices=available_divisions(), help="Division key")
    parser.add_argument("--weights", type=str, default=None, help="Path to optimized_weights.json")
    parser.add_argument("--industry-weights", type=str, default=None, help="Path to industry_weights.json")
    parser.add_argument("--asset-weights", type=str, default=None, help="Path to asset_rollup_weights.json")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip generating visuals")
    parser.add_argument("--skip-neighbors", action="store_true", help="Skip building account neighbors artifact")
    parser.add_argument("--no-als", action="store_true", help="Disable ALS vectors in neighbors")
    parser.add_argument("--strict", action="store_true", help="Fail on output validation errors")
    # Note: --neighbors-only logic omitted for brevity in this refactor, can be added back if needed
    
    args = parser.parse_args()

    if args.weights:
        os.environ["ICP_OPT_WEIGHTS_PATH"] = args.weights
    if args.industry_weights:
        os.environ["ICP_INDUSTRY_WEIGHTS_PATH"] = args.industry_weights
    if args.asset_weights:
        os.environ["ICP_ASSET_WEIGHTS_PATH"] = args.asset_weights
    if args.out:
        os.environ["ICP_OUT_PATH"] = args.out
        CLI_OPTS["out_path"] = args.out
    
    CLI_OPTS["skip_neighbors"] = args.skip_neighbors
    CLI_OPTS["no_als"] = args.no_als
    CLI_OPTS["skip_visuals"] = args.skip_visuals
    CLI_OPTS["strict_validation"] = args.strict
    CLI_OPTS["division"] = args.division

    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
        raise
