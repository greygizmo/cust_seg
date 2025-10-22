"""
goe_icp_scoring.py
---------------------------------
End-to-end script to score GoEngineer Digital-Manufacturing accounts
and generate key visuals.

This script performs the following steps:
1.  Loads customer data, sales data, and enriched revenue data.
2.  Cleans and normalizes company names for reliable merging.
3.  Aggregates sales data to calculate 24-month Gross Profit (GP24).
4.  Merges the datasets into a single master DataFrame.
5.  Engineers features and calculates ICP scores using the centralized `scoring_logic.py`.
6.  Saves the final scored data to `icp_scored_accounts.csv`.
7.  Generates and saves 10 key visualizations as PNG files.

Files expected in the SAME directory:
  1) JY - Customer Analysis - Customer Segmentation.xlsx (contains industry, revenue, and customer data)
  2) TR - Master Sales Log - Customer Segementation.xlsx (contains GP and sales data)
  3) enrichment_progress.csv (contains enriched annual revenue data)

Outputs:
    icp_scored_accounts.csv
    vis1_  vis10_  PNG charts

Requires: pandas, numpy, matplotlib, python-dateutil, scikit-learn, scipy
---------------------------------
Usage:
  $ python -m icp.cli.score_accounts
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm

# Make sure package imports work when running as a module
ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import the centralized scoring logic
from icp.scoring import calculate_scores, LICENSE_COL, DEFAULT_WEIGHTS
from icp.validation import ensure_columns, ensure_non_negative, log_validation
from icp.schema import (
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
    COL_REL_LICENSE,
    COL_REL_SAAS,
    COL_REL_MAINT,
    COL_HW_REV,
    COL_CONS_REV,
)
# Import the new industry scoring module
from icp.industry import build_industry_weights, save_industry_weights, load_industry_weights
import icp.data_access as da

# ---------------------------
# 0.  CONFIG   file names & weights
# ---------------------------

INDUSTRY_ENRICHMENT_FILE = ROOT / "data" / "raw" / "TR - Industry Enrichment.csv"  # Updated industry data
ASSET_WEIGHTS_FILE = ROOT / "artifacts" / "weights" / "asset_rollup_weights.json"


def load_weights():
    """
    Load optimized weights from the JSON file.
    If the file is not found or is invalid, it falls back to the default weights
    defined in `scoring_logic.py`. It also converts the weights from the
    optimizer's format to the format expected by the scoring script.
    """
    try:
        weights_path = ROOT / 'artifacts' / 'weights' / 'optimized_weights.json'
        if not weights_path.exists():
            # Fallback to legacy location if not found
            weights_path = ROOT / 'optimized_weights.json'
        with open(weights_path, 'r') as f:
            data = json.load(f)
            raw_weights = data.get('weights', {})
            
            # Convert from optimizer format to script format and calculate missing weights
            weights = {}
            weights['vertical'] = raw_weights.get('vertical_score', DEFAULT_WEIGHTS['vertical'])
            weights['size'] = raw_weights.get('size_score', DEFAULT_WEIGHTS['size'])
            weights['adoption'] = raw_weights.get('adoption_score', DEFAULT_WEIGHTS['adoption'])
            
            # Calculate relationship score as remainder to ensure sum = 1.0
            total_so_far = weights['vertical'] + weights['size'] + weights['adoption']
            weights['relationship'] = max(0.0, 1.0 - total_so_far)
            
            print(f"[INFO] Loaded optimized weights from {weights_path}")
            print("  - Optimization details: " + str(data.get("n_trials","Unknown")) + " trials")
            print(f"  - Converted weights: vertical={weights['vertical']:.3f}, size={weights['size']:.3f}, adoption={weights['adoption']:.3f}, relationship={weights['relationship']:.3f}")
            return weights
    except FileNotFoundError:
        print(f"[INFO] No optimized weights found. Using default weights.")
        return DEFAULT_WEIGHTS
    except json.JSONDecodeError:
        print(f"[WARN] Error reading optimized_weights.json. Using default weights.")
        return DEFAULT_WEIGHTS


# Load weights dynamically at the start of the script
WEIGHTS = load_weights()

# ---------------------------
# 1.  Utility helpers
# ---------------------------

def clean_name(x: str) -> str:
    """
    Aggressively normalizes a company name for matching purposes.
    - Converts to lowercase.
    - Removes leading customer ID numbers.
    - Removes common punctuation and extra spaces.
    """
    import re
    if pd.isna(x):
        return ""
    x = str(x).lower()
    
    # Remove leading customer ID numbers (e.g., "123456 Company Name" -> "Company Name") 
    x = re.sub(r'^\d+\s+', '', x)
    
    junk = {",", ".", "&", "  "}
    for j in junk:
        x = x.replace(j, " ")
    return " ".join(x.split())


def check_env():
    """Checks for required Azure SQL env vars and enrichment CSV presence."""
    required = ["AZSQL_SERVER", "AZSQL_DB"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[WARN] Missing environment variables: {missing}. Ensure .env is configured.")
    if not Path(INDUSTRY_ENRICHMENT_FILE).exists():
        print(f"[INFO] Industry enrichment file '{INDUSTRY_ENRICHMENT_FILE}' not found. Proceeding without it.")


def load_revenue() -> pd.DataFrame:
    """
    Loads and cleans the enriched revenue data from `enrichment_progress.csv`.
    It applies a prioritization logic to select the most reliable revenue figure
    for each company, preferring SEC filings over other sources.
    """
    if not os.path.exists(REVENUE_FILE):
        return pd.DataFrame()  # Return empty dataframe if file doesn't exist
    
    try:
        df = pd.read_csv(REVENUE_FILE)
        
        # Handle potential variations in the company name column
        company_col = None
        for col in ["company_name", "Company Name", "Compnay Name", "company name"]:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None:
            raise ValueError(f"Could not find company name column. Available columns: {df.columns.tolist()}")
        
        df["key"] = df[company_col].map(clean_name)
        print(f"[INFO] Using company column: '{company_col}'")
        
        # Implement prioritization logic for selecting the most reliable revenue data
        print("[INFO] Applying revenue data prioritization logic...")
        
        df["reliable_revenue"] = None
        df["revenue_source"] = "none"
        
        # Priority 1: sec_match (highest priority - SEC filings)
        sec_mask = df["source"] == "sec_match"
        df.loc[sec_mask, "reliable_revenue"] = pd.to_numeric(df.loc[sec_mask, "revenue_estimate"], errors="coerce")
        df.loc[sec_mask, "revenue_source"] = "sec_match"
        sec_count = sec_mask.sum()
        
        # Priority 2: pdl_estimate (second most reliable)
        pdl_mask = (df["source"] == "pdl_estimate") & (df["reliable_revenue"].isna())
        df.loc[pdl_mask, "reliable_revenue"] = pd.to_numeric(df.loc[pdl_mask, "revenue_estimate"], errors="coerce")
        df.loc[pdl_mask, "revenue_source"] = "pdl_estimate"
        pdl_count = pdl_mask.sum()
        
        # Priority 3: fmp_match, but filter out likely currency errors (e.g., > $1T)
        fmp_mask = (df["source"] == "fmp_match") & (df["reliable_revenue"].isna())
        fmp_revenue = pd.to_numeric(df.loc[fmp_mask, "revenue_estimate"], errors="coerce")
        valid_fmp_mask = fmp_mask & (fmp_revenue < 1000000000000)  # < $1 trillion
        df.loc[valid_fmp_mask, "reliable_revenue"] = fmp_revenue.loc[valid_fmp_mask]
        df.loc[valid_fmp_mask, "revenue_source"] = "fmp_match"
        fmp_count = valid_fmp_mask.sum()
        
        # Priority 4: Discard heuristic_estimate (unreliable)
        heuristic_count = (df["source"] == "heuristic_estimate").sum()
        
        # Remove rows where a reliable revenue figure could not be determined
        df = df.dropna(subset=["reliable_revenue"])
        
        print(f"[INFO] Revenue data prioritization results:")
        print(f"  - sec_match: {sec_count} customers")
        print(f"  - pdl_estimate: {pdl_count} customers")
        print(f"  - fmp_match (valid): {fmp_count} customers") 
        print(f"  - heuristic_estimate (discarded): {heuristic_count} customers")
        print(f"  - Total reliable revenue records: {len(df)}")
        
        return df[["key", "reliable_revenue", "revenue_source"]]
    except Exception as e:
        print("\nAn error occurred - see traceback above.\n")
        raise


def load_industry_enrichment() -> pd.DataFrame:
    """
    Loads updated industry data from the industry enrichment CSV file.
    This provides more accurate and up-to-date industry classifications.
    """
    enrichment_path = Path(INDUSTRY_ENRICHMENT_FILE)
    if not enrichment_path.exists():
        # Fallback to legacy root location if file hasn't been moved yet
        legacy = ROOT / "TR - Industry Enrichment.csv"
        if legacy.exists():
            enrichment_path = legacy
        else:
            print(f"[INFO] Industry enrichment file not found in '{INDUSTRY_ENRICHMENT_FILE}' or legacy root. Using original industry data.")
            return pd.DataFrame()
    
    try:
        df = pd.read_csv(enrichment_path)
        
        # Handle different Customer ID column names
        customer_id_col = None
        for col in ["Customer ID", "ID", "customer_id", "id"]:
            if col in df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            print(f"[WARN] No Customer ID column found in industry enrichment. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Rename to standard column name
        if customer_id_col != "Customer ID":
            df = df.rename(columns={customer_id_col: "Customer ID"})
        
        # Validate required columns
        required_cols = ["Customer ID", "Industry", "Industry Sub List"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing required columns in industry enrichment: {missing_cols}")
            return pd.DataFrame()

        print(f"[INFO] Loaded industry enrichment data for {len(df)} customers")

        # Include Reasoning and CRM Full Name (Customer) if they exist
        cols_to_return = ["Customer ID", "Industry", "Industry Sub List"]
        if "Reasoning" in df.columns:
            cols_to_return.append("Reasoning")
            print(f"[INFO] Including 'Reasoning' column from industry enrichment")
        if "Customer" in df.columns:
            # Preserve the CRM string for name-based matching (e.g., "439775 Compusult Limited")
            df = df.rename(columns={"Customer": "CRM Full Name"})
            cols_to_return.append("CRM Full Name")

        return df[cols_to_return]
    except Exception as e:
        print(f"[WARN] Error loading industry enrichment file: {e}")
        return pd.DataFrame()


def apply_industry_enrichment(df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies industry updates using Customer ID matching.
    Uses new industry data where available, falls back to original data otherwise.
    """
    if enrichment_df.empty:
        return df
    
    print(f"[INFO] Applying industry enrichment to {len(df)} customers...")
    
    # Ensure Customer ID columns have matching canonical string form (strip trailing '.0' only)
    def _canon_id_safe(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        # Only remove a single trailing '.0' introduced by Excel-like coercions; DO NOT cast to int
        return s.str.replace(r"\.0$", "", regex=True)

    df["Customer ID"] = _canon_id_safe(df["Customer ID"])
    enrichment_df["Customer ID"] = _canon_id_safe(enrichment_df["Customer ID"])
    
    # Merge on Customer ID to update industry data
    updated = df.merge(
        enrichment_df,
        on="Customer ID",
        how="left",
        suffixes=("_original", "_enriched")
    )
    
    # Use enriched data where available, fall back to original
    if "Industry_enriched" in updated.columns:
        updated["Industry"] = updated["Industry_enriched"].fillna(updated["Industry_original"])
        matches = updated["Industry_enriched"].notna().sum()
        print(f"[INFO] Updated Industry for {matches} customers")
    
    if "Industry Sub List_enriched" in updated.columns:
        updated["Industry Sub List"] = updated["Industry Sub List_enriched"].fillna(updated["Industry Sub List_original"])
        matches = updated["Industry Sub List_enriched"].notna().sum()
        print(f"[INFO] Updated Industry Sub List for {matches} customers")
    
    # Add Reasoning column if it exists in enrichment data
    if "Reasoning" in enrichment_df.columns:
        updated["Industry_Reasoning"] = updated["Reasoning"]
        reasoning_matches = updated["Industry_Reasoning"].notna().sum()
        print(f"[INFO] Added reasoning for {reasoning_matches} customers")
    
    # Secondary match by CRM Full Name for any rows still missing Industry
    if 'Industry' in updated.columns and 'CRM Full Name' in enrichment_df.columns:
        # Prepare a slim enrichment frame keyed by CRM Full Name
        slim = enrichment_df[['CRM Full Name', 'Industry', 'Industry Sub List']].rename(
            columns={
                'Industry': 'Industry_by_name',
                'Industry Sub List': 'Industry Sub List_by_name'
            }
        )
        # Normalize strings for exact match after stripping
        def _norm(s):
            return s.astype(str).str.strip()
        if 'CRM Full Name' not in updated.columns and 'CRM Full Name_original' in updated.columns:
            updated['CRM Full Name'] = updated['CRM Full Name_original']
        if 'CRM Full Name' in updated.columns:
            updated['CRM Full Name'] = _norm(updated['CRM Full Name'])
            slim['CRM Full Name'] = _norm(slim['CRM Full Name'])
            # Merge by name
            updated = updated.merge(slim, on='CRM Full Name', how='left')
            # Fill only where still missing
            if 'Industry' in updated.columns and 'Industry_by_name' in updated.columns:
                missing = updated['Industry'].isna()
                updated.loc[missing, 'Industry'] = updated.loc[missing, 'Industry_by_name']
            if 'Industry Sub List' in updated.columns and 'Industry Sub List_by_name' in updated.columns:
                missing = updated['Industry Sub List'].isna()
                updated.loc[missing, 'Industry Sub List'] = updated.loc[missing, 'Industry Sub List_by_name']

    # Clean up temporary columns
    cols_to_drop = [
        col for col in updated.columns 
        if col.endswith(("_original", "_enriched")) or col in ("Reasoning", "Industry_by_name", "Industry Sub List_by_name")
    ]
    updated = updated.drop(columns=[c for c in cols_to_drop if c in updated.columns])

    return updated

# ---------------------------
# 2b. Azure SQL assembly
# ---------------------------

ASSET_WEIGHTS_FILE = ROOT / "artifacts" / "weights" / "asset_rollup_weights.json"


def load_asset_weights():
    try:
        with open(ASSET_WEIGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print(f"[WARN] Could not load {ASSET_WEIGHTS_FILE}. Using default weight=1.0 for all rollups.")
        return {"focus_goals": ["Printer", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"], "weights": {}}


def assemble_master_from_db() -> pd.DataFrame:
    """Pull data from Azure SQL and assemble a customer-level master table."""
    engine = da.get_engine()

    customers = da.get_customers_since_2023(engine)

    def _canon_id_series(s):
        s = s.astype(str).str.strip()
        return s.str.replace(r"\.0$", "", regex=True)

    # Canonicalize IDs and derive Company Name from CRM Full Name (strip leading numeric ID)
    if 'Customer ID' in customers.columns:
        customers['Customer ID'] = _canon_id_series(customers['Customer ID'])
    if 'CRM Full Name' in customers.columns:
        customers['Company Name'] = customers['CRM Full Name'].astype(str).str.replace(r'^\d+\s+', '', regex=True).str.strip()

    # Profit aggregates
    profit_goal = da.get_profit_since_2023_by_goal(engine)
    profit_rollup = da.get_profit_since_2023_by_rollup(engine)
    profit_quarterly = da.get_quarterly_profit_by_goal(engine)

    # Assets & seats
    assets = da.get_assets_and_seats(engine)

    # Apply industry enrichment (CSV) to update Industry fields (enrichment is sole source)
    industry_enrichment = load_industry_enrichment()
    if not industry_enrichment.empty:
        customers = apply_industry_enrichment(customers, industry_enrichment)

    # Pivot profit by Goal into columns
    if not profit_goal.empty:
        p_goal = (
            profit_goal.pivot_table(index="Customer ID", columns="Goal", values="Profit_Since_2023", aggfunc="sum")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        if 'Customer ID' in p_goal.columns: p_goal['Customer ID'] = _canon_id_series(p_goal['Customer ID'])
    else:
        p_goal = pd.DataFrame()

    # Merge into base
    master = customers.copy()
    if 'Customer ID' in master.columns:
        master['Customer ID'] = _canon_id_series(master['Customer ID'])
        master = master.merge(p_goal, on="Customer ID", how="left")

    # Compute total profit since 2023 across all Goals
    value_cols = [
        c for c in master.columns
        if c not in ("Customer ID", "Company Name", "Industry", "Industry Sub List", "Industry_Reasoning")
    ]
    if value_cols:
        master["Profit_Since_2023_Total"] = master[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        master["Profit_Since_2023_Total"] = 0.0

    # Derive quarterly totals per customer (LastQ and T4Q)
    if isinstance(profit_quarterly, pd.DataFrame) and not profit_quarterly.empty:
        pq = profit_quarterly.copy()
        # Build a sortable quarter key like 20241, 20242, ...
        def qkey(qs: str) -> int:
            # Expect format 'YYYYQn'
            try:
                yr = int(qs[0:4])
                qn = int(qs[-1])
                return yr * 10 + qn
            except Exception:
                return 0
        pq["_qkey"] = pq["Quarter"].astype(str).map(qkey)
        # Sum profit per customer, per quarter across goals
        cust_q = pq.groupby(["Customer ID", "_qkey"])['Profit'].sum().reset_index()
        # Determine latest quarter key globally
        latest_qkey = cust_q['_qkey'].max()
        # Last quarter per customer
        lastq = cust_q[cust_q['_qkey'] == latest_qkey].set_index("Customer ID")["Profit"].rename("Profit_LastQ_Total")
        # Trailing 4 quarters per customer
        lastq = lastq.reset_index()
        lastq['Customer ID'] = lastq['Customer ID'].astype(str)
        t4q_keys = sorted(cust_q['_qkey'].unique())[-4:]
        t4q = cust_q[cust_q['_qkey'].isin(t4q_keys)].groupby("Customer ID")["Profit"].sum().rename("Profit_T4Q_Total")
        master = master.merge(lastq, on="Customer ID", how="left")
        t4q = t4q.reset_index()
        t4q['Customer ID'] = t4q['Customer ID'].astype(str)
        master = master.merge(t4q, on="Customer ID", how="left")
        # Previous quarter per customer (for QoQ growth)
        keys_sorted = sorted(cust_q['_qkey'].unique())
        if len(keys_sorted) >= 2:
            prev_key = keys_sorted[-2]
            prevq = cust_q[cust_q['_qkey'] == prev_key].set_index("Customer ID")["Profit"].rename("Profit_PrevQ_Total")

            prevq = prevq.reset_index()
            prevq['Customer ID'] = prevq['Customer ID'].astype(str)
            # QoQ growth as percent change, guard divide by zero
            master = master.merge(prevq, on="Customer ID", how="left")
            prev_safe = master["Profit_PrevQ_Total"].fillna(0)
            last_safe = master["Profit_LastQ_Total"].fillna(0)
            denom = prev_safe.replace(0, np.nan)
            qoq = (last_safe - prev_safe) / denom
            master["Profit_QoQ_Growth"] = qoq.fillna(0.0)
        else:
            master["Profit_PrevQ_Total"] = 0.0
            master["Profit_QoQ_Growth"] = 0.0
    else:
        master["Profit_LastQ_Total"] = 0.0
        master["Profit_T4Q_Total"] = 0.0
        master["Profit_PrevQ_Total"] = 0.0
        master["Profit_QoQ_Growth"] = 0.0

    # Aggregate assets/seats totals and per-goal seats for filters/signals
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        a = assets.copy()
        # Normalize types
        if 'Customer ID' in a.columns:
            a['Customer ID'] = _canon_id_series(a['Customer ID'])
        for _dc in ["first_purchase_date", "last_expiration_date"]:
            if _dc in a.columns:
                a[_dc] = pd.to_datetime(a[_dc], errors="coerce")

        # Compute aggregates separately to avoid dtype issues
        parts = []
        if 'active_assets' in a.columns:
            s = a.groupby('Customer ID')['active_assets'].sum().rename('active_assets_total')
            parts.append(s)
        if 'seats_sum' in a.columns:
            s = a.groupby('Customer ID')['seats_sum'].sum().rename('seats_sum_total')
            parts.append(s)
        if 'first_purchase_date' in a.columns:
            s = a.groupby('Customer ID')['first_purchase_date'].min().rename('EarliestPurchaseDate')
            parts.append(s)
        if 'last_expiration_date' in a.columns:
            s = a.groupby('Customer ID')['last_expiration_date'].max().rename('LatestExpirationDate')
            parts.append(s)
        if 'item_rollup' in a.columns:
            s = a.groupby('Customer ID')['item_rollup'].nunique().rename('Portfolio_Breadth')
            parts.append(s)
        if parts:
            agg = pd.concat(parts, axis=1).reset_index()
            master = master.merge(agg, on="Customer ID", how="left")

        # Seats by Goal → pivot columns Seats_<Goal>
        if 'Goal' in a.columns and 'seats_sum' in a.columns:
            seats_by_goal = a.pivot_table(index="Customer ID", columns="Goal", values="seats_sum", aggfunc="sum").fillna(0)
            seats_by_goal.columns = [f"Seats_{str(c)}" for c in seats_by_goal.columns]
            seats_by_goal = seats_by_goal.reset_index()
            master = master.merge(seats_by_goal, on="Customer ID", how="left")
    else:
        master["active_assets_total"] = 0
        master["seats_sum_total"] = 0
    # Attach assets and rollup profit for feature engineering
    master._assets_raw = assets
    master._profit_rollup_raw = profit_rollup

    return master

def check_files_exist():
    """Backward-compatible wrapper that now just checks env and enrichment availability."""
    check_env()


# ---------------------------
# 2.  Load & clean data sets
# ---------------------------

def load_customers() -> pd.DataFrame:
    """Loads the main customer data from the Excel file and applies industry enrichment."""
    df = pd.read_excel(CUSTOMER_FILE)
    needed = {
        "Customer ID",
        "Company Name",
        "Industry",  # Add Industry column requirement
        "Industry Sub List",
        "Big Box Count",
        "Small Box Count",
        LICENSE_COL,
        "Total Hardware Revenue",
        "Total Consumable Revenue",
    }
    if not needed.issubset(df.columns):
        print("[WARN] Missing some expected columns in customer file.")
    
    df["key"] = df["Company Name"].map(clean_name)
    
    # Ensure printer counts are numeric, filling missing values with 0
    df["Big Box Count"] = pd.to_numeric(df["Big Box Count"], errors="coerce").fillna(0)
    df["Small Box Count"] = pd.to_numeric(
        df["Small Box Count"], errors="coerce"
    ).fillna(0)
    
    # Apply industry enrichment
    industry_enrichment = load_industry_enrichment()
    df = apply_industry_enrichment(df, industry_enrichment)
    
    return df


def load_sales() -> pd.DataFrame:
    """Loads the sales data from the Excel file."""
    df = pd.read_excel(SALES_FILE)
    needed = {"Company Name", "Dates", "GP", "Revenue"}
    if not needed.issubset(df.columns):
        print("[WARN] Missing some expected columns in sales file.")
    
    df["key"] = df["Company Name"].map(clean_name)
    df["Dates"] = pd.to_datetime(df["Dates"])
    return df

# ---------------------------
# 3.  Build GP24 & merge master
# ---------------------------

def aggregate_gp24(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregates sales data to get total Gross Profit and Revenue over the last 24 months."""
    start_date = datetime(2023, 6, 1)  # 24-month window
    df_24 = sales[sales["Dates"] >= start_date]
    gp24 = (
        df_24.groupby("key")
        .agg(GP24=("GP", "sum"), Revenue24=("Revenue", "sum"))
        .reset_index()
    )
    return gp24


def merge_master(customers, gp24, revenue=None) -> pd.DataFrame:
    """
    Merges the customer, GP24, and revenue data into a single master DataFrame.
    
    It joins first on the normalized company name ('key').
    """
    cust = customers.copy()
    master = cust.merge(
        gp24[["key", "GP24", "Revenue24"]], on="key", how="left", suffixes=("","")
    )
    
    # Merge revenue data if available
    if revenue is not None and len(revenue) > 0:
        master = master.merge(revenue, on="key", how="left", suffixes=("",""))
        
        # Use the reliable_revenue column for scoring
        master["revenue_estimate"] = master["reliable_revenue"].fillna(0)
        reliable_count = len(master[master["revenue_estimate"] > 0])
        print(f"[INFO] Merged reliable revenue data for {reliable_count} customers")
        
        # Add revenue source information for tracking and diagnostics
        if "revenue_source" in master.columns:
            source_counts = master["revenue_source"].fillna("none").value_counts()
            print(f"[INFO] Revenue sources: {dict(source_counts)}")
    else:
        master["revenue_estimate"] = 0  # Default to 0 if no revenue data is available
    
    return master

# ---------------------------
# 4.  Feature engineering
# ---------------------------

FOCUS_GOALS = {"Printers", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"}


def _normalize_goal_name(x: str) -> str:
    return str(x).strip().lower()


def engineer_features(df: pd.DataFrame, asset_weights: dict) -> pd.DataFrame:
    """
    Engineers features for scoring using assets/seats and profit.

    Adds:
      - adoption_assets: weighted asset/seat signal (focus divisions)
      - adoption_profit: profit signal for focus divisions (Printer, Accessorials, Scanners, Geomagic, 3DP Training rollup)
      - relationship_profit: profit for software goals (CAD, CPE, Specialty Software)
      - printer_count: compatibility metric from Printer assets
    """
    # Preserve access to any attached raw attributes before copying
    _base_df = df
    df = df.copy()

    assets = getattr(_base_df, "_assets_raw", pd.DataFrame())
    profit_roll = getattr(_base_df, "_profit_rollup_raw", pd.DataFrame())

    def _canon_id_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        return s.str.replace(r"\.0$", "", regex=True)

    if 'Customer ID' in df.columns:
        df['Customer ID'] = _canon_id_series(df['Customer ID'])

    # Default zeros
    df["adoption_assets"] = 0.0
    df["adoption_profit"] = 0.0
    df["relationship_profit"] = 0.0
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
        if 'Customer ID' in a.columns:
            a['Customer ID'] = _canon_id_series(a['Customer ID'])
        a["Goal"] = a["Goal"].map(_normalize_goal_name)
        # Keep only focus goals; special handling for Training/Services: only 3DP Training rollup counts
        a_focus = a[a["Goal"].isin(focus_goals)].copy()
        a_focus.loc[:, "weighted_value"] = a_focus.apply(weighted_measure, axis=1)

        # Compute printer_count specifically from Printer assets (use asset_count)
        printer_assets = a_focus[a_focus["Goal"].isin({_normalize_goal_name("Printer"), _normalize_goal_name("Printers")})]
        printer_counts = (
            printer_assets.groupby("Customer ID")["asset_count"].sum().rename("printer_count")
        )

        adoption_assets = (
            a_focus.groupby("Customer ID")["weighted_value"].sum().rename("adoption_assets")
        )

        df = df.merge(adoption_assets, on="Customer ID", how="left")
        df = df.merge(printer_counts, on="Customer ID", how="left")
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

    # Build adoption_profit from profit_rollup: focus goals + 3DP Training rollup
    if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
        pr = profit_roll.copy()
        if 'Customer ID' in pr.columns:
            pr['Customer ID'] = _canon_id_series(pr['Customer ID'])
        pr["Goal"] = pr["Goal"].map(_normalize_goal_name)
        # Sum for focus goals
        focus_goals_norm = {_normalize_goal_name(g) for g in FOCUS_GOALS}
        mask_focus_goals = pr["Goal"].isin(focus_goals_norm)
        # Include only 3DP Training within Training/Services
        mask_3dp_training = (pr["Goal"] == _normalize_goal_name("Training/Services")) & (pr["item_rollup"].astype(str).str.strip().str.lower() == "3dp training")
        mask_focus = mask_focus_goals & (~(pr["Goal"] == _normalize_goal_name("Training/Services")) | mask_3dp_training)
        adoption_profit = (
            pr[mask_focus]
            .groupby("Customer ID")["Profit_Since_2023"]
            .sum()
            .rename("adoption_profit")
        )
        df = df.merge(adoption_profit, on="Customer ID", how="left")
        if 'adoption_profit_y' in df.columns:
            if 'adoption_profit' in df.columns:
                df['adoption_profit'] = df['adoption_profit_y'].combine_first(df['adoption_profit'])
            else:
                df['adoption_profit'] = df['adoption_profit_y']
            drop_cols = [c for c in ['adoption_profit_x','adoption_profit_y'] if c in df.columns]
            df = df.drop(columns=drop_cols)
        df["adoption_profit"] = df.get("adoption_profit", 0).fillna(0.0)

    # Relationship: software goals CAD, CPE, Specialty Software from goal-level pivot already merged
    sw_cols = [c for c in df.columns if c in ("CAD", "CPE", "Specialty Software")]
    if sw_cols:
        df["relationship_profit"] = df[sw_cols].fillna(0).sum(axis=1)
    else:
        if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
            sw_mask = profit_roll["Goal"].isin(["CAD", "CPE", "Specialty Software"])
            rel = (
                profit_roll[sw_mask]
                .groupby("Customer ID")["Profit_Since_2023"].sum().rename("relationship_profit")
            )
            df = df.merge(rel, on="Customer ID", how="left")
            if 'relationship_profit_y' in df.columns:
                if 'relationship_profit' in df.columns:
                    df['relationship_profit'] = df['relationship_profit_y'].combine_first(df['relationship_profit'])
                else:
                    df['relationship_profit'] = df['relationship_profit_y']
                drop_cols = [c for c in ['relationship_profit_x','relationship_profit_y'] if c in df.columns]
                df = df.drop(columns=drop_cols)
            df["relationship_profit"] = df.get("relationship_profit", 0).fillna(0.0)

    # Flag for scaling
    df["scaling_flag"] = (df["printer_count"] >= 4).astype(int)

    # Compatibility column for dashboard tiering
    if "relationship_profit" in df.columns:
        df[LICENSE_COL] = df["relationship_profit"].fillna(0.0)
    else:
        df[LICENSE_COL] = 0.0

    # Calculate all component and final scores via scoring_logic
    df = calculate_scores(df, WEIGHTS)
    return df

# ---------------------------
# 5.  Visual builder
# ---------------------------

def save_fig(filename: str):
    """Saves the current matplotlib figure to reports/figures."""
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def build_visuals(df: pd.DataFrame):
    """Generates and saves a standard set of 10 visualizations."""
    
    # 1: Histogram of final ICP Scores
    plt.figure()
    plt.hist(df["ICP_score"].dropna(), bins=30)
    plt.title("Distribution of ICP Scores")
    plt.xlabel("ICP Score")
    plt.ylabel("Number of Customers")
    save_fig("vis1_icp_hist.png")
    
    # 2: Total Profit since 2023 by industry vertical
    plt.figure()
    if "Industry" in df.columns and "Profit_Since_2023_Total" in df.columns:
        s = df.groupby("Industry")["Profit_Since_2023_Total"].sum()
        if not s.empty:
            s.nlargest(10).plot(kind="bar")
        else:
            print("[INFO] Skipping Profit by Vertical (Top 10): no data.")
    # 3: Scatter plot of printer count vs GP24
    plt.figure()
    if "Profit_Since_2023_Total" in df.columns:
        plt.scatter(df["printer_count"], df["Profit_Since_2023_Total"])
    plt.title("Printer Count vs Profit Since 2023")
    plt.xlabel("Printer Count")
    plt.ylabel("Profit ($)")
    save_fig("vis3_printers_gp24.png")
    
    # 4: Box plot of GP24 by CAD tier
    if 'cad_tier' in df.columns and hasattr(df['cad_tier'], 'cat'):
        if not df['cad_tier'].cat.categories.empty and not df['cad_tier'].isnull().all():
            plt.figure()
            if "Profit_Since_2023_Total" in df.columns:
                df.boxplot(column="Profit_Since_2023_Total", by="cad_tier")
            plt.title("Profit Since 2023 by CAD Tier")
            plt.suptitle("")
            plt.xlabel("CAD Tier")
            plt.ylabel("Profit ($)")
            save_fig("vis4_gp24_cadtier.png")
        else:
            print("[INFO] Skipping 'GP24 by CAD Tier' visual: No data to plot.")
    else:
        print("[INFO] Skipping 'GP24 by CAD Tier' visual: 'cad_tier' column not suitable for plotting.")
    
    # 5: Average ICP score by industry vertical
    if "Industry" in df.columns and "ICP_score" in df.columns:
        s3 = df.groupby("Industry")["ICP_score"].mean()
        if not s3.empty:
            s3.nlargest(10).plot(kind="bar")
    save_fig("vis5_icp_vertical.png")
    
    # 6: Count of scaling accounts (>=4 printers) by vertical
    plt.figure()
    if "Industry" in df.columns and "Customer ID" in df.columns:
        s4 = df[df["scaling_flag"] == 1].groupby("Industry")["Customer ID"].count()
        if not s4.empty:
            s4.nlargest(10).plot(kind="bar")
        else:
            print("[INFO] Skipping scaling accounts chart: no data.")

    plt.title("Scaling Accounts (>=4 Printers) per Vertical")
    plt.ylabel("Account Count")
    save_fig("vis6_scaling_vertical.png")
    
    # 7: Scatter plot of CAD spend vs ICP score
    if LICENSE_COL in df.columns:
        plt.figure()
        plt.scatter(df[LICENSE_COL], df["ICP_score"])
        plt.title("CAD Spend vs ICP Score")
        plt.xlabel("Total Software License Revenue ($)")
        plt.ylabel("ICP Score")
        save_fig("vis7_cad_icp.png")
    
    # 8: Scatter plot of printer count vs ICP score
    plt.figure()
    plt.scatter(df["printer_count"], df["ICP_score"])
    plt.title("Printer Count vs ICP Score")
    plt.xlabel("Printer Count")
    plt.ylabel("ICP Score")
    save_fig("vis8_printers_icp.png")
    
    # 9: Total Profit since 2023 by industry vertical (duplicate view)
    plt.figure()
    if "Industry" in df.columns and "Profit_Since_2023_Total" in df.columns:
        s2 = df.groupby("Industry")["Profit_Since_2023_Total"].sum()
        if not s2.empty:
            s2.nlargest(10).plot(kind="bar")
        else:
            print("[INFO] Skipping Profit by Vertical (Top 10): no data.")
    # 10: Customer count by CAD tier
    plt.figure()
    df["cad_tier"].value_counts().plot(kind="bar")
    plt.title("Customer Count by CAD Tier")
    plt.xlabel("CAD Tier")
    plt.ylabel("Number of Customers")
    save_fig("vis10_customers_cadtier.png")

# ---------------------------
# 6.  Main driver
# ---------------------------

def main():
    """Main function to execute the scoring pipeline using Azure SQL inputs."""
    check_env()

    print("Loading data from Azure SQL...")
    master = assemble_master_from_db()

    # --- Validations ---
    ok, missing = ensure_columns(master, [COL_CUSTOMER_ID, COL_COMPANY_NAME, COL_INDUSTRY])
    if not ok:
        log_validation("Missing required columns in master", missing, root=ROOT)
        print(f"[WARN] Missing required columns: {missing}")
    ok2, bad = ensure_non_negative(master, [COL_REL_LICENSE, COL_REL_SAAS, COL_REL_MAINT, COL_HW_REV, COL_CONS_REV])
    if not ok2 and bad:
        log_validation("Found negative values; clamping to zero", bad, root=ROOT)
        for c in bad:
            master[c] = master[c].clip(lower=0)

    print("Generating data-driven industry weights...")
    industry_weights_path = ROOT / "artifacts" / "weights" / "industry_weights.json"
    if not industry_weights_path.exists():
        print("[INFO] Building new industry weights from historical profit since 2023")
        industry_weights = build_industry_weights(master)
        industry_weights_path.parent.mkdir(parents=True, exist_ok=True)
        save_industry_weights(industry_weights, filepath=str(industry_weights_path))
    else:
        print("[INFO] Loading existing industry weights")
        _ = load_industry_weights(filepath=str(industry_weights_path))

    print("Engineering features & scores...")
    asset_weights = load_asset_weights()
    scored = engineer_features(master, asset_weights)

    # Define the columns for the final output CSV
    out_cols = [
        "Customer ID",
        "Company Name",
        "Industry",
        "Industry Sub List",
        "printer_count",
        "scaling_flag",
        LICENSE_COL,
        "cad_tier",
        "Profit_Since_2023_Total",
        "Profit_T4Q_Total",
        "Profit_LastQ_Total",
        "Profit_PrevQ_Total",
        "Profit_QoQ_Growth",
        "adoption_assets",
        "adoption_profit",
        "relationship_profit",
        "active_assets_total",
        "seats_sum_total",
        "Portfolio_Breadth",
        "EarliestPurchaseDate",
        "LatestExpirationDate",
        "ICP_score",
        "ICP_grade",
        "ICP_score_raw",
        "vertical_score",
        "size_score",
        "adoption_score",
        "relationship_score"
    ]
    
    # Dynamically add other software revenue columns to the output if they exist
    revenue_cols_to_check = ['Total Consumable Revenue', 'Total SaaS Revenue', 'Total Maintenance Revenue', 'Printer', 'Printer Accessorials', 'Scanners', 'Geomagic', 'CAD', 'CPE', 'Specialty Software']
    for col in revenue_cols_to_check:
        if col in scored.columns:
            out_cols.append(col)
    
    # Add industry enrichment columns if they exist
    industry_enrichment_cols = ['Industry_Reasoning']
    for col in industry_enrichment_cols:
        if col in scored.columns:
            out_cols.append(col)
    
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
    scored[out_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("Creating visualisations...")
    build_visuals(scored)
    print("Saved 10 PNG charts (vis1..vis10).")
    
    print("All done.")


# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ICP scoring pipeline")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: data/processed/icp_scored_accounts.csv)")
    parser.add_argument("--weights", type=str, default=None, help="Path to optimized_weights.json")
    parser.add_argument("--industry-weights", type=str, default=None, help="Path to industry_weights.json")
    parser.add_argument("--asset-weights", type=str, default=None, help="Path to asset_rollup_weights.json")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip generating visuals")
    args, unknown = parser.parse_known_args()

    # Note: core script uses fixed paths; we allow overrides via env-like globals where feasible.
    # We won't refactor full internals here; just set known paths if provided.
    try:
        if args.weights:
            os.environ["ICP_OPT_WEIGHTS_PATH"] = args.weights
        if args.industry_weights:
            os.environ["ICP_INDUSTRY_WEIGHTS_PATH"] = args.industry_weights
        if args.asset_weights:
            os.environ["ICP_ASSET_WEIGHTS_PATH"] = args.asset_weights
        # Run main
        main()
        # If custom out was requested and default out exists, copy it
        if args.out:
            src = ROOT / "data" / "processed" / "icp_scored_accounts.csv"
            if src.exists():
                dst = Path(args.out)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"[INFO] Wrote copy to {dst}")
        if args.skip_visuals:
            print("[INFO] --skip-visuals specified (no additional visuals rendered)")
    except Exception:
        print("\nAn error occurred - see traceback above.\n")
        raise











