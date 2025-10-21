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
  • icp_scored_accounts.csv
  • vis1_… vis10_… PNG charts

Requires: pandas, numpy, matplotlib, python-dateutil, scikit-learn, scipy
---------------------------------
Usage:
  $ python goe_icp_scoring.py
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm

# Import the centralized scoring logic
from scoring_logic import calculate_scores, LICENSE_COL, DEFAULT_WEIGHTS
# Import the new industry scoring module
from industry_scoring import build_industry_weights, save_industry_weights, load_industry_weights
import data_access as da

# ---------------------------
# 0.  CONFIG – file names & weights
# ---------------------------

INDUSTRY_ENRICHMENT_FILE = "TR - Industry Enrichment.csv"  # Updated industry data
ASSET_WEIGHTS_FILE = "asset_rollup_weights.json"


def load_weights():
    """
    Load optimized weights from the JSON file.
    If the file is not found or is invalid, it falls back to the default weights
    defined in `scoring_logic.py`. It also converts the weights from the
    optimizer's format to the format expected by the scoring script.
    """
    try:
        with open('optimized_weights.json', 'r') as f:
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
            
            print(f"[INFO] Loaded optimized weights from optimized_weights.json")
            print(f"  - Optimization details: {data.get('n_trials', 'Unknown')} trials, λ={data.get('lambda_param', 'Unknown')}")
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
    if not os.path.exists(INDUSTRY_ENRICHMENT_FILE):
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
        print(f"[WARN] Error loading revenue file: {e}")
        return pd.DataFrame()


def load_industry_enrichment() -> pd.DataFrame:
    """
    Loads updated industry data from the industry enrichment CSV file.
    This provides more accurate and up-to-date industry classifications.
    """
    if not os.path.exists(INDUSTRY_ENRICHMENT_FILE):
        print(f"[INFO] Industry enrichment file '{INDUSTRY_ENRICHMENT_FILE}' not found. Using original industry data.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(INDUSTRY_ENRICHMENT_FILE)
        
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
        
        # Include Reasoning column if it exists
        cols_to_return = ["Customer ID", "Industry", "Industry Sub List"]
        if "Reasoning" in df.columns:
            cols_to_return.append("Reasoning")
            print(f"[INFO] Including 'Reasoning' column from industry enrichment")
        
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
    
    # Ensure Customer ID columns have matching data types
    # Convert Excel floats to integers first, then to strings to avoid .0 suffix
    df["Customer ID"] = df["Customer ID"].fillna(0).astype(int).astype(str)
    enrichment_df["Customer ID"] = enrichment_df["Customer ID"].astype(str)
    
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
    
    # Clean up temporary columns
    cols_to_drop = [col for col in updated.columns if col.endswith(("_original", "_enriched")) or col == "Reasoning"]
    updated = updated.drop(columns=cols_to_drop)
    
    return updated

# ---------------------------
# 2b. Azure SQL assembly
# ---------------------------

ASSET_WEIGHTS_FILE = "asset_rollup_weights.json"


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

    # Profit aggregates
    profit_goal = da.get_profit_since_2023_by_goal(engine)
    profit_rollup = da.get_profit_since_2023_by_rollup(engine)
    profit_quarterly = da.get_quarterly_profit_by_goal(engine)

    # Assets & seats
    assets = da.get_assets_and_seats(engine)

    # Apply industry enrichment (CSV)
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
    else:
        p_goal = pd.DataFrame()

    # Merge into base
    master = customers.copy()
    if not p_goal.empty:
        master = master.merge(p_goal, on="Customer ID", how="left")

    # Compute total profit since 2023 across all Goals
    value_cols = [
        c for c in master.columns
        if c not in ("Customer ID", "Company Name", "Industry", "Industry Sub List", "Industry_Reasoning")
    ]
    if value_cols:
        master["Profit_Since_2023_Total"] = pd.to_numeric(master[value_cols], errors="coerce").fillna(0).sum(axis=1)
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
        t4q_keys = sorted(cust_q['_qkey'].unique())[-4:]
        t4q = cust_q[cust_q['_qkey'].isin(t4q_keys)].groupby("Customer ID")["Profit"].sum().rename("Profit_T4Q_Total")
        master = master.merge(lastq, on="Customer ID", how="left")
        master = master.merge(t4q, on="Customer ID", how="left")
    else:
        master["Profit_LastQ_Total"] = 0.0
        master["Profit_T4Q_Total"] = 0.0

    # Aggregate assets/seats totals and per-goal seats for filters/signals
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        agg = assets.groupby("Customer ID").agg(
            active_assets_total=("active_assets", "sum"),
            seats_sum_total=("seats_sum", "sum"),
            EarliestPurchaseDate=("first_purchase_date", "min"),
            LatestExpirationDate=("last_expiration_date", "max"),
            Portfolio_Breadth=("item_rollup", pd.Series.nunique)
        )
        master = master.merge(agg, on="Customer ID", how="left")
        # Seats by Goal → pivot columns Seats_<Goal>
        seats_by_goal = assets.pivot_table(index="Customer ID", columns="Goal", values="seats_sum", aggfunc="sum").fillna(0)
        # Rename columns
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

FOCUS_GOALS = {"Printer", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"}


def _normalize_goal_name(x: str) -> str:
    return str(x).strip()


def engineer_features(df: pd.DataFrame, asset_weights: dict) -> pd.DataFrame:
    """
    Engineers features for scoring using assets/seats and profit.

    Adds:
      - adoption_assets: weighted asset/seat signal (focus divisions)
      - adoption_profit: profit signal for focus divisions (Printer, Accessorials, Scanners, Geomagic, 3DP Training rollup)
      - relationship_profit: profit for software goals (CAD, CPE, Specialty Software)
      - printer_count: compatibility metric from Printer assets
    """
    df = df.copy()

    assets = getattr(df, "_assets_raw", pd.DataFrame())
    profit_roll = getattr(df, "_profit_rollup_raw", pd.DataFrame())

    # Default zeros
    df["adoption_assets"] = 0.0
    df["adoption_profit"] = 0.0
    df["relationship_profit"] = 0.0
    df["printer_count"] = 0.0

    # Build adoption_assets from assets table with weights
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        weights_cfg = asset_weights.get("weights", {})
        focus_goals = set(asset_weights.get("focus_goals", list(FOCUS_GOALS)))

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
        a["Goal"] = a["Goal"].map(_normalize_goal_name)
        # Keep only focus goals; special handling for Training/Services: only 3DP Training rollup counts
        a_focus = a[a["Goal"].isin(focus_goals)].copy()
        a_focus.loc[:, "weighted_value"] = a_focus.apply(weighted_measure, axis=1)

        # Compute printer_count specifically from Printer assets (use asset_count)
        printer_assets = a_focus[a_focus["Goal"] == "Printer"]
        printer_counts = (
            printer_assets.groupby("Customer ID")["asset_count"].sum().rename("printer_count")
        )

        adoption_assets = (
            a_focus.groupby("Customer ID")["weighted_value"].sum().rename("adoption_assets")
        )

        df = df.merge(adoption_assets, on="Customer ID", how="left")
        df = df.merge(printer_counts, on="Customer ID", how="left")
        df["adoption_assets"] = df["adoption_assets"].fillna(0.0)
        df["printer_count"] = df["printer_count"].fillna(0.0)

    # Build adoption_profit from profit_rollup: focus goals + 3DP Training rollup
    if isinstance(profit_roll, pd.DataFrame) and not profit_roll.empty:
        pr = profit_roll.copy()
        pr["Goal"] = pr["Goal"].map(_normalize_goal_name)
        # Sum for focus goals
        mask_focus_goals = pr["Goal"].isin(FOCUS_GOALS)
        # Include only 3DP Training within Training/Services
        mask_3dp_training = (pr["Goal"] == "Training/Services") & (pr["item_rollup"].astype(str).str.strip().str.lower() == "3dp training")
        mask_focus = mask_focus_goals & (~(pr["Goal"] == "Training/Services") | mask_3dp_training)
        adoption_profit = (
            pr[mask_focus]
            .groupby("Customer ID")["Profit_Since_2023"]
            .sum()
            .rename("adoption_profit")
        )
        df = df.merge(adoption_profit, on="Customer ID", how="left")
        df["adoption_profit"] = df["adoption_profit"].fillna(0.0)

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
            df["relationship_profit"] = df["relationship_profit"].fillna(0.0)

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

def save_fig(path):
    """Saves the current matplotlib figure to a file."""
    plt.tight_layout()
    plt.savefig(path, dpi=120)
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
        df.groupby("Industry")["Profit_Since_2023_Total"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total Profit Since 2023 by Vertical (Top 10)")
    plt.ylabel("Profit ($)")
    save_fig("vis2_gp24_vertical.png")
    
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
    plt.figure()
    df.groupby("Industry")["ICP_score"].mean().nlargest(10).plot(kind="bar")
    plt.title("Average ICP Score by Vertical (Top 10)")
    plt.ylabel("Mean ICP Score")
    save_fig("vis5_icp_vertical.png")
    
    # 6: Count of scaling accounts (>=4 printers) by vertical
    plt.figure()
    (
        df[df["scaling_flag"] == 1]
        .groupby("Industry")["Customer ID"]
        .count()
        .nlargest(10)
        .plot(kind="bar")
    )
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
        df.groupby("Industry")["Profit_Since_2023_Total"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total Profit Since 2023 by Vertical (Top 10)")
    plt.ylabel("Profit ($)")
    save_fig("vis9_rev24_vertical.png")
    
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
    """Main function to execute the entire scoring pipeline."""
    check_files_exist()
    
    print("Loading spreadsheets…")
    customers = load_customers()
    sales = load_sales()
    
    print("Aggregating GP24…")
    gp24 = aggregate_gp24(sales)
    
    print("Merging data sets…")
    revenue = load_revenue()
    master = merge_master(customers, gp24, revenue)
    
    print("Generating data-driven industry weights…")
    # Build or load industry weights based on historical performance
    if not os.path.exists("industry_weights.json"):
        print("[INFO] Building new industry weights from historical data")
        industry_weights = build_industry_weights(master)
        save_industry_weights(industry_weights)
    else:
        print("[INFO] Loading existing industry weights")
        industry_weights_data = load_industry_weights()
        # We'll pass this to scoring_logic via a global or parameter
    
    print("Engineering features & scores…")
    scored = engineer_features(master)
    
    # Define the columns for the final output CSV
    out_cols = [
        "Customer ID",
        "Company Name",
        "Industry",
        "Industry Sub List",
        "printer_count",
        LICENSE_COL,
        "cad_tier",
        "Profit_Since_2023_Total",
        "Profit_T4Q_Total",
        "Profit_LastQ_Total",
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
    
    # Save the scored data to a CSV file
    scored[out_cols].to_csv("icp_scored_accounts.csv", index=False)
    print("Saved icp_scored_accounts.csv")
    
    print("Saved 10 PNG charts (vis1..vis10).")
    
    print("All done.")


# ---------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n⚠ An error occurred – details below.\n")
        raise






