"""
goe_icp_scoring.py
---------------------------------
End-to-end script to score GoEngineer Digital-Manufacturing accounts
and generate key visuals.

Files expected in the SAME directory:
  1) JY - Customer Analysis - Customer Segmentation.xlsx (contains industry, revenue, and customer data)
  2) TR - Master Sales Log - Customer Segmentation.xlsx (contains GP and sales data)
Outputs:
  • icp_scored_accounts.csv
  • vis1_… vis10_… PNG charts
Requires: pandas, numpy, matplotlib, python-dateutil
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
from sklearn.preprocessing import MinMaxScaler
import json
from scipy.stats import norm

# --------------------------- #
# 0.  CONFIG – file names & weights
# --------------------------- #
# ASSET_FILE   = "TR - All SSYS Customer Assets - Customer Segmentation.xlsx"  # No longer needed for industry data
CUSTOMER_FILE = "JY - Customer Analysis - Customer Segmentation.xlsx"
SALES_FILE   = "TR - Master Sales Log - Customer Segementation.xlsx"
REVENUE_FILE = "customer_revenue_analysis.xlsx"  # New revenue data file

# Default weights (fallback if optimized weights file doesn't exist)
DEFAULT_WEIGHTS = {
    "vertical": 0.25,
    "size": 0.25,
    "adoption": 0.25,
    "relationship": 0.25,
}

def load_weights():
    """Load optimized weights from JSON file, or use defaults if not available."""
    try:
        with open('optimized_weights.json', 'r') as f:
            data = json.load(f)
            raw_weights = data.get('weights', {})
            
            # Convert from optimizer format to script format and calculate missing weights
            weights = {}
            weights['vertical'] = raw_weights.get('vertical_score', 0.25)
            weights['size'] = raw_weights.get('size_score', 0.25)
            weights['adoption'] = raw_weights.get('adoption_score', 0.25)
            
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

# Load weights dynamically
WEIGHTS = load_weights()

# Data-driven vertical weights based on actual Total Hardware + Consumable Revenue performance from JY spreadsheet
# Analysis shows these industries generate the highest combined hardware and consumable revenue
PERFORMANCE_VERTICAL_WEIGHTS = {
    "aerospace & defense": 1.0,
    "automotive & transportation": 1.0,
    "consumer goods": 1.0,
    "high tech": 1.0,
    "medical devices & life sciences": 1.0,
    "engineering services": 0.8,
    "heavy equip & ind. components": 0.8,
    "industrial machinery": 0.8,
    "mold, tool & die": 0.8,
    "other": 0.8,
    "building & construction": 0.6,
    "chemicals & related products": 0.6,
    "dental": 0.6,
    "manufactured products": 0.6,
    "services": 0.6,
    "education & research": 0.4,
    "electromagnetic": 0.4,
    "energy": 0.4,
    "packaging": 0.4,
    "plant & process": 0.4,
    "shipbuilding": 0.4,
}
# Any industry not in this list will receive a default score of 0.3

LICENSE_COL = "Total Software License Revenue"

# --------------------------- #
# 1.  Utility helpers
# --------------------------- #
def clean_name(x: str) -> str:
    """Aggressive company-name normaliser."""
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

def check_files_exist():
    required_files = (CUSTOMER_FILE, SALES_FILE)
    for f in required_files:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] Cannot find '{f}' in current directory.")
    
    # Revenue file is optional
    if not os.path.exists(REVENUE_FILE):
        print(f"[INFO] Revenue file '{REVENUE_FILE}' not found. Will use printer-count-based size scoring.")

def load_revenue() -> pd.DataFrame:
    """Load and clean revenue data with prioritization logic"""
    if not os.path.exists(REVENUE_FILE):
        return pd.DataFrame()  # Return empty dataframe if file doesn't exist
    
    try:
        df = pd.read_excel(REVENUE_FILE)
        
        # Handle potential column name variations
        company_col = None
        for col in ["Company Name", "Compnay Name", "company name", "company_name"]:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None:
            raise ValueError(f"Could not find company name column. Available columns: {df.columns.tolist()}")
        
        df["key"] = df[company_col].map(clean_name)
        print(f"[INFO] Using company column: '{company_col}'")
        
        # Implement prioritization logic for reliable revenue
        print("[INFO] Applying revenue data prioritization logic...")
        
        # Initialize reliable_revenue column
        df["reliable_revenue"] = None
        df["revenue_source"] = "none"
        
        # Priority 1: pdl_estimate (most reliable)
        pdl_mask = df["source"] == "pdl_estimate"
        df.loc[pdl_mask, "reliable_revenue"] = pd.to_numeric(df.loc[pdl_mask, "revenue_exact"], errors="coerce")
        df.loc[pdl_mask, "revenue_source"] = "pdl_estimate"
        pdl_count = pdl_mask.sum()
        
        # Priority 2: fmp_income but only if < $1 trillion (to filter currency errors)
        fmp_mask = (df["source"] == "fmp_income") & (df["reliable_revenue"].isna())
        fmp_revenue = pd.to_numeric(df.loc[fmp_mask, "revenue_exact"], errors="coerce")
        valid_fmp_mask = fmp_mask & (fmp_revenue < 1000000000000)  # < $1 trillion
        df.loc[valid_fmp_mask, "reliable_revenue"] = fmp_revenue.loc[valid_fmp_mask]
        df.loc[valid_fmp_mask, "revenue_source"] = "fmp_income"
        fmp_count = valid_fmp_mask.sum()
        
        # Priority 3: Discard heuristic_estimate (all garbage)
        heuristic_count = (df["source"] == "heuristic_estimate").sum()
        
        # Remove rows where reliable_revenue is still None or NaN
        df = df.dropna(subset=["reliable_revenue"])
        
        print(f"[INFO] Revenue data prioritization results:")
        print(f"  - pdl_estimate: {pdl_count} customers")
        print(f"  - fmp_income (valid): {fmp_count} customers") 
        print(f"  - heuristic_estimate (discarded): {heuristic_count} customers")
        print(f"  - Total reliable revenue records: {len(df)}")
        
        return df[["key", "reliable_revenue", "revenue_source"]]
    except Exception as e:
        print(f"[WARN] Error loading revenue file: {e}")
        return pd.DataFrame()

# --------------------------- #
# 2.  Load & clean data sets
# --------------------------- #
def load_assets() -> pd.DataFrame:
    df = pd.read_excel(ASSET_FILE)
    # Since we're not using this file for industry enrichment anymore, 
    # just return an empty dataframe if required columns are missing
    if "Company Name" not in df.columns:
        print("[INFO] Assets file doesn't have required columns. Skipping assets data.")
        return pd.DataFrame(columns=["key"])
    
    df["key"] = df["Company Name"].map(clean_name)
    return df

def load_customers() -> pd.DataFrame:
    df = pd.read_excel(CUSTOMER_FILE)
    needed = {
        "Customer ID",
        "Company Name",
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
    # ensure numeric counts
    df["Big Box Count"] = pd.to_numeric(df["Big Box Count"], errors="coerce").fillna(0)
    df["Small Box Count"] = pd.to_numeric(
        df["Small Box Count"], errors="coerce"
    ).fillna(0)
    return df

def load_sales() -> pd.DataFrame:
    df = pd.read_excel(SALES_FILE)
    needed = {"Company Name", "Dates", "GP", "Revenue"}
    if not needed.issubset(df.columns):
        print("[WARN] Missing some expected columns in sales file.")
    df["key"] = df["Company Name"].map(clean_name)
    df["Dates"] = pd.to_datetime(df["Dates"])
    return df

# --------------------------- #
# 3.  Build GP24 & merge master
# --------------------------- #
def aggregate_gp24(sales: pd.DataFrame) -> pd.DataFrame:
    start_date = datetime(2023, 6, 1)  # 24-month window
    df_24 = sales[sales["Dates"] >= start_date]
    gp24 = (
        df_24.groupby("key")
        .agg(GP24=("GP", "sum"), Revenue24=("Revenue", "sum"))
        .reset_index()
    )
    return gp24

def merge_master(assets, customers, gp24, revenue=None) -> pd.DataFrame:
    # Join on Customer ID (preferred) then key
    cust = customers.copy()
    master = cust.merge(
        gp24[["key", "GP24", "Revenue24"]], on="key", how="left", suffixes=("","")
    )
    
    # Skip industry enrichment from assets file since JY file already has good industry data
    # The Industry column from the JY file contains proper industry categories
    # No need to merge or override with assets data
    
    # Merge revenue data if available
    if revenue is not None and len(revenue) > 0:
        master = master.merge(revenue, on="key", how="left", suffixes=("",""))
        # Use the reliable_revenue column and rename for consistency
        master["revenue_exact"] = master["reliable_revenue"].fillna(0)
        reliable_count = len(master[master["revenue_exact"] > 0])
        print(f"[INFO] Merged reliable revenue data for {reliable_count} customers")
        
        # Add revenue source information for tracking
        if "revenue_source" in master.columns:
            source_counts = master["revenue_source"].fillna("none").value_counts()
            print(f"[INFO] Revenue sources: {dict(source_counts)}")
    else:
        master["revenue_exact"] = 0  # Default to 0 if no revenue data
        
    return master

# --------------------------- #
# 4.  Feature engineering
# --------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Core counts
    df["printer_count"] = df["Big Box Count"] + df["Small Box Count"]
    df["scaling_flag"] = (df["printer_count"] >= 4).astype(int)

    # Relationship tier from software-license revenue (kept for visuals, not for scoring)
    bins = [-1, 5000, 25000, 100000, np.inf]
    labels = ["Bronze", "Silver", "Gold", "Platinum"]
    df["cad_tier"] = pd.cut(df[LICENSE_COL].fillna(0), bins=bins, labels=labels)

    # Scores for each criterion using the new performance-based weights
    v_lower = df["Industry"].astype(str).str.lower().str.strip()
    df["vertical_score"] = v_lower.map(PERFORMANCE_VERTICAL_WEIGHTS).fillna(0.3)

    # --- New data-driven scores ---
    # 1. New Adoption Score (Printer Count + Consumable Revenue) using log-then-min-max
    def min_max_scale(series):
        min_val, max_val = series.min(), series.max()
        if max_val - min_val == 0: return 0.0
        return (series - min_val) / (max_val - min_val)

    if 'Total Consumable Revenue' not in df.columns:
        df['Total Consumable Revenue'] = 0

    # Ensure non-negative values for log transformation to avoid RuntimeWarning
    printer_count_safe = np.maximum(df['printer_count'].fillna(0), 0)
    consumable_revenue_safe = np.maximum(df['Total Consumable Revenue'].fillna(0), 0)
    
    printer_score = min_max_scale(np.log1p(printer_count_safe))
    consumable_score = min_max_scale(np.log1p(consumable_revenue_safe))
    df['adoption_score'] = 0.5 * printer_score + 0.5 * consumable_score

    # 2. New Relationship Score (All Software-related Revenue) using log-then-min-max
    relationship_cols = ['Total Software License Revenue', 'Total SaaS Revenue', 'Total Maintenance Revenue']
    for col in relationship_cols:
        if col not in df.columns:
            df[col] = 0

    df['relationship_feature'] = df[relationship_cols].fillna(0).sum(axis=1)
    # Ensure non-negative values for log transformation to avoid RuntimeWarning
    relationship_feature_safe = np.maximum(df['relationship_feature'], 0)
    df['relationship_score'] = min_max_scale(np.log1p(relationship_feature_safe))
    
    # --- Data-Driven Size Score ---
    # Size score based on empirical analysis showing $1B+ companies are the sweet spot
    if "revenue_exact" in df.columns and df["revenue_exact"].sum() > 0:
        revenue_values = df["revenue_exact"].fillna(0)
        has_reliable_revenue = revenue_values > 0
        
        # Initialize with neutral default score for all customers
        df["size_score"] = 0.5  # Neutral default score
        
        # Apply data-driven revenue-based scoring based on engaged customer analysis
        # Analysis showed $250M-$1B companies have highest performance among engaged customers
        conditions = [
            (revenue_values >= 250_000_000) & (revenue_values < 1_000_000_000),  # $250M-$1B (Sweet Spot)
            (revenue_values >= 1_000_000_000),      # $1B+ (Excellent but harder to penetrate)
            (revenue_values >= 50_000_000),         # $50M-$250M (Moderate)
            (revenue_values >= 10_000_000),         # $10M-$50M (Lower)
            (revenue_values > 0)                    # $0-$10M (Lower)
        ]
        
        scores = [1.0, 0.9, 0.6, 0.4, 0.4]
        
        # Apply scoring tiers only to customers with reliable revenue data
        for condition, score in zip(conditions, scores):
            mask = has_reliable_revenue & condition
            df.loc[mask, "size_score"] = score
        
        reliable_count = has_reliable_revenue.sum()
        neutral_count = len(df) - reliable_count
        sweet_spot_count = (has_reliable_revenue & (revenue_values >= 250_000_000) & (revenue_values < 1_000_000_000)).sum()
        excellent_count = (has_reliable_revenue & (revenue_values >= 1_000_000_000)).sum()
        
        print(f"[INFO] Data-driven size scoring applied (based on engaged customer analysis):")
        print(f"  - {reliable_count} customers with reliable revenue data")
        print(f"  - {sweet_spot_count} customers in sweet spot ($250M-$1B)")
        print(f"  - {excellent_count} customers in excellent tier ($1B+)")
        print(f"  - {neutral_count} customers with neutral default score (0.5)")
    else:
        # If no revenue data at all, use neutral scoring for all customers
        df["size_score"] = 0.5  # Neutral default for all
        print("[INFO] No revenue data available - using neutral size scores for all customers")

    # Final ICP weighted score 0-100 (pain criteria removed)
    df["ICP_score_raw"] = (
        df["vertical_score"] * WEIGHTS["vertical"]
        + df["size_score"] * WEIGHTS["size"]
        + df["adoption_score"] * WEIGHTS["adoption"]
        + df["relationship_score"] * WEIGHTS["relationship"]
    ) * 100

    # Monotonic normalization for bell-curve shape
    ranks = df['ICP_score_raw'].rank(method='first')
    n = len(ranks)
    p = (ranks - 0.5) / n
    z = np.sqrt(2) * norm.ppf(p) # Use scipy's norm.ppf for inverse cdf
    
    df['ICP_score'] = (50 + 15 * z).clip(0, 100)

    # Create the target variable for the optimizer as per user specification
    df['Total Hardware + Consumable Revenue'] = df['Total Hardware Revenue'].fillna(0) + df['Total Consumable Revenue'].fillna(0)

    return df

# --------------------------- #
# 5.  Visual builder
# --------------------------- #
def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def build_visuals(df: pd.DataFrame):
    # 1 Histogram ICP
    plt.figure()
    plt.hist(df["ICP_score"].dropna(), bins=30)
    plt.title("Distribution of ICP Scores")
    plt.xlabel("ICP Score")
    plt.ylabel("Number of Customers")
    save_fig("vis1_icp_hist.png")

    # 2 GP24 by vertical
    plt.figure()
    df.groupby("Industry")["GP24"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total GP24 by Vertical (Top 10)")
    plt.ylabel("GP24 ($)")
    save_fig("vis2_gp24_vertical.png")

    # 3 printer count vs GP24
    plt.figure()
    plt.scatter(df["printer_count"], df["GP24"])
    plt.title("Printer Count vs GP24")
    plt.xlabel("Printer Count")
    plt.ylabel("GP24 ($)")
    save_fig("vis3_printers_gp24.png")

    # 4 GP24 by CAD tier
    # Check if 'cad_tier' column exists and is categorical with categories
    if 'cad_tier' in df.columns and hasattr(df['cad_tier'], 'cat'):
        # Ensure there are observed categories to plot
        if not df['cad_tier'].cat.categories.empty and not df['cad_tier'].isnull().all():
            plt.figure()
            df.boxplot(column="GP24", by="cad_tier")
            plt.title("GP24 by CAD Tier")
            plt.suptitle("")
            plt.xlabel("CAD Tier")
            plt.ylabel("GP24 ($)")
            save_fig("vis4_gp24_cadtier.png")
        else:
            print("[INFO] Skipping 'GP24 by CAD Tier' visual: No data to plot.")
    else:
        print("[INFO] Skipping 'GP24 by CAD Tier' visual: 'cad_tier' column not suitable for plotting.")

    # 5 mean ICP by vertical
    plt.figure()
    df.groupby("Industry")["ICP_score"].mean().nlargest(10).plot(kind="bar")
    plt.title("Average ICP Score by Vertical (Top 10)")
    plt.ylabel("Mean ICP Score")
    save_fig("vis5_icp_vertical.png")

    # 6 scaling accounts per vertical
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

    # 7 CAD spend vs ICP
    if LICENSE_COL in df.columns:
        plt.figure()
        plt.scatter(df[LICENSE_COL], df["ICP_score"])
        plt.title("CAD Spend vs ICP Score")
        plt.xlabel("Total Software License Revenue ($)")
        plt.ylabel("ICP Score")
        save_fig("vis7_cad_icp.png")

    # 8 printer count vs ICP
    plt.figure()
    plt.scatter(df["printer_count"], df["ICP_score"])
    plt.title("Printer Count vs ICP Score")
    plt.xlabel("Printer Count")
    plt.ylabel("ICP Score")
    save_fig("vis8_printers_icp.png")

    # 9 Revenue24 by vertical
    plt.figure()
    df.groupby("Industry")["Revenue24"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total Revenue24 by Vertical (Top 10)")
    plt.ylabel("Revenue24 ($)")
    save_fig("vis9_rev24_vertical.png")

    # 10 customer count by CAD tier
    plt.figure()
    df["cad_tier"].value_counts().plot(kind="bar")
    plt.title("Customer Count by CAD Tier")
    plt.xlabel("CAD Tier")
    plt.ylabel("Number of Customers")
    save_fig("vis10_customers_cadtier.png")

# --------------------------- #
# 6.  Main driver
# --------------------------- #
def main():
    check_files_exist()
    print("Loading spreadsheets…")
    # Skip loading assets since we're not using it for industry enrichment anymore
    # assets = load_assets()
    customers = load_customers()
    sales = load_sales()

    print("Aggregating GP24…")
    gp24 = aggregate_gp24(sales)

    print("Merging data sets…")
    revenue = load_revenue()
    # Pass empty dataframe for assets since we don't need it
    empty_assets = pd.DataFrame(columns=["key"])
    master = merge_master(empty_assets, customers, gp24, revenue)

    print("Engineering features & scores…")
    scored = engineer_features(master)

    out_cols = [
        "Customer ID",
        "Company Name",
        "Industry",
        "printer_count",
        "Big Box Count",
        "Small Box Count",
        LICENSE_COL,
        "cad_tier",
        "GP24",
        "Revenue24",
        "Total Hardware + Consumable Revenue",
        "ICP_score",
        "ICP_score_raw",
        "vertical_score",
        "size_score",
        "adoption_score",
        "relationship_score"
    ]
    
    # Add revenue columns to output if they exist
    revenue_cols_to_check = ['Total Consumable Revenue', 'Total SaaS Revenue', 'Total Maintenance Revenue']
    for col in revenue_cols_to_check:
        if col in scored.columns:
            out_cols.append(col)

    scored[out_cols].to_csv("icp_scored_accounts.csv", index=False)
    print("✅  Saved icp_scored_accounts.csv")

    print("Creating visualisations…")
    build_visuals(scored)
    print("✅  Saved 10 PNG charts (vis1_… vis10_…).")

    print("All done.")

# --------------------------- #
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n⚠ An error occurred – details below.\n")
        raise
