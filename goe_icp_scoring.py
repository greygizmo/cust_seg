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
from sklearn.preprocessing import MinMaxScaler
import json
from scipy.stats import norm

# Import the centralized scoring logic
from scoring_logic import calculate_scores, LICENSE_COL, DEFAULT_WEIGHTS
# Import the new industry scoring module
from industry_scoring import build_industry_weights, save_industry_weights, load_industry_weights

# ---------------------------
# 0.  CONFIG – file names & weights
# ---------------------------

CUSTOMER_FILE = "JY - Customer Analysis - Customer Segmentation.xlsx"
SALES_FILE   = "TR - Master Sales Log - Customer Segementation.xlsx"
REVENUE_FILE = "enrichment_progress.csv"  # New revenue data file
INDUSTRY_ENRICHMENT_FILE = "TR - Industry Enrichment.csv"  # Updated industry data


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


def check_files_exist():
    """Checks for the existence of required input files and exits if not found."""
    required_files = (CUSTOMER_FILE, SALES_FILE)
    for f in required_files:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] Cannot find '{f}' in current directory.")
    
    # Revenue file is optional but recommended
    if not os.path.exists(REVENUE_FILE):
        print(f"[INFO] Revenue file '{REVENUE_FILE}' not found. Size scoring will be based on printer counts.")


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
# 2.  Load & clean data sets
# ---------------------------

def load_customers() -> pd.DataFrame:
    """Loads the main customer data from the Excel file and applies industry enrichment."""
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers all features required for scoring and analysis.
    - Calculates total printer count.
    - Creates a target variable for the optimization script.
    - Calls the centralized `calculate_scores` function to generate all ICP scores.
    """
    df = df.copy()
    
    # Core counts
    df["printer_count"] = df["Big Box Count"] + df["Small Box Count"]
    df["scaling_flag"] = (df["printer_count"] >= 4).astype(int)
    
    # Create the target variable for the optimizer (historical hardware/consumable revenue)
    df['Total Hardware + Consumable Revenue'] = df['Total Hardware Revenue'].fillna(0) + df['Total Consumable Revenue'].fillna(0)
    
    # Use the centralized scoring logic to calculate all component and final scores
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
    
    # 2: Total GP24 by industry vertical
    plt.figure()
    df.groupby("Industry")["GP24"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total GP24 by Vertical (Top 10)")
    plt.ylabel("GP24 ($)")
    save_fig("vis2_gp24_vertical.png")
    
    # 3: Scatter plot of printer count vs GP24
    plt.figure()
    plt.scatter(df["printer_count"], df["GP24"])
    plt.title("Printer Count vs GP24")
    plt.xlabel("Printer Count")
    plt.ylabel("GP24 ($)")
    save_fig("vis3_printers_gp24.png")
    
    # 4: Box plot of GP24 by CAD tier
    if 'cad_tier' in df.columns and hasattr(df['cad_tier'], 'cat'):
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
    
    # 9: Total Revenue24 by industry vertical
    plt.figure()
    df.groupby("Industry")["Revenue24"].sum().nlargest(10).plot(kind="bar")
    plt.title("Total Revenue24 by Vertical (Top 10)")
    plt.ylabel("Revenue24 ($)")
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
        "Big Box Count",
        "Small Box Count",
        LICENSE_COL,
        "cad_tier",
        "GP24",
        "Revenue24",
        "Total Hardware + Consumable Revenue",
        "ICP_score",
        "ICP_grade",
        "ICP_score_raw",
        "vertical_score",
        "size_score",
        "adoption_score",
        "relationship_score"
    ]
    
    # Dynamically add other software revenue columns to the output if they exist
    revenue_cols_to_check = ['Total Consumable Revenue', 'Total SaaS Revenue', 'Total Maintenance Revenue']
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
    print("✅  Saved icp_scored_accounts.csv")
    
    print("Creating visualisations…")
    build_visuals(scored)
    print("✅  Saved 10 PNG charts (vis1_… vis10_…).")
    
    print("All done.")


# ---------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n⚠ An error occurred – details below.\n")
        raise



