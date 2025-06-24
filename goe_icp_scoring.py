"""
goe_icp_scoring.py
---------------------------------
End-to-end script to score GoEngineer Digital-Manufacturing accounts
and generate key visuals.

Files expected in the SAME directory:
  1) TR - All SSYS Customer Assets - Customer Segmentation.xlsx
  2) JY - Customer Analysis - Customer Segmentation.xlsx
  3) TR - Master Sales Log - Customer Segmentation.xlsx
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

# --------------------------- #
# 0.  CONFIG – file names & weights
# --------------------------- #
ASSET_FILE   = "TR - All SSYS Customer Assets - Customer Segmentation.xlsx"
CUSTOMER_FILE = "JY - Customer Analysis - Customer Segmentation.xlsx"
SALES_FILE   = "TR - Master Sales Log - Customer Segementation.xlsx"
REVENUE_FILE = "customer_revenue_analysis.xlsx"  # New revenue data file

# Business weights (sum = 1.0) - Pain criteria removed
WEIGHTS = dict(vertical=0.333,  # 0.30 / 0.90 = 0.333
               size=0.222,      # 0.20 / 0.90 = 0.222  
               adoption=0.278,  # 0.25 / 0.90 = 0.278
               relationship=0.167)  # 0.15 / 0.90 = 0.167

VERTICAL_WEIGHTS = {
    "aerospace & defense": 1.00,
    "aerospace": 1.00,
    "automotive & transportation": 0.90,
    "automotive": 0.90,
    "medical devices & life sci.": 0.85,
    "medical devices": 0.85,
    "high tech": 0.80,
    "consumer / cpg": 0.80,
    "consumer goods": 0.80,
    "industrial machinery": 0.70,
    "government": 0.75,
}



LICENSE_COL = "Total Software License Revenue"

# --------------------------- #
# 1.  Utility helpers
# --------------------------- #
def clean_name(x: str) -> str:
    """Aggressive company-name normaliser."""
    if pd.isna(x):
        return ""
    x = str(x).lower()
    junk = {",", ".", "&", "  "}
    for j in junk:
        x = x.replace(j, " ")
    return " ".join(x.split())

def check_files_exist():
    required_files = (ASSET_FILE, CUSTOMER_FILE, SALES_FILE)
    for f in required_files:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] Cannot find '{f}' in current directory.")
    
    # Revenue file is optional
    if not os.path.exists(REVENUE_FILE):
        print(f"[INFO] Revenue file '{REVENUE_FILE}' not found. Will use printer-count-based size scoring.")

def load_revenue() -> pd.DataFrame:
    """Load revenue data if available"""
    if not os.path.exists(REVENUE_FILE):
        return pd.DataFrame()  # Return empty dataframe if file doesn't exist
    
    try:
        df = pd.read_excel(REVENUE_FILE)
        df["key"] = df["Compnay Name"].map(clean_name)  # Note: typo in source column name
        return df[["key", "revenue_exact"]]
    except Exception as e:
        print(f"[WARN] Error loading revenue file: {e}")
        return pd.DataFrame()

# --------------------------- #
# 2.  Load & clean data sets
# --------------------------- #
def load_assets() -> pd.DataFrame:
    df = pd.read_excel(ASSET_FILE)
    # expect: Customer ID, Company Name, Big-/Small-box counts, Industry
    needed = {"Customer ID", "Company Name", "Industry (New)"}
    if not needed.issubset(df.columns):
        print("[WARN] Missing some expected columns in assets file.")
    df["key"] = df["Company Name"].map(clean_name)
    return df

def load_customers() -> pd.DataFrame:
    df = pd.read_excel(CUSTOMER_FILE)
    needed = {
        "Customer ID",
        "Company Name",
        "Industry",
        "Big Box Count",
        "Small Box Count",
        LICENSE_COL,
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
    # Some industry enrichment if missing
    master = master.merge(
        assets[["key", "Industry (New)"]], on="key", how="left", suffixes=("","")
    )
    master["Industry"].fillna(master["Industry (New)"], inplace=True)
    master.drop(columns=["Industry (New)"], inplace=True, errors="ignore")
    
    # Merge revenue data if available
    if revenue is not None and len(revenue) > 0:
        master = master.merge(revenue, on="key", how="left", suffixes=("",""))
        master["revenue_exact"] = master["revenue_exact"].fillna(0)
        print(f"[INFO] Merged revenue data for {len(master[master['revenue_exact'] > 0])} customers")
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

    # Relationship tier from software-license revenue
    bins = [-1, 5000, 25000, 100000, np.inf]
    labels = ["Bronze", "Silver", "Gold", "Platinum"]
    df["cad_tier"] = pd.cut(df[LICENSE_COL].fillna(0), bins=bins, labels=labels)

    # Scores for each criterion
    v_lower = df["Industry"].astype(str).str.lower()
    df["vertical_score"] = v_lower.map(VERTICAL_WEIGHTS).fillna(0.5)

    # Size score - use revenue if available, otherwise printer count
    if "revenue_exact" in df.columns and df["revenue_exact"].sum() > 0:
        # Revenue-based scoring ($50M-$500M sweet spot)
        revenue_values = df["revenue_exact"].fillna(0)
        df["size_score"] = np.where(
            revenue_values.between(50000000, 500000000),  # $50M-$500M sweet spot
            1.0,  # Score in sweet spot
            0.6   # Score outside sweet spot
        )
        print("[INFO] Using revenue-based size scoring")
    else:
        # Fallback to printer-count-based scoring
        df["size_score"] = np.where(df["printer_count"].between(2, 3), 1.0, 0.5)
        print("[INFO] Using printer-count-based size scoring")

    tier_map = {"Platinum": 1.0, "Gold": 0.9, "Silver": 0.7, "Bronze": 0.5}
    df["relationship_score"] = df["cad_tier"].map(tier_map).fillna(0.2)

    # Adoption score based on scaling flag
    df["adoption_score"] = df["scaling_flag"].astype(float)

    # Final ICP weighted score 0-100 (pain criteria removed)
    df["ICP_score"] = (
        df["vertical_score"] * WEIGHTS["vertical"]
        + df["size_score"] * WEIGHTS["size"]
        + df["adoption_score"] * WEIGHTS["adoption"]
        + df["relationship_score"] * WEIGHTS["relationship"]
    ) * 100

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
    plt.figure()
    df.boxplot(column="GP24", by="cad_tier")
    plt.title("GP24 by CAD Tier")
    plt.suptitle("")
    plt.xlabel("CAD Tier")
    plt.ylabel("GP24 ($)")
    save_fig("vis4_gp24_cadtier.png")

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
    assets = load_assets()
    customers = load_customers()
    sales = load_sales()

    print("Aggregating GP24…")
    gp24 = aggregate_gp24(sales)

    print("Merging data sets…")
    revenue = load_revenue()
    master = merge_master(assets, customers, gp24, revenue)

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
        "ICP_score",
    ]
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
