"""Visualization functions for ICP scoring."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from icp.schema import COL_INDUSTRY, COL_CUSTOMER_ID, LICENSE_COL

def save_fig(filename: str, root_path: Path):
    """Saves the current matplotlib figure to reports/figures."""
    out_dir = root_path / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def build_visuals(df: pd.DataFrame, root_path: Path):
    """Generates and saves a standard set of 10 visualizations."""
    
    # 1: Histogram of final ICP Scores (hardware)
    plt.figure()
    if "ICP_score_hardware" in df.columns:
        plt.hist(df["ICP_score_hardware"].dropna(), bins=30)
    plt.title("Distribution of Hardware ICP Scores")
    plt.xlabel("Hardware ICP Score")
    plt.ylabel("Number of Customers")
    save_fig("vis1_icp_hist.png", root_path)
    
    # 2: Total Profit since 2023 by industry vertical
    plt.figure()
    if COL_INDUSTRY in df.columns and "Profit_Since_2023_Total" in df.columns:
        s = df.groupby(COL_INDUSTRY)["Profit_Since_2023_Total"].sum()
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
    save_fig("vis3_printers_gp24.png", root_path)
    
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
            save_fig("vis4_gp24_cadtier.png", root_path)
        else:
            print("[INFO] Skipping 'GP24 by CAD Tier' visual: No data to plot.")
    else:
        print("[INFO] Skipping 'GP24 by CAD Tier' visual: 'cad_tier' column not suitable for plotting.")
    
    # 5: Average hardware ICP score by industry vertical
    if COL_INDUSTRY in df.columns and "ICP_score_hardware" in df.columns:
        s3 = df.groupby(COL_INDUSTRY)["ICP_score_hardware"].mean()
        if not s3.empty:
            s3.nlargest(10).plot(kind="bar")
    save_fig("vis5_icp_vertical.png", root_path)
    
    # 6: Count of scaling accounts (>=4 printers) by vertical
    plt.figure()
    if COL_INDUSTRY in df.columns and COL_CUSTOMER_ID in df.columns:
        s4 = df[df["scaling_flag"] == 1].groupby(COL_INDUSTRY)[COL_CUSTOMER_ID].count()
        if not s4.empty:
            s4.nlargest(10).plot(kind="bar")
        else:
            print("[INFO] Skipping scaling accounts chart: no data.")

    plt.title("Scaling Accounts (>=4 Printers) per Vertical")
    plt.ylabel("Account Count")
    save_fig("vis6_scaling_vertical.png", root_path)
    
    # 7: Scatter plot of CAD spend vs Hardware ICP score
    if LICENSE_COL in df.columns and "ICP_score_hardware" in df.columns:
        plt.figure()
        plt.scatter(df[LICENSE_COL], df["ICP_score_hardware"])
        plt.title("CAD Spend vs Hardware ICP Score")
        plt.xlabel("Total Software License Revenue ($)")
        plt.ylabel("Hardware ICP Score")
        save_fig("vis7_cad_icp.png", root_path)
    
    # 8: Scatter plot of printer count vs Hardware ICP score
    plt.figure()
    if "ICP_score_hardware" in df.columns:
        plt.scatter(df["printer_count"], df["ICP_score_hardware"])
    plt.title("Printer Count vs Hardware ICP Score")
    plt.xlabel("Printer Count")
    plt.ylabel("Hardware ICP Score")
    save_fig("vis8_printers_icp.png", root_path)
    
    # 9: Total Profit since 2023 by industry vertical (duplicate view)
    plt.figure()
    if COL_INDUSTRY in df.columns and "Profit_Since_2023_Total" in df.columns:
        s2 = df.groupby(COL_INDUSTRY)["Profit_Since_2023_Total"].sum()
        if not s2.empty:
            s2.nlargest(10).plot(kind="bar")
        else:
            print("[INFO] Skipping Profit by Vertical (Top 10): no data.")
    # 10: Customer count by CAD tier (deprecated)
    if 'cad_tier' in df.columns:
        plt.figure()
        df["cad_tier"].value_counts().plot(kind="bar")
        plt.title("Customer Count by CAD Tier")
        plt.xlabel("CAD Tier")
        plt.ylabel("Number of Customers")
        save_fig("vis10_customers_cadtier.png", root_path)
    else:
        print("[INFO] Skipping 'Customer Count by CAD Tier' visual: cad_tier deprecated.")
