"""Data loading and assembly functions."""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

import icp.data_access as da
from icp.schema import (
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
    COL_INDUSTRY_SUBLIST,
    canonicalize_customer_id,
)
from icp.etl.cleaner import clean_name
from icp.features.engineering import _attach_helper_frame

# TODO: Move these to configuration
ROOT = Path(__file__).resolve().parents[3]
INDUSTRY_ENRICHMENT_FILE = ROOT / "data" / "raw" / "TR - Industry Enrichment.csv"
ASSET_WEIGHTS_FILE = ROOT / "artifacts" / "weights" / "asset_rollup_weights.json"
REVENUE_FILE = ROOT / "enrichment_progress.csv" # Assuming this is the file name based on usage

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
        for col in [COL_COMPANY_NAME, "company_name", "Compnay Name", "company name"]:
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
        
        print("[INFO] Revenue data prioritization results:")
        print(f"  - sec_match: {sec_count} customers")
        print(f"  - pdl_estimate: {pdl_count} customers")
        print(f"  - fmp_match (valid): {fmp_count} customers") 
        print(f"  - heuristic_estimate (discarded): {heuristic_count} customers")
        print(f"  - Total reliable revenue records: {len(df)}")
        
        return df[["key", "reliable_revenue", "revenue_source"]]
    except Exception as e:
        print(f"\\nAn error occurred - see traceback above: {e}\\n")
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
        for col in [COL_CUSTOMER_ID, "ID", "customer_id", "id"]:
            if col in df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            print(f"[WARN] No Customer ID column found in industry enrichment. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Rename to standard column name
        if customer_id_col != COL_CUSTOMER_ID:
            df = df.rename(columns={customer_id_col: COL_CUSTOMER_ID})
        
        # Validate required columns
        required_cols = [COL_CUSTOMER_ID, COL_INDUSTRY, COL_INDUSTRY_SUBLIST]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing required columns in industry enrichment: {missing_cols}")
            return pd.DataFrame()

        print(f"[INFO] Loaded industry enrichment data for {len(df)} customers")

        # Include Reasoning and CRM Full Name (Customer) if they exist
        cols_to_return = [COL_CUSTOMER_ID, COL_INDUSTRY, COL_INDUSTRY_SUBLIST]
        if "Reasoning" in df.columns:
            cols_to_return.append("Reasoning")
            print("[INFO] Including 'Reasoning' column from industry enrichment")
        if "Cleaned Customer Name" in df.columns:
            cols_to_return.append("Cleaned Customer Name")
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
    df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])
    enrichment_df[COL_CUSTOMER_ID] = canonicalize_customer_id(enrichment_df[COL_CUSTOMER_ID])
    
    # Merge on Customer ID to update industry data
    updated = df.merge(
        enrichment_df,
        on=COL_CUSTOMER_ID,
        how="left",
        suffixes=("_original", "_enriched")
    )
    
    # Use enriched data where available, fall back to original
    industry_enriched = f"{COL_INDUSTRY}_enriched"
    industry_orig = f"{COL_INDUSTRY}_original"
    sub_enriched = f"{COL_INDUSTRY_SUBLIST}_enriched"
    sub_orig = f"{COL_INDUSTRY_SUBLIST}_original"

    if industry_enriched in updated.columns:
        updated[COL_INDUSTRY] = updated[industry_enriched].fillna(updated.get(industry_orig))
        matches = updated[industry_enriched].notna().sum()
        print(f"[INFO] Updated Industry for {matches} customers")

    if sub_enriched in updated.columns:
        updated[COL_INDUSTRY_SUBLIST] = updated[sub_enriched].fillna(updated.get(sub_orig))
        matches = updated[sub_enriched].notna().sum()
        print(f"[INFO] Updated Industry Sub List for {matches} customers")
    
    # Add Reasoning column if it exists in enrichment data
    if "Reasoning" in enrichment_df.columns:
        updated["Industry_Reasoning"] = updated["Reasoning"]
        reasoning_matches = updated["Industry_Reasoning"].notna().sum()
        print(f"[INFO] Added reasoning for {reasoning_matches} customers")
    else:
        updated["Industry_Reasoning"] = pd.NA
    
    # Secondary match by CRM Full Name for any rows still missing Industry
    if COL_INDUSTRY in updated.columns and 'CRM Full Name' in enrichment_df.columns:
        # Prepare a slim enrichment frame keyed by CRM Full Name
        slim = enrichment_df[['CRM Full Name', COL_INDUSTRY, COL_INDUSTRY_SUBLIST]].rename(
            columns={
                COL_INDUSTRY: 'Industry_by_name',
                COL_INDUSTRY_SUBLIST: 'Industry Sub List_by_name'
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
            if COL_INDUSTRY in updated.columns and 'Industry_by_name' in updated.columns:
                missing = updated[COL_INDUSTRY].isna()
                updated.loc[missing, COL_INDUSTRY] = updated.loc[missing, 'Industry_by_name']
            if COL_INDUSTRY_SUBLIST in updated.columns and 'Industry Sub List_by_name' in updated.columns:
                missing = updated[COL_INDUSTRY_SUBLIST].isna()
                updated.loc[missing, COL_INDUSTRY_SUBLIST] = updated.loc[missing, 'Industry Sub List_by_name']

    # Tertiary fallback: match on cleaned company names (CRM label or standalone company name)
    try:
        if 'Cleaned Customer Name' in enrichment_df.columns:
            name_source_col = 'Cleaned Customer Name'
        elif 'CRM Full Name' in enrichment_df.columns:
            name_source_col = 'CRM Full Name'
        else:
            name_source_col = None

        if name_source_col:
            if COL_COMPANY_NAME in updated.columns:
                updated['_company_match_key'] = updated[COL_COMPANY_NAME].map(clean_name)
            elif 'CRM Full Name' in updated.columns:
                updated['_company_match_key'] = updated['CRM Full Name'].map(clean_name)
            else:
                updated['_company_match_key'] = None

            slim = enrichment_df[[name_source_col, COL_INDUSTRY, COL_INDUSTRY_SUBLIST]].copy()
            if "Reasoning" in enrichment_df.columns:
                slim["Industry_Reasoning_by_name"] = enrichment_df["Reasoning"]
            slim['_company_match_key'] = slim[name_source_col].map(clean_name)
            slim = slim.dropna(subset=['_company_match_key']).drop_duplicates(subset=['_company_match_key'])
            keep_cols = ['_company_match_key', COL_INDUSTRY, COL_INDUSTRY_SUBLIST]
            if "Industry_Reasoning_by_name" in slim.columns:
                keep_cols.append("Industry_Reasoning_by_name")
            updated = updated.merge(
                slim[keep_cols],
                on="_company_match_key",
                how="left",
                suffixes=("", "_name_fallback"),
            )
            if f"{COL_INDUSTRY}_name_fallback" in updated.columns:
                mask_ind = updated[COL_INDUSTRY].isna() & updated[f"{COL_INDUSTRY}_name_fallback"].notna()
                updated.loc[mask_ind, COL_INDUSTRY] = updated.loc[mask_ind, f"{COL_INDUSTRY}_name_fallback"]
            if f"{COL_INDUSTRY_SUBLIST}_name_fallback" in updated.columns:
                mask_sub = updated[COL_INDUSTRY_SUBLIST].isna() & updated[f"{COL_INDUSTRY_SUBLIST}_name_fallback"].notna()
                updated.loc[mask_sub, COL_INDUSTRY_SUBLIST] = updated.loc[mask_sub, f"{COL_INDUSTRY_SUBLIST}_name_fallback"]
            if "Industry_Reasoning_by_name" in updated.columns:
                mask_reason = updated["Industry_Reasoning"].isna() & updated["Industry_Reasoning_by_name"].notna()
                updated.loc[mask_reason, "Industry_Reasoning"] = updated.loc[mask_reason, "Industry_Reasoning_by_name"]
    except Exception as e:
        print(f"[WARN] Name-based industry enrichment failed: {e}")

    # Clean up temporary columns
    drop_exact = {"Reasoning", "Industry_by_name", "Industry Sub List_by_name", "_company_match_key", "Industry_Reasoning_by_name"}
    cols_to_drop = [
        col for col in updated.columns
        if col.endswith(("_original", "_enriched", "_name_fallback")) or col in drop_exact
    ]
    updated = updated.drop(columns=[c for c in cols_to_drop if c in updated.columns])

    return updated

def load_asset_weights():
    override = os.environ.get("ICP_ASSET_WEIGHTS_PATH")
    weights_path = Path(override) if override else ASSET_WEIGHTS_FILE
    try:
        with weights_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print(f"[WARN] Could not load {weights_path}. Using default weight=1.0 for all rollups.")
        return {"focus_goals": ["Printer", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"], "weights": {}}

def assemble_master_from_db() -> pd.DataFrame:
    """Pull data from Azure SQL and assemble a customer-level master table."""
    engine = da.get_engine()

    customers = da.get_customers_since_2023(engine)

    # Canonicalize IDs and derive Company Name from CRM Full Name (strip leading numeric ID)
    if COL_CUSTOMER_ID in customers.columns:
        customers[COL_CUSTOMER_ID] = canonicalize_customer_id(customers[COL_CUSTOMER_ID])
        # Drop invalid IDs immediately
        customers = customers.dropna(subset=[COL_CUSTOMER_ID])
        customers = customers[customers[COL_CUSTOMER_ID].astype(str).str.strip() != ""]
        
    if 'CRM Full Name' in customers.columns:
        customers[COL_COMPANY_NAME] = (
            customers['CRM Full Name']
            .astype(str)
            .str.replace(r'^\d+\s+', '', regex=True)
            .str.strip()
        )

    # Enrich with AM Sales Rep, AM Territory, and EDU assets from customer headers
    try:
        print("  - Fetching customer headers...")
        cust_hdr = da.get_customer_headers(engine)
        print(f"  - Customer headers fetched: {len(cust_hdr)} rows")
        if not cust_hdr.empty:
            if COL_CUSTOMER_ID in cust_hdr.columns:
                cust_hdr[COL_CUSTOMER_ID] = canonicalize_customer_id(cust_hdr[COL_CUSTOMER_ID])
            # Keep only required columns for join
            keep_cols = [c for c in [COL_CUSTOMER_ID,'am_sales_rep','cre_sales_rep','AM_Territory','CAD_Territory','edu_assets'] if c in cust_hdr.columns]
            if keep_cols:
                customers = customers.merge(cust_hdr[keep_cols], on=COL_CUSTOMER_ID, how='left')
    except Exception as e:
        print(f"[WARN] Could not load customer headers: {e}")

    # Identify 'cold' customers (own hardware assets but no recent sales)
    try:
        print("  - Fetching assets for cold customer identification...")
        assets_all = da.get_assets_and_seats(engine)
        print(f"  - Assets fetched: {len(assets_all)} rows")
        # Backfill Goal using product tags when missing
        try:
            tags = da.get_product_tags(engine)
            if not tags.empty:
                tags = tags.dropna(subset=["item_rollup"]).copy()
                tags["item_rollup_norm"] = tags["item_rollup"].astype(str).str.strip().str.lower()
                tag_map = dict(zip(tags["item_rollup_norm"], tags["Goal"]))
                assets_all["item_rollup_norm"] = assets_all["item_rollup"].astype(str).str.strip().str.lower()
                fill = assets_all["Goal"].fillna(assets_all["item_rollup_norm"].map(tag_map))
                assets_all["Goal"] = fill
        except Exception as e:
            print(f"[WARN] Could not backfill asset Goal from tags: {e}")
        assets = assets_all.copy()
        if COL_CUSTOMER_ID in assets.columns:
            assets[COL_CUSTOMER_ID] = canonicalize_customer_id(assets[COL_CUSTOMER_ID])
        # Hardware goals only (Printers, Printer Accessorials, Scanners)
        hw_goals = { 'Printers', 'Printer Accessorials', 'Scanners' }
        if 'Goal' in assets.columns:
            assets_hw = assets[assets['Goal'].isin({g.lower() for g in hw_goals}) | assets['Goal'].isin(hw_goals)]
        else:
            assets_hw = assets
        # Sum assets per customer and select those with >0 assets
        if not assets_hw.empty and 'asset_count' in assets_hw.columns:
            hw_counts = assets_hw.groupby(COL_CUSTOMER_ID)['asset_count'].sum().rename('hw_asset_count')
            warm_ids = set(customers[COL_CUSTOMER_ID].astype(str)) if COL_CUSTOMER_ID in customers.columns else set()
            candidates = hw_counts[hw_counts > 0].reset_index()
            cold_ids = set(candidates[COL_CUSTOMER_ID].astype(str)) - warm_ids
            if cold_ids:
                cold_df = pd.DataFrame({COL_CUSTOMER_ID: sorted(cold_ids)})
                # Add Company Name via entityid from cust_hdr if available
                try:
                    if 'cust_hdr' not in locals() or cust_hdr is None or cust_hdr.empty:
                        cust_hdr = da.get_customer_headers(engine)
                        if not cust_hdr.empty and COL_CUSTOMER_ID in cust_hdr.columns:
                            cust_hdr[COL_CUSTOMER_ID] = canonicalize_customer_id(cust_hdr[COL_CUSTOMER_ID])
                    if isinstance(cust_hdr, pd.DataFrame) and not cust_hdr.empty:
                        cols = [c for c in [COL_CUSTOMER_ID,'entityid','am_sales_rep','cre_sales_rep','AM_Territory','CAD_Territory','edu_assets'] if c in cust_hdr.columns]
                        cold_df = cold_df.merge(cust_hdr[cols], on=COL_CUSTOMER_ID, how='left')
                        if 'entityid' in cold_df.columns:
                            cold_df[COL_COMPANY_NAME] = (
                                cold_df['entityid']
                                .astype(str)
                                .str.replace(r'^\d+\s+', '', regex=True)
                                .str.strip()
                            )
                            cold_df = cold_df.drop(columns=['entityid'])
                except Exception as e:
                    print(f"[WARN] Could not enrich cold customers with headers: {e}")
                # Mark activity segment (warm/cold)
                customers['activity_segment'] = 'warm'
                cold_df['activity_segment'] = 'cold'
                # Union back into customers
                customers = pd.concat([customers, cold_df], ignore_index=True, sort=False)
    except Exception as e:
        print(f"[WARN] Could not derive cold customers: {e}")

    # Enrich with primary contact info (Name, email, phone)
    try:
        print("  - Fetching primary contacts...")
        contacts_rp = da.get_primary_contacts(engine)
        print(f"  - Primary contacts fetched: {len(contacts_rp)} rows")
        if not contacts_rp.empty:
            if COL_CUSTOMER_ID in contacts_rp.columns:
                contacts_rp[COL_CUSTOMER_ID] = canonicalize_customer_id(contacts_rp[COL_CUSTOMER_ID])
            contacts_rp = contacts_rp.dropna(subset=[COL_CUSTOMER_ID]).copy()
            contacts_rp = contacts_rp.drop_duplicates(subset=[COL_CUSTOMER_ID], keep='first')
            # Rename to RP_ designated columns
            rp_cols_map = {
                'Name': 'RP_Primary_Name',
                'email': 'RP_Primary_Email',
                'phone': 'RP_Primary_Phone'
            }
            available = [COL_CUSTOMER_ID] + [c for c in rp_cols_map if c in contacts_rp.columns]
            contacts_rp = contacts_rp[available].rename(columns=rp_cols_map)
            customers = customers.merge(contacts_rp, on=COL_CUSTOMER_ID, how='left')

        # Account-level Primary Contact
        print("  - Fetching account primary contacts...")
        contacts_acct = da.get_account_primary_contacts(engine)
        print(f"  - Account primary contacts fetched: {len(contacts_acct)} rows")
        if not contacts_acct.empty:
            if COL_CUSTOMER_ID in contacts_acct.columns:
                contacts_acct[COL_CUSTOMER_ID] = canonicalize_customer_id(contacts_acct[COL_CUSTOMER_ID])
            contacts_acct = contacts_acct.dropna(subset=[COL_CUSTOMER_ID]).copy()
            contacts_acct = contacts_acct.drop_duplicates(subset=[COL_CUSTOMER_ID], keep='first')
            acct_cols_map = {
                'Name': 'Primary_Contact_Name',
                'email': 'Primary_Contact_Email',
                'phone': 'Primary_Contact_Phone'
            }
            available2 = [COL_CUSTOMER_ID] + [c for c in acct_cols_map if c in contacts_acct.columns]
            contacts_acct = contacts_acct[available2].rename(columns=acct_cols_map)
            customers = customers.merge(contacts_acct, on=COL_CUSTOMER_ID, how='left')

        # Backward-compatible generic contact fields mapped to RP Primary when present
        for src, dst in [
            ('RP_Primary_Name','Name'),
            ('RP_Primary_Email','email'),
            ('RP_Primary_Phone','phone'),
        ]:
            if src in customers.columns and dst not in customers.columns:
                customers[dst] = customers[src]
    except Exception as e:
        print(f"[WARN] Could not load primary contacts: {e}")

    # Enrich with shipping address fields
    try:
        print("  - Fetching shipping addresses...")
        ship = da.get_customer_shipping(engine)
        print(f"  - Shipping addresses fetched: {len(ship)} rows")
        if not ship.empty:
            if COL_CUSTOMER_ID in ship.columns:
                ship[COL_CUSTOMER_ID] = canonicalize_customer_id(ship[COL_CUSTOMER_ID])
            keep_ship = [c for c in [COL_CUSTOMER_ID,'ShippingAddr1','ShippingAddr2','ShippingCity','ShippingState','ShippingZip','ShippingCountry'] if c in ship.columns]
            if keep_ship:
                customers = customers.merge(ship[keep_ship], on=COL_CUSTOMER_ID, how='left')
    except Exception as e:
        print(f"[WARN] Could not load shipping addresses: {e}")

    # Profit aggregates
    print("  - Fetching profit by goal...")
    profit_goal = da.get_profit_since_2023_by_goal(engine)
    print(f"  - Profit by goal fetched: {len(profit_goal)} rows")
    
    print("  - Fetching profit by rollup...")
    profit_rollup = da.get_profit_since_2023_by_rollup(engine)
    print(f"  - Profit by rollup fetched: {len(profit_rollup)} rows")
    # Backfill Goal for rollups using product tags when missing
    try:
        if 'tags' not in locals():
            tags = da.get_product_tags(engine)
        if not profit_rollup.empty and not tags.empty:
            tags = tags.dropna(subset=["item_rollup"]).copy()
            tags["item_rollup_norm"] = tags["item_rollup"].astype(str).str.strip().str.lower()
            tag_map = dict(zip(tags["item_rollup_norm"], tags["Goal"]))
            profit_rollup["item_rollup_norm"] = profit_rollup["item_rollup"].astype(str).str.strip().str.lower()
            profit_rollup["Goal"] = profit_rollup["Goal"].fillna(profit_rollup["item_rollup_norm"].map(tag_map))
    except Exception as e:
        print(f"[WARN] Could not backfill profit rollup Goal from tags: {e}")
    # Rebuild profit_goal from rollup (after backfill) for consistency
    try:
        if not profit_rollup.empty:
            profit_goal = (
                profit_rollup.groupby([COL_CUSTOMER_ID, "Goal"])["Profit_Since_2023"]
                .sum()
                .reset_index()
            )
    except Exception as e:
        print(f"[WARN] Could not rebuild profit_goal from rollup: {e}")
    
    print("  - Fetching quarterly profit by goal...")
    profit_quarterly = da.get_quarterly_profit_by_goal(engine)
    print(f"  - Quarterly profit by goal fetched: {len(profit_quarterly)} rows")
    try:
        print("  - Fetching GP history windows...")
        gp_last90 = da.get_profit_last_days(engine, 90)
        print(f"  - GP last 90 days fetched: {len(gp_last90)} rows")
    except Exception as e:
        print(f"[WARN] Could not fetch GP last 90 days: {e}")
        gp_last90 = pd.DataFrame()
    try:
        gp_monthly12 = da.get_monthly_profit_last_n(engine, 12)
        print(f"  - Monthly GP (12M) fetched: {len(gp_monthly12)} rows")
    except Exception as e:
        print(f"[WARN] Could not fetch monthly GP history: {e}")
        gp_monthly12 = pd.DataFrame()

    # Assets & seats
    # assets = da.get_assets_and_seats(engine) # Already fetched above as assets_all
    assets = assets_all

    # Apply industry enrichment (CSV) to update Industry fields (enrichment is sole source)
    industry_enrichment = load_industry_enrichment()
    if not industry_enrichment.empty:
        customers = apply_industry_enrichment(customers, industry_enrichment)

    goal_columns: list[str] = []
    # Pivot profit by Goal into columns
    if not profit_goal.empty:
        p_goal = (
            profit_goal.pivot_table(index=COL_CUSTOMER_ID, columns="Goal", values="Profit_Since_2023", aggfunc="sum")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        if COL_CUSTOMER_ID in p_goal.columns:
            p_goal[COL_CUSTOMER_ID] = canonicalize_customer_id(p_goal[COL_CUSTOMER_ID])
        goal_columns = [c for c in p_goal.columns if c != COL_CUSTOMER_ID]
    else:
        p_goal = pd.DataFrame()

    # Merge into base
    master = customers.copy()
    if COL_CUSTOMER_ID in master.columns:
        master[COL_CUSTOMER_ID] = canonicalize_customer_id(master[COL_CUSTOMER_ID])
        master = master.merge(p_goal, on=COL_CUSTOMER_ID, how="left")

    # CRE Training subset: only include Training/Services item_rollups that
    # belong to CRE super-division: 'Success Plan' and 'Training'.
    try:
        pr = profit_rollup.copy()
        if COL_CUSTOMER_ID in pr.columns:
            pr[COL_CUSTOMER_ID] = canonicalize_customer_id(pr[COL_CUSTOMER_ID])
        pr["Goal"] = pr["Goal"].astype(str)
        pr["item_rollup"] = pr.get("item_rollup", "").astype(str)
        # Normalize goal label to match downstream label mapping
        def _norm_goal(g: str) -> str:
            return str(g).strip()
        pr["Goal_norm"] = pr["Goal"].map(_norm_goal)
        allowed_train = {"success plan", "training"}
        mask_cre_train = (
            pr["Goal_norm"].str.lower() == "training/services".lower()
        ) & (pr["item_rollup"].str.strip().str.lower().isin(allowed_train))
        if mask_cre_train.any():
            cre_train = (
                pr.loc[mask_cre_train]
                .groupby(COL_CUSTOMER_ID)["Profit_Since_2023"].sum().rename("CRE_Training")
                .reset_index()
            )
            master = master.merge(cre_train, on=COL_CUSTOMER_ID, how="left")
            master["CRE_Training"] = master.get("CRE_Training", 0).fillna(0.0)
        else:
            master["CRE_Training"] = 0.0
    except Exception as e:
        print(f"[WARN] Could not compute CRE_Training from rollups: {e}")
        master["CRE_Training"] = 0.0

    # Compute total profit since 2023 across all Goals using a non-overlapping source
    # Previous method summed goal columns which caused overcounting if items mapped to multiple goals.
    try:
        print("  - Fetching profit by customer/rollup (no goal join) for accurate totals...")
        profit_clean = da.get_profit_since_2023_by_customer_rollup(engine)
        if not profit_clean.empty:
            if COL_CUSTOMER_ID in profit_clean.columns:
                profit_clean[COL_CUSTOMER_ID] = canonicalize_customer_id(profit_clean[COL_CUSTOMER_ID])
            total_gp = profit_clean.groupby(COL_CUSTOMER_ID)["Profit_Since_2023"].sum().rename("Profit_Since_2023_Total")
            master = master.merge(total_gp, on=COL_CUSTOMER_ID, how="left")
            master["Profit_Since_2023_Total"] = master["Profit_Since_2023_Total"].fillna(0.0)
        else:
            master["Profit_Since_2023_Total"] = 0.0
    except Exception as e:
        print(f"[WARN] Could not calculate accurate total profit: {e}")
        master["Profit_Since_2023_Total"] = 0.0

    # Derive quarterly totals per customer (LastQ and T4Q)
    # Always use the total quarterly query (no goal join) to avoid overcounting
    print("  - Fetching quarterly profit total (no goal join)...")
    pq = da.get_quarterly_profit_total(engine)
    
    if isinstance(pq, pd.DataFrame) and not pq.empty:
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
        # Sum profit per customer, per quarter (already unique per customer/quarter from query)
        cust_q = pq.groupby([COL_CUSTOMER_ID, "_qkey"])['Profit'].sum().reset_index()
        # Canonicalize Customer ID strings to match master
        cust_q[COL_CUSTOMER_ID] = canonicalize_customer_id(cust_q[COL_CUSTOMER_ID])
        # Determine current and completed quarter keys
        now_ts = pd.Timestamp.now()
        current_qkey = now_ts.year * 10 + ((now_ts.month - 1)//3 + 1)
        latest_qkey = cust_q['_qkey'].max()
        print(f"[INFO] Latest quarter key detected: {latest_qkey}")
        # This quarter (partial)
        thisq = cust_q[cust_q['_qkey'] == current_qkey].set_index(COL_CUSTOMER_ID)["Profit"].rename("Profit_ThisQ_Total")
        master = master.merge(thisq.reset_index(), on=COL_CUSTOMER_ID, how="left")
        # Completed quarters exclude current quarter
        completed_keys = sorted([k for k in cust_q['_qkey'].unique() if k < current_qkey])
        # Trailing 4 completed quarters per customer
        if completed_keys:
            t4_keys = completed_keys[-4:]
            t4q = cust_q[cust_q['_qkey'].isin(t4_keys)].groupby(COL_CUSTOMER_ID)["Profit"].sum().rename("Profit_T4Q_Total")
            master = master.merge(t4q.reset_index(), on=COL_CUSTOMER_ID, how="left")
        else:
            master["Profit_T4Q_Total"] = 0.0
        # Previous quarter per customer (for QoQ growth) using completed quarters only
        if len(completed_keys) >= 2:
            last_completed = completed_keys[-1]
            prev_completed = completed_keys[-2]
            lastq_comp = cust_q[cust_q['_qkey'] == last_completed].set_index(COL_CUSTOMER_ID)["Profit"].rename("_tmp_LastComp")
            prevq_comp = cust_q[cust_q['_qkey'] == prev_completed].set_index(COL_CUSTOMER_ID)["Profit"].rename("Profit_PrevQ_Total")
            master = master.merge(prevq_comp.reset_index(), on=COL_CUSTOMER_ID, how="left")
            tmp = lastq_comp.reset_index()
            master = master.merge(tmp, on=COL_CUSTOMER_ID, how="left")
            prev_safe = master["Profit_PrevQ_Total"].fillna(0)
            last_safe = master["_tmp_LastComp"].fillna(0)
            denom = prev_safe.replace(0, np.nan)
            master["Profit_QoQ_Growth"] = ((last_safe - prev_safe) / denom).fillna(0.0)
            master["Profit_QoQ_Delta"] = (last_safe - prev_safe)
        else:
            master["Profit_PrevQ_Total"] = 0.0
            master["Profit_QoQ_Growth"] = 0.0
            master["Profit_QoQ_Delta"] = 0.0
    else:
        master["Profit_LastQ_Total"] = 0.0
        master["Profit_T4Q_Total"] = 0.0
        master["Profit_PrevQ_Total"] = 0.0
        master["Profit_QoQ_Growth"] = 0.0

    # Aggregate assets/seats totals and per-goal seats for filters/signals
    if isinstance(assets, pd.DataFrame) and not assets.empty:
        a = assets.copy()
        # Normalize types
        if COL_CUSTOMER_ID in a.columns:
            a[COL_CUSTOMER_ID] = canonicalize_customer_id(a[COL_CUSTOMER_ID])
        for _dc in ["first_purchase_date", "last_purchase_date", "last_expiration_date"]:
            if _dc in a.columns:
                a[_dc] = pd.to_datetime(a[_dc], errors="coerce")

        # Compute aggregates separately to avoid dtype issues
        parts = []
        if 'active_assets' in a.columns:
            s = a.groupby(COL_CUSTOMER_ID)['active_assets'].sum().rename('active_assets_total')
            parts.append(s)
        if 'seats_sum' in a.columns:
            s = a.groupby(COL_CUSTOMER_ID)['seats_sum'].sum().rename('seats_sum_total')
            parts.append(s)
        if 'first_purchase_date' in a.columns:
            s = a.groupby(COL_CUSTOMER_ID)['first_purchase_date'].min().rename('EarliestPurchaseDate')
            parts.append(s)
        
        if parts:
            agg = pd.concat(parts, axis=1).reset_index()
            master = master.merge(agg, on=COL_CUSTOMER_ID, how="left")

    # Attach helper frames for downstream feature engineering (adoption/relationship, ALS, momentum)
    try:
        _attach_helper_frame(master, "_assets_raw", assets_all if "assets_all" in locals() else pd.DataFrame())
    except Exception as e:
        print(f"[WARN] Could not attach assets frame: {e}")
    try:
        _attach_helper_frame(master, "_profit_rollup_raw", profit_rollup if "profit_rollup" in locals() else pd.DataFrame())
    except Exception as e:
        print(f"[WARN] Could not attach profit rollup frame: {e}")
    try:
        _attach_helper_frame(master, "_profit_goal_raw", profit_goal if "profit_goal" in locals() else pd.DataFrame())
    except Exception as e:
        print(f"[WARN] Could not attach profit goal frame: {e}")
    try:
        _attach_helper_frame(master, "_gp_last90", gp_last90 if "gp_last90" in locals() else pd.DataFrame())
    except Exception as e:
        print(f"[WARN] Could not attach GP last90 frame: {e}")
    try:
        _attach_helper_frame(master, "_gp_monthly12", gp_monthly12 if "gp_monthly12" in locals() else pd.DataFrame())
    except Exception as e:
        print(f"[WARN] Could not attach monthly GP frame: {e}")

    return master
