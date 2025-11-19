import os
import sys
import pandas as pd
from sqlalchemy import text
sys.path.append("src")
import icp.data_access as da
from icp.schema import COL_CUSTOMER_ID

def main():
    icp_db = os.getenv("ICP_AZSQL_DB", "").strip()
    if not icp_db:
        print("ICP_AZSQL_DB is not set.")
        sys.exit(1)
        
    scored_path = "data/processed/icp_scored_accounts.csv"
    print(f"Loading {scored_path}...")
    scored = pd.read_csv(scored_path)
    
    # Replicate logic from score_accounts.py
    out_cols = scored.columns.tolist() # In the real script this is filtered, but here we take all
    
    print(f"[INFO] Writing scored accounts to {icp_db}.dbo.customer_icp")
    try:
        engine = da.get_engine(database=icp_db)
        
        # Ensure no null IDs before writing
        scored_db = scored.copy()
        scored_db = scored_db.dropna(subset=[COL_CUSTOMER_ID])
        
        # Rename timestamp column if present to avoid SQL reserved word conflicts
        if "run_timestamp_utc" in scored_db.columns:
            scored_db = scored_db.rename(columns={"run_timestamp_utc": "run_ts_utc"})
        
        # Explicitly drop the table first to avoid schema conflicts (e.g. timestamp columns)
        with engine.connect() as conn:
            print("Dropping table...")
            conn.execute(text("DROP TABLE IF EXISTS dbo.customer_icp"))
            conn.commit()

        print("Writing to SQL...")
        scored_db.to_sql(
            "customer_icp",
            engine,
            schema="dbo",
            if_exists="replace",
            index=False,
            chunksize=5000,
        )
        print("[INFO] Wrote scored accounts to dbo.customer_icp")
    except Exception as e:
        print(f"[WARN] Failed to write scored accounts to database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
