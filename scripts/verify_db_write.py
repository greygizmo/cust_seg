import os
import sys
import pandas as pd
from sqlalchemy import text
sys.path.append("src")
import icp.data_access as da

def main():
    icp_db = os.getenv("ICP_AZSQL_DB", "").strip()
    if not icp_db:
        print("ICP_AZSQL_DB is not set.")
        sys.exit(1)
    
    print(f"Checking database: {icp_db}")
    try:
        engine = da.get_engine(database=icp_db)
        with engine.connect() as conn:
            # Check if table exists and get row count
            result = conn.execute(text("SELECT COUNT(*) FROM dbo.customer_icp")).scalar()
            print(f"Row count in dbo.customer_icp: {result}")
            
            # Check a sample row's run_ts_utc
            # We need to see if run_ts_utc column exists and what the value is
            try:
                ts = conn.execute(text("SELECT TOP 1 run_ts_utc FROM dbo.customer_icp")).scalar()
                print(f"Sample run_ts_utc: {ts}")
            except Exception as e:
                print(f"Could not read run_ts_utc: {e}")
                
    except Exception as e:
        print(f"Database check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
