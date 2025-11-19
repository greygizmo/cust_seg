import os
import sys
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.append("src")

import icp.data_access as da

def test_connection():
    print("--- Testing Database Connection ---")
    try:
        engine = da.get_engine()
        print(f"Engine created: {engine}")
        
        print("Attempting to connect...")
        with engine.connect() as conn:
            print("Connection established!")
            
            print("Running simple query (SELECT 1)...")
            result = conn.execute(text("SELECT 1")).fetchone()
            print(f"Query result: {result}")
            
            print("Running get_customers_since_2023 query (limit 5)...")
            # Manually constructing the query from data_access.py but with TOP 5
            sql = text(
                """
                SELECT TOP 5
                    s.CompanyId AS [Customer ID],
                    MAX(CAST(s.New_Business AS varchar(500))) AS [CRM Full Name]
                FROM dbo.table_saleslog_detail s
                WHERE s.Rec_Date >= '2023-01-01'
                GROUP BY s.CompanyId
                """
            )
            df = pd.read_sql(sql, conn)
            print(f"Data fetched: {len(df)} rows")
            print(df)
            
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
