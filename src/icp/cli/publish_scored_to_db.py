"""
Publish an existing icp_scored_accounts.csv into SQL as dbo.customer_icp.

Usage (from repo root):
    set PYTHONPATH=src
    python -m icp.cli.publish_scored_to_db --db db-goeng-icp-prod
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from icp import data_access as da


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish an existing icp_scored_accounts.csv to SQL as dbo.customer_icp"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(ROOT / "data" / "processed" / "icp_scored_accounts.csv"),
        help="Path to scored CSV (default: data/processed/icp_scored_accounts.csv)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("ICP_AZSQL_DB") or "",
        help="Target database name (default: ICP_AZSQL_DB env var)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="dbo",
        help="Target schema (default: dbo)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="customer_icp",
        help="Target table name (default: customer_icp)",
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        default="replace",
        choices=["fail", "replace", "append"],
        help="Behavior if the target table already exists (default: replace)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: scored CSV not found at {csv_path}")
        return

    target_db = args.db.strip()
    if not target_db:
        print("Error: target DB not provided. Set ICP_AZSQL_DB or pass --db.")
        return

    print(f"[INFO] Loading scored accounts from {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: scored CSV is empty; nothing to publish.")
        return

    print(f"[INFO] Connecting to database '{target_db}'...")
    try:
        engine = da.get_engine(database=target_db)
    except ModuleNotFoundError:
        print("Error: sqlalchemy is required to publish to SQL but is not installed.")
        return

    print(
        f"[INFO] Writing {len(df):,} rows to "
        f"{target_db}.{args.schema}.{args.table} (if_exists={args.if_exists})"
    )
    try:
        df.to_sql(
            args.table,
            engine,
            schema=args.schema,
            if_exists=args.if_exists,
            index=False,
            chunksize=5000,
        )
        print(f"[INFO] Successfully wrote to {target_db}.{args.schema}.{args.table}")
    except Exception as exc:
        print(f"Error: failed to write to {target_db}.{args.schema}.{args.table}: {exc}")


if __name__ == "__main__":
    main()
