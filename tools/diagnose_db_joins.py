"""
Diagnostics for Azure SQL joins used by the ICP pipeline.

Checks:
- Row counts of key tables
- Join coverage: sales -> items_category_limited (by Item_internalid = internalId)
- Mapping coverage: items_category_limited.Item_Rollup -> analytics_product_tags.item_rollup
- Rows with non-null Goal via current join path
- Sample of unmapped keys for quick inspection
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data_access as da


def main():
    engine = da.get_engine()
    with engine.connect() as con:
        n_sales = con.execute(
            text("SELECT COUNT(*) FROM dbo.table_saleslog_detail WHERE Rec_Date >= :d"),
            {"d": da.SINCE_DATE},
        ).scalar()
        n_items = con.execute(text("SELECT COUNT(*) FROM dbo.items_category_limited")).scalar()
        n_tags = con.execute(text("SELECT COUNT(*) FROM dbo.analytics_product_tags")).scalar()
        print(f"rows sales since {da.SINCE_DATE}: {n_sales:,}")
        print(f"rows items_category_limited: {n_items:,}")
        print(f"rows analytics_product_tags: {n_tags:,}")

        joined = con.execute(
            text(
                """
                SELECT COUNT(*)
                FROM dbo.table_saleslog_detail s
                INNER JOIN dbo.items_category_limited icl
                  ON s.Item_internalid = icl.internalId
                WHERE s.Rec_Date >= :d
                """
            ),
            {"d": da.SINCE_DATE},
        ).scalar()
        null_item = con.execute(
            text(
                "SELECT COUNT(*) FROM dbo.table_saleslog_detail WHERE Rec_Date >= :d AND Item_internalid IS NULL"
            ),
            {"d": da.SINCE_DATE},
        ).scalar()
        tot_left, matched = con.execute(
            text(
                """
                SELECT 
                  COUNT(*) AS total_rows,
                  SUM(CASE WHEN icl.internalId IS NOT NULL THEN 1 ELSE 0 END) AS matched_items
                FROM dbo.table_saleslog_detail s
                LEFT JOIN dbo.items_category_limited icl
                  ON s.Item_internalid = icl.internalId
                WHERE s.Rec_Date >= :d
                """
            ),
            {"d": da.SINCE_DATE},
        ).fetchone()
        print(f"sales joined to items (INNER) since {da.SINCE_DATE}: {joined:,}")
        print(f"sales with NULL Item_internalid: {null_item:,}")
        print(f"LEFT join coverage matched/total: {matched:,} / {tot_left:,}")

        # Distinct internal IDs in sales not in items table
        missing_internal = con.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT s.Item_internalid
                  FROM dbo.table_saleslog_detail s
                  WHERE s.Rec_Date >= :d AND s.Item_internalid IS NOT NULL
                  EXCEPT
                  SELECT DISTINCT icl.internalId FROM dbo.items_category_limited icl
                ) x
                """
            ),
            {"d": da.SINCE_DATE},
        ).scalar()
        print(f"distinct Item_internalid in sales not in items table: {missing_internal:,}")

        # items -> tags mapping gaps
        missing_tags = con.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT icl.Item_Rollup FROM dbo.items_category_limited icl
                ) i
                LEFT JOIN (
                  SELECT DISTINCT item_rollup FROM dbo.analytics_product_tags
                ) t ON i.Item_Rollup = t.item_rollup
                WHERE t.item_rollup IS NULL
                """
            )
        ).scalar()
        print(f"distinct Item_Rollup without tag (no Goal): {missing_tags:,}")

        goal_rows = con.execute(
            text(
                """
                SELECT COUNT(*)
                FROM dbo.table_saleslog_detail s
                INNER JOIN dbo.items_category_limited icl
                  ON s.Item_internalid = icl.internalId
                LEFT JOIN dbo.analytics_product_tags t
                  ON icl.Item_Rollup = t.item_rollup
                WHERE s.Rec_Date >= :d AND t.Goal IS NOT NULL
                """
            ),
            {"d": da.SINCE_DATE},
        ).scalar()
        print(f"sales rows with non-null Goal (current path): {goal_rows:,}")

        # Peek at a few missing item IDs
        sample_missing = pd.read_sql(
            text(
                """
                SELECT TOP 10 s.Item_internalid
                FROM dbo.table_saleslog_detail s
                WHERE s.Rec_Date >= :d AND s.Item_internalid IS NOT NULL
                EXCEPT
                SELECT DISTINCT icl.internalId FROM dbo.items_category_limited icl
                """
            ),
            con,
            params={"d": da.SINCE_DATE},
        )
        if not sample_missing.empty:
            print("Example Item_internalid not in items_category_limited:")
            print(sample_missing.to_string(index=False))

        # Column list for visibility
        cols_sales = pd.read_sql(
            text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='table_saleslog_detail'"),
            con,
        )["COLUMN_NAME"].tolist()
        print("saleslog_detail columns:", cols_sales)


if __name__ == "__main__":
    main()
