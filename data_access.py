"""
Data access layer for Azure SQL.

Loads connection settings from environment (.env) and exposes helpers to
pull customer master, profit (GP+Term_GP) since 2023 and per-quarter,
and asset/seat aggregates joined to item_rollup/Goal via items_category_limited
and analytics_product_tags.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; env vars may be provided by the host
    pass


def _build_connection_url() -> str:
    """
    Build an SQLAlchemy connection URL for Azure SQL using pyodbc.

    Supports either SQL auth (AZSQL_USER/AZSQL_PWD) or interactive AAD
    (when no user/pass provided) via the ODBC Driver 18 parameters.
    """
    server = os.getenv("AZSQL_SERVER", "").strip()
    database = os.getenv("AZSQL_DB", "").strip()
    user = os.getenv("AZSQL_USER", "").strip()
    pwd = os.getenv("AZSQL_PWD", "").strip()

    if not server or not database:
        raise RuntimeError("Missing AZSQL_SERVER or AZSQL_DB environment variables")

    driver = "ODBC Driver 18 for SQL Server"

    if user and pwd:
        # SQL authentication using DSN-less ODBC connection to safely handle special characters
        from urllib.parse import quote_plus
        odbc = (
            "DRIVER={" + driver + "};SERVER=" + server + ";DATABASE=" + database + ";" +
            "UID=" + user + ";PWD=" + pwd + ";Encrypt=yes;TrustServerCertificate=no"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"
    else:
        # AAD interactive or MSI — rely on ODBC authentication parameter
        # Note: some environments require 'Authentication=ActiveDirectoryInteractive'
        # and omit UID/PWD entirely.
        return (
            f"mssql+pyodbc://@{server}/{database}?"
            f"driver={driver.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=no"
            f"&Authentication=ActiveDirectoryInteractive"
        )


def get_engine():
    """Create and return a SQLAlchemy engine for Azure SQL."""
    url = _build_connection_url()
    engine = create_engine(url, fast_executemany=True)
    return engine


SINCE_DATE = "2023-01-01"


def get_customers_since_2023(engine=None) -> pd.DataFrame:
    """
    Distinct customers seen in SalesLog since 2023-01-01.

    Returns columns:
      - Customer ID
      - Company Name
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT DISTINCT
            s.CompanyId       AS [Customer ID],
            s.[New Business]  AS [Company Name]
        FROM dbo.table_saleslog_detail s
        WHERE s.Rec_Date >= :since_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_profit_since_2023_by_goal(engine=None) -> pd.DataFrame:
    """
    Profit (GP + Term_GP) since 2023-01-01 grouped by customer and Goal.

    Returns: [Customer ID, Goal, Profit_Since_2023]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            t.Goal      AS Goal,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit_Since_2023
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        LEFT JOIN dbo.analytics_product_tags t
            ON icl.Item_Rollup = t.item_rollup
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, t.Goal
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_profit_since_2023_by_rollup(engine=None) -> pd.DataFrame:
    """
    Profit (GP + Term_GP) since 2023-01-01 grouped by customer and item_rollup (with Goal).

    Returns: [Customer ID, item_rollup, Goal, Profit_Since_2023]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            icl.Item_Rollup AS item_rollup,
            t.Goal      AS Goal,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit_Since_2023
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        LEFT JOIN dbo.analytics_product_tags t
            ON icl.Item_Rollup = t.item_rollup
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, icl.Item_Rollup, t.Goal
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_quarterly_profit_by_goal(engine=None) -> pd.DataFrame:
    """
    Quarterly Profit (GP + Term_GP) since 2023-01-01 grouped by customer and Goal.

    Returns: [Customer ID, Quarter, Goal, Profit]
             Quarter format: YYYYQn (e.g., 2024Q3)
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date)) AS [Quarter],
            t.Goal AS Goal,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        LEFT JOIN dbo.analytics_product_tags t
            ON icl.Item_Rollup = t.item_rollup
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date)), t.Goal
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_assets_and_seats(engine=None) -> pd.DataFrame:
    """
    Asset aggregates per customer, item_rollup, and Goal with seats.

    Returns: [Customer ID, item_rollup, Goal, asset_count, seats_sum,
              active_assets, first_purchase_date, last_expiration_date]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            p.Customer_Internal_Id AS [Customer ID],
            p.item_rollup,
            t.Goal,
            COUNT(*) AS asset_count,
            SUM(COALESCE(p.Number_of_Seats, 0)) AS seats_sum,
            SUM(CASE WHEN p.Status = 'Active' AND (p.Expires IS NULL OR p.Expires >= GETDATE()) THEN 1 ELSE 0 END) AS active_assets,
            MIN(p.Purchase_Date) AS first_purchase_date,
            MAX(p.Expires) AS last_expiration_date
        FROM dbo.table_Product_Info_cleaned_headers p
        LEFT JOIN dbo.analytics_product_tags t
            ON p.item_rollup = t.item_rollup
        GROUP BY p.Customer_Internal_Id, p.item_rollup, t.Goal
        """
    )
    return pd.read_sql(sql, engine)
