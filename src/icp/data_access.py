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
        # AAD interactive or MSI â€” rely on ODBC authentication parameter
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
      - CRM Full Name (New_Business) — e.g., "439775 Compusult Limited"
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT 
            s.CompanyId AS [Customer ID],
            MAX(CAST(s.New_Business AS varchar(500))) AS [CRM Full Name]
        FROM dbo.table_saleslog_detail s
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


# Removed master/fallback industry access per requirements: enrichment CSV is sole source of industry data
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


def get_quarterly_profit_total(engine=None) -> pd.DataFrame:
    """
    Quarterly Profit (GP + Term_GP) since 2023-01-01 grouped by customer (all goals combined).

    Returns: [Customer ID, Quarter, Profit]
    Quarter format: YYYYQn (e.g., 2024Q3)
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date)) AS [Quarter],
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit
        FROM dbo.table_saleslog_detail s
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date))
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_profit_last_days(engine=None, days: int = 90) -> pd.DataFrame:
    """
    Sum of GP (GP + Term_GP) in the last N days per customer.

    Returns: [Customer ID, GP_Last_ND]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS GP_Last_ND
        FROM dbo.table_saleslog_detail s
        WHERE s.Rec_Date >= DATEADD(DAY, -:days, GETDATE())
        GROUP BY s.CompanyId
        """
    )
    return pd.read_sql(sql, engine, params={"days": int(days)})


def get_monthly_profit_last_n(engine=None, months: int = 12) -> pd.DataFrame:
    """
    Monthly GP per customer for the last N months.

    Returns: [Customer ID, Year, Month, Profit]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            s.CompanyId AS [Customer ID],
            YEAR(s.Rec_Date) AS [Year],
            MONTH(s.Rec_Date) AS [Month],
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit
        FROM dbo.table_saleslog_detail s
        WHERE s.Rec_Date >= DATEADD(MONTH, -:months, GETDATE())
        GROUP BY s.CompanyId, YEAR(s.Rec_Date), MONTH(s.Rec_Date)
        """
    )
    return pd.read_sql(sql, engine, params={"months": int(months)})


def get_assets_and_seats(engine=None) -> pd.DataFrame:
    """
    Asset aggregates per customer, item_rollup, and Goal with seats.

    Returns: [Customer ID, item_rollup, Goal, asset_count, seats_sum,
              active_assets, first_purchase_date, last_expiration_date]
    """
    engine = engine or get_engine()
    # Switch to the unified all-products view to include hardware assets
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
            MAX(p.Purchase_Date) AS last_purchase_date,
            MAX(p.Expires) AS last_expiration_date
        FROM dbo.table_All_Product_Info_cleaned_headers p
        LEFT JOIN dbo.analytics_product_tags t
            ON p.item_rollup = t.item_rollup
        GROUP BY p.Customer_Internal_Id, p.item_rollup, t.Goal
        """
    )
    return pd.read_sql(sql, engine)


def get_customer_headers(engine=None) -> pd.DataFrame:
    """
    Fetch customer header attributes from NetSuite cleaned headers.

    Returns: [Customer ID, entityid, am_sales_rep, AM_Territory, edu_assets]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            CAST(c.internalid AS varchar(50))      AS [Customer ID],
            c.entityid,
            c.am_sales_rep,
            c.AM_Territory,
            c.edu_assets
        FROM dbo.customer_cleaned_headers c
        """
    )
    return pd.read_sql(sql, engine)


def get_primary_contacts(engine=None) -> pd.DataFrame:
    """
    Fetch primary Hardware contacts from NetSuite contact headers.

    Filters to rows marked as RP Primary Contact.
    Returns: [Customer ID, Name, email, phone]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            CAST(ch.[Company ID] AS varchar(50)) AS [Customer ID],
            ch.[Name] AS [Name],
            ch.[email] AS [email],
            ch.[phone] AS [phone]
        FROM dbo.contact_clean_headers ch
        WHERE (
            TRY_CAST(ch.[RP Primary Contact] AS int) = 1
            OR UPPER(LTRIM(RTRIM(CAST(ch.[RP Primary Contact] AS nvarchar(10))))) IN ('TRUE','YES','Y','1')
        )
        """
    )
    return pd.read_sql(sql, engine)


def get_account_primary_contacts(engine=None) -> pd.DataFrame:
    """
    Fetch account-level primary contacts (general Primary Contact flag).

    Returns: [Customer ID, Name, email, phone]
    """
    engine = engine or get_engine()
    sql = text(
        """
        SELECT
            CAST(ch.[Company ID] AS varchar(50)) AS [Customer ID],
            ch.[Name] AS [Name],
            ch.[email] AS [email],
            ch.[phone] AS [phone]
        FROM dbo.contact_clean_headers ch
        WHERE (
            TRY_CAST(ch.[Primary Contact] AS int) = 1
            OR UPPER(LTRIM(RTRIM(CAST(ch.[Primary Contact] AS nvarchar(10))))) IN ('TRUE','YES','Y','1')
        )
        """
    )
    return pd.read_sql(sql, engine)


def get_customer_shipping(engine=None) -> pd.DataFrame:
    """
    Fetch customer shipping address fields.

    Preferred source: dbo.customer_customerOnly; falls back to dbo.customer_cleaned_headers.

    Returns: [Customer ID, ShippingAddr1, ShippingAddr2, ShippingCity, ShippingState, ShippingZip, ShippingCountry]
    """
    engine = engine or get_engine()
    # Try primary source first
    try:
        sql = text(
            """
            SELECT
                CAST(c.internalid AS varchar(50)) AS [Customer ID],
                c.ShippingAddr1,
                c.ShippingAddr2,
                c.ShippingCity,
                c.ShippingState,
                c.ShippingZip,
                c.ShippingCountry
            FROM dbo.customer_customerOnly c
            """
        )
        return pd.read_sql(sql, engine)
    except Exception:
        # Fallback to cleaned headers if customer_customerOnly not available
        try:
            sql2 = text(
                """
                SELECT
                    CAST(c.internalid AS varchar(50)) AS [Customer ID],
                    c.ShippingAddr1,
                    c.ShippingAddr2,
                    c.ShippingCity,
                    c.ShippingState,
                    c.ShippingZip,
                    c.ShippingCountry
                FROM dbo.customer_cleaned_headers c
                """
            )
            return pd.read_sql(sql2, engine)
        except Exception:
            # No shipping available
            return pd.DataFrame(columns=[
                'Customer ID','ShippingAddr1','ShippingAddr2','ShippingCity','ShippingState','ShippingZip','ShippingCountry'
            ])
