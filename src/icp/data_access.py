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
from icp.schema import COL_CUSTOMER_ID
try:
    from sqlalchemy import create_engine, text
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    def create_engine(*args, **kwargs):  # type: ignore
        raise ModuleNotFoundError("sqlalchemy is required for database connectivity")

    def text(sql: str):  # type: ignore
        raise ModuleNotFoundError("sqlalchemy is required for database connectivity")

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; env vars may be provided by the host
    pass


def _build_connection_url(database_override: Optional[str] = None) -> str:

    server = (os.getenv("AZSQL_SERVER") or "").strip()
    database = (database_override or os.getenv("AZSQL_DB") or "").strip()
    user = (os.getenv("AZSQL_USER") or "").strip()
    pwd = (os.getenv("AZSQL_PWD") or "").strip()

    if not server or not database:
        raise RuntimeError("Missing AZSQL_SERVER or AZSQL_DB environment variables")

    driver = "ODBC Driver 18 for SQL Server"

    if user and pwd:
        print(f"[DEBUG] Connecting to {server}/{database} using SQL Authentication (User: {user})")
        # SQL authentication using DSN-less ODBC connection to safely handle special characters
        from urllib.parse import quote_plus
        odbc = (
            "DRIVER={" + driver + "};SERVER=" + server + ";DATABASE=" + database + ";" +
            "UID=" + user + ";PWD=" + pwd + ";Encrypt=yes;TrustServerCertificate=no"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"
    else:
        print(f"[DEBUG] Connecting to {server}/{database} using ActiveDirectoryInteractive")
        # AAD interactive or MSI â€” rely on ODBC authentication parameter
        # Note: some environments require 'Authentication=ActiveDirectoryInteractive'
        # and omit UID/PWD entirely.
        return (
            f"mssql+pyodbc://@{server}/{database}?"
            f"driver={driver.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=no"
            f"&Authentication=ActiveDirectoryInteractive"
        )


def get_engine(database: Optional[str] = None):
    """Create and return a SQLAlchemy engine for Azure SQL.

    If `database` is provided, it overrides AZSQL_DB for this engine only.
    """
    url = _build_connection_url(database_override=database)
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
        f"""
        SELECT 
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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
    
    WARNING: This joins on analytics_product_tags. If an item_rollup maps to multiple Goals,
    the profit will be duplicated for each Goal. Use with caution for aggregations.
    
    Returns: [Customer ID, item_rollup, Goal, Profit_Since_2023]
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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


def get_profit_since_2023_by_customer_rollup(engine=None) -> pd.DataFrame:
    """
    Profit (GP + Term_GP) since 2023-01-01 grouped by customer and item_rollup.
    Does NOT join with Goals, ensuring no duplication of GP.

    Returns: [Customer ID, item_rollup, Profit_Since_2023]
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
            icl.Item_Rollup AS item_rollup,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit_Since_2023
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, icl.Item_Rollup
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": SINCE_DATE})


def get_product_tags(engine=None) -> pd.DataFrame:
    """
    Fetch the mapping from item_rollup to Goal.
    
    Returns: [item_rollup, Goal]
    """
    engine = engine or get_engine()
    sql = text("SELECT DISTINCT item_rollup, Goal FROM dbo.analytics_product_tags")
    return pd.read_sql(sql, engine)


def get_quarterly_profit_by_goal(engine=None) -> pd.DataFrame:
    """
    Quarterly Profit (GP + Term_GP) since 2023-01-01 grouped by customer and Goal.

    Returns: [Customer ID, Quarter, Goal, Profit]
             Quarter format: YYYYQn (e.g., 2024Q3)
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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


def get_quarterly_profit_by_rollup(engine=None) -> pd.DataFrame:
    """
    Quarterly Profit (GP + Term_GP) since 2023-01-01 grouped by customer, Goal, and item_rollup.

    Returns: [Customer ID, Quarter, Goal, item_rollup, Profit]
             Quarter format: YYYYQn (e.g., 2024Q3)
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
            CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date)) AS [Quarter],
            t.Goal AS Goal,
            icl.Item_Rollup AS item_rollup,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS Profit
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        LEFT JOIN dbo.analytics_product_tags t
            ON icl.Item_Rollup = t.item_rollup
        WHERE s.Rec_Date >= :since_date
        GROUP BY s.CompanyId, CONCAT(YEAR(s.Rec_Date),'Q', DATEPART(QUARTER, s.Rec_Date)), t.Goal, icl.Item_Rollup
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
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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
        f"""
        SELECT
            s.CompanyId AS [{COL_CUSTOMER_ID}],
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
        f"""
        SELECT
            p.Customer_Internal_Id AS [{COL_CUSTOMER_ID}],
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


def get_tx_for_features(engine=None, months_back: int = 18) -> pd.DataFrame:
    """
    Transaction-like daily aggregates for feature engineering from SalesLog.

    Returns columns suitable for feature builders (lowercase after caller renames):
      - [Customer ID], [date], invoice_id (synthetic per-day ID), item_rollup as product_id
      - Goal, super_division (Software for CAD/CPE/Specialty Software, else Hardware)
      - division (Goal proxy), sub_division (item_rollup proxy)
      - net_revenue (using GP+Term_GP as a stable proxy)

    Note: Uses profit as a proxy for spend to avoid requiring external revenue columns.
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            s.CompanyId                                   AS [{COL_CUSTOMER_ID}],
            CAST(s.Rec_Date AS date)                      AS [date],
            CONCAT('D', CONVERT(varchar(10), CAST(s.Rec_Date AS date), 23)) AS invoice_id,
            icl.Item_Rollup                               AS item_rollup,
            t.Goal                                        AS Goal,
            CASE WHEN t.Goal IN ('CAD','CPE','Specialty Software') THEN 'Software' ELSE 'Hardware' END AS super_division,
            t.Goal                                        AS division,
            icl.Item_Rollup                               AS sub_division,
            SUM(COALESCE(s.GP,0) + COALESCE(s.Term_GP,0)) AS net_revenue
        FROM dbo.table_saleslog_detail s
        INNER JOIN dbo.items_category_limited icl
            ON s.Item_internalid = icl.internalId
        LEFT JOIN dbo.analytics_product_tags t
            ON icl.Item_Rollup = t.item_rollup
        WHERE s.Rec_Date >= DATEADD(MONTH, -:months_back, GETDATE())
        GROUP BY s.CompanyId, CAST(s.Rec_Date AS date), icl.Item_Rollup, t.Goal
        """
    )
    return pd.read_sql(sql, engine, params={"months_back": int(months_back)})


def get_customer_headers(engine=None) -> pd.DataFrame:
    """
    Fetch customer header attributes from NetSuite cleaned headers.

    Returns: [Customer ID, entityid, am_sales_rep, AM_Territory, edu_assets]
    """
    engine = engine or get_engine()
    sql = text(
        f"""
        SELECT
            CAST(c.internalid AS varchar(50))      AS [{COL_CUSTOMER_ID}],
            c.entityid,
            c.am_sales_rep,
            c.CAD_Territory,
            c.salesrep_Name AS cre_sales_rep,
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
        f"""
        SELECT
            CAST(ch.[Company ID] AS varchar(50)) AS [{COL_CUSTOMER_ID}],
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
        f"""
        SELECT
            CAST(ch.[Company ID] AS varchar(50)) AS [{COL_CUSTOMER_ID}],
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
            f"""
            SELECT
                CAST(c.internalid AS varchar(50)) AS [{COL_CUSTOMER_ID}],
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
                f"""
                SELECT
                    CAST(c.internalid AS varchar(50)) AS [{COL_CUSTOMER_ID}],
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
                COL_CUSTOMER_ID,'ShippingAddr1','ShippingAddr2','ShippingCity','ShippingState','ShippingZip','ShippingCountry'
            ])
def get_sales_detail_since_2022(engine=None) -> pd.DataFrame:
    """
    Alias for get_tx_for_features to support legacy calls.
    """
    return get_tx_for_features(engine, months_back=36)
