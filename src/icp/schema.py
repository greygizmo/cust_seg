"""
Canonical column names and schema helpers for ICP scoring.

Use these constants instead of hardcoded strings. The `unify_columns` helper
renames common aliases in-place to canonical names for downstream logic.
"""
from typing import Dict, List
import pandas as pd

# Canonical column names
COL_CUSTOMER_ID = "Customer ID"
COL_COMPANY_NAME = "Company Name"
COL_INDUSTRY = "Industry"
COL_INDUSTRY_SUBLIST = "Industry Sub List"

# Hardware adoption inputs (legacy)
COL_BIG_BOX = "Big Box Count"
COL_SMALL_BOX = "Small Box Count"
COL_HW_REV = "Total Hardware Revenue"
COL_CONS_REV = "Total Consumable Revenue"

# Relationship (software) inputs (fallback)
COL_REL_LICENSE = "Total Software License Revenue"
COL_REL_SAAS = "Total SaaS Revenue"
COL_REL_MAINT = "Total Maintenance Revenue"

# Profit-based features (preferred)
COL_PROFIT_SINCE_2023_TOTAL = "Profit_Since_2023_Total"
COL_RELATIONSHIP_PROFIT = "relationship_profit"

# Derived/feature columns
COL_ADOPTION_ASSETS = "adoption_assets"
COL_ADOPTION_PROFIT = "adoption_profit"


# Aliases mapping: alias -> canonical
ALIASES: Dict[str, str] = {
    # Customer/Company
    "customer id": COL_CUSTOMER_ID,
    "id": COL_CUSTOMER_ID,
    "crm full name": "CRM Full Name",  # preserved for other flows
    "company": COL_COMPANY_NAME,

    # Industry variations
    "industry sublist": COL_INDUSTRY_SUBLIST,
    "industry_sub_list": COL_INDUSTRY_SUBLIST,

    # Hardware/consumables revenue
    "hardware_revenue": COL_HW_REV,
    "consumable_revenue": COL_CONS_REV,

    # Software revenue
    "software_license_revenue": COL_REL_LICENSE,
    "saas_revenue": COL_REL_SAAS,
    "maintenance_revenue": COL_REL_MAINT,

    # Profit
    "profit since 2023 total": COL_PROFIT_SINCE_2023_TOTAL,
    "relationship profit": COL_RELATIONSHIP_PROFIT,
}


def unify_columns(df: pd.DataFrame, extra_aliases: Dict[str, str] | None = None) -> pd.DataFrame:
    """
    Rename common alias columns to their canonical names, in-place-safe (returns copy).

    Args:
        df: input DataFrame
        extra_aliases: optional additional alias mapping

    Returns:
        DataFrame with standardized columns
    """
    mapping = {**ALIASES}
    if extra_aliases:
        mapping.update({k: v for k, v in extra_aliases.items() if isinstance(k, str) and isinstance(v, str)})

    # Build case-insensitive map of current columns
    lower_cols = {c.lower(): c for c in df.columns}
    renames: Dict[str, str] = {}
    for alias_lower, canonical in mapping.items():
        if alias_lower in lower_cols and canonical not in df.columns:
            renames[lower_cols[alias_lower]] = canonical

    if renames:
        return df.rename(columns=renames)
    return df


def canonicalize_customer_id(series: pd.Series) -> pd.Series:
    """Normalize Customer ID values without losing leading zeros."""
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


REQUIRED_MIN_COLUMNS: List[str] = [
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
]


# ---------------------------------------------------------------------------
# Canonical schema for `data/processed/icp_scored_accounts.csv`
# ---------------------------------------------------------------------------

# Identity and ownership
ICP_SCHEMA_IDENTITY: List[str] = [
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    "account_id",
    "activity_segment",
]

ICP_SCHEMA_OWNERSHIP: List[str] = [
    "am_sales_rep",
    "AM_Territory",
    "edu_assets",
]

ICP_SCHEMA_CONTACTS: List[str] = [
    "RP_Primary_Name",
    "RP_Primary_Email",
    "RP_Primary_Phone",
    "Primary_Contact_Name",
    "Primary_Contact_Email",
    "Primary_Contact_Phone",
    # Back-compat generic fields (map to RP Primary)
    "Name",
    "email",
    "phone",
]

ICP_SCHEMA_SHIPPING: List[str] = [
    "ShippingAddr1",
    "ShippingAddr2",
    "ShippingCity",
    "ShippingState",
    "ShippingZip",
    "ShippingCountry",
]

# Scores and grades (division-aware)
ICP_SCHEMA_SCORES: List[str] = [
    "ICP_score_hardware",
    "ICP_grade_hardware",
    "ICP_score_cre",
    "ICP_grade_cre",
]

# Context columns
ICP_SCHEMA_CONTEXT: List[str] = [
    COL_INDUSTRY,
    COL_INDUSTRY_SUBLIST,
    "Industry_Reasoning",
]

# Headline components (adoption/relationship)
ICP_SCHEMA_COMPONENT_SCORES: List[str] = [
    "Hardware_score",
    "Software_score",
]

# Cross-division and CRE/HW balance signals
ICP_SCHEMA_CROSS_DIVISION: List[str] = [
    "cross_division_balance_score",
    "hw_to_sw_cross_sell_score",
    "sw_to_hw_cross_sell_score",
    "training_to_hw_ratio",
    "training_to_cre_ratio",
]

# Profit aggregates
ICP_SCHEMA_PROFIT_AGG: List[str] = [
    "GP_LastQ_Total",
    "GP_PrevQ_Total",
    "GP_QoQ_Growth",
    "GP_T4Q_Total",
    "GP_Since_2023_Total",
]

# Hardware totals (base columns only; dynamic printer subdivisions are appended separately)
ICP_SCHEMA_HARDWARE_TOTALS: List[str] = [
    "Qty_Printers",
    "GP_Printers",
    "Qty_Printer Accessories",
    "GP_Printer Accessories",
    "Qty_Scanners",
    "GP_Scanners",
    "Qty_Geomagic",
    "GP_Geomagic",
]

# CRE / software totals and training subset
ICP_SCHEMA_CRE_TOTALS: List[str] = [
    "Seats_CAD",
    "GP_CAD",
    "Seats_CPE",
    "GP_CPE",
    "Seats_Specialty Software",
    "GP_Specialty Software",
    # Training/Services subset
    "GP_Training/Services_Success_Plan",
    "GP_Training/Services_Training",
]

ICP_SCHEMA_CRE_ADOPTION_REL: List[str] = [
    "cre_adoption_assets",
    "cre_adoption_profit",
    "relationship_profit",
    "cre_relationship_profit",
]

# Operational and portfolio metrics
ICP_SCHEMA_OPERATIONAL: List[str] = [
    "scaling_flag",
    COL_REL_LICENSE,
    "active_assets_total",
    "seats_sum_total",
    "Portfolio_Breadth",
    "EarliestPurchaseDate",
    "LatestPurchaseDate",
    "LatestExpirationDate",
    "Days_Since_First_Purchase",
    "Days_Since_Last_Purchase",
    "Days_Since_Last_Expiration",
]

ICP_SCORED_ACCOUNTS_SCHEMA_GROUPS: Dict[str, List[str]] = {
    "identity": ICP_SCHEMA_IDENTITY,
    "ownership": ICP_SCHEMA_OWNERSHIP,
    "contacts": ICP_SCHEMA_CONTACTS,
    "shipping": ICP_SCHEMA_SHIPPING,
    "scores": ICP_SCHEMA_SCORES,
    "context": ICP_SCHEMA_CONTEXT,
    "component_scores": ICP_SCHEMA_COMPONENT_SCORES,
    "cross_division": ICP_SCHEMA_CROSS_DIVISION,
    "profit_aggregates": ICP_SCHEMA_PROFIT_AGG,
    "hardware_totals": ICP_SCHEMA_HARDWARE_TOTALS,
    "cre_totals": ICP_SCHEMA_CRE_TOTALS,
    "cre_adoption_relationship": ICP_SCHEMA_CRE_ADOPTION_REL,
    "operational": ICP_SCHEMA_OPERATIONAL,
}


def get_icp_scored_accounts_base_order() -> List[str]:
    """
    Return the canonical, grouped base column order for
    `data/processed/icp_scored_accounts.csv`.

    Dynamic feature columns (e.g., spend dynamics, printer subdivision rollups,
    percentiles, *_printers, *_cre, *_pctl) are appended in the CLI after this
    base order.
    """
    ordered: List[str] = []
    for group_name in (
        "identity",
        "ownership",
        "contacts",
        "shipping",
        "scores",
        "context",
        "component_scores",
        "cross_division",
        "profit_aggregates",
        "hardware_totals",
        "cre_totals",
        "cre_adoption_relationship",
        "operational",
    ):
        ordered.extend(ICP_SCORED_ACCOUNTS_SCHEMA_GROUPS[group_name])
    return ordered
