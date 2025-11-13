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
