# features/product_taxonomy.py
from __future__ import annotations

import pandas as pd

REQUIRED_TXN_COLS = {
    "date",
    "account_id",
    "product_id",
    "invoice_id",
    "net_revenue",
}
# list_price_revenue is optional

REQUIRED_PROD_COLS = {
    "product_id",
    "super_division",
    "division",
    "sub_division",
}


def validate_and_join_products(txn: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures required columns exist and enforces Super/Div/SubDiv integrity.
    Returns transactions joined to product hierarchy.
    """
    missing_txn = REQUIRED_TXN_COLS - set(map(str.lower, txn.columns))
    if missing_txn:
        raise ValueError(f"Transactions missing required columns: {missing_txn}")

    missing_prod = REQUIRED_PROD_COLS - set(map(str.lower, prod.columns))
    if missing_prod:
        raise ValueError(f"Products missing required columns: {missing_prod}")

    # Normalize column names
    t = txn.rename(columns=str.lower).copy()
    p = prod.rename(columns=str.lower).copy()

    # Ensure date dtype
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    if t["date"].isna().any():
        raise ValueError("Transactions contain non-parseable dates.")

    # Basic integrity
    for col in ("super_division", "division", "sub_division"):
        if p[col].isna().any():
            raise ValueError(f"Products contain nulls in {col}.")

    # Join
    merged = t.merge(p, on="product_id", how="left", validate="many_to_one")

    if merged["super_division"].isna().any():
        # Surface orphan product_ids early
        bad = merged.loc[merged["super_division"].isna(), "product_id"].unique()[:10]
        raise ValueError(f"Some product_ids not in product map, e.g., {bad}")

    return merged
