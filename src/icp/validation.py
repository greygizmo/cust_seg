"""
Data validation schemas using Pandera.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

try:
    import pandera.pandas as pa

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None  # type: ignore

from icp.schema import (
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
)

# Define schemas
if PANDERA_AVAILABLE:
    class MasterDataSchema(pa.DataFrameModel):  # type: ignore
        """Schema for the assembled master dataframe before feature engineering."""
        pass
    
    class ScoredDataSchema(pa.DataFrameModel):  # type: ignore
        """Schema for the final scored accounts dataframe."""
        # We can enforce key columns here
        pass
else:
    # Dummy classes for type checking when pandera is missing
    class MasterDataSchema:  # type: ignore[no-redef]
        pass

    class ScoredDataSchema:  # type: ignore[no-redef]
        pass

def get_master_schema():
    if not PANDERA_AVAILABLE:
        return None
    return pa.DataFrameSchema({
        COL_CUSTOMER_ID: pa.Column(str, coerce=True, nullable=False),
        COL_COMPANY_NAME: pa.Column(str, nullable=True),
        COL_INDUSTRY: pa.Column(str, nullable=True),
        "Profit_Since_2023_Total": pa.Column(float, coerce=True, nullable=True),
        "active_assets_total": pa.Column(float, coerce=True, nullable=True),
    }, coerce=True)

def get_scored_schema():
    if not PANDERA_AVAILABLE:
        return None
    return pa.DataFrameSchema({
        COL_CUSTOMER_ID: pa.Column(str, coerce=True, nullable=False),
        "ICP_score_hardware": pa.Column(float, coerce=True, nullable=True),
        "ICP_grade_hardware": pa.Column(str, nullable=True),
        "ICP_score_cre": pa.Column(float, coerce=True, nullable=True),
        "ICP_grade_cre": pa.Column(str, nullable=True),
        "printer_count": pa.Column(float, coerce=True, nullable=True),
        "GP_Since_2023_Total": pa.Column(float, coerce=True, nullable=True),
    }, coerce=True)

def validate_master(df: pd.DataFrame) -> pd.DataFrame:
    """Validates the master dataframe against the schema."""
    if not PANDERA_AVAILABLE:
        print("[WARN] Pandera not installed. Skipping validation.")
        return df
        
    schema = get_master_schema()
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print(f"[WARN] Master data validation failed with {len(err.failure_cases)} errors.")
        print(err.failure_cases.head())
        return df

def validate_scored(df: pd.DataFrame) -> pd.DataFrame:
    """Validates the scored dataframe against the schema."""
    if not PANDERA_AVAILABLE:
        print("[WARN] Pandera not installed. Skipping validation.")
        return df
        
    schema = get_scored_schema()
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print(f"[WARN] Scored data validation failed with {len(err.failure_cases)} errors.")
        print(err.failure_cases.head())
        return df

def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> Tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def ensure_non_negative(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[bool, list[str]]:
    bad = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if (s < 0).any():
                bad.append(c)
    return (len(bad) == 0, bad)


def log_validation(summary: str, details: list[str] | None = None, root: Path | None = None) -> Path:
    """Append a validation log entry under reports/logs/."""
    root = root or Path.cwd()
    out_dir = root / "reports" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {summary}\n")
        if details:
            for line in details:
                f.write(f"    - {line}\n")
    return path
