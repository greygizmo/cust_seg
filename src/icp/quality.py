"""Data quality and validation schemas using Pandera."""

import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd
from typing import Optional

from icp.schema import (
    COL_CUSTOMER_ID,
    COL_COMPANY_NAME,
    COL_INDUSTRY,
)

# ---------------------------------------------------------------------------
# Scored Accounts Schema
# ---------------------------------------------------------------------------
class ScoredAccountsSchema(pa.DataFrameModel):
    """Validation schema for the main scored accounts CSV."""
    
    # Identity
    customer_id: Series[str] = pa.Field(alias=COL_CUSTOMER_ID, coerce=True)
    company_name: Series[str] = pa.Field(alias=COL_COMPANY_NAME, nullable=True)
    industry: Series[str] = pa.Field(alias=COL_INDUSTRY, nullable=True)
    
    # Scores (ensure they are floats between 0 and 100, or null if not scored)
    # We use dynamic aliases based on the schema constants, but for class attributes
    # we need fixed names. We'll map them in the Config.
    
    # For simplicity in this initial version, we'll validate the existence of key score columns
    # and their types.
    
    class Config:
        coerce = True
        strict = False  # Allow extra columns (there are many feature columns)

# ---------------------------------------------------------------------------
# Neighbors Schema
# ---------------------------------------------------------------------------
class NeighborsSchema(pa.DataFrameModel):
    """Validation schema for account neighbors."""
    
    account_id: Series[str] = pa.Field(coerce=True)
    neighbor_account_id: Series[str] = pa.Field(coerce=True)
    sim_overall: Series[float] = pa.Field(ge=-1.0, le=1.0)  # Cosine similarity range
    neighbor_rank: Series[int] = pa.Field(ge=1)
    
    class Config:
        coerce = True
        strict = False  # Allow additional columns like sim_numeric, sim_categorical, etc.

# ---------------------------------------------------------------------------
# Playbooks Schema
# ---------------------------------------------------------------------------
class PlaybooksSchema(pa.DataFrameModel):
    """Validation schema for account playbooks."""
    
    customer_id: Series[str] = pa.Field(coerce=True)
    playbook_primary: Series[str] = pa.Field()
    playbook_tags: Series[str] = pa.Field(nullable=True)
    
    class Config:
        coerce = True
        strict = False # Allow extra columns like tags

# ---------------------------------------------------------------------------
# Validation Helper
# ---------------------------------------------------------------------------
def validate_outputs(
    scored_path: Optional[str] = None,
    neighbors_path: Optional[str] = None,
    playbooks_path: Optional[str] = None,
    raise_error: bool = False
) -> bool:
    """
    Validate output files against schemas.
    
    Args:
        scored_path: Path to scored accounts CSV
        neighbors_path: Path to neighbors CSV
        playbooks_path: Path to playbooks CSV
        raise_error: If True, raise SchemaError on failure. If False, print warning.
        
    Returns:
        True if all provided files are valid, False otherwise.
    """
    all_valid = True
    
    # Validate Scored Accounts
    if scored_path:
        try:
            df = pd.read_csv(scored_path)
            # Dynamic validation for score columns since they might vary or be renamed
            # But we can check for the ones in ICP_SCHEMA_SCORES
            # For now, just validate the base schema
            ScoredAccountsSchema.validate(df)
            print(f"[OK] Scored accounts valid: {scored_path}")
        except Exception as e:
            print(f"[WARN] Scored accounts invalid: {scored_path}")
            print(e)
            all_valid = False
            if raise_error:
                raise e

    # Validate Neighbors
    if neighbors_path:
        try:
            df = pd.read_csv(neighbors_path)
            NeighborsSchema.validate(df)
            print(f"[OK] Neighbors valid: {neighbors_path}")
        except Exception as e:
            print(f"[WARN] Neighbors invalid: {neighbors_path}")
            print(e)
            all_valid = False
            if raise_error:
                raise e

    # Validate Playbooks
    if playbooks_path:
        try:
            df = pd.read_csv(playbooks_path)
            PlaybooksSchema.validate(df)
            print(f"[OK] Playbooks valid: {playbooks_path}")
        except Exception as e:
            print(f"[WARN] Playbooks invalid: {playbooks_path}")
            print(e)
            all_valid = False
            if raise_error:
                raise e
                
    return all_valid
