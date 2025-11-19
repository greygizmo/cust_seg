import sys
import os
from pathlib import Path
sys.path.append("src")

from icp.quality import validate_outputs

def main():
    scored_path = Path("data/processed/icp_scored_accounts.csv")
    print(f"Validating {scored_path}...")
    
    import pandas as pd
    df = pd.read_csv(scored_path)
    print(f"Loaded {len(df)} rows.")
    bad = df[df["Customer ID"].isna()]
    if not bad.empty:
        print(f"Found {len(bad)} rows with null Customer ID:")
        print(bad.head())
        print(bad.index)
    
    try:
        validate_outputs(scored_path=str(scored_path), raise_error=True)
        print("Validation SUCCESS!")
    except Exception as e:
        print(f"Validation FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
