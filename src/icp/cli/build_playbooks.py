"""Generate rule-based playbooks and tags for scored accounts.

This CLI is intended to run *after* the main scoring + neighbors pipeline.
It reads:

- `data/processed/icp_scored_accounts.csv` (or a supplied path)
- `artifacts/account_neighbors.csv` (optional)

and writes a compact artifact with CRO/CFO-friendly playbooks:

- `artifacts/account_playbooks.csv`

The artifact is safe to import into Power BI or join into Streamlit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from icp.schema import COL_CUSTOMER_ID, COL_COMPANY_NAME, canonicalize_customer_id
from icp.playbooks.logic import compute_neighbor_flags, derive_playbooks


ROOT = Path(__file__).resolve().parents[3]
SCORED_DEFAULT = ROOT / "data" / "processed" / "icp_scored_accounts.csv"
NEIGHBORS_DEFAULT = ROOT / "artifacts" / "account_neighbors.csv"
OUT_DEFAULT = ROOT / "artifacts" / "account_playbooks.csv"


def _load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scored accounts file not found: {path}")
    df = pd.read_csv(path)
    if COL_CUSTOMER_ID not in df.columns:
        raise ValueError(f"Expected '{COL_CUSTOMER_ID}' in scored accounts.")
    df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])
    df["customer_id"] = df[COL_CUSTOMER_ID].astype(str)
    if COL_COMPANY_NAME not in df.columns:
        fallback = df.get("company_name", df.index.astype(str))
        df[COL_COMPANY_NAME] = fallback.astype(str)
    df[COL_COMPANY_NAME] = df[COL_COMPANY_NAME].fillna("").astype(str)
    return df


def _load_neighbors(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("account_id", "neighbor_account_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def build_playbooks(scored_path: Path, neighbors_path: Path, out_path: Path) -> Path:
    scored = _load_scored(scored_path)
    neighbors = _load_neighbors(neighbors_path)
    
    # Apply logic from the playbooks module
    scored_flags = compute_neighbor_flags(scored, neighbors)
    artifact = derive_playbooks(scored_flags)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(out_path, index=False)
    print(f"[INFO] Wrote playbooks artifact: {out_path}")
    return out_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based playbooks and tags from scored accounts.")
    parser.add_argument(
        "--in-scored",
        type=str,
        default=str(SCORED_DEFAULT),
        help=f"Path to scored accounts CSV (default: {SCORED_DEFAULT})",
    )
    parser.add_argument(
        "--neighbors",
        type=str,
        default=str(NEIGHBORS_DEFAULT),
        help=f"Path to neighbors CSV (default: {NEIGHBORS_DEFAULT}, optional).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_DEFAULT),
        help=f"Output CSV path for playbooks (default: {OUT_DEFAULT})",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    scored_path = Path(args.in_scored)
    neighbors_path = Path(args.neighbors)
    out_path = Path(args.out)
    build_playbooks(scored_path, neighbors_path, out_path)


if __name__ == "__main__":
    main()
