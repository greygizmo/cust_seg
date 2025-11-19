"""Generate preset call lists for sales teams."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from icp.reporting.call_lists import (
    normalize_portfolio,
    preset_top_ab_by_segment,
    preset_revenue_only_high_relationship,
    preset_heavy_fleet_expansion,
)

ROOT = Path(__file__).resolve().parents[3]


def _write_output(name: str, table: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    if table.empty:
        return
    csv_path = out_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    meta_out = {
        **meta,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output": str(csv_path),
    }
    with (out_dir / f"{name}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta_out, handle, indent=2)
    print(f"[INFO] Wrote {len(table):,} rows to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate preset call lists from scored accounts.")
    parser.add_argument(
        "--src",
        type=str,
        default=str(ROOT / "data" / "processed" / "icp_scored_accounts.csv"),
        help="Source scored accounts file (CSV or Parquet).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "reports" / "call_lists"),
        help="Root directory for dated call list exports.",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional YYYYMMDD run date override (defaults to today).",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise FileNotFoundError(f"Scored accounts not found at {src_path}")
    if src_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(src_path)
    else:
        df = pd.read_csv(src_path)

    prepared = normalize_portfolio(df)
    run_date = args.run_date or datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_root) / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    presets = [
        ("top_ab_by_segment",) + preset_top_ab_by_segment(prepared),
        ("revenue_only_high_relationship",) + preset_revenue_only_high_relationship(prepared),
        ("heavy_fleet_expansion",) + preset_heavy_fleet_expansion(prepared),
    ]
    for name, table, meta in presets:
        _write_output(name, table, meta, out_dir)


if __name__ == "__main__":
    main()
