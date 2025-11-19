"""Generate canonical asset_rollup_weights.json from database tags.

This CLI discovers all item_rollup values under focus divisions (Goals)
from the database and seeds them with even weights (1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from icp import data_access as da


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "artifacts" / "weights" / "asset_rollup_weights.json"

FOCUS_GOALS = [
    "Printer",
    "Printer Accessorials",
    "Scanners",
    "Geomagic",
    "Training/Services",
]


def generate_weights(out_path: Path) -> None:
    """Pull tags from DB and generate weighted JSON."""
    engine = da.get_engine()
    # Pull all (item_rollup, Goal) from analytics_product_tags
    query = (
        "SELECT item_rollup, Goal FROM dbo.analytics_product_tags WHERE Goal IS NOT NULL"
    )
    tags = pd.read_sql(query, engine)
    tags["Goal"] = tags["Goal"].astype(str).str.strip()
    tags["item_rollup"] = tags["item_rollup"].astype(str).str.strip()

    weights: dict[str, dict[str, float]] = {}
    for goal in FOCUS_GOALS:
        subset = tags[tags["Goal"].str.lower() == goal.lower()]
        # Sort for deterministic output
        rollups = sorted(subset["item_rollup"].dropna().unique())
        goal_map = {rollup: 1.0 for rollup in rollups}
        
        if not goal_map:
            goal_map = {"default": 1.0}
        weights[goal] = goal_map

    output = {
        "focus_goals": FOCUS_GOALS,
        "weights": weights,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"[INFO] Wrote {out_path} with evenly balanced sub-division weights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical asset_rollup_weights.json from database tags."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate_weights(Path(args.out))


if __name__ == "__main__":
    main()
