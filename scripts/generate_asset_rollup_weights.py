"""
Generate a canonical asset_rollup_weights.json by discovering all item_rollup
values under focus divisions (Goals) from the database and seeding them with
even weights (1.0).

Usage:
  python generate_asset_rollup_weights.py

Requires: valid .env for Azure SQL connection.
"""

import os
from pathlib import Path
import json
from collections import defaultdict

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Fallback: simple .env parser
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    k = k.strip().lstrip("\ufeff")
                    os.environ.setdefault(k, v.strip())

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
import icp.data_access as da

FOCUS_GOALS = [
    "Printer",
    "Printer Accessorials",
    "Scanners",
    "Geomagic",
    "Training/Services",
]


def main():
    engine = da.get_engine()
    # Pull all (item_rollup, Goal) from analytics_product_tags
    query = (
        "SELECT item_rollup, Goal FROM dbo.analytics_product_tags WHERE Goal IS NOT NULL"
    )
    tags = pd.read_sql(query, engine)
    tags['Goal'] = tags['Goal'].astype(str).str.strip()
    tags['item_rollup'] = tags['item_rollup'].astype(str).str.strip()

    weights = {}
    for goal in FOCUS_GOALS:
        subset = tags[tags['Goal'].str.lower() == goal.lower()]
        goal_map = {rollup: 1.0 for rollup in sorted(subset['item_rollup'].dropna().unique())}
        if not goal_map:
            goal_map = {"default": 1.0}
        weights[goal] = goal_map

    output = {
        "focus_goals": FOCUS_GOALS,
        "weights": weights,
    }

    out_path = ROOT / 'artifacts' / 'weights' / 'asset_rollup_weights.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path} with evenly balanced sub-division weights")


if __name__ == "__main__":
    main()


