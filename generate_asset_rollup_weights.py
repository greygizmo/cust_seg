"""
Generate a canonical asset_rollup_weights.json by discovering all item_rollup
values under focus divisions (Goals) from the database and seeding them with
even weights (1.0).

Usage:
  python generate_asset_rollup_weights.py

Requires: valid .env for Azure SQL connection.
"""

import json
from collections import defaultdict

import pandas as pd

import data_access as da

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

    with open("asset_rollup_weights.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("✓ Wrote asset_rollup_weights.json with evenly balanced sub-division weights")


if __name__ == "__main__":
    main()

