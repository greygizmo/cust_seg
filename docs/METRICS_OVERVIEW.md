# Metrics and Features Overview

This document reflects the metrics and artifacts produced by the current repo code paths. It covers the main ICP scoring output and the separate neighbors artifact (with ALS).

## ICP Scoring (Azure SQL pipeline)

Identity and context columns (if present in sources):
- Customer ID, Company Name, Industry, Industry Sub List, Industry_Reasoning
- AM Sales Rep, AM Territory, EDU Assets flag
- Contacts (RP Primary and Account Primary), Shipping address fields

Profit aggregates (customer-level):
- Profit_Since_2023_Total, Profit_T4Q_Total, Profit_LastQ_Total, Profit_PrevQ_Total, Profit_QoQ_Growth
- Months_Active_12M (derived from monthly profit activity)

Hardware/Software features (customer-level):
- adoption_assets (aggregated asset/seat signals across focus goals)
- adoption_profit (profit signals across Printer/Accessories/Scanners/Geomagic and 3DP Training rollup)
- relationship_profit (profit across software goals: CAD, CPE, Specialty Software)
- printer_count, scaling_flag

Goal- and rollup-level totals (merged to customer-level where available):
- Seats_CAD/CPE/Specialty Software; GP_CAD/CPE/Specialty Software
- Qty_Printers, GP_Printers; per-printer-subdivision qty/gp where available
- active_assets_total, seats_sum_total, Portfolio_Breadth
- Earliest/Latest purchase/expiration dates and corresponding day deltas

Scores and grades:
- Hardware_score (adoption), Software_score (relationship)
- vertical_score, size_score
- ICP_score_raw, ICP_score (0–100 normalized), ICP_grade (A–F)

Notes:
- The “List‑Builder” time‑series and mix features (spend_13w, trend/momentum, HW/SW share, top_subdivision, HHI, POV tags, etc.) are now ENABLED in the default pipeline via the Azure SQL feed and are appended to the scored CSV.

Output file:
- data/processed/icp_scored_accounts.csv

## Neighbors Artifact (Exact, blockwise)

Hybrid embedding composed from blocks:
- Numeric: all numeric columns from scored accounts after robust transforms (winsor, log1p/logit per config, robust z, L2 row norm)
- Categorical: one‑hot for industry, segment (activity_segment), territory, top_subdivision_12m (if present)
- Text: sentence embeddings of Industry_Reasoning (MiniLM) with TF‑IDF+SVD fallback
- ALS (collaborative): account vectors trained on composite implicit signals per (account,item_rollup) and per (account,Goal)

Neighbors output columns:
- account_id, neighbor_account_id, neighbor_rank
- sim_overall, sim_numeric, sim_categorical, sim_text, sim_als
- neighbor_account_name, neighbor_industry, neighbor_segment, neighbor_territory

Output file:
- artifacts/account_neighbors.csv

## Configuration (current)

In `config.toml`:

```toml
[similarity]
k_neighbors   = 15
use_text      = true
use_als       = true
w_numeric     = 0.50
w_categorical = 0.15
w_text        = 0.25
w_als         = 0.10
winsor_pct    = 0.01
log1p_cols    = [
  "GP_Since_2023_Total",
  "GP_T4Q_Total",
  "GP_LastQ_Total",
  "GP_PrevQ_Total",
  "Total Software License Revenue",
]
logit_cols    = [
  "hw_share_12m",
  "sw_share_12m",
  "recency_score",
  "magnitude_score",
  "cadence_score",
  "breadth_score_hw",
]
text_column   = "industry_reasoning"

# Memory/scale controls for exact neighbors
max_dense_accounts = 5000      # use blockwise above this N
row_block_size     = 512       # query rows per block in blockwise

[als]
# Composite strength weights for implicit signals
w_rev = 1.0
w_seats = 0.3
w_assets = 0.2
w_active = 0.1
w_recency = 0.3
recency_half_life_days = 180
alpha = 40.0
reg = 0.05
iterations = 20
use_bm25 = true
```

## CLI usage

- Baseline scoring (no neighbors/visuals):
  - `python -m icp.cli.score_accounts --skip-neighbors --skip-visuals`
- Build neighbors only (exact, k per config) from existing scored CSV:
  - `python -m icp.cli.score_accounts --neighbors-only --in-scored data/processed/icp_scored_accounts.csv`
- Disable ALS in neighbors (override config):
  - `python -m icp.cli.score_accounts --neighbors-only --no-als`
