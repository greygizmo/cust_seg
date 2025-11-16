# Metrics and Features Overview

This document summarizes the main artifacts and metrics produced by the current pipeline.

## ICP Scoring (Azure SQL pipeline)

Identity and context (when available):
- `Customer ID`, `Company Name`, `Industry`, `Industry Sub List`, `Industry_Reasoning`
- `am_sales_rep`, `AM_Territory`, `edu_assets`
- Contacts (RP Primary and Account Primary), shipping address fields

Profit aggregates (customer-level):
- `Profit_Since_2023_Total`, `Profit_T4Q_Total`, `Profit_LastQ_Total`, `Profit_PrevQ_Total`, `Profit_QoQ_Growth`
- `Months_Active_12M` (derived from monthly activity)

Division-aware features:
- `adoption_assets` (aggregated assets/seats across focus hardware goals)
- `adoption_profit` (profit from focus hardware goals, + CRE training subset as configured)
- `relationship_profit` (hardware/relationship profit for software goals: CAD, CPE, Specialty Software)
- `cre_adoption_assets`, `cre_adoption_profit`, `cre_relationship_profit` (division‑aware CRE signals)
- `printer_count` (exposed for BI; used as a hardware scaling flag)

Goal/rollup totals (joined to customer-level when available):
- Seats and GP by goal: `Seats_CAD/CPE/Specialty Software`, `GP_CAD/CPE/Specialty Software`
- Hardware totals: `Qty_Printers`, `GP_Printers`, plus per‑printer‑subdivision qty/GP where present
- Portfolio totals: `active_assets_total`, `seats_sum_total`, `Portfolio_Breadth`
- Earliest/latest purchase/expiration dates and day‑since metrics

Scores and grades (exported):
- `Hardware_score` (adoption component), `Software_score` (relationship component)
- `ICP_score_hardware`, `ICP_grade_hardware` (Hardware division, 0–100 normalized, A–F)
- `ICP_score_cre`, `ICP_grade_cre` (CRE division, 0–100 normalized, A–F)

Notes:
- List‑builder dynamics and mix (e.g., `spend_13w`, momentum scores, HW/SW share, cross‑division balance and breadth, top_subdivision, HHI, POV tags) are produced from Azure SQL and appended to the scored CSV for BI, but do not feed back into the ICP computation directly.

Output file:
- `data/processed/icp_scored_accounts.csv`

## Neighbors Artifact (exact, blockwise)

Hybrid embedding (blocks):
- Numeric: numeric columns from scored accounts after robust transforms (as configured)
- Categorical: one‑hot for industry, segment, territory, `top_subdivision_12m`
- Text: sentence embeddings of `Industry_Reasoning` (MiniLM) with TF‑IDF+SVD fallback
- ALS (collaborative): vectors trained on composite implicit signals per `(account, item_rollup)` and per `(account, Goal)`

Neighbors columns:
- `account_id`, `neighbor_account_id`, `neighbor_rank`
- `sim_overall`, `sim_numeric`, `sim_categorical`, `sim_text`, `sim_als`
- `neighbor_account_name`, `neighbor_industry`, `neighbor_segment`, `neighbor_territory`

Output file:
- `artifacts/account_neighbors.csv`

## Configuration (current)

Key sections in `config.toml`:

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
max_dense_accounts = 5000
row_block_size     = 512

[als]
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

