# Metrics and Features Overview

This document summarizes the main artifacts, features, and SQL tables used by the current pipeline.

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
- `cre_adoption_assets`, `cre_adoption_profit`, `cre_relationship_profit` (division-aware CRE signals)
- `printer_count` (exposed for BI; used as a hardware scaling flag)

Goal/rollup totals (joined to customer-level when available):
- Seats and GP by goal: `Seats_CAD/CPE/Specialty Software`, `GP_CAD/CPE/Specialty Software`
- Hardware totals: `Qty_Printers`, `GP_Printers`, plus per-printer-subdivision qty/GP where present
- Portfolio totals: `active_assets_total`, `seats_sum_total`, `Portfolio_Breadth`
- Earliest/latest purchase/expiration dates and day-since metrics

Scores and grades (exported):
- `Hardware_score` (adoption component), `Software_score` (relationship component)
- `ICP_score_hardware`, `ICP_grade_hardware` (Hardware division, 0�100 normalized, A�F)
- `ICP_score_cre`, `ICP_grade_cre` (CRE division, 0�100 normalized, A�F)

Notes:
- List-builder dynamics and mix (e.g., `spend_13w`, momentum scores, HW/SW share, cross-division balance and breadth, top_subdivision, HHI, POV tags) are produced from Azure SQL and appended to the scored CSV for BI, but do not feed back into the ICP computation directly.

Output file:
- `data/processed/icp_scored_accounts.csv`

## Source and Target Tables (SQL)

Source tables in `AZSQL_DB` (e.g., `db-goeng-netsuite-prod`):
- `dbo.table_saleslog_detail` � transactional GP/Term_GP (profit) by item and date; drives profit aggregates and spend dynamics.
- `dbo.items_category_limited` � item ? `item_rollup` mapping used to join transactions to rollups and goals.
- `dbo.analytics_product_tags` � goal, super_division, and taxonomy tags per `item_rollup`; used for division/goal classification.
- `dbo.table_Product_Info_cleaned_headers` � product/asset headers; drives assets/seats, adoption signals, and portfolio breadth.
- `dbo.customer_cleaned_headers` � customer attributes (AM, territory, EDU, fallback shipping) used in identity and ownership features.
- `dbo.customer_customerOnly` � preferred source for shipping address fields when available.
- `dbo.contact_clean_headers` � contact headers used to derive RP Primary and Account Primary contacts.

Target table in `ICP_AZSQL_DB` (e.g., `db-goeng-icp-prod`), when configured:
- `dbo.customer_icp` � full scored accounts table (same schema as `icp_scored_accounts.csv`), replaced on each scoring run.

## Neighbors Artifact (exact, blockwise)

Hybrid embedding (blocks):
- Numeric: numeric columns from scored accounts after robust transforms (as configured)
- Categorical: one-hot for industry, segment, territory, `top_subdivision_12m`
- Text: sentence embeddings of `Industry_Reasoning` (MiniLM) with TF-IDF+SVD fallback
- ALS (collaborative): vectors trained on composite implicit signals per `(account, item_rollup)` and per `(account, Goal)`

Neighbors columns:
- `account_id`, `neighbor_account_id`, `neighbor_rank`
- `sim_overall`, `sim_numeric`, `sim_categorical`, `sim_text`, `sim_als`
- `neighbor_account_name`, `neighbor_industry`, `neighbor_segment`, `neighbor_territory`

Output file:
- `artifacts/account_neighbors.csv`

## Playbooks Artifact (rule-based CRO/CFO motions)

The playbooks CLI (`python -m icp.cli.build_playbooks`) generates a compact artifact of rule-based tags and plays:

Input:
- Scored accounts (`data/processed/icp_scored_accounts.csv`)
- Neighbors (`artifacts/account_neighbors.csv`, optional)

Key columns:
- `Customer ID`, `Company Name`, `customer_id`
- `playbook_primary` – single headline motion to run for the account.
- `playbook_tags` – semicolon-delimited list of all matching tags (for example, "HW Expansion Sprint; Training & Services Attach").
- `playbook_rationale` – short natural-language explanation of *why* those tags fired.
- Convenience metrics copied from scored accounts:
  - `GP_Since_2023_Total`, `ICP_score_hardware`, `ICP_grade_hardware`, `ICP_score_cre`, `ICP_grade_cre`
  - `whitespace_score`, `hw_share_12m`, `sw_share_12m`, `CRE_Training`, `days_since_last_order`
  - `customer_segment`, `recency_bucket`
- Neighbor-based flags:
  - `is_hero` – A-grade, high-GP “hero” accounts.
  - `is_hero_neighbor` – accounts that appear as neighbors of at least one hero.
  - `is_hero_orphan_neighbor` – hero neighbors that are dormant/long-recency.
  - `hero_neighbor_count` – number of hero accounts that reference this account as a neighbor.

Output file:
- `artifacts/account_playbooks.csv`

## Pulse Artifacts (snapshots for trend tracking)

The pulse CLI (`python -m icp.cli.build_pulse_artifacts`) produces small snapshot tables summarizing portfolio, neighbors, and playbooks at a point in time.

Inputs:
- Scored accounts (`data/processed/icp_scored_accounts.csv`)
- Neighbors (`artifacts/account_neighbors.csv`)
- Playbooks (`artifacts/account_playbooks.csv`)

Outputs (all under `artifacts/`):

- `pulse_portfolio.csv` – one row per snapshot with:
  - `snapshot_ts_utc`, `as_of_date`
  - `accounts_total`, `gp_total`
  - `accounts_ab_hw`, `gp_ab_hw`
  - `accounts_ab_cre`, `gp_ab_cre`

- `pulse_neighbors.csv` – one row per snapshot with:
  - `snapshot_ts_utc`
  - `accounts_with_neighbors`, `neighbor_edges`, `avg_neighbors_per_account`
  - `inbound_neighbors_avg`, `inbound_neighbors_p95`
  - `hero_count`, `hero_neighbor_count`, `orphan_hero_neighbor_count`

- `pulse_playbooks.csv` – multiple rows per snapshot (one per `playbook_primary`), including:
  - `snapshot_ts_utc`
  - `playbook_primary`
  - `accounts`, `gp_total`
  - `hero_neighbor` (whether any hero neighbors are in that playbook)
  - `accounts_pct`, `gp_pct` (share of accounts and GP for that playbook at the snapshot)

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

## Validation (Pandera)

The pipeline now includes strict data validation using **Pandera** schemas defined in `src/icp/quality.py`.

Validated artifacts:
- **Scored Accounts**: Checks for existence of ID, Company, Industry, and Score columns.
- **Neighbors**: Checks for `source_account_id`, `neighbor_account_id`, `similarity_score` (0-1), and `rank`.
- **Playbooks**: Checks for `account_id` and `playbook_name`.

To enforce validation and fail on errors, use the `--strict` flag when running `score_accounts.py`.
