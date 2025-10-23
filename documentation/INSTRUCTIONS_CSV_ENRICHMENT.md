# ICP CSV Enrichment Instructions

## Objective

Produce a single CSV—`icp_scored_accounts.csv`—that already contains all list-builder features (spending patterns, velocity, momentum, adoption & mix, health/concentration), plus POV tags (including SW→HW whitespace) so Power BI needs only to display columns. No DAX measures. We integrate into the existing pipeline run by:

```
python goe_icp_scoring.py
```

(Per your README.)

## File & Module Plan

Create these new modules and wire them into `goe_icp_scoring.py`:

```
/features/
  __init__.py
  spend_dynamics.py          # all windows, slopes, acceleration, volatility, seasonality
  adoption_and_mix.py        # HW/SW 12M shares, breadth across HW sub-divisions, adoption score
  health_concentration.py    # discount%, month HHI, median interpurchase days
  sw_hw_whitespace.py        # SW dominance + SW→HW whitespace score
  pov_tags.py                # primary and all-tags, rule-based (no ML)
  product_taxonomy.py        # helpers to enforce SuperDiv/Div/SubDiv integrity

documentation/
  INSTRUCTIONS_CSV_ENRICHMENT.md  # (this file)
  TODO_CSV_ENRICHMENT.md          # task checklist
```

We append all features as columns to the per-account rows your pipeline already writes to `icp_scored_accounts.csv`. The only extra inputs required are transactions and a product taxonomy mapping with SuperDivision, Division, SubDivision.

## Config

Extend `config.toml` with a new section. (Do not break existing keys.)

```
[data_sources]
sales_path      = "data/sales_transactions.csv"      # Required columns: date, account_id, product_id, invoice_id, net_revenue, [list_price_revenue]
products_path   = "data/products.csv"                # Required columns: product_id, super_division, division, sub_division
as_of_date      = ""                                  # Optional (YYYY-MM-DD); default = max(txn date)

[momentum_weights]  # default no-ML momentum weights
w_trend     = 0.4
w_recency   = 0.3
w_magnitude = 0.2
w_cadence   = 0.1

[windows]
weeks_short = 13     # 13W window for short-term velocity
months_ltm  = 12     # LTM window for levels/HHI
weeks_year  = 52     # 52 week window for magnitude normalization
```

## Column Schema (new fields appended in CSV)

> All columns are per AccountID (flat). Names are snake_case.

### Level & Change (Section 2.1)

```
spend_13w, spend_13w_prior, delta_13w, delta_13w_pct,

spend_12m, spend_52w, yoy_13w_pct
```

### Cadence & Recency (2.2)

```
days_since_last_order, active_weeks_13w, purchase_streak_months, median_interpurchase_days
```

### Velocity & Acceleration (2.3 & 2.4)

```
slope_13w, slope_13w_prior, acceleration_13w
```

### Stability & Seasonality (2.5)

```
volatility_13w, seasonality_factor_13w
```

### Momentum (2.6)

```
trend_score (0–1; percentile of slope_13w), recency_score, magnitude_score, cadence_score,

momentum_score (w_trend*w + …), plus w_trend, w_recency, w_magnitude, w_cadence for transparency.
```

### Adoption & Mix (2.7) — NO Big/Small Box; Super/Div/SubDiv only

```
hw_spend_12m, sw_spend_12m, hw_share_12m, sw_share_12m

breadth_hw_subdiv_12m (count of HW sub-divisions with LTM spend)

max_hw_subdiv (constant per run), breadth_score_hw (breadth/max)

days_since_last_hw_order, recency_score_hw

hardware_adoption_score (0.5*hw_share_12m + 0.3*breadth_score_hw + 0.2*recency_score_hw)

consumables_to_hw_ratio (if SubDivision="Consumables" exists under Hardware)

top_subdivision_12m (label), top_subdivision_share_12m (0–1)
```

### Health & Concentration (2.8)

```
discount_pct (if list_price_revenue available), month_conc_hhi_12m (0–1, higher = bursty)
```

### SW whitespace (new for Section 3 POVs)

```
sw_dominance_score (0–1), sw_to_hw_whitespace_score (0–1)
```

### POV Tags (Section 3)

```
pov_primary (string, one tag)

pov_tags_all (comma-separated list)
```

### Timestamps

```
as_of_date (ISO), run_timestamp_utc
```

## Code (drop-in, well-commented)

> All code below is pure Python / pandas, no ML, and can be pasted as-is. Place in `/features/*.py` and import from `goe_icp_scoring.py` before writing the CSV (see patch at bottom).

*(See repository for concrete implementations.)*

## Testing & Acceptance

Contract: `icp_scored_accounts.csv` includes all columns listed above and retains all existing ICP columns.

Edge cases: accounts with no transactions have blanks (not zeros) for dynamic fields; tags default to `General ICP Match`.

Performance: the grouping is linear in rows; weekly resampling and per-account OLS slope is vectorized over small windows (13 points).

Determinism: given identical inputs and `as_of_date`, outputs are bit-for-bit reproducible.

## Final Notes

This design keeps one source of truth in the repo and one artifact for BI—exactly your constraint. The hook is surgical: we extend `goe_icp_scoring.py`—the script you already use to create `icp_scored_accounts.csv`.

We intentionally dropped all references to Big-/Small-Box and replaced adoption with HW share + breadth + recency, aligned to SuperDivision / Division / SubDivision.
