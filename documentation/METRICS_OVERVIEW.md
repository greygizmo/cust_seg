# Metrics and Features Overview

This document enumerates the metrics, features, and data points produced by the ICP scoring pipeline and by the similarity (neighbors) pipeline, including ALS.

## 1) ICP Scoring and Enrichment (Azure‑DB pipeline)

Key identity and context:
- Customer ID, Company Name, Industry, Industry Sub List, Industry_Reasoning
- AM Sales Rep, AM Territory, EDU Assets flag
- Contacts (RP Primary and Account Primary), Shipping address fields

Activity segment:
- activity_segment: warm (in sales log since 2023) or cold (owns hardware assets, no recent sales)

Profit & momentum aggregates:
- Profit_Since_2023_Total, Profit_T4Q_Total, Profit_LastQ_Total, Profit_PrevQ_Total, Profit_QoQ_Growth
- GP_Last_90D, Months_Active_12M, GP_Trend_Slope_12M

Hardware/Software features:
- adoption_assets, adoption_profit, relationship_profit, printer_count, scaling_flag
- Goal‑level seats and profit: Seats_CAD/CPE/Specialty Software, GP_CAD/CPE/Specialty Software

Printer subdivisions (quantities and profit):
- Qty_Printers_* and GP_Printers_* for {AM Software, AM Support, Consumables, FDM, FormLabs, Metals, P3, Polyjet, Post Processing, SAF, SLA, Spare Parts/Repair Parts/Time & Materials}

Assets / portfolio breadth and recency:
- active_assets_total, seats_sum_total, Portfolio_Breadth
- EarliestPurchaseDate, LatestPurchaseDate, LatestExpirationDate
- Days_Since_First_Purchase, Days_Since_Last_Purchase, Days_Since_Last_Expiration

Scores and grades:
- Hardware_score (alias adoption_score), Software_score (alias relationship_score)
- vertical_score, size_score, ICP_score_raw, ICP_score, ICP_grade

List‑builder feature columns (when CSV enrichment is enabled):
- Spend dynamics: spend_13w, spend_13w_prior, delta_13w, delta_13w_pct, spend_12m, spend_52w, yoy_13w_pct
- Seasonality and trend: slope_13w, slope_13w_prior, acceleration_13w, volatility_13w, seasonality_factor_13w
- Activity: days_since_last_order, active_weeks_13w, purchase_streak_months, median_interpurchase_days
- Momentum scores: trend_score, recency_score, magnitude_score, cadence_score, momentum_score (+ weight components)
- Adoption/mix: hw_spend_12m, sw_spend_12m, hw_share_12m, sw_share_12m, breadth_hw_subdiv_12m, max_hw_subdiv, breadth_score_hw, days_since_last_hw_order, recency_score_hw, hardware_adoption_score, consumables_to_hw_ratio, top_subdivision_12m, top_subdivision_share_12m
- Health/concentration/discount: month_conc_hhi_12m, discount_pct
- POV/whitespace: sw_dominance_score, sw_to_hw_whitespace_score, pov_primary, pov_tags_all
- Timestamps: as_of_date (ISO date), run_timestamp_utc (RFC3339)

## 2) Similarity (Neighbors) Pipeline

Hybrid embedding blocks (combined via weighted concatenation and L2‑normalization):

1. Numeric block (content‑based)
   - All numeric features from the scored accounts (excluding IDs and text/categorical columns), after robust transforms:
     - Winsorize at p1/p99
     - log1p for scale‑heavy metrics (configurable)
     - logit for bounded ratio features (configurable)
     - Robust z‑score and L2 normalize rows

2. Categorical block (content‑based)
   - One‑hot encodings for: industry, segment (warm/cold), territory, top_subdivision_12m (if present)
   - L2 normalized row vectors

3. Text block (content‑based)
   - Sentence embedding of Industry_Reasoning via Sentence‑Transformers (all‑MiniLM‑L6‑v2)
   - TF‑IDF + SVD fallback if the transformer is unavailable
   - L2 normalized row vectors

4. ALS block (collaborative)
   - Composite implicit strength per (account_id, item_rollup) combining:
     - Profit_Since_2023 (log1p)
     - seats_sum (log1p), asset_count, active_assets
     - Recency weight per rollup using exponential decay over days since last purchase/expiry
   - Weights are configurable in `[als]` (w_rev, w_seats, w_assets, w_active, w_recency, recency_half_life_days)
   - The resulting account vectors are L2 normalized.

Overall similarity:
- The four blocks are weighted per `[similarity]` (w_numeric, w_categorical, w_text, w_als), concatenated, and L2 normalized; cosine similarity is then computed.
- Top‑K neighbors are exported to `artifacts/account_neighbors.csv` with overall and per‑block similarities so sellers can see why accounts are similar.

## 3) Artifacts

- `data/processed/icp_scored_accounts.csv` — main scored accounts table powering the dashboard
- `artifacts/account_neighbors.csv` — top‑K nearest neighbors per account for Power BI

## 4) Configuration

`config.toml` sections relevant to similarity:

```toml
[similarity]
k_neighbors   = 25
use_text      = true
use_als       = true
w_numeric     = 0.50
w_categorical = 0.15
w_text        = 0.25
w_als         = 0.10
winsor_pct    = 0.01
log1p_cols    = ["GP_Since_2023_Total","GP_T4Q_Total","GP_LastQ_Total","GP_PrevQ_Total","Total Software License Revenue"]
logit_cols    = ["hw_share_12m","sw_share_12m","recency_score","magnitude_score","cadence_score","breadth_score_hw"]
text_column   = "industry_reasoning"

[als]
w_rev = 1.0
w_seats = 0.3
w_assets = 0.2
w_active = 0.1
w_recency = 0.3
recency_half_life_days = 180
```

