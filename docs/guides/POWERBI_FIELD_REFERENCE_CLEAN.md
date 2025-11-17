# Power BI Field Reference: Scored Accounts and Neighbors

This guide documents the key columns emitted by the scoring pipeline for:
- `data/processed/icp_scored_accounts.csv` (scored accounts)
- `artifacts/account_neighbors.csv` (neighbors)

It focuses on what you can safely use in Power BI after the latest schema cleanup.

## Scored Accounts (`icp_scored_accounts.csv`)

### Identity and Ownership
- `Customer ID` — Canonical ID (string; preserves leading zeros).
- `Company Name` — Customer display name.
- `activity_segment` — Warm/Cold activity flag.
- `am_sales_rep`, `cre_sales_rep`, `AM_Territory`, `CAD_Territory`, `edu_assets` — Hardware/CAD account owners and territories, plus EDU assets.

### Contacts (Optional)
- `RP_Primary_*` — Designated primary hardware contact.
- `Primary_Contact_*` — Account primary contact.
- `Name`, `email`, `phone` — Generic fallbacks.

### Shipping (Optional)
- `ShippingAddr1`, `ShippingAddr2`, `ShippingCity`, `ShippingState`, `ShippingZip`, `ShippingCountry`.

### Scores and Grades
- `ICP_score_hardware`, `ICP_grade_hardware` — Hardware division ICP score/grade (0–100, A–F).
- `ICP_score_cre`, `ICP_grade_cre` — CRE division ICP score/grade (0–100, A–F).

### Component Scores
- `Hardware_score` — Adoption component (0–1) for hardware.
- `Software_score` — Relationship component (0–1) for software/CRE.

### Industry Context
- `Industry` — Canonical industry label.
- `Industry Sub List` — Granular industry subcategory.
- `Industry_Reasoning` — Free-text reasoning from enrichment (when available).

### Profit Aggregates (GP = GP + Term_GP)
- `GP_LastQ_Total`, `GP_PrevQ_Total`, `GP_QoQ_Growth`.
- `GP_T4Q_Total`, `GP_Since_2023_Total`.

### Hardware Division Signals

Hardware totals and printer rollups (quantities and GP):
- `Qty_Printers`, `GP_Printers`.
- `Qty_Printers_<Subdivision>`, `GP_Printers_<Subdivision>` — Per-printer subdivision (e.g., FDM, SAF, SLA).
- `Qty_Printer Accessories`, `GP_Printer Accessories`.
- `Qty_Scanners`, `GP_Scanners`.
- `Qty_Geomagic`, `GP_Geomagic`.

Hardware adoption and mix:
- `adoption_assets`, `adoption_profit` — Hardware adoption inputs for scoring.
- `hw_spend_12m`, `sw_spend_12m` — 12‑month HW/SW spend.
- `hw_share_12m`, `sw_share_12m` — Share of 12‑month spend by HW vs SW.
- `breadth_hw_subdiv_12m`, `max_hw_subdiv`, `breadth_score_hw` — Hardware breadth metrics.
- `days_since_last_hw_order`, `recency_score_hw` — Hardware recency signals.
- `hardware_adoption_score` — Composite hardware adoption metric.

### CRE Division Signals

Software seats and profit (goal-level):
- `Seats_CAD`, `GP_CAD`.
- `Seats_CPE`, `GP_CPE`.
- `Seats_Specialty Software`, `GP_Specialty Software`.

CRE rollups (dynamic, when present):
- `Seats_CAD_<Rollup>`, `GP_CAD_<Rollup>`.
- `Seats_Specialty Software_<Rollup>`, `GP_Specialty Software_<Rollup>`.

Training subset (CRE):
- `GP_Training/Services_Success_Plan`.
- `GP_Training/Services_Training` — Dedicated CRE training signals, when available.

Division adoption inputs (for traceability):
- `cre_adoption_assets`, `cre_adoption_profit` — CRE adoption inputs.
- `relationship_profit`, `cre_relationship_profit` — Relationship/profit signals (hardware vs CRE).

CRE dynamics and breadth (scoped to CRE signals):
- `spend_13w_cre`, `spend_13w_prior_cre`, `delta_13w_cre`, `delta_13w_pct_cre`.
- `spend_12m_cre`, `spend_52w_cre`, `yoy_13w_pct_cre`.
- `slope_13w_cre`, `slope_13w_prior_cre`, `acceleration_13w_cre`, `volatility_13w_cre`, `seasonality_factor_13w_cre`.
- `days_since_last_cre_order`, `recency_score_cre`.
- `breadth_cre_rollup_12m`, `max_cre_rollup`, `breadth_score_cre`.

### List-Builder Dynamics and Mix (Company-Wide)

Core dynamics:
- `spend_13w`, `spend_13w_prior`, `delta_13w`, `delta_13w_pct`, `spend_12m`, `spend_52w`, `yoy_13w_pct`.
- `days_since_last_order`, `active_weeks_13w`, `purchase_streak_months`, `median_interpurchase_days`.
- `slope_13w`, `slope_13w_prior`, `acceleration_13w`, `volatility_13w`, `seasonality_factor_13w`.

Momentum and recency:
- `trend_score`, `recency_score`, `magnitude_score`, `cadence_score`, `momentum_score`.
- `w_trend`, `w_recency`, `w_magnitude`, `w_cadence` — Weights used to build `momentum_score`.

Percentile helpers (where present):
- `*_pctl` — Percentile versions of key dynamics/adoption metrics (0–100, across the population).

### Cross-Division Levers

Momentum by division:
- `hw_spend_13w`, `hw_spend_13w_prior`, `hw_delta_13w`, `hw_delta_13w_pct`.
- `sw_spend_13w`, `sw_spend_13w_prior`, `sw_delta_13w`, `sw_delta_13w_pct`.

Portfolio breadth:
- `super_division_breadth_12m` — # of high-level divisions active in last 12 months.
- `division_breadth_12m` — # of divisions active in last 12 months.
- `software_division_breadth_12m` — # of software divisions active in last 12 months.

Opportunity scores:
- `cross_division_balance_score` — Balance between HW and SW share (higher is more balanced).
- `hw_to_sw_cross_sell_score` — Opportunity to cross-sell SW into HW-heavy accounts.
- `sw_to_hw_cross_sell_score` — Opportunity to cross-sell HW into SW-heavy accounts.
- `training_to_hw_ratio` — CRE training subset relative to hardware footprint.
- `training_to_cre_ratio` — CRE training subset relative to CRE footprint.

### Concentration, Discount, and Whitespace

- `discount_pct` — Average discount level.
- `month_conc_hhi_12m` — 12‑month concentration (HHI) across months.
- `sw_dominance_score` — Whether software dominates spend.
- `sw_to_hw_whitespace_score` — SW‑to‑HW whitespace opportunity.

### POV (Point of View)

- `pov_primary` — Primary POV tag (text).
- `pov_tags_all` — All POV tags concatenated.

### Timestamps

- `as_of_date` — Scoring cutoff date.
- `run_timestamp_utc` — Pipeline run timestamp (UTC).

## Neighbors (`artifacts/account_neighbors.csv`)

Columns:
- `account_id`, `neighbor_account_id`, `neighbor_rank`.
- `sim_overall`, `sim_numeric`, `sim_categorical`, `sim_text`, `sim_als`.
- `neighbor_account_name`, `neighbor_industry`, `neighbor_segment`, `neighbor_territory`.

Notes:
- The neighbors pipeline uses exact blockwise cosine and a hybrid embedding across numeric, categorical, text, and ALS blocks.
