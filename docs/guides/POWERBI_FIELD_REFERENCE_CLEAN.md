# Power BI Field Reference: Scored Accounts and Neighbors

This guide documents columns emitted by the scoring pipeline for:
- `data/processed/icp_scored_accounts.csv` (Scored Accounts)
- `artifacts/account_neighbors.csv` (Neighbors)

It explains meaning, provenance, and typical usage.

## Scored Accounts (icp_scored_accounts.csv)

Identity and ownership
- `Customer ID` — Canonical ID (string; preserves leading zeros)
- `Company Name` — Customer display name
- `activity_segment` — Warm/Cold activity flag
- `am_sales_rep`, `AM_Territory`, `edu_assets`

Contacts (optional)
- `RP_Primary_*` — Designated primary hardware contact
- `Primary_Contact_*` — Account primary contact
- `Name`, `email`, `phone` — Generic fallbacks

Shipping (optional)
- `ShippingAddr1`, `ShippingAddr2`, `ShippingCity`, `ShippingState`, `ShippingZip`, `ShippingCountry`

Scores and grades
- `ICP_score` — Overall score 0–100 (rank-normalized to mean 50, sd 15)
- `ICP_grade` — A–F grade via target distribution (A 10%, B 20%, C 40%, D 20%, F 10%)
- `ICP_score_hardware`, `ICP_grade_hardware` — Division-specific score/grade
- `ICP_score_cre`, `ICP_grade_cre` — Division-specific score/grade

Component scores (division-aware)
- `vertical_score` — Industry fit (0–1) from division weights
- `size_score` — Discrete bands by revenue (0–1)
- `Hardware_score` — Adoption component (0–1)
- `Software_score` — Relationship component (0–1)
- `ICP_score_raw` — Pre-normalization weighted sum (×100)

Industry context
- `Industry`, `Industry Sub List`, `Industry_Reasoning`

Profit aggregates (GP = GP + Term_GP)
- `GP_LastQ_Total`, `GP_PrevQ_Total`, `GP_QoQ_Growth`, `GP_T4Q_Total`, `GP_Since_2023_Total`

Hardware totals and printer rollups (qty and GP)
- `Qty_Printers`, `GP_Printers`
- `Qty_Printers_<Subdivision>`, `GP_Printers_<Subdivision>`
- `Qty_Printer Accessories`, `GP_Printer Accessories`
- `Qty_Scanners`, `GP_Scanners`
- `Qty_Geomagic`, `GP_Geomagic`

Software seats and profit (goal-level)
- `Seats_CAD`, `GP_CAD`; `Seats_CPE`, `GP_CPE`; `Seats_Specialty Software`, `GP_Specialty Software`
- CRE rollups (dynamic): `Seats_CAD_<Rollup>`, `GP_CAD_<Rollup>`, `Seats_Specialty Software_<Rollup>`, `GP_Specialty Software_<Rollup>`
- Training subset (CRE): `GP_Training/Services_Success_Plan`, `GP_Training/Services_Training` (if available)

Division adoption inputs (for traceability)
- `adoption_assets`, `adoption_profit` (Hardware)
- `cre_adoption_assets`, `cre_adoption_profit` (CRE)
- `relationship_profit`, `cre_relationship_profit`

List-builder dynamics and mix (BI features)
- `spend_13w`, `spend_13w_prior`, `delta_13w`, `delta_13w_pct`, `spend_12m`, `spend_52w`, `yoy_13w_pct`
- `days_since_last_order`, `active_weeks_13w`, `purchase_streak_months`, `median_interpurchase_days`
- `slope_13w`, `slope_13w_prior`, `acceleration_13w`, `volatility_13w`, `seasonality_factor_13w`
- `trend_score`, `recency_score`, `magnitude_score`, `cadence_score`, `momentum_score` (+ weights)
- Hardware mix: `hw_spend_12m`, `sw_spend_12m`, `hw_share_12m`, `sw_share_12m`
- Breadth and recency: `breadth_hw_subdiv_12m`, `max_hw_subdiv`, `breadth_score_hw`, `days_since_last_hw_order`, `recency_score_hw`, `hardware_adoption_score`
- Concentration/discount/whitespace: `discount_pct`, `month_conc_hhi_12m`, `sw_dominance_score`, `sw_to_hw_whitespace_score`
- POV: `pov_primary`, `pov_tags_all`

Timestamps
- `as_of_date` — Scoring cutoff
- `run_timestamp_utc` — Pipeline run timestamp

## Neighbors (artifacts/account_neighbors.csv)

Columns
- `account_id`, `neighbor_account_id`, `neighbor_rank`
- `sim_overall`, `sim_numeric`, `sim_categorical`, `sim_text`, `sim_als`
- `neighbor_account_name`, `neighbor_industry`, `neighbor_segment`, `neighbor_territory`

Notes
- The neighbors pipeline uses exact blockwise cosine and a hybrid embedding across numeric/categorical/text/ALS blocks.

