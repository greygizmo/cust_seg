
# Features Used in Scoring

This file summarizes the main features currently used by the baseline scoring pipeline. Additional feature modules exist and may be enabled later.

## Adoption (hardware-forward)
- `adoption_assets`: aggregate of asset_count, seats_sum, active_assets across focus hardware goals
- `adoption_profit`: profit since 2023 across Printer/Accessories/Scanners/Geomagic, plus 3DP Training rollup within Training/Services
- `printer_count` and `scaling_flag` (>=4 printers) as auxiliary signals

## Relationship (software-forward)
- `relationship_profit`: profit since 2023 across CAD, CPE, Specialty Software
- Per-goal seats and profit: `Seats_CAD/CPE/Specialty Software`, `GP_CAD/CPE/Specialty Software`

## Activity and breadth
- `active_assets_total`, `seats_sum_total`, `Portfolio_Breadth`
- Key dates and day-since metrics for earliest/latest purchase/expiration
- `Months_Active_12M` from monthly profit activity

## Scores
- `Hardware_score` (adoption) and `Software_score` (relationship)
- `ICP_score_hardware`, `ICP_grade_hardware` (Hardware division)
- `ICP_score_cre`, `ICP_grade_cre` (CRE division)

## Active Time-Series & Enrichment Features
The pipeline now enriches the output with advanced signals for BI and List Building:
- **Spend Dynamics**: `spend_13w`, `spend_12m`, `delta_13w_pct`, `slope_13w`.
- **Momentum**: `momentum_score`, `recency_score`, `cadence_score`.
- **Cross-Division**: `cross_division_balance_score`, `hw_to_sw_cross_sell_score`.
- **Whitespace**: `sw_to_hw_whitespace_score`, `sw_dominance_score`.
- **POV**: `pov_primary`, `pov_tags_all`.

These features are appended to `icp_scored_accounts.csv` after the core scoring logic.
