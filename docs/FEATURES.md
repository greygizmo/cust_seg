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
- `vertical_score`, `size_score`
- `ICP_score_raw`, `ICP_score` (0–100), `ICP_grade` (A–F)

## Optional time-series features (currently disabled)
Feature modules are available for spend dynamics, momentum, health/concentration, POV tags, and whitespace analysis. These are disabled in the default run to reduce memory, but can be enabled in code when needed.

