# Power BI Field Reference: Scored Accounts and Neighbors

This guide documents all columns emitted by the scoring pipeline for:
- `data/processed/icp_scored_accounts.csv` (Scored Accounts)
- `artifacts/account_neighbors.csv` (Account Neighbors)

It explains what each column means, where it comes from, and how to use it.

---

## Scored Accounts (icp_scored_accounts.csv)

Identity and ownership
- `Customer ID` â€“ Unique account identifier from NetSuite/SalesLog.
- `Company Name` â€“ Humanâ€‘readable account name (customer header), cleaned of any leading numeric ID.
- `activity_segment` â€“ Warm if seen in Sales Log since 2023â€‘01â€‘01; Cold if not, but owns hardware assets.
- `am_sales_rep` â€“ Account Manager owner (from NetSuite customer headers).
- `AM_Territory` â€“ Territory string for the account.
- `edu_assets` â€“ EDU assets flag (Yes/No in the CSV for BI friendliness).

Key contacts (optional presence)
- `RP_Primary_Name`, `RP_Primary_Email`, `RP_Primary_Phone` â€“ Designated primary hardware contact.
- `Primary_Contact_Name`, `Primary_Contact_Email`, `Primary_Contact_Phone` â€“ General primary contact.
- `Name`, `email`, `phone` â€“ Backâ€‘compat generic fields (mirror the designated contacts where available).

Shipping address (optional presence)
- `ShippingAddr1`, `ShippingAddr2`, `ShippingCity`, `ShippingState`, `ShippingZip`, `ShippingCountry` â€“ Latest known shipping fields.

Scores and grades
- `ICP_score` â€“ Overall Ideal Customer Profile score on a 0â€“100 scale (rankâ€‘normalized to N(50,15)).
- `ICP_grade` â€“ Letter grade bucket (Aâ€“F) from percentile targets (A=10%, B=20%, C=40%, D=20%, F=10%).
- `Hardware_score` â€“ Adoption component (divisionâ€‘aware) â€“ recent footprint and breadth.
- `Software_score` â€“ Relationship component (divisionâ€‘aware) â€“ CAD/Specialty adoption and profitability.
- `vertical_score` â€“ Industry fit from dynamic weights (Empiricalâ€‘Bayes Ã— strategic blend).
- `size_score` â€“ Size/scale proxy derived from revenue bands.
- `ICP_score_raw` â€“ Unâ€‘normalized weighted score before mapping to 0â€“100.

Notes on adoption/relationship naming
- `Hardware_score` and `Software_score` are aliases of the underlying `adoption_score` and `relationship_score` components and are division-aware. In CRE, `Hardware_score` still refers to the adoption component (CAD/Specialty adoption depth), while `Software_score` captures relationship/profitability (Specialty + CRE training subset).


Industry context
- `Industry` â€“ Current industry classification (enriched from CSV when available).
- `Industry Sub List` â€“ Subâ€‘industry tags for context (enriched).
- `Industry_Reasoning` â€“ Notes explaining enrichment decisions when present.

Recent profitability (GP = Gross Profit + Term_GP)
- `GP_PrevQ_Total` â€“ Profit in the completed quarter prior to the most recent quarter.
- `GP_QoQ_Growth` â€“ Quarterâ€‘overâ€‘quarter growth vs. prior completed quarter.
- `GP_T4Q_Total` â€“ Profit across the last four completed quarters.
- `GP_Since_2023_Total` â€“ Profit summed since 2023â€‘01â€‘01.

Hardware totals and printer rollups (quantity and GP)
- `Qty_Printers`, `GP_Printers` â€“ Totals for printer division.
- Rollups: `Qty_Printers_<Rollup>`, `GP_Printers_<Rollup>` for each printer subdivision (e.g., `FDM`, `FormLabs`, `Polyjet`, `SLA`, `SAF`, etc.).
- Additional hardware: `Qty_Printer Accessories`, `GP_Printer Accessories`, `Qty_Scanners`, `GP_Scanners`, `Qty_Geomagic`, `GP_Geomagic`.

Software seats and profit (goalâ€‘level)
- `Seats_CAD`, `GP_CAD` â€“ CAD seat totals and profit.
- `Seats_CPE`, `GP_CPE` â€“ CPE seat totals and profit.
- `Seats_Specialty Software`, `GP_Specialty Software` â€“ Specialty Software seats and profit.

CRE rollups (seats and GP)
- Seats by rollup (dynamic): `Seats_CAD_<Rollup>`, `Seats_Specialty Software_<Rollup>` â€“ per item_rollup seat totals.
- GP by rollup (dynamic): `GP_CAD_<Rollup>`, `GP_Specialty Software_<Rollup>` â€“ per item_rollup GP totals.
- Training/Services (CREâ€‘only subset): when present in data, `GP_Training/Services_Success_Plan`, `GP_Training/Services_Training`.
- CRE aggregate signals (divisionâ€‘aware inputs to scoring): `cre_adoption_assets`, `cre_adoption_profit`, `cre_relationship_profit`.

Adoption helpers and totals
- `scaling_flag` â€“ 1 if printer_count â‰¥ 4 (multiâ€‘printer scaling signal), else 0.
- `Total Software License Revenue` â€“ Relationship fallback feature when division config permits.
- `active_assets_total` â€“ Count of active assets across all goals.
- `seats_sum_total` â€“ Sum of seats across all goals.
- `Portfolio_Breadth` â€“ Count of unique rollups engaged by the account.

Lifecycle dates and daysâ€‘since metrics
- `EarliestPurchaseDate`, `LatestPurchaseDate`, `LatestExpirationDate` â€“ Lifecycle anchors.
- `Days_Since_First_Purchase`, `Days_Since_Last_Purchase`, `Days_Since_Last_Expiration` â€“ Age/recency indicators.

Momentum, mix, health, whitespace, POV
- Spending dynamics: `spend_13w`, `spend_13w_prior`, `delta_13w`, `delta_13w_pct`, `spend_12m`, `spend_52w`, `yoy_13w_pct`.
- Cadence/recency/trend: `days_since_last_order`, `active_weeks_13w`, `purchase_streak_months`, `median_interpurchase_days`, `slope_13w`, `slope_13w_prior`, `acceleration_13w`, `volatility_13w`, `seasonality_factor_13w`, `trend_score`, `recency_score`, `magnitude_score`, `cadence_score`, `momentum_score`, `w_trend`, `w_recency`, `w_magnitude`, `w_cadence`.
- Mix/adoption: `hw_spend_12m`, `sw_spend_12m`, `hw_share_12m`, `sw_share_12m`, `breadth_hw_subdiv_12m`, `max_hw_subdiv`, `breadth_score_hw`, `days_since_last_hw_order`, `recency_score_hw`, `hardware_adoption_score`, `consumables_to_hw_ratio`, `top_subdivision_12m`, `top_subdivision_share_12m`.
- Health/whitespace: `discount_pct`, `month_conc_hhi_12m`, `sw_dominance_score`, `sw_to_hw_whitespace_score`.
- POV tags: `pov_primary`, `pov_tags_all`.

Printerâ€‘only dynamics
- Suffix `_printers` indicates dynamics computed only on printer division (e.g., `spend_12m_printers`).

Percentiles (0â€“100)
- Any column ending with `_pctl` is a percentile rank for easier BI interpretation (e.g., `spend_12m_pctl`).

Timestamps
- `as_of_date` â€“ The analysis anchor date (e.g., latest transaction date considered).
- `run_timestamp_utc` â€“ UTC timestamp when the pipeline produced the row.

Dynamic naming conventions
- Rollup columns are discovered from product taxonomy and emitted when present in data.
- Column slugs follow `Seats_<Goal>_<RollupSlug>` and `GP_<Goal>_<RollupSlug>` (nonâ€‘alphanumeric characters mapped to underscores; consistent with printer rollups).
- Training/Services for CRE includes only `Success Plan` and `Training` rollups.

Notes
- Profit fields (GP_*) are derived from Azure SQL SalesLog since 2023â€‘01â€‘01.
- Some contact/address columns are optional and may be blank.

---

## Account Neighbors (account_neighbors.csv)

Identification
- `account_id` â€“ Source account (string form of `Customer ID`).
- `neighbor_account_id` â€“ Recommended similar account.
- `neighbor_rank` â€“ 1 is most similar; higher numbers are nextâ€‘closest matches.

Similarities (0â€“1 typical; higher is more similar)
- `sim_overall` â€“ Combined similarity.
- `sim_numeric` â€“ Quantitative profile similarity (scores, amounts, ratios).
- `sim_categorical` â€“ Category flags (industry, segment, territory, top hardware subdivision).
- `sim_text` â€“ Similarity of enrichment text when available.
- `sim_als` â€“ Collaborative lookâ€‘alike similarity from product/rollup interactions (ALS embeddings).

Neighbor metadata
- `neighbor_account_name`, `neighbor_industry`, `neighbor_segment`, `neighbor_territory`.

Notes
- Uses weights from `config.toml` ([similarity] block). ALS components are optional.
- Exact neighbor computation (no approximation), with memoryâ€‘friendly blocking.

---

## ICP Weights: What They Are and How We Use Them

Major criteria
- Vertical (industry fit), Size (scale proxy), Adoption (footprint/activity), Relationship (software depth/profit).
- Divisionâ€‘aware signals feed Adoption and Relationship differently for Hardware vs. CRE.

Where weights come from
- Default perâ€‘division component weights live in `src/icp/divisions.py` and can be overridden by `artifacts/divisions/<division>.json`.
- Optimized weights are stored in `artifacts/weights/optimized_weights.json` and are loaded at runtime. If absent, defaults are used.

Current optimization process (baseline)
- The script `src/icp/cli/optimize_weights.py` uses Optuna to tune weights to:
  - Maximize predictive power (correlation with a target revenue/profit proxy).
  - Match the business grade distribution (A/B/C/D/F targets).
- Today this uses a single target and dataset; historically biased toward hardware revenue proxies.

Recommended improvements
- Separate optimized weights per superâ€‘division:
  - Hardware target: future hardware profit/revenue (e.g., nextâ€‘quarter or T4Q GP for printers/accessories/scanners/geomagic) and adoption stability.
  - CRE target: future software profit/revenue (CAD + Specialty + CREâ€‘Training subset) and seat retention/expansion.
- Divisionâ€‘aware training sets and objectives:
  - Use timeâ€‘based CV (train on prior windows, validate on the next quarter) to reduce leakage.
  - Multiâ€‘objective score: maximize Spearman correlation with future outcomes, penalize deviation from target grade distribution, and add stability penalties across territories/industries.
  - Monotonic constraints or priors to avoid degenerate solutions (e.g., minimum mass on Adoption for Hardware, minimum mass on Relationship for CRE).
- Robustness controls:
  - Regularize toward division defaults; shrinkage when sample sizes are small.
  - Guardrails for `size_score` to prevent overweighting scale where it doesnâ€™t signal opportunity.
- Output format:
  - Save `optimized_weights.json` as a mapping per division (e.g., `{ "hardware": {...}, "cre": {...} }`) with metadata (train window, target choice, CV metrics).
  - Update the loader in scoring to pick weights by `--division`.

What ICP measures (intent)
- A forwardâ€‘looking fit score that blends structural fit (industry, size) and actionable signals (adoption, relationship) to prioritize where we win and can grow next.
- For Hardware, Adoption deserves higher weight; for CRE, Relationship often carries more signal. Let the perâ€‘division optimization confirm and calibrate these intuitions against future outcomes.

Next steps to implement
- Add `--division` to the optimization CLI and choose divisionâ€‘specific targets.
- Build timeâ€‘split CV and perâ€‘division output format.
- Wire scoring to load perâ€‘division optimized weights with fallback to defaults.
- Document chosen targets and CV metrics alongside the saved weights for transparency.

---

## Divisionâ€‘Specific Weight Tuning Strategy (Deep Dive)

Situation
- We score two superâ€‘divisions with distinct buying signals:
  - Hardware: success correlates with multiâ€‘printer adoption/breadth and nearâ€‘term GP in printers/accessories/scanners/geomagic.
  - CRE: success correlates with software depth and profitability (CAD + Specialty), plus CREâ€‘specific training rollups (Success Plan, Training).
- Objective: surface bestâ€‘fit accounts to prioritize resources (coverage, pipeline creation, expansion) with interpretable, stable scores and the desired A/B/C/D/F grade mix.

Data available (from Azure SQL and engineered features)
- Transaction aggregates by goal and rollup: quarterly GP since 2023, lastâ€‘90â€‘days GP, monthly GP.
- Assets & seats by goal/rollup (active flags, date ranges).
- Enriched industry labels and tiers; dynamic industry weights via Empiricalâ€‘Bayes shrinkage.
- Feature blocks used for downstream prioritization and diagnostics (momentum, mix, health, whitespace, POV tags, percentiles).

What the ICP should measure
- A forwardâ€‘looking fit: the likelihood and magnitude of profitable engagement for the specific superâ€‘division, not just current spend.
- A calibrated blend of structural fit (industry, size) and actionable signals (adoption/relationship) that generalizes across territories and reduces false positives.

Proposed approach (per superâ€‘division)
- Separate optimizations per division with divisionâ€‘specific targets and priors; save weights per division.
- Use timeâ€‘based splits to avoid leakage: train on historical asâ€‘of snapshots; predict the next quarter (or next two quarters) outcomes.
- Optimize a composite objective that balances predictive power and business shape:
  - Predictive: maximize Spearman correlation with future outcome (robust to monotonic relationships) and topâ€‘decile lift (precision@k).
  - Distribution: penalize deviation from target grade mix (A/B/C/D/F) via KLâ€‘divergence to targets.
  - Stability: add a penalty for large weight variance across territories/industries (groupâ€‘wise score variance regularization).

Hardware ICP (printers and related divisions)
- Target/outcome (next 1â€“2 quarters):
  - Primary: GP in Hardware divisions (Printers, Printer Accessories, Scanners, Geomagic).
  - Secondary classification: any hardware purchase (conversion) to capture lift when amounts are sparse.
- Priors and bounds (example search box):
  - Adoption: 0.40â€“0.70 (printers + revenue fallback), Relationship: 0.15â€“0.35, Vertical: 0.15â€“0.40, Size: 0.00â€“0.15. Sum=1.
  - Rationale: adoption/breadth are the strongest forward signals in hardware; size helps but should not dominate.
- Method details:
  - Train/validation windows: rolling quarters (e.g., train up to 2024Q2, validate 2024Q3; repeat).
  - Objective: minimize composite loss L = âˆ’corr_spearman âˆ’ Î±Â·lift@10% + Î²Â·KL(grade||target) + Î³Â·stability.
  - Select weights by CVâ€‘mean performance with variance guard (choose the most stable among the top performers).

CRE ICP (CAD, Specialty Software, CREâ€‘Training subset)
- Target/outcome (next 1â€“2 quarters):
  - Primary: GP in CAD + Specialty Software + CRE Training rollups (Success Plan, Training) and/or seat growth.
  - Secondary: renewal/expansion classification if available (e.g., seat increases, upsell flags).
- Priors and bounds (example search box):
  - Relationship: 0.35â€“0.60, Adoption: 0.20â€“0.50, Vertical: 0.15â€“0.40, Size: 0.00â€“0.15. Sum=1.
  - Rationale: software relationship depth/profitability carries stronger forward signal; adoption still matters (CAD/Specialty footprints).
- Method details:
  - Same rolling timeâ€‘based CV; put more weight on expansionâ€‘related outcomes where available.
  - Objective mirrors Hardware but weights lift where â€œconversionâ€ is seat expansion/renewal.

Industry and grade calibration
- Industry weights are already recomputed with Empiricalâ€‘Bayes shrinkage plus strategic blending; keep refreshing on a cadence (monthly/quarterly).
- Maintain grade mix per division to keep sales motions balanced; optionally display grade distribution by territory for fairness.

Guardrails and fairness
- Apply lower bounds on Adoption (Hardware) and Relationship (CRE) to prevent degenerate, â€œsizeâ€‘onlyâ€ solutions.
- Regularize toward division defaults when sample sizes are small; shrink across territories.
- Monitor topâ€‘decile lift and precision across industries/territories to detect bias.

Implementation plan
- Data snapshots: generate component scores and outcomes at historical asâ€‘of dates; assemble perâ€‘division training sets.
- Extend optimizer CLI to accept `--division`, `--as-of`, and horizon configuration; output perâ€‘division weights with metadata (train window, CV, targets).
- Update scoring loader to pick optimized weights for the requested division with fallback to defaults.
- Add monitoring artifacts: gains charts, topâ€‘k coverage, grade distribution by territory/industry.

---

## Optimization Artifacts and Metrics

What is saved
- `artifacts/weights/optimized_weights.json` â€“ Stores perâ€‘division weights and evaluation metadata under `meta`.
- `artifacts/weights/*_industry_weights.json` â€“ Divisionâ€‘specific dynamic industry weights used to compute `vertical_score`.

Key metrics recorded (per study)
- `spearman_mean` â€“ Mean Spearman correlation between weighted score and future GP across time folds.
- `kl_divergence_mean` â€“ Mean KL divergence of the achieved A/B/C/D/F mix versus the target grade distribution.
- `stability_std_spearman_mean` â€“ Mean standard deviation of withinâ€‘group Spearman correlations (lower is more stable) using the chosen `group_col` (default `Industry`).
- `lifts` â€“ Perâ€‘grade lift values such as `lift@A`, `lift@B`, â€¦ computed as share of outcome captured within each grade versus baseline.
- `lift_curves` â€“ Cumulative lift curve points and `auc` for each horizon (e.g., `"1"`, `"2"` for 1Q and 2Q).

How labels are built (no ML)
- The optimizer reads `data/processed/icp_scored_accounts.csv` and uses its `as_of_date` to build futureâ€‘quarter labels from Azure SQL:
  - Hardware: Profit (GP + Term_GP) in next quarter for goals: Printers, Printer Accessorials, Scanners, Geomagic.
  - CRE: Profit in next quarter for goals: CAD, Specialty Software, plus Training/Services restricted to rollups `Success Plan` and `Training`.

Bounds and weights
- Size is fixed to 0.0 in v1.5 (both divisions). Search ranges are the same for both divisions to avoid overfitting:
  - Vertical: [0.15, 0.45]
  - Adoption: [0.20, 0.55]
  - Relationship: [0.20, 0.55]

Running studies (examples)
- Score accounts at a historical cutoff to populate `as_of_date` and refresh industry weights:
  - `set ICP_AS_OF_DATE=2024-06-30`
  - `python -m icp.cli.score_accounts --division hardware --skip-visuals --skip-neighbors`
  - `python -m icp.cli.score_accounts --division cre --skip-visuals --skip-neighbors`
- Optimize per division with time folds (1Q,2Q) and 5,000 trials:
  - `python -m icp.cli.optimize_weights --division hardware --n-trials 5000 --horizons 1,2`
  - `python -m icp.cli.optimize_weights --division cre --n-trials 5000 --horizons 1,2`

History tracking
- Append this runâ€™s metadata to a persistent history in the same file:
  - `python -m icp.cli.optimize_weights --division hardware --n-trials 5000 --horizons 1,2 --append-history`
  - `python -m icp.cli.optimize_weights --division cre --n-trials 5000 --horizons 1,2 --append-history`
- The optimizer writes per-division meta under `meta.<division>` and, when `--append-history` is used, appends an entry to `meta_history` with `division`, `as_of_date`, `run_timestamp_utc`, lifts, AUCs, and label counts.

Operational tips
- If lifts or correlations are near zero, choose an earlier `ICP_AS_OF_DATE` to ensure nonâ€‘zero future quarters exist in the label window.
- To force industry weight refresh, remove `artifacts/weights/hardware_industry_weights.json` or `artifacts/weights/cre_industry_weights.json` before scoring.
- After locking weights, unset `ICP_AS_OF_DATE` and reâ€‘run scoring for production outputs.


## Division-Specific Scores (New)

To support both Hardware and CRE scoring in one table without breaking existing visuals, the CSV now includes:
- ICP_score_hardware, ICP_grade_hardware � Overall score/grade computed with Hardware division weights and signals.
- ICP_score_cre, ICP_grade_cre � Overall score/grade computed with CRE division weights and signals.

Compatibility notes
- Legacy ICP_score and ICP_grade remain and mirror the Hardware score/grade for backward compatibility with existing visuals and measures.
- All other component columns (e.g., Hardware_score, Software_score, ertical_score) remain unchanged and are division-aware as documented above.
