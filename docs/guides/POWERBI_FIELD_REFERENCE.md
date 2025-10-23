# Power BI Field Reference: Scored Accounts and Neighbors

This guide explains every column in the two core artifacts that power the dashboard:
- `data/processed/icp_scored_accounts.csv` (Scored Accounts)
- `artifacts/account_neighbors.csv` (Account Neighbors)

It’s written for sellers and managers to understand what each field means, where it comes from, and how to use it in conversations and targeting.

---

## Scored Accounts (icp_scored_accounts.csv)

Identity and ownership
- `Customer ID` — Unique account identifier from NetSuite/SalesLog.
- `Company Name` — Human‑readable account name (customer header), cleaned of any leading numeric ID.
- `activity_segment` — Warm if seen in Sales Log since Jan 1, 2023; Cold if not, but owns hardware assets.
- `am_sales_rep` — Account Manager owner (from NetSuite customer headers).
- `AM_Territory` — Territory string for the account.
- `edu_assets` — EDU assets flag (1/0 or Yes/No) if applicable.

Key contacts (optional presence)
- `RP_Primary_Name`, `RP_Primary_Email`, `RP_Primary_Phone` — Designated primary hardware contact.
- `Primary_Contact_Name`, `Primary_Contact_Email`, `Primary_Contact_Phone` — General primary contact.
- `Name`, `email`, `phone` — Back‑compat generic fields (mirror the designated contacts where available).

Shipping address (optional presence)
- `ShippingAddr1`, `ShippingAddr2`, `ShippingCity`, `ShippingState`, `ShippingZip`, `ShippingCountry` — Latest known shipping fields.

Scores and grades
- `ICP_score` — Overall Ideal Customer Profile score on a 0–100 scale. Higher is better.
  - Built from four component scores (Vertical, Size, Adoption, Relationship) using weights in `artifacts/weights/optimized_weights.json` (falls back to defaults if absent).
  - Normalized to a bell‑curve 0–100 distribution for easy reading.
- `ICP_grade` — Letter grade bucket (A–F) derived from the normalized ICP_score.
- `Hardware_score` — Adoption component; how strong the account’s hardware footprint and recent activity are.
- `Software_score` — Relationship component; strength of CAD/CPE/Specialty Software adoption and profitability.
- `vertical_score` — Vertical fit based on industry performance weights.
- `size_score` — Size fit; revenue/scale context proxy.
- `ICP_score_raw` — Un‑normalized score prior to mapping to 0–100.

Industry context
- `Industry` — Current industry classification (optionally enriched from CSV).
- `Industry Sub List` — Sub‑industry tags for additional context (CSV enrichment).
- `Industry_Reasoning` — Free‑text notes explaining enrichment decisions (if provided).

Recent profitability (GP = Gross Profit + Term_GP)
- `GP_PrevQ_Total` — Profit last quarter prior to the most recent quarter.
- `GP_QoQ_Growth` — Quarter‑over‑quarter growth vs. prior quarter (ratio or delta; positive is growth).
- `GP_T4Q_Total` — Profit across trailing four quarters (T4Q).
- `GP_Since_2023_Total` — Profit summed since Jan 1, 2023.

Hardware totals and printer rollups
- `Qty_Printers`, `GP_Printers` — Total printer quantity and profit.
- Per‑subdivision rollups (quantity and profit). Each pair describes activity in that hardware category:
  - `Qty_Printers_AM_Software`, `GP_Printers_AM_Software`
  - `Qty_Printers_AM_Support`, `GP_Printers_AM_Support`
  - `Qty_Printers_Consumables`, `GP_Printers_Consumables`
  - `Qty_Printers_FDM`, `GP_Printers_FDM`
  - `Qty_Printers_FormLabs`, `GP_Printers_FormLabs`
  - `Qty_Printers_Metals`, `GP_Printers_Metals`
  - `Qty_Printers_P3`, `GP_Printers_P3`
  - `Qty_Printers_Polyjet`, `GP_Printers_Polyjet`
  - `Qty_Printers_Post_Processing`, `GP_Printers_Post_Processing`
  - `Qty_Printers_SAF`, `GP_Printers_SAF`
  - `Qty_Printers_SLA`, `GP_Printers_SLA`
  - `Qty_Printers_Spare_Parts_Repair_Parts_Time_and_Materials`, `GP_Printers_Spare_Parts_Repair_Parts_Time_and_Materials`
- `Qty_Printer Accessories`, `GP_Printer Accessories` — Totals for printer accessories.
- `Qty_Scanners`, `GP_Scanners` — Scanner totals.
- `Qty_Geomagic`, `GP_Geomagic` — Geomagic totals.

Software seats and profit (goal‑level)
- `Seats_CAD`, `GP_CAD` — CAD seats and profit.
- `Seats_CPE`, `GP_CPE` — CPE seats and profit.
- `Seats_Specialty Software`, `GP_Specialty Software` — Specialty software seats and profit.

Adoption helpers and totals
- `scaling_flag` — 1 if printer_count ≥ 4 (multi‑printer scaling signal), else 0.
- `Total Software License Revenue` — Most recent total software license revenue (used as a backup for relationship scoring).
- `active_assets_total` — Total active assets across the portfolio.
- `seats_sum_total` — Total seats across products.
- `Portfolio_Breadth` — Count of unique rollups engaged by the account (breadth of portfolio).

Lifecycle dates and “days since” metrics
- `EarliestPurchaseDate` — First recorded purchase date for the account.
- `LatestPurchaseDate` — Most recent purchase date.
- `LatestExpirationDate` — Most recent expiration date (for term‑based products).
- `Days_Since_First_Purchase` — Days from earliest purchase to today.
- `Days_Since_Last_Purchase` — Days since the most recent purchase.
- `Days_Since_Last_Expiration` — Days since the most recent expiration.

How to use these in Power BI
- Prioritization: Sort by `ICP_grade` or `ICP_score` within your territory to focus outreach.
- Hardware expansion: Filter by `scaling_flag = 1`, then check printer subdivisions to spot upgrade paths.
- Software opportunity: Use `Software_score`, `Seats_*` and `GP_*` goal fields to identify CAD/CPE/Specialty gaps.
- Account context: `Industry`, `Industry Sub List`, and `Industry_Reasoning` help tailor messaging.
- Health check: Review `GP_T4Q_Total` trends and `Days_Since_Last_Purchase` to preempt churn.

Notes
- Profit fields (GP_*) are derived from Azure SQL SalesLog and cover activity since 2023‑01‑01.
- Missing contact/address fields simply indicate unavailable data — not a data error.

---

## Account Neighbors (account_neighbors.csv)

Identification
- `account_id` — Source account (string form of `Customer ID`).
- `neighbor_account_id` — Recommended similar account.
- `neighbor_rank` — 1 is most similar; higher numbers are next‑closest matches.

Similarities (0–1 typical; higher is more similar)
- `sim_overall` — Combined similarity (weighted mixture of all blocks below).
- `sim_numeric` — How similar the quantitative profile is (scores, amounts, ratios).
- `sim_categorical` — Match on category flags (industry, warm/cold segment, territory, top hardware subdivision if present).
- `sim_text` — Similarity of industry reasoning text (NLP embedding) when available.
- `sim_als` — Collaborative look‑alike similarity from product/rollup interactions (ALS embeddings).

Neighbor metadata (for the recommended account)
- `neighbor_account_name` — Friendly name of the neighbor account.
- `neighbor_industry` — Industry label.
- `neighbor_segment` — Warm/Cold flag (same concept as `activity_segment`).
- `neighbor_territory` — Territory string.

How to use these in Power BI
- Look‑alike targeting: For a high‑value win, find neighbors with similar profiles and replicate the play.
- Conversation starters: Use `neighbor_industry` and `neighbor_territory` to bring relevant proof points.
- Block diagnostics: Compare `sim_numeric` vs `sim_als` — high `sim_als` suggests collaborative/product usage patterns are driving the match.

Notes
- Overall similarity uses weights from `config.toml` ([similarity] block). ALS components are optional and included when enabled.
- Exact neighbor search is used (no approximation), computed in blocks to keep memory reasonable.

---

## FAQ for Sellers
- Why is my account “Cold” but listed? Because we found hardware assets/seats but no recent SalesLog activity since 2023 — a re‑engagement opportunity.
- Are scores static? No. They’re recalculated when data updates; ICP scores reflect the latest industry weighting, account size, and adoption/relationship signals.
- Does higher `sim_als` mean a stronger upsell? It indicates similar buying patterns; combine with `sim_numeric` and recent profit to prioritize.

If you need a one‑pager or training module derived from this guide, we can provide a condensed version with examples tailored to your territory.
