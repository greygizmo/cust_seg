# Sales Playbook: Building Call Lists

This guide shows how to build call lists using the ICP Dashboard.

## Where to Go
- Launch: `streamlit run apps/streamlit/app.py`
- In the app, open the **Execution Hub** tab and use the **Opportunity Watchlist** and **Playbooks** sections as your call list builder.

## Filters to Start With
- Segment: choose Strategic/Growth/Core based on goals.
- Adoption band: focus on higher `whitespace_score` for expansion and lower `whitespace_score` for maintenance.
- Relationship band: use `momentum_segment` and `recency_bucket` to find cross-sell opportunities.
- Toggles (via filters and columns):
  - Revenue-only (no printers): focus on higher profit and low hardware adoption.
  - Heavy fleet (>=10 weighted): filter on high printer-related metrics and `scaling_flag`.

## Export & CRM
- Use **Download watchlist** or the individual playbook download buttons to export CSV call lists.
- Save exports under `reports/call_lists/YYYYMMDD/`.
- Import CSV into CRM using the "ICP Call List" template mapping (see CRM docs).

## Tags & Talking Points
- Upgrade Likely: high adoption, low relationship → discuss CAD/CPE packages.
- Consumables Focus: high printers, steady usage → auto-replenishment offers.
- Printer Expansion: heavy fleet, newer models in portfolio → trade-in/expansion.
- Cross-Division Push: filter for high `hw_to_sw_cross_sell_score` or `sw_to_hw_cross_sell_score` to pinpoint hardware-heavy or software-heavy accounts ready for the complementary motion.
- Balanced Portfolio Wins: `cross_division_balance_score` near 1.0 highlights accounts with healthy HW/SW mix—prioritize for success story references.
- Training Attachment: sort by `training_to_hw_ratio` or the new `training_to_cre_ratio` to see who has adopted services relative to the hardware or CRE footprint and target low-ratio accounts for enablement packages.

## Tips
- Sort by ICP Grade (A/B first), then by Adoption.
- Use Recent Profit to avoid stale accounts.
