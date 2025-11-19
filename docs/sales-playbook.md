# Sales Playbook: Building Call Lists

This guide shows how to build call lists using the ICP Dashboard.

## Where to Go
- Launch: `make dashboard` (or `streamlit run apps/dashboard.py`)
- In the app:
  - Use **Portfolio Overview** for high-level metrics.
  - Use **Account Explorer** to filter and find accounts.
  - Use **Neighbor Visualizer** to find accounts that look like your best A/B customers.

## Filters to Start With
- Segment: choose Strategic/Growth/Core based on goals.
- Adoption band: focus on higher `whitespace_score` for expansion and lower `whitespace_score` for maintenance.
- Relationship band: use `momentum_segment` and `recency_bucket` to find cross-sell opportunities.
- Toggles (via filters and columns):
  - Revenue-only (no printers): focus on higher profit and low hardware adoption.
  - Heavy fleet (>=10 weighted): filter on high printer-related metrics and `scaling_flag`.

## Export & CRM
- Use **Download watchlist** or the individual playbook download buttons to export CSV call lists.
- In **Call List Builder**, use the HW/CRE tabs and the download buttons to export division-specific lists; copy the generated email list when needed.
- Save exports under `reports/call_lists/YYYYMMDD/`.
- Import CSV into CRM using the "ICP Call List" template mapping (see CRM docs).

## Tags & Talking Points
- Upgrade Likely: high adoption, low relationship � discuss CAD/CPE packages and CRE/software add-ons.
- Consumables Focus: high printers, steady usage � auto-replenishment offers and contract attach.
- Printer Expansion: heavy fleet, newer models in portfolio � trade-in/expansion to higher-end systems.
- Cross-Division Push: filter for high `hw_to_sw_cross_sell_score` or `sw_to_hw_cross_sell_score` to pinpoint hardware-heavy or software-heavy accounts ready for the complementary motion.
- Balanced Portfolio Wins: `cross_division_balance_score` near 1.0 highlights accounts with healthy HW/SW mix�prioritize for success story references.
- Training Attachment: sort by `training_to_hw_ratio` or `training_to_cre_ratio` to see who has adopted services relative to the hardware or CRE footprint and target low-ratio accounts for enablement packages.
- Look-alike Replication: in **Look-alike Lab**, pick a hero A/B account, stage high-similarity neighbors with high whitespace or lower GP, then use Call List Builder to work those neighbors with the same play that worked on the hero.
- Orphan Look-alikes: ask your manager which territories have many dormant/long-recency neighbors to your best accounts (Manager HQ). Focus reactivation plays there first.

## Tips
- Sort by ICP Grade (A/B first), then by Adoption.
- Use Recent Profit to avoid stale accounts.

