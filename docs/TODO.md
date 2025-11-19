# TODO Checklist

This checklist captures engineering improvements and sales enablement features. Each item includes concrete steps and Done criteria.

## Schema & Constants
- [ ] Centralize column names and config
  - [x] Add `src/icp/schema.py` with constants (e.g., `COL_BIG_BOX`, `COL_SW_LICENSE`, etc.) and docstrings.
  - [ ] Replace string literals in `src/icp/scoring.py`, `src/icp/cli/score_accounts.py`, `src/icp/industry.py`.
  - [x] Add a mapping helper to handle legacy column aliases safely.
  - [ ] Update README to reference schema and expected inputs.
  - Done when: no hardcoded column strings remain in core modules and README points at the schema.

## Data Validation
- [ ] Add lightweight DF validators
  - [x] Create `src/icp/validation.py` with checks (presence, dtype, non-negativity, % missing).
  - [x] Call validations in the main pipeline before `calculate_scores()` (see `src/icp/cli/score_accounts.py`).
  - [x] Log issues to `reports/logs/validation_YYYYMMDD.log` and summarize counts.
  - [x] Surface warnings in Streamlit "Scoring Details".
  - Done when: bad inputs produce clear warnings and safe fallbacks, with logs saved and surfaced in all UIs.

## Tests
- [ ] Add pytest with focused unit tests
  - [x] Add scoring helper tests for adoption/relationship invariants (zero-only, identical series + 0.5, revenue-only sqrt cap, min-max behavior).
  - [x] Add `tests/test_schema.py` for required column presence mapping.
  - [x] Add `tests/test_industry.py` for EB shrinkage pathways.
  - [x] Document how to run tests in README.
  - Done when: tests pass locally and cover key edge cases including industry weights.

## CLI Parameters
- [ ] Parameterize paths and flags
  - [x] Add `argparse`/`typer` to `src/icp/cli/score_accounts.py`:
    - [x] `--out`, `--weights`, `--industry-weights`, `--asset-weights`, `--skip-visuals`.
  - [x] Add params to `src/icp/cli/optimize_weights.py`:
    - [x] `--n-trials`, `--lambda`, `--include-size`, `--out`.
  - [x] Update README usage examples.
  - Done when: CLIs accept overrides and default to repo standards.

## CI/CD & Quality
- [x] Add GitHub Actions
  - [x] Workflow: setup Python, install deps (with cache), run ruff lint, run pytest.
  - [x] Add `ruff` config (line length, ignores) and wire pre-commit (optional).
  - [x] Artifact: upload `coverage.xml` (optional scope).
  - Done when: PRs run lint/tests automatically.

## Sales Enablement: Call List Builder (Dashboard)
- [ ] Add a "Call List Builder" page in Streamlit
  - [x] Filters: segment, industry, region, account owner, adoption band, relationship band.
  - [x] Toggles: "Revenue-only" (no printers), "Heavy fleet (>=10 weighted)", "Newly improved score".
  - [x] Columns: rank, company, segment, `adoption_score`, `relationship_score`, `ICP_grade`, last profit, suggested playbook.
  - [x] Export: CSV/Excel to `reports/call_lists/`.
  - [x] Quick actions: "Copy email list", "Copy CSV path".
  - Done when: sales can build/export call lists interactively.

## Sales Enablement: Weekly Call List Exports
- [ ] Add CLI to generate weekly lists
  - [x] New `src/icp/cli/export_call_lists.py` with presets:
    - [x] "Top A/B by segment", "Revenue-only with high relationship", "Heavy fleet expansion".
  - [x] Write to `reports/call_lists/{yyyymmdd}/*.csv` with metadata (filters, counts).
  - [x] Add a PowerShell `scripts/export_call_lists.ps1` wrapper.
  - [x] Document how to schedule (Windows Task Scheduler example).
  - Done when: running one command produces dated call list packs.

## Similarity Search & Look-alike Experiences
- [x] Harden similarity inputs and configuration
  - [x] Inspect `artifacts/account_neighbors.csv` and confirm it includes `account_id`, `neighbor_account_id`, similarity score, and any additional columns needed for BI joins.
  - [x] Audit which numeric fields from `icp_scored_accounts` are feeding the numeric block (GP windows, HW/CRE shares, whitespace scores, training/discount metrics, momentum scores) and document them in `docs/guides/POWERBI_FIELD_REFERENCE_CLEAN.md`.
  - [x] Review and, if needed, tune `[similarity]` block weights in `config.toml` (`w_numeric`, `w_categorical`, `w_text`, `w_als`, `log1p_cols`, `logit_cols`) so no single metric dominates and HW/CRE signals are appropriately balanced.
  - [ ] (Optional) Define alternate similarity profiles (e.g., `config_hw.toml`, `config_cre.toml`) for HW- and CRE-emphasized neighbors, reusing the same engine.
  - Done when: similarity configuration is documented, stable, and clearly aligned to HW/CRE and leadership use cases.

- [x] Streamlit: Make Look-alike Lab directly actionable for sellers
  - [x] In the Look-alike Lab tab, add an anchor account selector driven by the current filters (search/slicer over A/B accounts in scope).
  - [x] For the selected anchor, show a neighbors table with columns: similarity score, GP windows (since 2023, T4Q, LastQ), HW vs CRE GP and shares, whitespace score, CRE_Training, days since last order, ICP grades (HW and CRE).
  - [x] Compute and display a concise “Why these neighbors?” explanation per anchor, highlighting 2–3 shared traits (industry, subdivision, ICP band) and 2–3 biggest opportunity gaps (e.g., CRE training, HW breadth, whitespace).
  - [x] Add row checkboxes plus a “Send to Call List Builder” button that pushes selected neighbors into a staged call list stored in Streamlit session state.
  - [x] Update the Call List Builder tab to consume the staged neighbor list (when present) and show a focused view filtered to those accounts while preserving existing division/territory filters.
  - Done when: a HW or CRE rep can pick a hero account, see its closest neighbors with clear context, and one-click stage those neighbors into a call list from within the app.

- [x] Streamlit: Division-aware neighbor views for HW and CRE
  - [x] Add a HW/CRE lens toggle inside the Look-alike Lab that chooses which division metrics (e.g., `ICP_score_hardware` vs `ICP_score_cre`, HW vs CRE GP and whitespace) are emphasized in the neighbors table.
  - [x] Ensure neighbor explanations and suggested plays use division-specific language (e.g., “CRE training whitespace”, “HW fleet expansion”) based on the selected lens.
  - [x] Validate that CRE sellers see equal value: test scenarios where neighbors surface CRE-heavy look-alikes (high training, high seats) and neighbors with CRE whitespace.
  - Done when: both HW and CRE sellers can use the same Look-alike Lab but see neighbors through a division-appropriate lens without confusing HW-only metrics for company-wide KPIs.

- [x] Streamlit: Manager & leader neighbor workflows
  - [x] Extend the Manager HQ tab to surface “hero” accounts per territory (top A-grade accounts by GP and growth for both AM_Territory and CAD_Territory).
  - [x] For each hero, compute counts and GP of neighbors in the same territory that are underpenetrated (e.g., lower grade or lower GP but similar profile, higher whitespace score) and show them in a Manager HQ table.
  - [x] Add territory-level metrics for “orphan look-alikes” (neighbors of top accounts that are cold/dormant or have long recency) to focus attention on neglected opportunities.
  - [x] Add a simple neighbor-based QBR panel: quarter-over-quarter GP growth for activated look-alikes vs the rest of the portfolio, segmented by HW and CRE.
  - Done when: frontline managers can use Streamlit to identify which hero accounts to replicate, which territories have unworked look-alikes, and how neighbor-driven motions are performing over time.

- [x] Power BI: Neighbor model integration and Similar Accounts page
  - [x] Import `account_neighbors.csv` as a table (e.g., `icp_account_neighbors`) and establish relationships to `icp_scored_accounts v1.5` by Customer ID / Account ID.
  - [x] Add calculated columns or measures to flag neighbors of the currently selected account, expose similarity score, and compute neighbor whitespace (GP gap to anchor or gap to best-in-cluster).
  - [x] Build a “Similar Accounts” report page with:
    - [x] A slicer for anchor account (by name/ID, filtered to A/B accounts in the user’s territory).
    - [x] A neighbors table showing similarity, GP metrics, HW/CRE metrics, whitespace, and key tags/playbook hints.
    - [x] Bookmarks or buttons to apply the neighbor set as a filter to the existing Call List visuals on the same or another page.
  - Done when: a seller can replicate the Streamlit Look-alike Lab behavior inside Power BI and drive a neighbor-filtered call list without leaving the dashboard.
  
  - [x] Power BI & leadership: Neighbor-driven territory and uplift views
    - [x] Create territory-level visuals that count neighbors of top A accounts per territory (AM and CAD) and aggregate their GP and whitespace.
    - [x] Add a KPI-style measure estimating potential uplift if neighbors performed at the median GP of their top-3 neighbors (for both HW and CRE), exposed at territory and division levels.
    - [x] Include a simple “Orphan look-alikes by territory” visual (count of neighbors with low activity or long recency) for leadership risk reviews.
    - [x] Integrate these visuals into QBR-ready pages and link them from the main dashboard navigation.
    - Done when: leadership can see where neighbor-driven motions could move the needle and how much upside exists by territory/division.
  
  - [x] Documentation & enablement for similarity experiences
    - [x] Extend `docs/guides/POWERBI_DASHBOARD_GUIDE.md` with a “Look-alike Lab in Power BI” section that explains the Similar Accounts page, how to choose anchors, and how to interpret similarity/whitespace metrics for HW and CRE.
    - [x] Update README and/or a Streamlit-specific guide to walk through Look-alike Lab flows for individual reps (HW and CRE) and managers (hero + neighbor campaigns, orphan look-alikes).
    - [x] Enrich `docs/sales-playbook.md` with neighbor-centric plays: “Replicate my best account”, “Activate renewal look-alikes”, “Rescue orphan look-alikes”, including example filters and talking points.
    - [ ] Add one or two QBR agenda examples highlighting how to tell the story of neighbor activation and uplift using the new visuals.
    - Done when: a new seller or manager can read the docs and immediately understand how to use similarity search in both Streamlit and Power BI to drive concrete motions.

## Sales Enablement: Propensity Tags & Playbooks
- [x] Add rule-based tags and talking points
  - [x] In `scoring.py` or `src/icp/insights.py`, create rules:
    - [x] "Upgrade Likely" (high adoption, low relationship), "Cross-sell CAD", "Consumables Focus", "Printer Expansion".
  - [x] Add `playbook` column with short talking points and next-step suggestions.
  - [x] Show tags in dashboard and include in exports.
  - Done when: each account has zero/more tags and a playbook string.

## Dashboard UX for Sales
- [x] Add "Prospect Explorer" and "Account 360"
  - [x] Prospect Explorer: table with live filters + detail panel (history snapshots, enrichment source).
  - [x] Account 360: single account view with trend sparklines, adoption breakdown, recent profit, tags/playbook.
  - [x] Add "Quick filters" buttons (A-grade only, revenue-only).
  - Done when: sales can inspect one account deeply and navigate prospects easily.

## Docs & Onboarding
- [ ] Create "How to Build My Call List" guide
  - [x] Add `docs/sales-playbook.md` with: where to click, filters to use, what tags mean, how to export and import to CRM.
  - [ ] Add screenshots/gifs of the new pages.
  - [ ] Add section in README linking to sales docs.
  - Done when: a salesperson can follow the guide without help.

## Data & Artifacts Hygiene
- [ ] Finish moving locked files and tighten ignores
  - [ ] Move `TR - Industry Enrichment.csv` to `data/raw/` (close any processes locking it).
  - [x] Confirm `.gitignore` excludes `archive/`, `data/raw/`, `data/interim/`, `.venv/`.
  - [ ] Add `reports/powerbi/` ignore if not versioning PBIX.
  - Done when: repo clean status and no large data tracked.

## Stretch Goals
- [ ] CRM-friendly export format
  - [ ] Add "Salesforce/HubSpot import" output schemes (column names/order).
  - [ ] Include a validation check to confirm required columns present.
- [ ] Score change deltas
  - [ ] Maintain a snapshot history and compute week-over-week ICP score deltas for "hot movers".
  - [ ] Add "Risers/Fallers" preset to call list exports.
- [ ] Simple propensity model
  - [ ] Train a logistic model using past conversions (if available) to add a `propensity_score` alongside rule-based tags.
