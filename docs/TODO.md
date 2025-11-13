# TODO Checklist

This checklist captures engineering improvements and sales enablement features. Each item includes concrete steps and Done criteria.

## Schema & Constants
- [ ] Centralize column names and config
  - [ ] Add `src/icp/schema.py` with constants (e.g., `COL_BIG_BOX`, `COL_SW_LICENSE`, etc.) and docstrings.
  - [ ] Replace string literals in `src/icp/scoring.py`, `src/icp/cli/score_accounts.py`, `src/icp/industry.py`.
  - [ ] Add a mapping helper to handle legacy column aliases safely.
  - [ ] Update README to reference schema and expected inputs.
  - Done when: no hardcoded column strings remain in core modules.

## Data Validation
- [ ] Add lightweight DF validators
  - [ ] Create `src/icp/validation.py` with checks (presence, dtype, non-negativity, % missing).
  - [ ] Call validations in `assemble_master_from_db()` and before `calculate_scores()`.
  - [ ] Log issues to `reports/logs/validation_YYYYMMDD.log` and summarize counts.
  - [ ] Surface warnings in CLI output and Streamlit “Scoring Details”.
  - Done when: bad inputs produce clear warnings and safe fallbacks, with logs saved.

## Tests
- [ ] Add pytest with focused unit tests
  - [ ] Add `tests/test_scoring.py` for adoption/relationship invariants (zero-only, identical series → 0.5, heavy fleet bonus, revenue-only sqrt cap, min-max behavior).
  - [ ] Add `tests/test_schema.py` for required column presence mapping.
  - [ ] Add `tests/test_industry.py` for EB shrinkage pathways.
  - [ ] Document how to run tests in README.
  - Done when: tests pass locally and cover key edge cases.

## CLI Parameters
- [ ] Parameterize paths and flags
  - [ ] Add `argparse`/`typer` to `src/icp/cli/score_accounts.py`:
    - [ ] `--out`, `--weights`, `--industry-weights`, `--asset-weights`, `--skip-visuals`.
  - [ ] Add params to `src/icp/cli/optimize_weights.py`:
    - [ ] `--n-trials`, `--lambda`, `--include-size`, `--out`.
  - [ ] Update README usage examples.
  - Done when: CLIs accept overrides and default to repo standards.

## CI/CD & Quality
- [ ] Add GitHub Actions
  - [ ] Workflow: setup Python, install deps (with cache), run ruff lint, run pytest.
  - [ ] Add `ruff` config (line length, ignores) and wire pre-commit (optional).
  - [ ] Artifact: upload `coverage.xml` (optional scope).
  - Done when: PRs run lint/tests automatically.

## Sales Enablement: Call List Builder (Dashboard)
- [ ] Add a “Call List Builder” page in Streamlit
  - [ ] Filters: segment, industry, region, account owner, adoption band, relationship band.
  - [ ] Toggles: “Revenue-only” (no printers), “Heavy fleet (≥10 weighted)”, “Newly improved score”.
  - [ ] Columns: rank, company, segment, adoption_score, relationship_score, ICP_grade, last profit, suggested playbook.
  - [ ] Export: CSV/Excel to `reports/call_lists/`.
  - [ ] Quick actions: “Copy email list”, “Copy CSV path”.
  - Done when: sales can build/export call lists interactively.

## Sales Enablement: Weekly Call List Exports
- [ ] Add CLI to generate weekly lists
  - [ ] New `src/icp/cli/export_call_lists.py` with presets:
    - [ ] “Top A/B by segment”, “Revenue-only with high relationship”, “Heavy fleet expansion”.
  - [ ] Write to `reports/call_lists/{yyyymmdd}/…csv` with metadata (filters, counts).
  - [ ] Add a PowerShell `scripts/export_call_lists.ps1` wrapper.
  - [ ] Document how to schedule (Windows Task Scheduler example).
  - Done when: running one command produces dated call list packs.

## Sales Enablement: Propensity Tags & Playbooks
- [ ] Add rule-based tags and talking points
  - [ ] In `scoring.py` or `src/icp/insights.py`, create rules:
    - [ ] “Upgrade Likely” (high adoption, low relationship), “Cross-sell CAD”, “Consumables Focus”, “Printer Expansion”.
  - [ ] Add `playbook` column with short talking points and next-step suggestions.
  - [ ] Show tags in dashboard and include in exports.
  - Done when: each account has zero/more tags and a playbook string.

## Dashboard UX for Sales
- [ ] Add “Prospect Explorer” and “Account 360”
  - [ ] Prospect Explorer: table with live filters + detail panel (history snapshots, enrichment source).
  - [ ] Account 360: single account view with trend sparklines, adoption breakdown, recent profit, tags/playbook.
  - [ ] Add “Quick filters” buttons (A-grade only, heavy fleet, revenue-only).
  - Done when: sales can inspect one account deeply and navigate prospects easily.

## Docs & Onboarding
- [ ] Create “How to Build My Call List” guide
  - [ ] Add `docs/sales-playbook.md` with: where to click, filters to use, what tags mean, how to export and import to CRM.
  - [ ] Add screenshots/gifs of the new pages.
  - [ ] Add section in README linking to sales docs.
  - Done when: a salesperson can follow the guide without help.

## Data & Artifacts Hygiene
- [ ] Finish moving locked files and tighten ignores
  - [ ] Move `TR - Industry Enrichment.csv` to `data/raw/` (close any processes locking it).
  - [ ] Confirm `.gitignore` excludes `archive/`, `data/raw/`, `data/interim/`, `.venv/`.
  - [ ] Add `reports/powerbi/` ignore if not versioning PBIX.
  - Done when: repo clean status and no large data tracked.

## Stretch Goals
- [ ] CRM-friendly export format
  - [ ] Add “Salesforce/HubSpot import” output schemes (column names/order).
  - [ ] Include a validation check to confirm required columns present.
- [ ] Score change deltas
  - [ ] Maintain a snapshot history and compute week-over-week ICP score deltas for “hot movers”.
  - [ ] Add “Risers/Fallers” preset to call list exports.
- [ ] Simple propensity model
  - [ ] Train a logistic model using past conversions (if available) to add a “propensity_score” alongside rule-based tags.

