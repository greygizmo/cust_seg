# ICP Scoring and Neighbors

End-to-end pipeline to score GoEngineer Digital Manufacturing accounts from Azure SQL, plus an exact (blockwise) neighbors artifact with optional ALS embeddings.

## Outputs
- `data/processed/icp_scored_accounts.csv` - baseline scored accounts with division scores and grades
- `dbo.customer_icp` (optional) - when `ICP_AZSQL_DB` is set, the same scored accounts are written to this database table
- `reports/figures/*.png` - batch visuals
- `artifacts/account_neighbors.csv` - exact top-K neighbors per account (optional stage)
- `artifacts/account_playbooks.csv` - rule-based playbooks and tags per account (CRO/CFO-friendly motions)
- `artifacts/pulse_*.csv` - compact “pulse” snapshots for portfolio, neighbors, and playbooks

## Quick start
> **Note for Windows Users**: If `make` is not installed, use `.\make` or `.\make.bat` in PowerShell/CMD to run the commands below.

- **Unified Pipeline**:
  - `make pipeline` (runs scoring, playbooks, and call lists)
- Baseline scoring (no neighbors/visuals):
  - `make score` (or `python -m icp.cli.score_accounts --skip-neighbors`)
- Build rule-based playbooks/tags:
  - `make playbooks` (or `python -m icp.cli.build_playbooks`)
- Generate dated call lists:
  - `make call-lists` (or `python -m icp.cli.export_call_lists`)
- Generate asset weights from DB:
  - `make weights` (or `python -m icp.cli.generate_weights`)
- Launch the Interactive Dashboard:
  - `make dashboard` (or `streamlit run apps/dashboard.py`)

## Configuration
Edit `config.toml`:
- `[similarity]` - `k_neighbors` (default 25), block weights (numeric/categorical/text/ALS), curated numeric/text columns, and memory controls `max_dense_accounts`, `row_block_size`
- `[als]` - `alpha`, `reg`, `iterations`, `use_bm25`, and composite strength weights

Environment variables:
- `AZSQL_DB` - source database for NetSuite-backed inputs (e.g., `db-goeng-netsuite-prod`)
- `ICP_AZSQL_DB` - optional target database for scored accounts (e.g., `db-goeng-icp-prod`); when set, `dbo.customer_icp` is replaced on each scoring run

## Architecture & Modules
The codebase is modularized for maintainability:
- `src/icp/cli/` - Command-line interfaces (orchestrators)
- `src/icp/etl/` - Data loading (`loader.py`) and cleaning (`cleaner.py`)
- `src/icp/features/` - Feature engineering logic (`engineering.py`)
- `src/icp/reporting/` - Visualization and reporting (`visuals.py`)
- `src/icp/config.py` - Centralized configuration management
- `src/icp/validation.py` - Input validation (Pandera)
- `src/icp/quality.py` - Output validation (Pandera)
- `apps/dashboard.py` - Interactive Streamlit dashboard

## Schema & columns
- `src/icp/schema.py` centralizes canonical column names (for example `COL_CUSTOMER_ID`, `COL_INDUSTRY`, `COL_REL_LICENSE`) and helpers such as `unify_columns`.
- `docs/guides/POWERBI_FIELD_REFERENCE.md` mirrors these constants; extend `schema.py` whenever you introduce a new canonical field or alias.

## Running tests
- Install dependencies (`pip install -r requirements.txt` plus `pytest` if needed).
- Run `pytest tests` to execute focused unit tests across scoring helpers, schema aliasing, feature engineering, and industry weights/shrinkage.
- Use `make test` for a quick test run.
- Use `make lint` to run code quality checks (ruff).
- Use `make type-check` to run static type analysis (mypy).

## CLI usage
```
python -m icp.cli.score_accounts \
  --division hardware \
  --out data/processed/icp_scored_accounts.csv \
  --weights artifacts/weights/optimized_weights.json \
  --industry-weights artifacts/weights/industry_weights.json \
  --asset-weights artifacts/weights/asset_rollup_weights.json \
  [--skip-visuals] [--skip-neighbors] [--no-als] [--strict]
```
- `--strict` enforces strict validation of output files, failing the pipeline if data quality issues are found.
- `--out` writes the scored CSV to the provided path (also respected via `ICP_OUT_PATH`).
- `--skip-visuals` suppresses matplotlib output for CI/headless runs.
- `--neighbors-only --in-scored <path>` builds neighbors from an existing CSV without recomputing the scores.
- Generate dated call list packs for sales operations:
  - `python -m icp.cli.export_call_lists --src data/processed/icp_scored_accounts.csv --out-root reports/call_lists`

### Scheduling weekly call list exports
- Windows Task Scheduler example action:
  ```
  Program/script: python.exe
  Arguments: -m icp.cli.export_call_lists --src data/processed/icp_scored_accounts.csv --out-root reports/call_lists
  Start in: D:\path\to\repo
  ```
- The script writes dated folders under `reports/call_lists/YYYYMMDD/`.

## Documentation
All docs are in `docs/` and reflect the current behavior:
- `docs/METRICS_OVERVIEW.md` - pipeline outputs, neighbor details, and config keys
- `docs/mermaid-charts/` - architecture and pipeline diagrams (including the neighbors stage)
- `docs/guides/POWERBI_FIELD_REFERENCE.md` - column reference for sales/BI consumers
- `docs/guides/POWERBI_DASHBOARD_GUIDE.md` - recommended Power BI model, measures, and dashboard pages
 - `docs/sales-playbook.md` - step-by-step guide for sellers on how to build and work call lists (Streamlit + Power BI)
