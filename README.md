# ICP Scoring and Neighbors

End-to-end pipeline to score GoEngineer Digital Manufacturing accounts from Azure SQL, plus an exact (blockwise) neighbors artifact with optional ALS embeddings.

## Outputs
- `data/processed/icp_scored_accounts.csv` - baseline scored accounts with division scores and grades
- `dbo.customer_icp` (optional) - when `ICP_AZSQL_DB` is set, the same scored accounts are written to this database table
- `reports/figures/*.png` - batch visuals
- `artifacts/account_neighbors.csv` - exact top-K neighbors per account (optional stage)

## Quick start
- Baseline scoring (no neighbors/visuals):
  - `python -m icp.cli.score_accounts --skip-neighbors --skip-visuals`
- Build neighbors later from the saved CSV (exact, blockwise):
  - `python -m icp.cli.score_accounts --neighbors-only --in-scored data/processed/icp_scored_accounts.csv`
- Disable ALS in neighbors (override config):
  - `python -m icp.cli.score_accounts --neighbors-only --no-als`

## Configuration
Edit `config.toml`:
- `[similarity]` - `k_neighbors` (default 15), `use_text`, `use_als`, block weights, and memory controls `max_dense_accounts`, `row_block_size`
- `[als]` - `alpha`, `reg`, `iterations`, `use_bm25`, and composite strength weights

Environment variables:
- `AZSQL_DB` - source database for NetSuite-backed inputs (e.g., `db-goeng-netsuite-prod`)
- `ICP_AZSQL_DB` - optional target database for scored accounts (e.g., `db-goeng-icp-prod`); when set, `dbo.customer_icp` is replaced on each scoring run

## Notes on features
- The main pipeline computes vertical/adoption/relationship components and writes division-specific ICP scores and grades (`ICP_score_hardware`/`ICP_grade_hardware`, `ICP_score_cre`/`ICP_grade_cre`), plus the raw blended score (`ICP_score_raw`).
- Time-series "List Builder" features (spend dynamics, momentum, POV/whitespace) exist in `/features` but are currently disabled in the default run to keep laptop memory headroom. If enabled later, those columns will be appended.

## Schema & columns
- `src/icp/schema.py` centralizes canonical column names (for example `COL_CUSTOMER_ID`, `COL_INDUSTRY`, `COL_REL_LICENSE`) and helpers such as `unify_columns`.
- `docs/guides/POWERBI_FIELD_REFERENCE.md` mirrors these constants; extend `schema.py` whenever you introduce a new canonical field or alias.

## Running tests
- Install dependencies (`pip install -r requirements.txt` plus `pytest` if needed).
- Run `pytest tests` to execute focused unit tests across scoring helpers, schema aliasing, feature engineering, and industry weights/shrinkage.

## CLI usage
```
python -m icp.cli.score_accounts \
  --division hardware \
  --out data/processed/icp_scored_accounts.csv \
  --weights artifacts/weights/optimized_weights.json \
  --industry-weights artifacts/weights/industry_weights.json \
  --asset-weights artifacts/weights/asset_rollup_weights.json \
  [--skip-visuals] [--skip-neighbors] [--no-als]
```
- `--out` writes the scored CSV to the provided path (also respected via `ICP_OUT_PATH`).
- `--skip-visuals` suppresses matplotlib output for CI/headless runs.
- `--neighbors-only --in-scored <path>` builds neighbors from an existing CSV without recomputing the scores.
- Generate dated call list packs for sales operations:
  - `python -m icp.cli.export_call_lists --src data/processed/icp_scored_accounts.csv --out-root reports/call_lists`

### Scheduling weekly call list exports
- Windows Task Scheduler example action:
  ```
  Program/script: powershell.exe
  Arguments: -ExecutionPolicy Bypass -File scripts/export_call_lists.ps1 --src data\processed\icp_scored_accounts.csv --out-root reports\call_lists
  Start in: D:\path\to\repo
  ```
- The script sets `PYTHONPATH=src` automatically and writes dated folders under `reports/call_lists/YYYYMMDD/`.

## Documentation
All docs are in `docs/` and reflect the current behavior:
- `docs/METRICS_OVERVIEW.md` - pipeline outputs, neighbor details, and config keys
- `docs/mermaid-charts/` - architecture and pipeline diagrams (including the neighbors stage)
- `docs/guides/POWERBI_FIELD_REFERENCE.md` - column reference for sales/BI consumers
- `docs/guides/POWERBI_DASHBOARD_GUIDE.md` - recommended Power BI model, measures, and dashboard pages
 - `docs/sales-playbook.md` - step-by-step guide for sellers on how to build and work call lists (Streamlit + Power BI)
