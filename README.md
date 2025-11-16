# ICP Scoring and Neighbors

End-to-end pipeline to score GoEngineer Digital Manufacturing accounts from Azure SQL, plus an exact (blockwise) neighbors artifact with optional ALS embeddings.

## Outputs
- `data/processed/icp_scored_accounts.csv` — baseline scored accounts with division scores and grades
- `dbo.customer_icp` (optional) — when `ICP_AZSQL_DB` is set, the same scored accounts are written to this database table
- `reports/figures/*.png` — batch visuals
- `artifacts/account_neighbors.csv` — exact Top‑K neighbors per account (optional step)

## Quick start
- Baseline scoring (no neighbors/visuals):
  - `python -m icp.cli.score_accounts --skip-neighbors --skip-visuals`
- Build neighbors later from the saved CSV (exact, blockwise):
  - `python -m icp.cli.score_accounts --neighbors-only --in-scored data/processed/icp_scored_accounts.csv`
- Disable ALS in neighbors (override config):
  - `python -m icp.cli.score_accounts --neighbors-only --no-als`

## Configuration
Edit `config.toml`:
- `[similarity]` — `k_neighbors` (default 15), `use_text`, `use_als`, block weights, and memory controls `max_dense_accounts`, `row_block_size`
- `[als]` — `alpha`, `reg`, `iterations`, `use_bm25`, and composite strength weights

Environment variables:
- `AZSQL_DB` — source database for NetSuite-backed inputs (e.g., `db-goeng-netsuite-prod`)
- `ICP_AZSQL_DB` — optional target database for scored accounts (e.g., `db-goeng-icp-prod`); when set, `dbo.customer_icp` is replaced on each scoring run

## Notes on features
- The main pipeline computes vertical/adoption/relationship components and writes division-specific ICP scores and grades (`ICP_score_hardware`/`ICP_grade_hardware`, `ICP_score_cre`/`ICP_grade_cre`), plus the raw blended score (`ICP_score_raw`).
- Time-series “List‑Builder” features (spend dynamics, momentum, POV/whitespace) exist in `/features` but are currently disabled in the default run to keep laptop memory headroom. If enabled later, those columns will be appended.

## Documentation
All docs are in `docs/` and reflect the current behavior:
- `docs/METRICS_OVERVIEW.md` — what the pipeline outputs, neighbors details, and config keys
- `docs/mermaid-charts/` — architecture and pipeline diagrams (include neighbors stage)

