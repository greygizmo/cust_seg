# ICP Scoring and Neighbors

End-to-end pipeline to score GoEngineer Digital Manufacturing accounts from Azure SQL, plus an exact (blockwise) neighbors artifact with optional ALS embeddings.

## Outputs
- `data/processed/icp_scored_accounts.csv` — baseline scored accounts with component scores and grades
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

## Notes on features
- The main pipeline computes vertical/size/adoption/relationship and writes an ICP score and grade.
- Time-series “List‑Builder” features (spend dynamics, momentum, POV/whitespace) exist in `/features` but are currently disabled in the default run to keep laptop memory headroom. If enabled later, those columns will be appended.

## Documentation
All docs are in `docs/` and reflect the current behavior:
- `docs/METRICS_OVERVIEW.md` — what the pipeline outputs, neighbors details, and config keys
- `docs/mermaid-charts/` — architecture and pipeline diagrams (include neighbors stage)

