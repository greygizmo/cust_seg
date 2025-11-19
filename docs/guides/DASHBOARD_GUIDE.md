# ICP Scoring Dashboard Guide

The ICP Scoring Dashboard is an interactive tool built with **Streamlit** to help you explore the results of the scoring pipeline.

## How to Run

Ensure you have installed the dependencies (`pip install -r requirements.txt`) and have run the scoring pipeline at least once to generate `data/processed/icp_scored_accounts.csv`.

```bash
make dashboard
# OR
streamlit run apps/dashboard.py
```

## Features

### 1. Portfolio Overview
- **Metrics**: View total accounts, average hardware/software scores, and total Gross Profit (since 2023).
- **Charts**:
    - **Score Distribution**: Histogram of Hardware Scores.
    - **Top Industries**: Bar chart of the most common industries in the scored dataset.

### 2. Account Explorer
- **Data Table**: A searchable, sortable table of all scored accounts.
- **Filters**:
    - **Industry**: Select one or multiple industries to filter the view.
    - **Score Range**: Use the slider to filter accounts by Hardware Score.

### 3. Neighbor Visualizer
- **Account Selection**: Search for an account by ID or Name.
- **Neighbors Table**: View the top-K similar accounts (neighbors) for the selected account.
- **Details**: See the similarity score, rank, and key attributes of each neighbor.

## Troubleshooting

- **"Could not find data..."**: Ensure you have run `python -m icp.cli.score_accounts` successfully.
- **"No neighbors artifact found"**: Ensure you ran the scoring pipeline *without* `--skip-neighbors`.
