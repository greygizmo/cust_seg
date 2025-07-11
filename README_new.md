# ICP Scoring and Segmentation Dashboard

An interactive Streamlit dashboard for real-time customer segmentation and analysis, featuring a data-driven, optimizable scoring model for GoEngineer Digital Manufacturing accounts.

## Overview

This dashboard allows you to:
- **Analyze customer segments** based on configurable revenue thresholds (Small Business, Mid-Market, Enterprise).
- **Adjust scoring weights** in real-time, with defaults provided by an automated optimization process.
- **Visualize the impact** of weight changes on customer ICP scores and segment performance.
- **Identify high-value customers** with dynamic filtering and data-driven recommendations.
- **Export updated scores** and segment data for further analysis.

## Features

### 🏢 Customer Segmentation
- **Configurable Thresholds**: Define segments by annual revenue.
- **Segment Selector**: Filter the entire dashboard view by segment (All, Small Business, Mid-Market, Large Enterprise).
- **Comparison Analytics**: View charts comparing key metrics like average ICP score, customer count, and revenue across segments.
- **Segment-Specific Insights**: Get tailored metrics and strategic recommendations for each segment.

### 🎛️ Interactive Weight Controls
- **Optimized Defaults**: Weights are pre-loaded from `optimized_weights.json` for a data-driven starting point.
- **Real-time Adjustment**: Sliders for four main criteria:
    - **Vertical Weight**: Importance of customer's industry.
    - **Size Weight**: Importance of customer's annual revenue.
    - **Adoption Weight**: Importance of technology adoption (printer count, consumable revenue).
    - **Relationship Weight**: Importance of software revenue.

### 📊 Real-time Visualizations
1. **ICP Score Distribution**: Enhanced histogram and box plot showing the spread of scores and key statistical markers.
2. **Segment Comparison**: Multi-panel chart comparing performance across segments.
3. **Weight Distribution Radar**: Visual representation of the current scoring weights.
4. **Score by Industry**: Bar chart of average scores across top industry verticals.
5. **Diagnostic Plots**: Scatter plots to ensure normalized scores correlate with raw data inputs.

## Setup Instructions

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Base Data**
    Run the scoring script to process the raw data files and generate `icp_scored_accounts.csv`.
    ```bash
    python goe_icp_scoring.py
    ```

3.  **Optimize Weights (Recommended)**
    Run the optimization script to find the best weights based on historical revenue data. This creates the `optimized_weights.json` file, which the dashboard will use automatically.
    ```bash
    python run_optimization.py
    ```

4.  **Launch Dashboard**
    ```bash
    streamlit run streamlit_icp_dashboard.py
    ```

## Scoring Methodology

The scoring logic is centralized in `scoring_logic.py` and uses a data-driven approach.

### Individual Component Scores (0-1)

Component scores are calculated based on empirical data and then normalized.

-   **Vertical Score**: Mapped from a dictionary (`PERFORMANCE_VERTICAL_WEIGHTS`) based on the historical revenue performance of different industries.
-   **Size Score**: Determined by tiers of reliable, enriched annual revenue data. Higher revenue generally leads to a higher score.
-   **Adoption Score**: A composite score derived from `log1p`-transformed printer count and consumable revenue, which are then combined and scaled.
-   **Relationship Score**: A score derived from the `log1p`-transformed sum of all software-related revenue (licenses, SaaS, maintenance), which is then scaled.

### Final ICP Score Calculation

1.  **Raw Score**: The component scores are multiplied by their respective weights (from the dashboard sliders or `optimized_weights.json`) and summed.
    ```
    Raw Score = (Vertical * W_v) + (Size * W_s) + (Adoption * W_a) + (Relationship * W_r)
    ```
2.  **Normalization**: The raw scores are converted to a percentile rank and then mapped to a normal (bell curve) distribution using the inverse of the cumulative distribution function (`norm.ppf`). This creates a more intuitive and statistically robust final `ICP_score` between 0 and 100.
3.  **Grading**: Customers are assigned an A-F grade based on their percentile rank in the final score distribution.

## Weight Optimization

The script `run_optimization.py` uses the `optuna` library to find the ideal set of weights. Its goal is to solve a multi-objective problem:

1.  **Maximize Revenue Correlation**: It tries to find weights that make the final ICP score as predictive of historical customer revenue as possible (measured by Spearman correlation).
2.  **Match Target Distribution**: It simultaneously tries to shape the final scores into a predefined A-F grade distribution (e.g., 10% A's, 20% B's, etc.), measured by KL Divergence.

The `lambda_param` in the script controls the trade-off between these two goals. The best-performing set of weights is saved to `optimized_weights.json`.

## File Structure

```
├── streamlit_icp_dashboard.py    # Main dashboard application
├── goe_icp_scoring.py           # Generates the base icp_scored_accounts.csv
├── scoring_logic.py             # Centralized, data-driven scoring functions
├── run_optimization.py          # Runs the weight optimization
├── optimize_weights.py          # The optimization objective function
├── optimized_weights.json       # Output of the optimization, used by the dashboard
├── requirements.txt             # Python dependencies
├── icp_scored_accounts.csv      # Input data for the dashboard
├── SEGMENTATION_FEATURES.md     # Summary of segmentation feature implementation
└── README.md                    # This file
```

## Troubleshooting

-   **Data Not Loading**: Ensure `icp_scored_accounts.csv` exists. If not, run `python goe_icp_scoring.py`.
-   **"Optimized Weights Not Found" Warning**: This means `optimized_weights.json` is missing. The dashboard will fall back to default weights. Run `python run_optimization.py` to generate it.
-   **Weight Validation Errors**: The four main weights must sum to 1.0. Adjust the sliders until the sum is correct.

## Support

For questions or issues, please refer to the relevant scripts or contact the data analytics team.
