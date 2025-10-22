# ICP Scoring and Segmentation Dashboard

An interactive Streamlit dashboard for real-time customer segmentation and analysis, featuring a data-driven, optimizable scoring model for GoEngineer Digital Manufacturing accounts.

## Overview

This dashboard allows you to:
- **Analyze customer segments** based on configurable revenue thresholds (Small Business, Mid-Market, Enterprise).
- **Adjust scoring weights** in real-time, with defaults provided by an automated optimization process.
- **Visualize the impact** of weight changes on customer ICP scores and segment performance.
- **Identify high-value customers** with dynamic filtering and data-driven recommendations.
- **Export updated scores** and segment data for further analysis.

## Recent Major Update: Hardware Adoption Score Algorithm

**As of July 2025, the Hardware Adoption Score logic has been significantly improved for more accurate and business-aligned ICP scoring.**

### 🚀 What Changed?
- **Weighted Printer Score:**
  - Big Box printers are now valued at 2x the weight of Small Box printers, reflecting their higher investment and engagement.
- **Comprehensive Revenue:**
  - The adoption score now includes both `Total Hardware Revenue` and `Total Consumable Revenue` for a complete picture of hardware engagement.
- **Percentile-Based Scaling:**
  - Both the weighted printer score and total hardware+consumable revenue are converted to percentile ranks (0-1) across all customers, ensuring fair comparison between different units.
- **Business Rules for True Adoption:**
  - **If a customer has zero printers AND zero hardware/consumable revenue, their adoption score is set to 0.0.**
  - **If a customer has revenue but no printers, their adoption score is capped at 0.4.**
  - **Only customers with actual printer investment can achieve high adoption scores.**
- **50/50 Weighting:**
  - The final adoption score is a 50/50 blend of the printer percentile and the revenue percentile (subject to the above business rules).

### 💡 Why This Matters
- **No more "phantom adopters":** Customers with no printers and no spend now get a true zero for adoption.
- **Revenue-only customers are recognized, but capped:** They can't outrank true hardware adopters.
- **Big Box investment is rewarded:** Customers with more significant hardware investment are prioritized.
- **ICP grades are now highly predictive of hardware sales potential.**

### 🔑 Impact on Sales Prioritization
- Hardware sales teams can now trust that high ICP grades reflect real, tangible hardware engagement.
- The adoption score is now the dominant factor in the optimized ICP model (50% weight), with industry and software relationship as supporting factors.

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
    ```
apps/streamlit/app.py          # Streamlit dashboard
src/icp/cli/score_accounts.py  # Generate icp_scored_accounts.csv
src/icp/scoring.py             # Centralized scoring logic
src/icp/cli/optimize_weights.py# Run weight optimization
artifacts/weights/*.json       # Optimized and industry weights
src/icp/industry.py            # Industry weights builder
scripts/clean/*                # Data cleanup utilities
configs/default.toml           # App/pipeline config
README.md                      # Project overview
```

## Data Processing Pipeline

1. **Industry Data Cleanup**: `cleanup_industry_data.py` standardizes industry classifications using fuzzy matching
2. **Industry Scoring**: `industry_scoring.py` calculates performance-based weights for each industry
3. **ICP Scoring**: `goe_icp_scoring.py` processes all data and generates scored accounts
4. **Weight Optimization**: `run_optimization.py` finds optimal component weights
5. **Dashboard**: `streamlit_icp_dashboard.py` provides interactive analysis

## Key Improvements

### Adoption Score Algorithm (Latest)
- **Percentile Scaling Fix**: Excludes zero-value customers from percentile calculations to prevent distribution compression
- **Square Root Scaling**: Revenue-only customers use square root curve for better distribution within 0.0-0.5 range
- **Heavy Fleet Bonus**: Customers with 10+ weighted printers receive +0.05 bonus
- **60/40 Blend**: Printer customers use 60% printer percentile + 40% revenue percentile
- **Zero-Everything Rule**: Customers with no printers AND no revenue get 0.0 adoption score

### Distribution Quality
- **Before**: Standard deviation of 0.008 (extremely compressed)
- **After**: Standard deviation of 0.116 (14x better distribution)
- **Result**: Much more granular and predictive adoption scoring
## Scoring Reference

- Adoption Score
  - Inputs (preferred): doption_assets and doption_profit (profit since 2023 for focus divisions)
  - Inputs (fallback): Big Box Count, Small Box Count, Total Hardware Revenue, Total Consumable Revenue
  - Scaling helpers
    - Percentile scale P/R: ranks only among non-zero values; zero values stay 0.0; if all values identical, returns 0.5 for all
  - Preferred (DB) mode
    - P = percentile(adoption_assets)
    - R = percentile(adoption_profit)
    - If adoption_assets == 0 and adoption_profit > 0: adoption = 0.5 * sqrt(R)
    - If adoption_assets > 0: adoption = (0.6 * P) + (0.4 * R)
    - If both are 0: adoption = 0.0
  - Legacy fallback
    - WeightedPrinters = (2.0 * Big Box Count) + (1.0 * Small Box Count)
    - TotalAdoptionRevenue = Total Hardware Revenue + Total Consumable Revenue (both clamped at >= 0)
    - P = percentile(WeightedPrinters)
    - R = percentile(TotalAdoptionRevenue)
    - If WeightedPrinters == 0 and TotalAdoptionRevenue > 0: adoption = 0.5 * sqrt(R)
    - If WeightedPrinters > 0: adoption = (0.6 * P) + (0.4 * R)
    - Heavy fleet bonus: if WeightedPrinters >= 10, add +0.05 and clip to [0, 1]

- Relationship Score
  - Inputs (preferred): elationship_profit (sum of profits for software-related goals)
  - Inputs (fallback): Total Software License Revenue, Total SaaS Revenue, Total Maintenance Revenue
  - Transform
    - Compute feature = relationship_profit (preferred), else sum of software revenues (all clamped at >= 0)
    - Apply log1p(feature)
    - Min-max scale to [0, 1] across the dataset

- Final ICP Score
  - Raw = (vertical_score * Wv) + (size_score * Ws) + (adoption_score * Wa) + (relationship_score * Wr)
  - Normalize: rank ? percentile ? inverse normal (ppf) ? scaled to mean 50, std 15, clipped to [0, 100]
  - Grade: A�F assigned by target percentile cutoffs

