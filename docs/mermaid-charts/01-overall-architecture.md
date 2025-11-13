# Overall System Architecture

```mermaid
graph TB
    %% Define main system components (updated repo)
    subgraph "Data Sources"
        AzureCustomers[Azure SQL: Customers Since 2023]
        AzureProfitGoal[Azure SQL: Profit Since 2023 by Goal]
        AzureProfitRollup[Azure SQL: Profit Since 2023 by Rollup]
        AzureQuarterly[Azure SQL: Quarterly Profit by Goal]
        AzureAssets[Azure SQL: Assets & Seats]
        IndustryCSV[data/raw/TR - Industry Enrichment.csv (optional)<br/>Updated Industry Classifications]
    end

    subgraph "Processing & Orchestration"
        MainPipeline[src/icp/cli/score_accounts.py<br/>End-to-End Data Assembly & Scoring]
        IndustryBuilder[src/icp/industry.py<br/>Build/Load Industry Weights]
        SchemaValidation[src/icp/schema.py & validation.py<br/>Column constants & data checks]
    end

    subgraph "Scoring Engine"
        ScoringLogic[src/icp/scoring.py<br/>Centralized Scoring Functions<br/>- Vertical (Industry)
- Size (Revenue)
- Adoption (Assets/Profit or Printers/Revenue)
- Relationship (Software Profit/Revenue)]
        OptimizationCLI[src/icp/cli/optimize_weights.py]
        OptimizationCore[src/icp/optimization.py<br/>Optuna Objective]
    end

    subgraph "Analytics & Visualization"
        Dashboard[apps/streamlit/app.py<br/>Interactive Dashboard + Call List Builder]
        VisualOutputs[reports/figures/*.png]
    end

    subgraph "Configuration & Storage"
        ConfigFiles[config.toml<br/>artifacts/industry/strategic_industry_tiers.json]
        OptimizedWeights[artifacts/weights/optimized_weights.json]
        IndustryWeights[artifacts/weights/industry_weights.json]
        ScoredData[data/processed/icp_scored_accounts.csv]
        NeighborsCSV[artifacts/account_neighbors.csv]
    end

    %% Data flow connections
    AzureCustomers --> MainPipeline
    AzureProfitGoal --> MainPipeline
    AzureProfitRollup --> MainPipeline
    AzureQuarterly --> MainPipeline
    AzureAssets --> MainPipeline
    IndustryCSV --> MainPipeline

    MainPipeline --> IndustryBuilder
    IndustryBuilder --> IndustryWeights
    IndustryWeights --> ScoringLogic
    ScoringLogic --> MainPipeline

    MainPipeline --> ScoredData
    MainPipeline --> VisualOutputs
    ScoredData --> Dashboard
    ScoredData --> OptimizationCLI

    %% Similarity & Neighbors
    subgraph "Similarity & Neighbors"
        SimilarityBuilder[features/similarity_build.py<br/>Exact blockwise Top-K]
        ALSVectors[features/als_prep.py + als_embed.py<br/>Rollup + Goal vectors]
    end

    ScoredData --> SimilarityBuilder
    AzureProfitRollup --> ALSVectors
    AzureAssets --> ALSVectors
    ALSVectors --> SimilarityBuilder
    SimilarityBuilder --> NeighborsCSV

    OptimizationCLI --> OptimizationCore --> OptimizedWeights
    OptimizedWeights --> ScoringLogic
    OptimizedWeights --> Dashboard
    Dashboard --> VisualOutputs

    %% Styles
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef scoring fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef analytics fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class AzureCustomers,AzureProfitGoal,AzureProfitRollup,AzureQuarterly,AzureAssets,IndustryCSV dataSource
    class MainPipeline,IndustryBuilder,SchemaValidation processing
    class ScoringLogic,OptimizationCLI,OptimizationCore scoring
    class Dashboard,VisualOutputs analytics
    class ConfigFiles,OptimizedWeights,IndustryWeights,ScoredData,NeighborsCSV storage
    class SimilarityBuilder,ALSVectors processing
```

## System Overview

This architecture represents the updated **ICP Scoring System**. It assembles data directly from Azure SQL, enriches industry classifications (optional CSV), computes scores via centralized logic, and surfaces results in a Streamlit dashboard with a Call List Builder.

### Key Features:
- **Data-Driven Scoring**: Uses historical revenue data to calculate industry performance weights
- **Machine Learning Optimization**: Employs Optuna to find optimal scoring weights
- **Interactive Dashboard**: Real-time weight adjustment and visualization
- **Multi-Source Data Integration**: Combines customer, sales, and enriched revenue data
- **Automated Pipeline**: End-to-end processing from raw data to scored accounts

### Architecture Layers:
1. **Data Sources**: Azure SQL (customers, profit, assets) + optional enrichment CSV
2. **Processing**: Orchestrated by `src/icp/cli/score_accounts.py` (validation, enrichment, features)
3. **Scoring Engine**: `src/icp/scoring.py` with ML‑optimized weights (`artifacts/weights`)
4. **Analytics**: `apps/streamlit/app.py` with Call List Builder and docs integration
5. **Storage**: `data/processed`, `reports/figures`, `artifacts/weights`, configs in `configs/`

### Data Flow:
1. Raw data is cleaned and standardized
2. Industry weights are calculated using historical performance data
3. Customer scores are computed using the optimized weighting system
4. Results are presented through an interactive dashboard
5. The system continuously improves through ML optimization




