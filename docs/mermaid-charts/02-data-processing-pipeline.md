# Data Processing Pipeline

```mermaid
graph TD
    %% Updated pipeline aligned with current repo
    subgraph "Input Data Sources"
        AzureCustomers[Azure SQL: Customers Since 2023]
        AzureProfitGoal[Azure SQL: Profit Since 2023 by Goal]
        AzureProfitRollup[Azure SQL: Profit Since 2023 by Rollup]
        AzureQuarterly[Azure SQL: Quarterly Profit by Goal]
        AzureAssets[Azure SQL: Assets & Seats]
        IndustryFile[data/raw/TR - Industry Enrichment.csv (optional)]
    end

    subgraph "Stage 1: Data Loading (DB)"
        LoadCustomers[Load Customers]
        LoadProfit[Load Profit (goal/rollup/quarterly)]
        LoadAssets[Load Assets & Seats]
        LoadIndustry[Load Industry Enrichment (optional)]
    end

    subgraph "Stage 2: Standardization & Validation"
        IDCanonicalize[Canonicalize Customer ID<br/>strip trailing '.0', preserve leading zeros]
        DataValidation[Validate & Clamp<br/>required columns, non-negative values<br/>log to reports/logs]
    end

    subgraph "Stage 3: Industry Enrichment"
        EnrichmentProcess[Apply enrichment by Customer ID<br/>fallback by CRM Full Name when required]
    end

    subgraph "Stage 4: Data Integration"
        ProfitAggregation[Aggregate Profit
- LatestQ, T4Q, PrevQ, QoQ]
        AssetsAggregation[Aggregate Assets & Seats
- active_assets_total, seats_sum_total
- portfolio breadth]
        MasterMerge[Master DataFrame by Customer ID]
    end

    subgraph "Stage 5: Feature Engineering"
        AdoptionPreferred[Adoption (preferred)
- adoption_assets, adoption_profit]
        AdoptionLegacy[Adoption (legacy fallback)
- weighted printers, HW+Consumable revenue]
        Relationship[Relationship feature
- relationship_profit (preferred)
- fallback software revenues]
    end

    subgraph "Stage 6: Industry Weights"
        BuildWeights[src/icp/industry.py
- empirical-bayes shrinkage
- strategic blending]
        SaveWeights[artifacts/weights/industry_weights.json]
    end

    subgraph "Stage 7: Scoring"
        CalculateScores[src/icp/scoring.py
- vertical/size/adoption/relationship
- normalization & grading]
        ApplyWeights[Use optimized_weights.json if available]
    end

    subgraph "Stage 8: Outputs"
        ScoredDataset[data/processed/icp_scored_accounts.csv]
        VisualizationGeneration[reports/figures/*.png]
        MetadataStorage[artifacts/weights/*.json]
    end

    subgraph "Stage 9: Neighbors (Exact)"
        BuildVectors[Build blocks: numeric/categorical/text]
        BuildALS[Train ALS (rollup + goal) vectors]
        BlockwiseTopK[Compute Top-K with blockwise exact cosine]
        NeighborsOut[artifacts/account_neighbors.csv]
    end

    %% Flow
    AzureCustomers --> LoadCustomers --> IDCanonicalize
    AzureProfitGoal --> LoadProfit
    AzureProfitRollup --> LoadProfit
    AzureQuarterly --> LoadProfit
    AzureAssets --> LoadAssets
    IndustryFile --> LoadIndustry

    IDCanonicalize --> DataValidation --> EnrichmentProcess
    LoadIndustry --> EnrichmentProcess

    EnrichmentProcess --> ProfitAggregation --> MasterMerge
    LoadAssets --> AssetsAggregation --> MasterMerge

    MasterMerge --> AdoptionPreferred
    MasterMerge --> AdoptionLegacy
    MasterMerge --> Relationship

    MasterMerge --> BuildWeights --> SaveWeights
    SaveWeights --> CalculateScores

    AdoptionPreferred --> CalculateScores
    AdoptionLegacy --> CalculateScores
    Relationship --> CalculateScores
    CalculateScores --> ApplyWeights --> ScoredDataset
    CalculateScores --> VisualizationGeneration
    SaveWeights --> MetadataStorage

    %% Neighbors flow (separate step)
    ScoredDataset --> BuildVectors --> BlockwiseTopK --> NeighborsOut
    LoadAssets --> BuildALS
    LoadProfit --> BuildALS
    BuildALS --> BlockwiseTopK

    %% Styles
    classDef inputData fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef stage fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef outputs fill:#f5f5f5,stroke:#616161,stroke-width:2px

    class AzureCustomers,AzureProfitGoal,AzureProfitRollup,AzureQuarterly,AzureAssets,IndustryFile inputData
    class LoadCustomers,LoadProfit,LoadAssets,LoadIndustry,IDCanonicalize,DataValidation,EnrichmentProcess,ProfitAggregation,AssetsAggregation,MasterMerge,AdoptionPreferred,AdoptionLegacy,Relationship,BuildWeights,SaveWeights,CalculateScores,ApplyWeights stage
    class ScoredDataset,VisualizationGeneration,MetadataStorage outputs
```

## Data Processing Pipeline Details

This pipeline assembles data from Azure SQL, applies optional industry enrichment, and produces actionable ICP scores through an 8‑stage process.

### Stage 1: Data Loading (DB)
- Customers, profit (goal/rollup/quarterly), assets & seats
- Optional enrichment CSV for updated industry classifications

### Stage 2: Standardization & Validation
- Canonicalize Customer ID (strip trailing .0, preserve leading zeros)
- Validate presence and non-negativity; log issues to reports/logs

### Stage 3: Industry Enrichment
- Apply enrichment by Customer ID; preserve original fields and Reasoning

### Stage 4: Data Integration
- Aggregate profit (LatestQ, T4Q, PrevQ, QoQ)
- Aggregate assets & seats; compute portfolio breadth
- Merge all by Customer ID into a master dataset

### Stage 5: Feature Engineering
- Preferred adoption signals: adoption_assets and adoption_profit
- Legacy fallback: weighted printers + HW/Consumable revenue
- Relationship: relationship_profit preferred; fallback to software revenues

### Stage 6: Industry Weights
- Empirical‑Bayes shrinkage of industry success metric
- Blend with strategic tiers from artifacts/industry/strategic_industry_tiers.json

### Stage 7: Scoring & Normalization
- Compute component scores; apply optimized/default weights
- Normalize to 0–100 bell‑curve distribution; assign A–F grades

### Stage 8: Outputs
- Scored dataset: data/processed/icp_scored_accounts.csv
- Visualizations: reports/figures/*.png
- Weights & metadata: artifacts/weights/*.json

### Stage 9: Neighbors (Exact, blockwise)
- Inputs: scored dataset; optional Azure SQL aggregates for ALS vectors
- Blocks: numeric, categorical, text, ALS (configurable weights)
- Exact cosine Top‑K computed blockwise to avoid NxN memory
- Output: artifacts/account_neighbors.csv

Config knobs (config.toml):
- [similarity] k_neighbors, use_text, use_als, w_* block weights
- [similarity] max_dense_accounts, row_block_size (memory/scale)
- [als] alpha, reg, iterations, use_bm25, composite weights

### Key Quality Controls:
- **Minimum Sample Sizes**: Ensures statistical significance
- **Missing Data Handling**: Graceful fallbacks for incomplete data
- **Source Prioritization**: Uses highest quality data available
- **Validation Checks**: Prevents processing of invalid data




