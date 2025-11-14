# File Relationships & Dependencies

```mermaid
graph TB
    %% Define all Python files and their relationships
    subgraph "Main Scripts (Executable)"
        MainScoring[src/icp/cli/score_accounts.py<br/>PRIMARY EXECUTABLE<br/>End-to-end data assembly & scoring]
        Optimization[src/icp/cli/optimize_weights.py<br/>OPTIONAL EXECUTABLE<br/>Weight optimization with Optuna]
        Dashboard[apps/streamlit/app.py<br/>WEB APP EXECUTABLE<br/>Interactive dashboard + Call List Builder]
    end

    subgraph "Utility Modules (Imported)"
        IndustryUtils[scripts/clean/cleanup_industry_data.py]
        IndustryScoring[src/icp/industry.py]
        ScoringLogic[src/icp/scoring.py]
        OptimizationObj[src/icp/optimization.py]
    end

    subgraph "Configuration Files (Read)"
        ConfigFile[configs/default.toml]
        StrategicConfig[artifacts/industry/strategic_industry_tiers.json]
        Requirements[requirements.txt]
    end

    subgraph "Data Files (Read/Write)"
        subgraph "Input Data"
            AzureSources[Azure SQL Sources<br/>Customers, Profit, Assets/Seats<br/>Read by main scoring]
            IndustryEnrich[data/raw/TR - Industry Enrichment.csv<br/>Industry classifications<br/>Read by main scoring (optional)]
        end

        subgraph "Generated Data Files"
            ScoredAccounts[data/processed/icp_scored_accounts.csv<br/>Scored customer dataset]
            OptimizedWeights[artifacts/weights/optimized_weights.json]
            IndustryWeights[artifacts/weights/{division}_industry_weights.json]
            Visualizations[reports/figures/vis1-vis10.png]
        end
    end

    subgraph "Import Relationships"
    MainScoring --> IndustryUtils
    MainScoring --> IndustryScoring
    MainScoring --> ScoringLogic

        Dashboard --> ScoringLogic
        Dashboard --> NameUtils

        Optimization --> OptimizationObj

        IndustryScoring --> ScoringLogic
    end

    subgraph "Data Read Relationships"
        MainScoring --> AzureSources
        MainScoring --> IndustryEnrich
        MainScoring --> ConfigFile

        IndustryScoring --> StrategicConfig
        IndustryScoring --> ScoredAccounts

        Optimization --> ScoredAccounts
        Optimization --> StrategicConfig

        Dashboard --> ScoredAccounts
        Dashboard --> OptimizedWeights
        Dashboard --> IndustryWeights
        Dashboard --> Visualizations

        ScoringLogic --> OptimizedWeights
        ScoringLogic --> IndustryWeights
    end

    subgraph "Data Write Relationships"
        MainScoring --> ScoredAccounts
        MainScoring --> Visualizations
        MainScoring --> IndustryWeights

        IndustryScoring --> IndustryWeights

        Optimization --> OptimizedWeights
    end

    subgraph "Execution Dependencies"
        AzureSources -.-> MainScoring
        IndustryEnrich -.-> MainScoring

        MainScoring -.-> ScoredAccounts
        MainScoring -.-> Visualizations
        ScoredAccounts -.-> Optimization
        ScoredAccounts -.-> Dashboard

        Optimization -.-> OptimizedWeights
        OptimizedWeights -.-> Dashboard
        OptimizedWeights -.-> ScoringLogic
    end

    %% Style definitions
    classDef executable fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    classDef module fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef config fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef inputData fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef outputData fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef importRel stroke:#28a745,stroke-width:4px
    classDef readRel stroke:#007bff,stroke-width:4px
    classDef writeRel stroke:#dc3545,stroke-width:4px
    classDef execRel stroke:#6f42c1,stroke-width:4px

    %% Apply styles
    class MainScoring,Optimization,Dashboard executable
    class NameUtils,IndustryUtils,IndustryScoring,ScoringLogic,OptimizationObj module
    class ConfigFile,StrategicConfig,Requirements config
    class CustomerData,SalesData,RevenueData,IndustryEnrich inputData
    class ScoredAccounts,OptimizedWeights,IndustryWeights,Visualizations outputData

    %% Apply relationship styles using linkStyle
    linkStyle 15,16,17,18 stroke:#28a745,stroke-width:4px
    linkStyle 19,20,21,22,23,24,25,26,27,28,29,30 stroke:#007bff,stroke-width:4px
    linkStyle 31,32,33,34,35 stroke:#dc3545,stroke-width:4px
    linkStyle 36,37,38,39,40,41,42,43,44 stroke:#6f42c1,stroke-width:4px
```

## File Relationships & Dependencies Legend

### Relationship Types:

#### **Import Relationships** (Green)
- `script.py --> module.py`: Import statement relationships
- Shows Python module dependencies
- Must be satisfied for code execution

#### **Read Relationships** (Blue)
- `script.py --> file.ext`: Files read during execution
- Includes configuration files, input data, cached results
- Required for proper functioning

#### **Write Relationships** (Red)
- `script.py --> file.ext`: Files created or modified during execution
- Generated datasets, optimized weights, visualizations
- Output artifacts of processing

#### **Execution Dependencies** (Purple)
- `input --> script.py --> output`: Processing pipeline dependencies
- Shows the required execution order
- Dashed lines indicate optional/conditional dependencies

### Execution Order:

1. **src/icp/cli/score_accounts.py** (Required - Main Processing)
   - Reads: Azure SQL sources, optional enrichment CSV, config
   - Writes: Scored accounts, industry weights, visualizations
   - Imports: All utility modules and scoring logic

2. **src/icp/cli/optimize_weights.py** (Optional - Weight Optimization)
   - Reads: Scored accounts dataset
   - Writes: Optimized weights JSON
   - Imports: Optimization objective function
   - Depends on: Main scoring completion

3. **apps/streamlit/app.py** (Interactive Analysis)
   - Reads: Scored accounts, optimized weights, industry weights, visualizations
   - No file writes (analysis only)
   - Imports: Scoring logic and name utilities
   - Depends on: Main scoring completion

### Critical Dependencies:

#### **Hard Dependencies** (Must Exist):
- `src/icp/cli/score_accounts.py`  `src/icp/scoring.py` (core scoring)
- `src/icp/cli/score_accounts.py`  `src/icp/industry.py` (industry weights)
- `apps/streamlit/app.py`  `data/processed/icp_scored_accounts.csv` (data source)
- `src/icp/scoring.py`  `artifacts/weights/{division}_industry_weights.json` (industry scores)

#### **Soft Dependencies** (Fallback Available):
- `artifacts/weights/optimized_weights.json` (falls back to DEFAULT_WEIGHTS)
- Industry enrichment CSV optional (original industry used if absent)

#### **Generated Dependencies**:
- All output files are generated by the processing pipeline
- Optimization and dashboard depend on main scoring completion
- Industry weights can be generated standalone or as part of main pipeline

### File Categories:

#### **Source Code Files**:
- `.py` files containing Python code
- Executable scripts and importable modules
- Business logic and algorithms

#### **Configuration Files**:
- `.toml`, `.json` files with settings
- Strategic priorities and system parameters
- Dependency specifications

#### **Data Files**:
- `.xlsx`, `.csv` files with customer and sales data
- Input datasets and generated results
- Analysis outputs and cached computations

This dependency map ensures proper execution order and helps with troubleshooting, maintenance, and system understanding.




