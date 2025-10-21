# File Relationships & Dependencies

```mermaid
graph TB
    %% Define all Python files and their relationships
    subgraph "Main Scripts (Executable)"
        MainScoring[goe_icp_scoring.py<br/>PRIMARY EXECUTABLE<br/>End-to-end processing pipeline<br/>Execution Order: 1st]
        Optimization[run_optimization.py<br/>OPTIONAL EXECUTABLE<br/>Weight optimization with Optuna<br/>Execution Order: 3rd (after main scoring)]
        Dashboard[streamlit_icp_dashboard.py<br/>WEB APP EXECUTABLE<br/>Interactive dashboard<br/>Execution Order: 4th (after all processing)]
    end

    subgraph "Utility Modules (Imported)"
        NameUtils[normalize_names.py<br/>Name standardization functions<br/>Used by main scoring and dashboard]
        IndustryUtils[cleanup_industry_data.py<br/>Industry classification utilities<br/>Used by main scoring]
        IndustryScoring[industry_scoring.py<br/>Data-driven industry weights<br/>Used by main scoring and scoring logic]
        ScoringLogic[scoring_logic.py<br/>CORE SCORING ENGINE<br/>Component score calculations<br/>Used by main scoring and dashboard]
        OptimizationObj[optimize_weights.py<br/>Optimization objective function<br/>Used by optimization script]
    end

    subgraph "Configuration Files (Read)"
        ConfigFile[config.toml<br/>System configuration<br/>Read by main scoring]
        StrategicConfig[strategic_industry_tiers.json<br/>Strategic industry priorities<br/>Read by industry scoring]
        Requirements[requirements.txt<br/>Python dependencies<br/>Read by pip install]
    end

    subgraph "Data Files (Read/Write)"
        subgraph "Input Data Files"
            CustomerData[JY - Customer Analysis.xlsx<br/>Customer master data<br/>Read by main scoring]
            SalesData[TR - Master Sales Log.xlsx<br/>Historical sales data<br/>Read by main scoring]
            RevenueData[enrichment_progress.csv<br/>Enriched revenue data<br/>Read by main scoring]
            IndustryEnrich[TR - Industry Enrichment.csv<br/>Industry classifications<br/>Read by main scoring]
        end

        subgraph "Generated Data Files"
            ScoredAccounts[icp_scored_accounts.csv<br/>FINAL OUTPUT<br/>Scored customer dataset<br/>Written by main scoring<br/>Read by dashboard and optimization]
            OptimizedWeights[optimized_weights.json<br/>ML-optimized weights<br/>Written by optimization<br/>Read by scoring logic and dashboard]
            IndustryWeights[industry_weights.json<br/>Industry performance weights<br/>Written by main scoring/industry scoring<br/>Read by scoring logic]
            Visualizations[vis1-vis10.png<br/>Analysis charts<br/>Written by main scoring<br/>Read by dashboard]
        end
    end

    subgraph "Import Relationships"
        MainScoring --> NameUtils
        MainScoring --> IndustryUtils
        MainScoring --> IndustryScoring
        MainScoring --> ScoringLogic

        Dashboard --> ScoringLogic
        Dashboard --> NameUtils

        Optimization --> OptimizationObj

        IndustryScoring --> ScoringLogic
    end

    subgraph "Data Read Relationships"
        MainScoring --> CustomerData
        MainScoring --> SalesData
        MainScoring --> RevenueData
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
        CustomerData -.-> MainScoring
        SalesData -.-> MainScoring
        RevenueData -.-> MainScoring
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

1. **goe_icp_scoring.py** (Required - Main Processing)
   - Reads: Customer data, sales data, revenue data, industry data, config
   - Writes: Scored accounts, industry weights, visualizations
   - Imports: All utility modules and scoring logic

2. **run_optimization.py** (Optional - Weight Optimization)
   - Reads: Scored accounts dataset
   - Writes: Optimized weights JSON
   - Imports: Optimization objective function
   - Depends on: Main scoring completion

3. **streamlit_icp_dashboard.py** (Interactive Analysis)
   - Reads: Scored accounts, optimized weights, industry weights, visualizations
   - No file writes (analysis only)
   - Imports: Scoring logic and name utilities
   - Depends on: Main scoring completion

### Critical Dependencies:

#### **Hard Dependencies** (Must Exist):
- `goe_icp_scoring.py` → `scoring_logic.py` (core scoring functions)
- `goe_icp_scoring.py` → `industry_scoring.py` (industry weight calculation)
- `streamlit_icp_dashboard.py` → `icp_scored_accounts.csv` (data source)
- `scoring_logic.py` → `industry_weights.json` (industry scores)

#### **Soft Dependencies** (Fallback Available):
- `optimized_weights.json` (falls back to DEFAULT_WEIGHTS)
- `enrichment_progress.csv` (falls back to printer-based revenue estimation)
- `TR - Industry Enrichment.csv` (uses original industry data)

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




