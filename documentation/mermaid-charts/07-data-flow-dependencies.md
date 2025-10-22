# Data Flow & Dependencies

```mermaid
graph TB
    %% Define main data sources
    subgraph "External Data Sources"
        CustomerExcel[JY - Customer Analysis.xlsx<br/>Customer master data<br/>Industry, revenue, printer counts]
        SalesExcel[TR - Master Sales Log.xlsx<br/>Historical sales transactions<br/>GP, revenue, dates]
        RevenueCSV[enrichment_progress.csv<br/>Enriched revenue data<br/>SEC, PDL, FMP sources]
        IndustryCSV[TR - Industry Enrichment.csv<br/>Updated industry classifications<br/>With reasoning and validation]
    end

    subgraph "Configuration Files"
        ConfigTOML[config.toml<br/>System configuration<br/>File paths, parameters]
        StrategicJSON[strategic_industry_tiers.json<br/>Strategic priorities<br/>Industry tiers, blend weights]
        RequirementsTXT[requirements.txt<br/>Python dependencies<br/>Package versions]
    end

    subgraph "Core Processing Scripts"
        NameCleaningScript[normalize_names.py<br/>Standardize company names<br/>Create matching keys]
        IndustryCleanupScript[cleanup_industry_data.py<br/>Standardize industry names<br/>Fuzzy matching logic]
        IndustryScoringScript[industry_scoring.py<br/>Calculate industry weights<br/>Empirical-Bayes shrinkage]
        MainScoringScript[goe_icp_scoring.py<br/>End-to-end processing<br/>Orchestrates all steps]
    end

    subgraph "Scoring & Optimization"
        ScoringLogicModule[scoring_logic.py<br/>Core scoring functions<br/>Component score calculations]
        OptimizationScript[run_optimization.py<br/>Weight optimization<br/>Optuna study execution]
        OptimizationFunction[optimize_weights.py<br/>Objective function<br/>Multi-objective optimization]
    end

    subgraph "Dashboard & Visualization"
        DashboardScript[streamlit_icp_dashboard.py<br/>Interactive dashboard<br/>Real-time analysis]
        VisualizationOutputs[Generated Charts<br/>vis1-vis10 PNG files<br/>Analysis visualizations]
    end

    subgraph "Generated Data Files"
        ScoredAccountsCSV[icp_scored_accounts.csv<br/>Final scored dataset<br/>All customer scores and metadata]
        OptimizedWeightsJSON[optimized_weights.json<br/>ML-optimized weights<br/>Optimization metadata]
        IndustryWeightsJSON[industry_weights.json<br/>Data-driven industry scores<br/>Processing metadata]
        EnrichmentBackup[TR - Industry Enrichment_backup_*.csv<br/>Historical backups<br/>Version control]
    end

    %% Define dependency relationships
    CustomerExcel --> NameCleaningScript
    CustomerExcel --> MainScoringScript
    SalesExcel --> NameCleaningScript
    SalesExcel --> MainScoringScript
    RevenueCSV --> MainScoringScript
    IndustryCSV --> IndustryCleanupScript
    IndustryCSV --> MainScoringScript

    ConfigTOML --> MainScoringScript
    StrategicJSON --> IndustryScoringScript

    NameCleaningScript --> MainScoringScript
    IndustryCleanupScript --> IndustryScoringScript
    IndustryCleanupScript --> MainScoringScript

    IndustryScoringScript --> MainScoringScript
    IndustryScoringScript --> IndustryWeightsJSON

    MainScoringScript --> ScoredAccountsCSV
    MainScoringScript --> VisualizationOutputs
    ScoredAccountsCSV --> DashboardScript
    ScoredAccountsCSV --> OptimizationScript

    ScoredAccountsCSV --> OptimizationScript
    OptimizationScript --> OptimizationFunction
    OptimizationFunction --> OptimizedWeightsJSON

    OptimizedWeightsJSON --> ScoringLogicModule
    OptimizedWeightsJSON --> DashboardScript
    IndustryWeightsJSON --> ScoringLogicModule
    ScoringLogicModule --> MainScoringScript
    ScoringLogicModule --> DashboardScript

    DashboardScript --> VisualizationOutputs

    %% Additional relationships for optimization
    ScoredAccountsCSV --> OptimizationFunction
    StrategicJSON --> MainScoringScript

    %% Style definitions with clear visual separation
    classDef dataSource fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef config fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef processing fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef scoring fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef dashboard fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef output fill:#f9fbe7,stroke:#689f38,stroke-width:2px

    %% Apply styles
    class CustomerExcel,SalesExcel,RevenueCSV,IndustryCSV dataSource
    class ConfigTOML,StrategicJSON,RequirementsTXT config
    class NameCleaningScript,IndustryCleanupScript,IndustryScoringScript,MainScoringScript processing
    class ScoringLogicModule,OptimizationScript,OptimizationFunction scoring
    class DashboardScript,VisualizationOutputs dashboard
    class ScoredAccountsCSV,OptimizedWeightsJSON,IndustryWeightsJSON,EnrichmentBackup output
```

## Data Flow & Dependencies Overview

This diagram shows the complete data flow and interdependencies between all components of the ICP scoring system.

### Data Flow Paths:

#### 1. Raw Data  Processing Pipeline
```
External Data Sources  Processing Scripts  Generated Data Files
```
- Customer and sales data are cleaned and standardized
- Industry classifications are enriched and validated
- Revenue data is prioritized and merged
- All data flows through the main scoring script

#### 2. Configuration  Processing
```
Configuration Files  Processing Scripts
```
- Strategic priorities guide industry scoring
- System configuration controls processing parameters
- Dependencies specify required packages

#### 3. Processing  Scoring Engine
```
Generated Data Files  Scoring & Optimization  Dashboard
```
- Industry weights feed into the scoring logic
- Optimized weights enhance the scoring model
- Scoring functions power the dashboard calculations

#### 4. Processing  Output Generation
```
Processing Scripts  Generated Data Files  Visualizations
```
- Scored accounts dataset is the central output
- Optimization produces improved weights
- Industry analysis generates performance insights

### Key Dependency Relationships:

#### Critical Path Dependencies:
1. **Name normalization** must complete before data merging
2. **Industry cleanup** must complete before industry scoring
3. **Main scoring script** requires all input data and processed weights
4. **Dashboard** requires the scored accounts dataset
5. **Optimization** requires the scored accounts dataset

#### Optional Dependencies:
1. **Industry enrichment** enhances but doesn't block processing
2. **Revenue enrichment** improves but isn't required for core scoring
3. **Weight optimization** improves but doesn't block basic scoring

#### Circular Dependencies:
1. **Industry weights** are used by scoring logic
2. **Scoring logic** is used by the main processing script
3. **Main processing script** generates the data used for industry scoring

### File Dependencies:

#### Input Dependencies:
- `goe_icp_scoring.py` depends on 4+ Excel/CSV files
- `industry_scoring.py` depends on processed customer data
- `run_optimization.py` depends on scored accounts data

#### Output Dependencies:
- `icp_scored_accounts.csv` is required by dashboard and optimization
- `optimized_weights.json` is required by scoring logic and dashboard
- `industry_weights.json` is required by scoring logic

#### Configuration Dependencies:
- `strategic_industry_tiers.json` configures industry priorities
- `config.toml` provides system-wide settings
- `requirements.txt` specifies runtime dependencies

### Processing Order:

1. **Data Collection**: Gather all input files
2. **Data Standardization**: Clean names and industries
3. **Industry Analysis**: Generate data-driven weights
4. **Core Processing**: Calculate all scores and metrics
5. **Optimization**: Improve weights using ML (optional)
6. **Visualization**: Generate charts and dashboard

### Recovery Points:
- **Backup files** preserve previous industry enrichment states
- **Fallback weights** ensure system works without optimization
- **Error handling** in scripts prevents complete failures
- **Caching** in dashboard improves performance

This dependency structure ensures the system is robust, maintainable, and can operate at reduced capacity even if some components fail.




