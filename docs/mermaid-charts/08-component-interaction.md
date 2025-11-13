# Component Interaction Diagram

```mermaid
graph TB
    %% Define main Python modules and their key functions/classes
    subgraph "Data Processing Layer"
        IndustryCleaner[scripts/clean/cleanup_industry_data.py<br/>Industry standardization utilities]
        IndustryScorer[src/icp/industry.py<br/>build_industry_weights()<br/>save_industry_weights()<br/>load_industry_weights()]
        MainProcessor[src/icp/cli/score_accounts.py<br/>main() - orchestrates data assembly & scoring<br/>assemble_master_from_db(), engineer_features(), build_visuals()]
    end

    subgraph "Scoring Engine Layer"
        ScoringLogic[src/icp/scoring.py<br/>calculate_scores(), calculate_grades(), load_dynamic_industry_weights()<br/>Constants: DEFAULT_WEIGHTS, TARGET_GRADE_DISTRIBUTION]
    end

    subgraph "Optimization Layer"
        OptimizationRunner[src/icp/cli/optimize_weights.py<br/>run_optimization() - CLI wrapper]
        ObjectiveFunction[src/icp/optimization.py<br/>objective() - Optuna objective function<br/>Spearman correlation + KL divergence]</n+    end

    subgraph "Dashboard Layer"
        DashboardApp[apps/streamlit/app.py<br/>Main dashboard + Call List Builder<br/>Docs integration]
        StreamlitComponents[Streamlit UI Components<br/>st.sidebar, st.slider, st.selectbox<br/>st.columns, st.plotly_chart<br/>st.dataframe, st.download_button]
    end

    subgraph "External Libraries"
        PandasLib[pandas<br/>DataFrame operations<br/>Data manipulation and analysis<br/>CSV/Excel file handling]
        NumpyLib[numpy<br/>Numerical computations<br/>Array operations<br/>Statistical functions]
        ScipyLib[scipy.stats<br/>norm.ppf() - Normal distribution<br/>spearmanr() - Correlation<br/>norm - CDF/PDF functions]
        OptunaLib[optuna<br/>create_study()<br/>suggest_float() - Parameter suggestion<br/>Trial pruning and optimization]
        StreamlitLib[streamlit<br/>Web app framework<br/>Interactive widgets<br/>Data visualization]
        PlotlyLib[plotly<br/>Interactive charts<br/>px.bar, px.pie, px.histogram<br/>go.Figure - Advanced charts]
        SklearnLib[sklearn.preprocessing<br/>MinMaxScaler<br/>Standardization functions]
    end

    subgraph "Data Storage Layer"
        InputFiles[Inputs<br/>Azure SQL sources<br/>CSV: Industry Enrichment (optional)
JSON: Strategic config]
        OutputFiles[Generated Artifacts<br/>data/processed/icp_scored_accounts.csv<br/>artifacts/weights/optimized_weights.json
artifacts/weights/industry_weights.json
reports/figures/vis1-vis10.png]
        CacheFiles[Cache & Backups
archive/*]
    end

    %% Define function call relationships
    MainProcessor --> IndustryCleaner
    MainProcessor --> IndustryScorer
    MainProcessor --> ScoringLogic

    IndustryScorer --> PandasLib
    IndustryScorer --> NumpyLib

    ScoringLogic --> PandasLib
    ScoringLogic --> NumpyLib
    ScoringLogic --> ScipyLib

    OptimizationRunner --> ObjectiveFunction
    OptimizationRunner --> OptunaLib
    OptimizationRunner --> PandasLib

    ObjectiveFunction --> ScipyLib
    ObjectiveFunction --> PandasLib
    ObjectiveFunction --> NumpyLib

    DashboardApp --> ScoringLogic
    DashboardApp --> StreamlitLib
    DashboardApp --> PlotlyLib
    DashboardApp --> PandasLib

    StreamlitComponents --> StreamlitLib

    %% Data flow relationships
    InputFiles --> MainProcessor
    InputFiles --> IndustryScorer
    InputFiles --> OptimizationRunner

    MainProcessor --> OutputFiles
    IndustryScorer --> OutputFiles
    OptimizationRunner --> OutputFiles

    OutputFiles --> DashboardApp
    CacheFiles --> DashboardApp

    %% Library dependencies
    MainProcessor --> PandasLib
    MainProcessor --> NumpyLib
    MainProcessor --> SklearnLib

    DashboardApp --> NumpyLib
    OptimizationRunner --> NumpyLib

    %% Style definitions
    classDef processing fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef scoring fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef optimization fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef dashboard fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef libraries fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    classDef storage fill:#ffecb3,stroke:#f57c00,stroke-width:2px

    %% Apply styles
    class NameNormalizer,IndustryCleaner,IndustryScorer,MainProcessor processing
    class ScoringLogic scoring
    class OptimizationRunner,ObjectiveFunction optimization
    class DashboardApp,StreamlitComponents dashboard
    class PandasLib,NumpyLib,ScipyLib,OptunaLib,StreamlitLib,PlotlyLib,SklearnLib libraries
    class InputFiles,OutputFiles,CacheFiles storage
```

## Component Interaction Details

This diagram shows how the Python modules and external libraries interact to form the complete ICP scoring system.

### Layer Architecture:

#### 1. Data Processing Layer
- **normalize_names.py**: Provides name standardization for data merging
- **cleanup_industry_data.py**: Handles industry classification standardization
- **industry_scoring.py**: Calculates data-driven industry weights using Empirical-Bayes
- **goe_icp_scoring.py**: Main orchestrator that calls all other processing functions

#### 2. Scoring Engine Layer
- **scoring_logic.py**: Core business logic containing:
  - `calculate_scores()`: Main scoring function with 4 component calculations
  - `calculate_grades()`: Converts scores to A-F grades
  - `percentile_scale()`: Handles zero-value exclusion for fair scaling
  - Constants for default weights and target distributions

#### 3. Optimization Layer
- **run_optimization.py**: Configures and executes Optuna studies
- **optimize_weights.py**: Contains the objective function that:
  - Calculates Spearman correlation between scores and revenue
  - Computes KL divergence for grade distribution matching
  - Applies business constraints (weight limits, normalization)

#### 4. Dashboard Layer
- **streamlit_icp_dashboard.py**: Main dashboard application with:
  - Real-time weight adjustment via sliders
  - Interactive charts using Plotly
  - Data export functionality
  - Segment filtering and analysis

### Key Function Call Flows:

#### Main Processing Pipeline:
1. `goe_icp_scoring.py:main()` calls:
   - `normalize_names.py` functions for name standardization
   - `industry_scoring.py:build_industry_weights()` for industry analysis
   - `scoring_logic.py:calculate_scores()` for final scoring

#### Optimization Process:
1. `run_optimization.py:run_optimization()` calls:
   - `optimize_weights.py:objective()` for each Optuna trial
   - Pandas for data loading and manipulation
   - Optuna for parameter suggestion and study management

#### Dashboard Operation:
1. `streamlit_icp_dashboard.py:main()` calls:
   - `scoring_logic.py:calculate_scores()` for real-time recalculation
   - Plotly functions for chart generation
   - Streamlit components for UI rendering

### External Library Dependencies:

#### Core Data Science Stack:
- **pandas**: Data manipulation, CSV/Excel handling, DataFrame operations
- **numpy**: Numerical computations, array operations
- **scipy.stats**: Statistical functions (norm.ppf, spearmanr)

#### Machine Learning & Optimization:
- **optuna**: Hyperparameter optimization framework
- **sklearn**: Preprocessing functions (MinMaxScaler)

#### Web Application & Visualization:
- **streamlit**: Web app framework, interactive widgets
- **plotly**: Interactive charts and visualizations

### Data Flow Patterns:

#### Input  Processing  Output:
```
Input Files  Processing Functions  Generated Files  Dashboard
```

#### Configuration  Processing  Results:
```
Strategic Config  Industry Scoring  Industry Weights  Scoring Logic
```

#### Optimization Loop:
```
Scored Data  Optimization  Better Weights  Improved Scoring
```

### Component Coupling:

#### Tightly Coupled:
- `scoring_logic.py` and `streamlit_icp_dashboard.py` (real-time interaction)
- `industry_scoring.py` and `scoring_logic.py` (shared data structures)

#### Loosely Coupled:
- Optimization components (can be run independently)
- Dashboard components (can operate with default weights)

#### Independent:
- Name normalization (pure utility functions)
- Industry cleanup (standalone processing)

This architecture ensures modularity, testability, and maintainability while supporting the complex interactions required for a sophisticated customer segmentation system.




