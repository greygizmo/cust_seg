# Overall System Architecture

```mermaid
graph TB
    %% Define main system components
    subgraph "Data Sources"
        CustomerData[JY - Customer Analysis - Customer Segmentation.xlsx<br/>Contains: Industry, Revenue, Customer Details]
        SalesData[TR - Master Sales Log - Customer Segementation.xlsx<br/>Contains: GP24, Revenue24, Historical Sales Data]
        RevenueData[enrichment_progress.csv<br/>Contains: Enriched Annual Revenue Data]
        IndustryData[TR - Industry Enrichment.csv<br/>Contains: Updated Industry Classifications]
    end

    subgraph "Data Processing Pipeline"
        NameCleaning[normalize_names.py<br/>Standardizes Company Names for Matching]
        IndustryCleanup[cleanup_industry_data.py<br/>Standardizes Industry Classifications]
        IndustryScoring[industry_scoring.py<br/>Calculates Data-Driven Industry Weights]
        DataAggregation[goe_icp_scoring.py<br/>End-to-End Data Processing & Scoring]
    end

    subgraph "Scoring Engine"
        ScoringLogic[scoring_logic.py<br/>Centralized Scoring Functions<br/>- Vertical Score (Industry)<br/>- Size Score (Revenue)<br/>- Adoption Score (Hardware/Printer)<br/>- Relationship Score (Software)]
        WeightOptimization[Weight Optimization Process<br/>- run_optimization.py<br/>- optimize_weights.py<br/>- Uses Optuna for ML Optimization]
    end

    subgraph "Analytics & Visualization"
        Dashboard[streamlit_icp_dashboard.py<br/>Interactive Web Dashboard<br/>- Real-time Weight Adjustment<br/>- Customer Segmentation<br/>- Interactive Charts & Metrics<br/>- Export Functionality]
        VisualOutputs[Generated Visualizations<br/>- ICP Score Distribution<br/>- Segment Analysis<br/>- Industry Performance<br/>- Revenue Correlation Charts]
    end

    subgraph "Configuration & Storage"
        ConfigFiles[Configuration Files<br/>- config.toml<br/>- strategic_industry_tiers.json<br/>- requirements.txt]
        OptimizedWeights[optimized_weights.json<br/>Stores ML-Optimized Scoring Weights]
        IndustryWeights[industry_weights.json<br/>Stores Data-Driven Industry Scores]
        ScoredData[icp_scored_accounts.csv<br/>Final Scored Customer Dataset]
    end

    %% Define data flow connections
    CustomerData --> NameCleaning
    SalesData --> NameCleaning
    RevenueData --> NameCleaning
    IndustryData --> IndustryCleanup

    NameCleaning --> DataAggregation
    IndustryCleanup --> IndustryScoring
    IndustryScoring --> DataAggregation
    IndustryScoring --> IndustryWeights

    DataAggregation --> ScoredData
    ScoredData --> Dashboard
    ScoredData --> WeightOptimization

    WeightOptimization --> OptimizedWeights
    OptimizedWeights --> ScoringLogic
    OptimizedWeights --> Dashboard

    ScoringLogic --> DataAggregation
    ScoringLogic --> Dashboard

    Dashboard --> VisualOutputs

    %% Style definitions
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef scoring fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef analytics fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    %% Apply styles
    class CustomerData,SalesData,RevenueData,IndustryData dataSource
    class NameCleaning,IndustryCleanup,IndustryScoring,DataAggregation processing
    class ScoringLogic,WeightOptimization scoring
    class Dashboard,VisualOutputs analytics
    class ConfigFiles,OptimizedWeights,IndustryWeights,ScoredData storage
```

## System Overview

This architecture represents a comprehensive **Customer Segmentation and ICP (Ideal Customer Profile) Scoring System** for GoEngineer Digital Manufacturing. The system processes customer data through multiple stages to generate actionable insights for sales and marketing teams.

### Key Features:
- **Data-Driven Scoring**: Uses historical revenue data to calculate industry performance weights
- **Machine Learning Optimization**: Employs Optuna to find optimal scoring weights
- **Interactive Dashboard**: Real-time weight adjustment and visualization
- **Multi-Source Data Integration**: Combines customer, sales, and enriched revenue data
- **Automated Pipeline**: End-to-end processing from raw data to scored accounts

### Architecture Layers:
1. **Data Sources**: Raw Excel and CSV files containing customer information
2. **Data Processing**: Name standardization, industry classification, and data aggregation
3. **Scoring Engine**: Core ICP calculation logic with ML-optimized weights
4. **Analytics & Visualization**: Interactive dashboard with real-time analytics
5. **Configuration & Storage**: Persistent storage of optimized weights and results

### Data Flow:
1. Raw data is cleaned and standardized
2. Industry weights are calculated using historical performance data
3. Customer scores are computed using the optimized weighting system
4. Results are presented through an interactive dashboard
5. The system continuously improves through ML optimization




