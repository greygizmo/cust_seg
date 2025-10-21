# Data Processing Pipeline

```mermaid
graph TD
    %% Define the main pipeline stages
    subgraph "Input Data Sources"
        CustomerFile[JY - Customer Analysis - Customer Segmentation.xlsx]
        SalesFile[TR - Master Sales Log - Customer Segementation.xlsx]
        RevenueFile[enrichment_progress.csv]
        IndustryFile[TR - Industry Enrichment.csv]
    end

    subgraph "Stage 1: Data Loading"
        LoadCustomer[Load Customer Data<br/>- Customer ID, Company Name<br/>- Industry, Industry Sub List<br/>- Big Box Count, Small Box Count<br/>- Total Hardware Revenue, Consumable Revenue<br/>- Total Software License Revenue]
        LoadSales[Load Sales Data<br/>- Company Name, Dates<br/>- GP (Gross Profit), Revenue]
        LoadRevenue[Load Revenue Data<br/>- Company Name, Revenue Estimate<br/>- Source (SEC, PDL, FMP, Heuristic)<br/>- Confidence Score]
        LoadIndustry[Load Industry Data<br/>- Customer ID, Industry<br/>- Industry Sub List, Reasoning]
    end

    subgraph "Stage 2: Data Standardization"
        NameNormalization[Company Name Normalization<br/>- Remove leading customer IDs<br/>- Convert to lowercase<br/>- Remove punctuation<br/>- Standardize spacing<br/>- Create 'key' field for matching]
        IndustryStandardization[Industry Classification Cleanup<br/>- Apply fuzzy matching<br/>- Standardize industry names<br/>- Handle variations and synonyms<br/>- Validate classifications]
    end

    subgraph "Stage 3: Industry Enrichment"
        EnrichmentProcess[Industry Enrichment Process<br/>- Match on Customer ID<br/>- Apply enriched classifications<br/>- Preserve original data<br/>- Track changes and reasoning<br/>- Handle missing data gracefully]
    end

    subgraph "Stage 4: Data Integration"
        RevenuePrioritization[Revenue Data Prioritization<br/>- Priority 1: SEC filings (highest reliability)<br/>- Priority 2: PDL estimates<br/>- Priority 3: FMP data (filtered)<br/>- Priority 4: Discard heuristics<br/>- Create reliable_revenue field]
        GP24Aggregation[GP24 Calculation<br/>- Filter to last 24 months<br/>- Aggregate by company key<br/>- Calculate total GP24<br/>- Calculate total Revenue24]
        MasterMerge[Master DataFrame Creation<br/>- Left join on normalized company name<br/>- Merge GP24 and Revenue24<br/>- Add revenue estimates<br/>- Handle missing data]
    end

    subgraph "Stage 5: Feature Engineering"
        PrinterFeatures[Printer Count Features<br/>- Calculate total printer count<br/>- Big Box (2x weight) + Small Box (1x weight)<br/>- Create scaling flag (>=4 printers)<br/>- Calculate weighted printer score]
        RevenueFeatures[Revenue Features<br/>- Calculate total hardware + consumable revenue<br/>- Create target variable for optimization<br/>- Handle missing values (set to 0)]
        SoftwareFeatures[Software Revenue Features<br/>- Aggregate all software revenue types<br/>- License, SaaS, Maintenance revenue<br/>- Create relationship_feature<br/>- Calculate CAD tier classification]
    end

    subgraph "Stage 6: Industry Weight Calculation"
        PerformanceCalculation[Industry Performance Calculation<br/>- Calculate total performance per customer<br/>- Hardware + Consumable + Service revenue<br/>- Group by industry for aggregation]
        AdoptionMetrics[Adoption-Adjusted Success Metric<br/>- Calculate adoption rate per industry<br/>- Mean revenue among adopters<br/>- Success = adoption_rate × mean_among_adopters<br/>- Filter industries with minimum sample size]
        EmpiricalBayes[Empirical-Bayes Shrinkage<br/>- Apply shrinkage to handle small samples<br/>- Balance observed vs. global performance<br/>- Prevent overfitting to small industries<br/>- Calculate shrinkage factors]
        StrategicBlending[Strategic Score Blending<br/>- Load strategic industry tiers<br/>- Blend data-driven and strategic scores<br/>- Apply configurable blend weights<br/>- Final bucketing and normalization]
    end

    subgraph "Stage 7: ICP Score Calculation"
        ComponentScoring[Component Score Calculation<br/>- Vertical Score (Industry-based)<br/>- Size Score (Revenue-based)<br/>- Adoption Score (Hardware engagement)<br/>- Relationship Score (Software revenue)]
        WeightApplication[Weight Application & Normalization<br/>- Apply optimized or default weights<br/>- Normalize weights to sum to 1.0<br/>- Calculate raw weighted score<br/>- Convert to 0-100 scale]
        FinalNormalization[Final Score Normalization<br/>- Rank-based percentile conversion<br/>- Apply normal distribution transformation<br/>- Create bell curve distribution<br/>- Assign A-F letter grades]
    end

    subgraph "Stage 8: Output Generation"
        ScoredDataset[Generate icp_scored_accounts.csv<br/>- All customer data with scores<br/>- Component scores and final ICP<br/>- Industry classifications<br/>- Revenue and printer data<br/>- CAD tier and grade assignments]
        VisualizationGeneration[Generate PNG Visualizations<br/>- ICP Score Distribution Histogram<br/>- GP24 by Industry Vertical<br/>- Printer Count vs GP24 Scatter<br/>- Revenue24 by Industry<br/>- Score by Industry Analysis<br/>- Customer Segment Analysis]
        MetadataStorage[Store Metadata & Weights<br/>- Save industry_weights.json<br/>- Store processing timestamps<br/>- Record sample sizes and parameters<br/>- Track data quality metrics]
    end

    %% Define data flow connections
    CustomerFile --> LoadCustomer
    SalesFile --> LoadSales
    RevenueFile --> LoadRevenue
    IndustryFile --> LoadIndustry

    LoadCustomer --> NameNormalization
    LoadSales --> NameNormalization
    LoadRevenue --> NameNormalization
    LoadIndustry --> IndustryStandardization

    NameNormalization --> EnrichmentProcess
    IndustryStandardization --> EnrichmentProcess

    EnrichmentProcess --> RevenuePrioritization
    LoadRevenue --> RevenuePrioritization

    NameNormalization --> GP24Aggregation
    LoadSales --> GP24Aggregation

    RevenuePrioritization --> MasterMerge
    GP24Aggregation --> MasterMerge

    MasterMerge --> PrinterFeatures
    MasterMerge --> RevenueFeatures
    MasterMerge --> SoftwareFeatures

    RevenueFeatures --> PerformanceCalculation
    PerformanceCalculation --> AdoptionMetrics
    AdoptionMetrics --> EmpiricalBayes
    EmpiricalBayes --> StrategicBlending

    StrategicBlending --> ComponentScoring
    PrinterFeatures --> ComponentScoring
    RevenueFeatures --> ComponentScoring
    SoftwareFeatures --> ComponentScoring

    ComponentScoring --> WeightApplication
    WeightApplication --> FinalNormalization

    FinalNormalization --> ScoredDataset
    ScoredDataset --> VisualizationGeneration
    StrategicBlending --> MetadataStorage

    %% Style definitions
    classDef inputData fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef loading fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef standardization fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef enrichment fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef integration fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef engineering fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef industry fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    classDef scoring fill:#ffecb3,stroke:#f57c00,stroke-width:2px
    classDef output fill:#f5f5f5,stroke:#616161,stroke-width:2px

    %% Apply styles
    class CustomerFile,SalesFile,RevenueFile,IndustryFile inputData
    class LoadCustomer,LoadSales,LoadRevenue,LoadIndustry loading
    class NameNormalization,IndustryStandardization standardization
    class EnrichmentProcess enrichment
    class RevenuePrioritization,GP24Aggregation,MasterMerge integration
    class PrinterFeatures,RevenueFeatures,SoftwareFeatures engineering
    class PerformanceCalculation,AdoptionMetrics,EmpiricalBayes,StrategicBlending industry
    class ComponentScoring,WeightApplication,FinalNormalization scoring
    class ScoredDataset,VisualizationGeneration,MetadataStorage output
```

## Data Processing Pipeline Details

This pipeline transforms raw customer data into actionable ICP (Ideal Customer Profile) scores through an 8-stage process.

### Stage 1: Data Loading
- **Customer Data**: Industry classifications, printer counts, revenue breakdowns
- **Sales Data**: Historical GP and revenue over time
- **Revenue Data**: Enriched annual revenue estimates from multiple sources
- **Industry Data**: Updated industry classifications with reasoning

### Stage 2: Data Standardization
- **Name Normalization**: Creates consistent matching keys across datasets
- **Industry Cleanup**: Standardizes industry names and handles variations

### Stage 3: Industry Enrichment
- **Customer ID Matching**: Updates industry classifications using enriched data
- **Change Tracking**: Preserves original data and tracks reasoning for changes

### Stage 4: Data Integration
- **Revenue Prioritization**: Implements 4-tier priority system for revenue data quality
- **GP24 Calculation**: Aggregates last 24 months of sales performance
- **Master Merge**: Creates unified dataset using normalized company names

### Stage 5: Feature Engineering
- **Printer Features**: Weighted scoring (Big Box = 2x, Small Box = 1x)
- **Revenue Features**: Creates optimization target variable
- **Software Features**: Aggregates all software revenue types for relationship scoring

### Stage 6: Industry Weight Calculation
- **Performance Calculation**: Measures actual revenue performance by industry
- **Adoption Metrics**: Calculates adoption-adjusted success rates
- **Empirical-Bayes**: Prevents overfitting to small industry samples
- **Strategic Blending**: Combines data-driven and strategic priorities

### Stage 7: ICP Score Calculation
- **Component Scores**: Calculates four dimensions (Vertical, Size, Adoption, Relationship)
- **Weight Application**: Applies ML-optimized or default weights
- **Normalization**: Converts to 0-100 scale with bell curve distribution

### Stage 8: Output Generation
- **Scored Dataset**: Complete customer dataset with all scores and classifications
- **Visualizations**: 10 key charts for analysis and reporting
- **Metadata Storage**: Preserves weights and processing information

### Key Quality Controls:
- **Minimum Sample Sizes**: Ensures statistical significance
- **Missing Data Handling**: Graceful fallbacks for incomplete data
- **Source Prioritization**: Uses highest quality data available
- **Validation Checks**: Prevents processing of invalid data




