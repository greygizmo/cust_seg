# Industry Scoring Process

```mermaid
graph TD
    %% Define the industry scoring workflow
    subgraph "Input Data"
        MasterData[Scored Customer Data<br/>(assembled)<br/>Contains: Industry, GP24, Revenue24<br/>Total Hardware + Consumable Revenue<br/>Printer counts and software revenue]
        StrategicConfig[artifacts/industry/strategic_industry_tiers.json<br/>Contains: Tier definitions<br/>Industry to tier mapping<br/>Tier scores and blend weights]
    end

    subgraph "Data-Driven Performance Analysis"
        PerformanceCalculation[Calculate Performance per Customer<br/>Total Performance = Hardware Revenue +<br/>Consumable Revenue + Service Bureau Revenue<br/>Measures actual hardware engagement]
        IndustryGrouping[Group by Industry Classification<br/>Clean and standardize industry names<br/>Handle missing/blank industries<br/>Minimum sample size filter (default: 10 customers)]
        AdoptionMetrics[Calculate Adoption-Adjusted Success<br/>Success = (adoption_rate)  (mean_revenue_among_adopters)<br/>Adoption Rate = customers_with_revenue / total_customers<br/>Mean Among Adopters = avg performance of revenue-generating customers]
    end

    subgraph "Empirical-Bayes Shrinkage"
        GlobalAverage[Calculate Global Success Average<br/>Weighted average across all industries<br/>Weight by customer count<br/>Provides prior for shrinkage]
        ShrinkageCalculation[Apply Empirical-Bayes Shrinkage<br/>Formula: (n_i  success_i + k  global) / (n_i + k)<br/>k = shrinkage parameter (default: 20)<br/>Balances observed vs global performance]
        ShrinkageFactors[Calculate Shrinkage Factors<br/>Shrinkage Factor = k / (n_i + k)<br/>Higher for small industries<br/>Lower for large industries]
    end

    subgraph "Strategic Priority Integration"
        StrategicTiers[Load Strategic Industry Tiers<br/>Tier 1: High priority (score: 1.0)<br/>Tier 2: Medium priority (score: 0.7)<br/>Tier 3: Standard priority (score: 0.4)<br/>Fallback for unmapped industries]
        IndustryMapping[Map Industries to Strategic Tiers<br/>Dictionary lookup: industry  tier<br/>Default to tier_3 if not found<br/>Handles variations in naming]
        StrategicScores[Calculate Strategic Scores<br/>Direct mapping from tier to score<br/>Fixed business priorities<br/>Independent of historical performance]
    end

    subgraph "Hybrid Score Blending"
        BlendConfiguration[Load Blend Weights Configuration<br/>Data-Driven Weight (default: 0.7)<br/>Strategic Weight (default: 0.3)<br/>Configurable trade-off ratio]
        ScoreNormalization[Normalize Data-Driven Scores<br/>Convert to 0-1 scale using min-max<br/>Preserve relative differences<br/>Handle edge cases (all same = 0.5)]
        WeightedBlending[Calculate Blended Scores<br/>Blended = (data_weight  normalized_data) +<br/>(strategic_weight  strategic_score)<br/>Combines empirical evidence with business priorities]
        FinalBucketing[Apply Final Score Bucketing<br/>Round to 0.05 increments (0.00, 0.05, 0.10, ...)<br/>Minimum score = 0.30 (neutral fallback)<br/>Maximum score = 1.00 (cap at 100%)]
    end

    subgraph "Quality Control & Validation"
        SampleSizeFilter[Apply Minimum Sample Size Filter<br/>Remove industries with < n customers<br/>Default: 10 customers minimum<br/>Prevents overfitting to small samples]
        UnknownHandling[Handle Unknown Industries<br/>Assign neutral score (0.30)<br/>Consistent treatment across system<br/>Fallback for missing classifications]
        ValidationChecks[Validate Final Scores<br/>Ensure all industries have valid scores<br/>Check score range (0.30-1.00)<br/>Verify no missing values]
    end

    subgraph "Output Generation"
        WeightDictionary[Create Industry Weight Dictionary<br/>industry_name  final_score mapping<br/>Lowercase keys for consistency<br/>Include fallback entries]
        MetadataCreation[Generate Metadata<br/>Processing timestamp<br/>Method description<br/>Sample sizes per industry<br/>Blend parameters used]
        JSONExport[artifacts/weights/{division}_industry_weights.json<br/>Weights dictionary + metadata<br/>Used by scoring logic<br/>Cached for performance]
        ResultsDisplay[Display Processing Results<br/>Number of industries processed<br/>Sample size distribution<br/>Score range achieved<br/>Shrinkage statistics]
    end

    subgraph "Industry Score Examples"
        HighScoreExample[High-Scoring Industries<br/>Aerospace & Defense: 1.0<br/>Automotive & Transportation: 1.0<br/>Medical Devices: 1.0<br/>Based on strong historical performance]
        MediumScoreExample[Medium-Scoring Industries<br/>Engineering Services: 0.8<br/>Industrial Machinery: 0.8<br/>Balanced performance characteristics]
        LowScoreExample[Lower-Scoring Industries<br/>Education & Research: 0.4<br/>Building & Construction: 0.6<br/>Adjusted by strategic priorities or small samples]
    end

    %% Define data flow connections
    MasterData --> PerformanceCalculation
    PerformanceCalculation --> IndustryGrouping
    IndustryGrouping --> AdoptionMetrics

    AdoptionMetrics --> GlobalAverage
    AdoptionMetrics --> ShrinkageCalculation
    GlobalAverage --> ShrinkageCalculation

    ShrinkageCalculation --> ShrinkageFactors
    ShrinkageCalculation --> ScoreNormalization

    StrategicConfig --> StrategicTiers
    StrategicTiers --> IndustryMapping
    IndustryMapping --> StrategicScores

    ScoreNormalization --> WeightedBlending
    StrategicScores --> WeightedBlending
    BlendConfiguration --> WeightedBlending

    WeightedBlending --> FinalBucketing
    FinalBucketing --> WeightDictionary

    IndustryGrouping --> SampleSizeFilter
    SampleSizeFilter --> UnknownHandling
    UnknownHandling --> ValidationChecks

    WeightDictionary --> MetadataCreation
    MetadataCreation --> JSONExport

    JSONExport --> ResultsDisplay

    FinalBucketing --> HighScoreExample
    FinalBucketing --> MediumScoreExample
    FinalBucketing --> LowScoreExample

    %% Style definitions
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef dataDriven fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef shrinkage fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef strategic fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef blending fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef quality fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef output fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    classDef examples fill:#ffecb3,stroke:#f57c00,stroke-width:2px

    %% Apply styles
    class MasterData,StrategicConfig input
    class PerformanceCalculation,IndustryGrouping,AdoptionMetrics dataDriven
    class GlobalAverage,ShrinkageCalculation,ShrinkageFactors shrinkage
    class StrategicTiers,IndustryMapping,StrategicScores strategic
    class BlendConfiguration,ScoreNormalization,WeightedBlending,FinalBucketing blending
    class SampleSizeFilter,UnknownHandling,ValidationChecks quality
    class WeightDictionary,MetadataCreation,JSONExport,ResultsDisplay output
    class HighScoreExample,MediumScoreExample,LowScoreExample examples
```

## Industry Scoring Process Details

The industry scoring process creates data-driven weights for each industry based on their historical performance with GoEngineer products. This ensures the ICP scoring system rewards customers from industries that have historically been successful partners.

### Core Methodology:

#### 1. Adoption-Adjusted Success Metric
Instead of simple average revenue, the system calculates:
```
Success = (Adoption Rate)  (Mean Revenue Among Adopters)
```

This accounts for two factors:
- **Adoption Rate**: Percentage of customers in the industry who actually buy hardware
- **Value When Adopted**: Average revenue from customers who do adopt hardware

#### 2. Empirical-Bayes Shrinkage
Small industries with few customers can have unreliable performance metrics. The system applies shrinkage to balance:
- **Observed Performance**: Actual data for that industry
- **Global Average**: Overall performance across all industries

Formula: `(n_i  success_i + k  global) / (n_i + k)`
- `n_i`: Number of customers in industry i
- `k`: Shrinkage parameter (default: 20)
- Higher shrinkage for small industries, lower for large ones

#### 3. Strategic Priority Integration
The system blends data-driven scores with strategic business priorities:
- **Tier 1** (Score: 1.0): Aerospace, Automotive, Medical, High Tech
- **Tier 2** (Score: 0.7): Engineering Services, Industrial Machinery
- **Tier 3** (Score: 0.4): Education, Building & Construction

#### 4. Hybrid Blending
```
Blended Score = (0.7  Data-Driven) + (0.3  Strategic)
```
- **70% weight** on empirical evidence
- **30% weight** on strategic priorities
- Configurable blend weights in `artifacts/industry/strategic_industry_tiers.json`

#### 5. Quality Controls
- **Minimum Sample Size**: Industries need 10 customers to be scored
- **Unknown Handling**: Unmapped industries get neutral score (0.30)
- **Score Bucketing**: Final scores rounded to 0.05 increments
- **Validation**: Ensures all industries have valid scores in range 0.30-1.00

### Key Benefits:
- **Data-Driven**: Based on actual historical performance
- **Robust**: Handles small sample sizes with statistical rigor
- **Strategic**: Incorporates business priorities
- **Transparent**: Clear methodology with configurable parameters
- **Predictive**: Uses adoption-adjusted metrics for better forecasting

### Output:
- **industry_weights.json**: Contains the final industry  score mapping
- **Used by**: `scoring_logic.py` for vertical score calculation
- **Integrated with**: Overall ICP scoring system
- **Cached**: Processed once and reused for performance






