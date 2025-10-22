# ICP Scoring Methodology

```mermaid
graph TD
    %% Define the main scoring components
    subgraph "Input Data"
        CustomerRecord[Customer Record<br/>Industry, profit, assets/seats<br/>Printer counts (legacy), software revenue]
        OptimizedWeights[Optimized Weights (artifacts/weights)
        - Vertical, Size, Adoption, Relationship]
    end

    subgraph "Component Score Calculation"
        subgraph "1. Vertical Score"
            IndustryLookup[Industry Classification<br/>From Customer Record or Enriched Data]
            IndustryWeights[Load Industry Weights<br/>From industry_weights.json<br/>Data-driven performance scores]
            VerticalScore[Calculate Vertical Score<br/>0.0 - 1.0 scale<br/>Fallback to 0.3 for unknown industries]
        end

        subgraph "2. Size Score"
            RevenueData[Revenue Data<br/>Total Hardware + Consumable Revenue<br/>From enriched or estimated data]
            RevenueBrackets[Revenue Tier Classification<br/>$250M-$1B: 1.0<br/>$50M-$250M: 0.6<br/>$10M-$50M: 0.4<br/>$0-$10M: 0.4<br/>Missing: 0.5 (neutral)]
            SizeScore[Calculate Size Score<br/>Based on revenue brackets<br/>Higher revenue = higher score]
        end

        subgraph "3. Adoption Score"
            subgraph "Hardware Data Collection"
                PreferredSignals[Preferred Signals (DB)
                - adoption_assets (weighted)
                - adoption_profit (Profit since 2023)]
                LegacySignals[Legacy Signals (fallback)
                - Weighted Printers (2x/1x)
                - HW + Consumable Revenue]
            end

            subgraph "Weighted Printer Score"
                WeightedPrinterCalc[Calculate Weighted Printer Score<br/>Big Box  2 + Small Box  1<br/>Represents hardware investment level]
            end

            subgraph "Percentile Scaling"
                ZeroExclusion[Exclude Zero Values<br/>Only non-zero customers for percentiles<br/>Prevents distribution compression]
                PctPreferred[P/R from preferred signals]
                PctLegacy[P/R from legacy signals]
            end

            subgraph "Business Rules Application"
                ZeroEverythingRule[Zero Everything Rule<br/>No printers + no revenue = 0.0<br/>True non-adopters]
                RevenueOnlyRule[Revenue-Only Rule (legacy)
Revenue but no printers = 0.0-0.5
0.5 * sqrt(R)]
                PrinterRule[Assets/Profit (preferred)
If assets>0: 0.6*P + 0.4*R
Legacy printers: same blend]
                HeavyFleetBonus[Heavy Fleet Bonus<br/>10+ weighted printers = +0.05<br/>Rewards significant investment]
            end

            AdoptionScore[Final Adoption Score<br/>0.0 - 1.0 scale<br/>Reflects hardware engagement level]
        end

        subgraph "4. Relationship Score"
            SoftwareRevenue[Software Signals<br/>Preferred: relationship_profit (software goals)
Fallback: License + SaaS + Maintenance]
            TotalSoftware[Calculate relationship feature]
            LogTransformation[Log Transformation<br/>log1p(total_software_revenue)<br/>Handles wide range of values]
            MinMaxScaling[Min-Max Scaling<br/>0.0 - 1.0 normalization<br/>Preserves relative differences]
            RelationshipScore[Calculate Relationship Score<br/>0.0 - 1.0 scale<br/>Reflects software engagement]
        end
    end

    subgraph "Final ICP Score Calculation"
        ComponentScores[Component Scores<br/>Vertical (0-1) + Size (0-1)<br/>+ Adoption (0-1) + Relationship (0-1)]
        WeightApplication[Apply Component Weights<br/>Raw Score = (component  weight)<br/>Weights sum to 1.0]
        PercentileConversion[Convert to Percentile Rank<br/>Rank-based percentile calculation<br/>Handles ties with average ranking]
        NormalDistribution[Normal Distribution Transformation<br/>Inverse CDF (ppf) conversion<br/>Creates bell curve distribution]
        ScaleTo100[Scale to 0-100 Range<br/>Mean = 50, Std Dev = 15<br/>Industry standard scaling]
        FinalICP[Final ICP Score<br/>0-100 scale<br/>Bell curve distribution]
    end

    subgraph "Grade Assignment"
        TargetDistribution[Target Grade Distribution<br/>A: 10% (top 10%)<br/>B: 20% (next 20%)<br/>C: 40% (middle 40%)<br/>D: 20% (next 20%)<br/>F: 10% (bottom 10%)]
        PercentileBrackets[Grade Assignment Brackets<br/>F: 0-10th percentile<br/>D: 10-30th percentile<br/>C: 30-70th percentile<br/>B: 70-90th percentile<br/>A: 90-100th percentile]
        LetterGrade[Assign Letter Grade<br/>A, B, C, D, or F<br/>Based on percentile rank]
    end

    subgraph "Output Components"
        FinalScore[ICP_score (0-100)]
        Grade[ICP_grade (A-F)]
        ComponentBreakdown[Component Scores<br/>vertical_score, size_score<br/>adoption_score, relationship_score]
        RawScore[ICP_score_raw (weighted sum)]
    end

    %% Define data flow connections
    CustomerRecord --> IndustryLookup
    CustomerRecord --> RevenueData
    CustomerRecord --> PrinterCounts
    CustomerRecord --> HardwareRevenue
    CustomerRecord --> SoftwareRevenue

    OptimizedWeights --> WeightApplication

    IndustryLookup --> IndustryWeights --> VerticalScore
    RevenueData --> RevenueBrackets --> SizeScore

    PrinterCounts --> WeightedPrinterCalc
    HardwareRevenue --> TotalHardwareRevenue

    WeightedPrinterCalc --> PrinterPercentile
    TotalHardwareRevenue --> RevenuePercentile

    PrinterPercentile --> ZeroExclusion
    RevenuePercentile --> ZeroExclusion

    ZeroExclusion --> BusinessRulesApplication
    BusinessRulesApplication --> ZeroEverythingRule
    BusinessRulesApplication --> RevenueOnlyRule
    BusinessRulesApplication --> PrinterRule
    BusinessRulesApplication --> HeavyFleetBonus
    HeavyFleetBonus --> AdoptionScore

    SoftwareRevenue --> TotalSoftware --> LogTransformation --> MinMaxScaling --> RelationshipScore

    VerticalScore --> ComponentScores
    SizeScore --> ComponentScores
    AdoptionScore --> ComponentScores
    RelationshipScore --> ComponentScores

    ComponentScores --> WeightApplication --> PercentileConversion --> NormalDistribution --> ScaleTo100 --> FinalICP

    FinalICP --> PercentileBrackets --> LetterGrade

    FinalICP --> FinalScore
    LetterGrade --> Grade
    ComponentScores --> ComponentBreakdown
    WeightApplication --> RawScore

    %% Style definitions
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef vertical fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef size fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef adoption fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef relationship fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef calculation fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef output fill:#f5f5f5,stroke:#616161,stroke-width:2px

    %% Apply styles
    class CustomerRecord,OptimizedWeights input
    class IndustryLookup,IndustryWeights,VerticalScore vertical
    class RevenueData,RevenueBrackets,SizeScore size
    class PrinterCounts,HardwareRevenue,WeightedPrinterCalc,ZeroExclusion,PrinterPercentile,RevenuePercentile,BusinessRulesApplication,ZeroEverythingRule,RevenueOnlyRule,PrinterRule,HeavyFleetBonus,AdoptionScore adoption
    class SoftwareRevenue,TotalSoftware,LogTransformation,MinMaxScaling,RelationshipScore relationship
    class ComponentScores,WeightApplication,PercentileConversion,NormalDistribution,ScaleTo100,FinalICP,TargetDistribution,PercentileBrackets,LetterGrade calculation
    class FinalScore,Grade,ComponentBreakdown,RawScore output
```

## ICP Scoring Methodology Details

The ICP (Ideal Customer Profile) scoring system uses a sophisticated 4-component approach to evaluate customers based on their strategic value to GoEngineer Digital Manufacturing.

### Component Scores (0-1 Scale)

#### 1. Vertical Score (Industry Performance)
- **Purpose**: Measures the historical revenue performance of the customer's industry
- **Calculation**: Direct lookup from data-driven industry weights
- **Fallback**: 0.3 for unknown or unmapped industries
- **Source**: `industry_weights.json` generated by `industry_scoring.py`

#### 2. Size Score (Revenue-Based)
- **Purpose**: Evaluates customer size based on annual revenue
- **Tiers**:
  - $250M-$1B: 1.0 (Enterprise)
  - $50M-$250M: 0.6 (Large)
  - $10M-$50M: 0.4 (Medium)
  - $0-$10M: 0.4 (Small)
  - Missing: 0.5 (neutral)
- **Source**: Enriched revenue data or estimates

#### 3. Adoption Score (Hardware Engagement)
- **Purpose**: Measures hardware investment and technology adoption
- **Complex Logic**:
  - **Weighted Printer Score**: Big Box (2x) + Small Box (1x)
  - **Hardware Revenue**: Total hardware + consumable revenue
  - **Percentile Scaling**: Both metrics converted to percentiles
  - **Business Rules**:
    - No printers + no revenue = 0.0
    - Revenue but no printers = 0.0-0.5 (square root scaling)
    - With printers = 0.0-1.0 (60% printer + 40% revenue)
    - 10+ weighted printers = +0.05 bonus

#### 4. Relationship Score (Software Engagement)
- **Purpose**: Measures software relationship strength
- **Calculation**:
  - Sum all software revenue (License, SaaS, Maintenance)
  - Apply log1p transformation to handle wide value ranges
  - Min-max scaling to 0-1 range
- **Source**: Software revenue data from customer records

### Final Score Calculation

#### Raw Score
```
Raw Score = (Vertical  W_v) + (Size  W_s) + (Adoption  W_a) + (Relationship  W_r)
```
Where weights (W_v, W_s, W_a, W_r) sum to 1.0

#### Normalization Process
1. **Percentile Conversion**: Convert raw scores to percentile ranks
2. **Normal Distribution**: Apply inverse CDF to create bell curve
3. **Scaling**: Transform to 0-100 scale (mean=50, std dev=15)

#### Grade Assignment
- **A Grade**: Top 10% (90-100 percentile)
- **B Grade**: Next 20% (70-90 percentile)
- **C Grade**: Middle 40% (30-70 percentile)
- **D Grade**: Next 20% (10-30 percentile)
- **F Grade**: Bottom 10% (0-10 percentile)

### Key Features:
- **Data-Driven**: Industry weights based on actual historical performance
- **ML-Optimized**: Component weights optimized using Optuna
- **Business Rules**: Incorporates domain knowledge and strategic priorities
- **Robust Scaling**: Handles missing data and wide value ranges
- **Predictive**: Optimized for correlation with future revenue potential




