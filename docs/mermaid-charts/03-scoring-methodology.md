# ICP Scoring Methodology

```mermaid
graph TD
    subgraph "Input Data"
        CustomerRecord[Customer Record<br/>Industry, adoption_assets, adoption_profit<br/>relationship_profit (software goals)]
        OptimizedWeights[Optimized Weights (artifacts/weights)<br/>Vertical, Adoption, Relationship]
    end

    subgraph "Component Score Calculation"
        subgraph "1. Vertical Score"
            IndustryLookup[Industry Classification]
            IndustryWeights[Load Division Industry Weights<br/>artifacts/weights/{division}_industry_weights.json]
            VerticalScore[Map to 0.0–1.0<br/>Neutral for unknown per DivisionConfig]
        end

        subgraph "2. Adoption Score"
            PreferredSignals[Preferred signals (division)<br/>adoption_assets, adoption_profit]
            Percentiles[Percentile scaling P(assets), R(profit)<br/>exclude all-zero cohort]
            BusinessRules[If assets>0: 0.6*P + 0.4*R<br/>Else if profit>0: 0.5*sqrt(R)<br/>Else: 0.0]
            AdoptionScore[Adoption score (0–1)]
        end

        subgraph "3. Relationship Score"
            RelSignals[relationship_profit (preferred)<br/>fallback: license + SaaS + maintenance]
            LogMinMax[log1p + min-max (0–1)]
            RelationshipScore[Relationship score (0–1)]
        end
    end

    subgraph "Final ICP Score Calculation"
        ComponentScores[Vertical + Adoption + Relationship]
        WeightApplication[Weighted blend (sum=1.0)]
        PercentileConversion[Within-cohort rank + percentile]
        NormalDistribution[Inverse normal CDF]
        ScaleTo100[Scale to 0–100 (50±15), clip]
        FinalICP[Division ICP score (0–100)]
    end

    subgraph "Grade Assignment"
        TargetDistribution[Target A–F distribution<br/>A 10%, B 20%, C 40%, D 20%, F 10%]
        PercentileBrackets[Map percentiles to grades]
        LetterGrade[Assign A–F]
    end

    subgraph "Output Components"
        FinalScore[ICP_score_hardware / ICP_score_cre<br/>(0–100 per division)]
        Grade[ICP_grade_hardware / ICP_grade_cre<br/>(A–F per division)]
        ComponentBreakdown[vertical_score,<br/>adoption_score, relationship_score]
        RawScore[ICP_score_raw (pre-normalization)]
    end

    CustomerRecord --> IndustryLookup --> IndustryWeights --> VerticalScore
    CustomerRecord --> PreferredSignals --> Percentiles --> BusinessRules --> AdoptionScore
    CustomerRecord --> RelSignals --> LogMinMax --> RelationshipScore

    VerticalScore --> ComponentScores
    AdoptionScore --> ComponentScores
    RelationshipScore --> ComponentScores

    ComponentScores --> WeightApplication --> PercentileConversion --> NormalDistribution --> ScaleTo100 --> FinalICP
    FinalICP --> PercentileBrackets --> LetterGrade

    FinalICP --> FinalScore
    LetterGrade --> Grade
    ComponentScores --> ComponentBreakdown
    WeightApplication --> RawScore

    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef vertical fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef adoption fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef relationship fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef calculation fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef output fill:#f5f5f5,stroke:#616161,stroke-width:2px

    class CustomerRecord,OptimizedWeights input
    class IndustryLookup,IndustryWeights,VerticalScore vertical
    class PreferredSignals,Percentiles,BusinessRules,AdoptionScore adoption
    class RelSignals,LogMinMax,RelationshipScore relationship
    class ComponentScores,WeightApplication,PercentileConversion,NormalDistribution,ScaleTo100,FinalICP,TargetDistribution,PercentileBrackets,LetterGrade calculation
    class FinalScore,Grade,ComponentBreakdown,RawScore output
```

## ICP Scoring Methodology Details

### Component Scores (0–1)

#### 1. Vertical Score (Industry Performance)
- Purpose: Measure historical performance of the customer’s industry.
- Calculation: Direct lookup from division-specific weights.
- Fallback: Neutral per division (e.g., hardware 0.30, CRE 0.35).
- Source: `src/icp/industry.py` + `artifacts/weights/{division}_industry_weights.json`.

#### 2. Adoption Score (Division Adoption)
- Purpose: Division adoption depth and value.
- Inputs: `adoption_assets`, `adoption_profit` (division-aware).
- Percentiles: P = pct(assets), R = pct(profit) over non-zero cohort.
- Rules: if assets > 0 → 0.6*P + 0.4*R; else if profit > 0 → 0.5*sqrt(R); else 0.0.

#### 3. Relationship Score (Software Engagement)
- Purpose: Software relationship strength.
- Preferred: `relationship_profit` (division goals).
- Fallback: license + SaaS + maintenance totals.
- Transform: `log1p` then min–max to 0–1.

### Final Score and Grades

- Raw blend: Weighted sum of vertical/adoption/relationship (sum=1.0), then ×100 to produce `ICP_score_raw`.
- Normalization: rank + inverse normal CDF + scale to 0–100 (50±15), clipped.
- Division scores: `ICP_score_hardware` / `ICP_score_cre` and corresponding grades `ICP_grade_hardware` / `ICP_grade_cre` are computed separately per division.
- Grades: A–F by target distribution (A 10%, B 20%, C 40%, D 20%, F 10%) within each division.

