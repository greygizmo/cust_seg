# Weight Optimization Process (Division-Aware)

```mermaid
graph TD
    subgraph "Inputs"
        ScoredCSV[data/processed/icp_scored_accounts.csv<br/>Component columns: vertical/size/adoption/relationship]
        AzureQuarterly[Azure SQL: Quarterly Profit by Goal/Rollup]
        Config[CLI Args: division, n_trials, lambda, horizons]
    end

    subgraph "Label Building (future GP)"
        AsOf[as_of_date from scored CSV]
        HorizonKeys[Compute future quarter keys (e.g., +1Q, +2Q)]
        DivisionFilter[Select division goals<br/>Hardware: Printers, Accessories, Scanners, Geomagic<br/>CRE: CAD, Specialty + CRE Training subset]
        SumLabels[Sum profit at future quarter by account]
        y_future[Labels: future GP by horizon]
    end

    subgraph "Features"
        Xcols[Extract features X = {vertical,size,adoption,relationship}<br/>Fallbacks: Hardware_score/Software_score used if needed]
        Validate[Validate columns and non-null rows per horizon]
    end

    subgraph "Optuna Study"
        Suggest[Suggest weights per component (size fixed 0 in v1.5)<br/>Bounds: vertical 0.15–0.45, adoption 0.20–0.55, relationship 0.20–0.55]
        Normalize[Normalize weights to sum=1.0]
        Score[icp_scores = X · w]
        Spearman[Spearman corr(icp_scores, y_future)]
        Grades[Assign A–F by target distribution]
        KL[KL divergence grade vs target]
        Stability[Std of within-group Spearman by group (e.g., Industry)]
        LiftA[Lift at top-grade (lift@A)]
        Objective[Objective = λ·KL + (1-λ)·(-Spearman)
                 + w_stab·Stability - w_lift·(lift@A-1)]
        Best[Minimize objective across trials and horizons]
    end

    subgraph "Outputs"
        WeightsJSON[artifacts/weights/optimized_weights.json<br/>Per-division weights + metadata]
        Metrics[Report: Spearman, KL, Stability, Lift, AUC]
    end

    %% Flow
    ScoredCSV --> AsOf --> HorizonKeys --> DivisionFilter --> SumLabels --> y_future
    AzureQuarterly --> DivisionFilter
    ScoredCSV --> Xcols --> Validate
    Config --> HorizonKeys
    Validate --> Suggest --> Normalize --> Score --> Spearman --> Objective
    Score --> Grades --> KL --> Objective
    Score --> Stability --> Objective
    Score --> LiftA --> Objective
    Objective --> Best --> WeightsJSON --> Metrics

    %% Styles
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef stage fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef optuna fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef output fill:#f5f5f5,stroke:#616161,stroke-width:2px

    class ScoredCSV,AzureQuarterly,Config input
    class AsOf,HorizonKeys,DivisionFilter,SumLabels,y_future,Xcols,Validate stage
    class Suggest,Normalize,Score,Spearman,Grades,KL,Stability,LiftA,Objective,Best optuna
    class WeightsJSON,Metrics output
```

## Details

- Division-aware labels: future gross profit by division goals using Azure SQL, at user-selected horizons (e.g., 1Q, 2Q).
- Features: component columns from the scored CSV; fall back to alias columns when needed.
- Bounds: vertical ∈ [0.15,0.45], adoption ∈ [0.20,0.55], relationship ∈ [0.20,0.55], size=0.
- Objective: λ·KL + (1-λ)·(-Spearman) + w_stab·Stability - w_lift·(lift@A-1).
- Outputs: `optimized_weights.json` stores per-division weights and consolidated metadata/history.

