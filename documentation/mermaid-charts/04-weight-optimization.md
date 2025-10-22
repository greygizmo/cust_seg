# Weight Optimization Process

```mermaid
graph TD
    %% Define the optimization workflow
    subgraph "Optimization Setup"
        ScoredDataInput[icp_scored_accounts.csv<br/>Contains scored customer data with<br/>- Component scores (vertical, size, adoption, relationship)<br/>- Historical revenue data (target variable)<br/>- Customer segmentation data]
        OptimizationConfig[Optimization Configuration<br/>- Number of trials (default: 5000)<br/>- Lambda parameter (0.0-1.0 trade-off)<br/>- Include size flag (true/false)<br/>- Search space constraints]
    end

    subgraph "Data Preparation"
        EngagedCustomerFilter[Filter Engaged Customers<br/>Only customers with >$0 hardware/consumable revenue<br/>Creates training dataset for optimization]
        FeatureTargetSplit[Feature-Target Split<br/>X: Component scores (features)<br/>y: Historical hardware/consumable revenue (target)<br/>Minimum 50 engaged customers required]
        ValidationCheck[Data Validation<br/>- Check all required score columns exist<br/>- Validate no missing values in features<br/>- Ensure sufficient sample size]
    end

    subgraph "Optuna Optimization Engine"
        StudyCreation[Create Optuna Study<br/>- Direction: minimize objective<br/>- Pruning enabled for efficiency<br/>- Sampler: default (TPE)<br/>- Pruner: default hyperband]
        TrialLoop[Trial Generation Loop<br/>For each trial (up to n_trials):<br/>- Suggest weight values<br/>- Apply constraints<br/>- Calculate objective<br/>- Report result to Optuna]
    end

    subgraph "Trial Processing"
        subgraph "Weight Suggestion"
            VerticalWeight[Suggest Vertical Weight<br/>Range: 0.15 - 0.50<br/>Industry importance]
            SizeWeight[Suggest Size Weight<br/>Range: 0.15 - 0.50 (if enabled)<br/>Or set to 0.0 (if disabled)<br/>Revenue size importance]
            AdoptionWeight[Suggest Adoption Weight<br/>Range: 0.15 - 0.50<br/>Hardware engagement importance]
            RelationshipWeight[Suggest Relationship Weight<br/>Range: 0.15 - 0.50<br/>Software relationship importance]
        end

        subgraph "Constraint Application"
            SumConstraint[Sum Constraint Check<br/>Sum of weights must be 0.95-1.05<br/>Handles floating point precision]
            MaxWeightConstraint[Maximum Weight Constraint<br/>No single weight > 0.50<br/>Prevents component dominance]
            Normalization[Weight Normalization<br/>Normalize to exact sum of 1.0<br/>Ensures mathematical consistency]
        end

        subgraph "Objective Calculation"
            WeightedScoreCalc[Calculate Weighted Scores<br/>ICP_scores = X  weights<br/>Matrix multiplication of features and weights]
            SpearmanCorrelation[Calculate Spearman Correlation<br/>Correlation between ICP scores and revenue<br/>Measures predictive power<br/>Range: -1.0 to 1.0]
            GradeDistribution[Calculate Grade Distribution<br/>Apply grade assignment logic<br/>Count customers per grade (A-F)<br/>Compare to target distribution]
            KLDiveregence[Calculate KL Divergence<br/>Measure difference between<br/>actual vs target grade distribution<br/>Measures distribution match<br/>Range: 0.0 to infinity]
            ObjectiveValue[Calculate Combined Objective<br/>Objective =   KL + (1-)  (-correlation)<br/>Lambda controls trade-off<br/>Lower values are better]
        end
    end

    subgraph "Optimization Results"
        BestTrialSelection[Select Best Trial<br/>Trial with lowest objective value<br/>Best balance of correlation and distribution]
        WeightExtraction[Extract Optimized Weights<br/>Final normalized weights<br/>Constrained to business rules<br/>Ready for production use]
        MetadataCollection[Collect Optimization Metadata<br/>- Number of trials run<br/>- Best objective value achieved<br/>- Lambda parameter used<br/>- Optimization timestamp<br/>- Include size setting]
    end

    subgraph "Output Storage"
        WeightsJSON[Save optimized_weights.json<br/>Contains optimized weights and metadata<br/>Used by dashboard and scoring engine<br/>Includes optimization parameters]
        ResultsDisplay[Display Optimization Results<br/>- Final optimized weights<br/>- Optimization statistics<br/>- Trade-off achieved<br/>- Performance metrics]
    end

    subgraph "Trade-off Analysis"
        subgraph "Correlation vs Distribution Balance"
            PureCorrelation[ = 0.0<br/>Pure revenue correlation maximization<br/>May ignore grade distribution target<br/>Risk: unrealistic grade spread]
            PureDistribution[ = 1.0<br/>Pure distribution matching<br/>May sacrifice predictive power<br/>Risk: poor revenue correlation]
            BalancedApproach[ = 0.25 (default)<br/>75% weight on correlation<br/>25% weight on distribution<br/>Balanced predictive and business requirements]
        end
    end

    %% Define data flow connections
    ScoredDataInput --> EngagedCustomerFilter
    OptimizationConfig --> StudyCreation

    EngagedCustomerFilter --> FeatureTargetSplit --> ValidationCheck --> StudyCreation

    StudyCreation --> TrialLoop
    TrialLoop --> VerticalWeight
    TrialLoop --> SizeWeight
    TrialLoop --> AdoptionWeight
    TrialLoop --> RelationshipWeight

    VerticalWeight --> SumConstraint
    SizeWeight --> SumConstraint
    AdoptionWeight --> SumConstraint
    RelationshipWeight --> SumConstraint

    SumConstraint --> MaxWeightConstraint --> Normalization --> WeightedScoreCalc

    WeightedScoreCalc --> SpearmanCorrelation
    WeightedScoreCalc --> GradeDistribution

    SpearmanCorrelation --> ObjectiveValue
    GradeDistribution --> KLDiveregence --> ObjectiveValue

    ObjectiveValue --> TrialLoop
    TrialLoop --> BestTrialSelection --> WeightExtraction --> MetadataCollection --> WeightsJSON

    WeightsJSON --> ResultsDisplay

    %% Style definitions
    classDef setup fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef preparation fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef optimization fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef trial fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef constraint fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef objective fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef results fill:#f9fbe7,stroke:#689f38,stroke-width:2px
    classDef output fill:#ffecb3,stroke:#f57c00,stroke-width:2px
    classDef tradeoff fill:#f5f5f5,stroke:#616161,stroke-width:2px

    %% Apply styles
    class ScoredDataInput,OptimizationConfig setup
    class EngagedCustomerFilter,FeatureTargetSplit,ValidationCheck preparation
    class StudyCreation,TrialLoop optimization
    class VerticalWeight,SizeWeight,AdoptionWeight,RelationshipWeight,SpearmanCorrelation,GradeDistribution,ObjectiveValue trial
    class SumConstraint,MaxWeightConstraint,Normalization constraint
    class WeightedScoreCalc,KLDiveregence objective
    class BestTrialSelection,WeightExtraction,MetadataCollection results
    class WeightsJSON,ResultsDisplay output
    class PureCorrelation,PureDistribution,BalancedApproach tradeoff
```

## Weight Optimization Process Details

The weight optimization process uses **Optuna** (a hyperparameter optimization framework) to find the optimal weights for the four ICP scoring components. This ensures the scoring system is both predictive of future revenue and aligned with business requirements.

### Multi-Objective Optimization

The optimization balances two competing goals:

1. **Maximize Revenue Correlation**: Find weights that make ICP scores highly predictive of historical customer revenue
2. **Match Target Distribution**: Ensure the final grade distribution matches business requirements (A=10%, B=20%, etc.)

### Key Components:

#### Lambda Parameter ()
- **Controls the trade-off between the two objectives**
- ** = 0.0**: Pure correlation optimization
- ** = 1.0**: Pure distribution matching
- ** = 0.25** (default): 75% correlation, 25% distribution

#### Search Space Constraints:
- **Individual weights**: 0.15 - 0.50 range
- **Sum constraint**: Weights must sum to 1.0
- **Maximum weight**: No single component > 50%
- **Size weight option**: Can be disabled (set to 0)

#### Optimization Algorithm:
- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Pruner**: Hyperband (prunes unpromising trials)
- **Direction**: Minimize combined objective function

### Objective Function:
```
Objective =   KL_Divergence + (1-)  (-Spearman_Correlation)
```

Where:
- **KL_Divergence**: Measures difference between actual and target grade distributions
- **Spearman_Correlation**: Measures correlation between ICP scores and revenue
- ****: Trade-off parameter (0.0-1.0)

### Output:
- **optimized_weights.json**: Contains the optimized weights and metadata
- **Performance metrics**: Final correlation achieved, distribution match quality
- **Optimization statistics**: Number of trials, best objective value, parameters used

### Business Rules Implementation:
- **Engaged customers only**: Training on customers with actual hardware/consumable revenue
- **Minimum sample size**: Requires at least 50 engaged customers
- **Pruning constraints**: Eliminates invalid weight combinations early
- **Production ready**: Optimized weights directly usable by scoring engine and dashboard

This optimization process ensures the ICP scoring system is both statistically robust and aligned with business objectives, creating a data-driven customer segmentation that maximizes predictive accuracy while maintaining practical grade distributions.




