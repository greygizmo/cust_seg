import optuna
import numpy as np
from scipy.stats import spearmanr, kstest, norm
import pandas as pd

# Define the target distribution for A-F grades
# A=10%, B=20%, C=40%, D=20%, F=10%
TARGET_GRADE_DISTRIBUTION = {
    'A': 0.10,
    'B': 0.20,
    'C': 0.40,
    'D': 0.20,
    'F': 0.10
}
TARGET_CUMULATIVE_DISTRIBUTION = np.cumsum([
    TARGET_GRADE_DISTRIBUTION['F'],
    TARGET_GRADE_DISTRIBUTION['D'],
    TARGET_GRADE_DISTRIBUTION['C'],
    TARGET_GRADE_DISTRIBUTION['B'],
    TARGET_GRADE_DISTRIBUTION['A']
])

def calculate_grades(scores):
    """Assigns A-F grades based on percentile cutoffs."""
    ranks = scores.rank(pct=True)
    grades = np.select(
        [
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[0],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[1],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[2],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[3],
            ranks > TARGET_CUMULATIVE_DISTRIBUTION[3]
        ],
        ['F', 'D', 'C', 'B', 'A'],
        default='C'
    )
    return grades

def objective(trial, X, y, lambda_param, weight_names, include_size=True):
    """
    Objective function for Optuna to minimize.
    It balances two goals:
    1. Maximize Spearman correlation between ICP score and revenue.
    2. Minimize the difference between the resulting score distribution and a target bell curve.
    """
    # Define constrained search space for weights based on business rules.
    # If `include_size` is False, we hard-set size weight to 0.0 and re-optimize the
    # remaining three criteria so that they still sum to 1.0.  Otherwise, we leave
    # size as a tunable parameter (legacy behaviour).

    w_vertical = trial.suggest_float('vertical_score', 0.10, 0.40)

    if include_size:
        w_size = trial.suggest_float('size_score', 0.10, 0.40)
    else:
        w_size = 0.0  # Size locked out

    w_adoption = trial.suggest_float('adoption_score', 0.10, 0.40)

    # Sample relationship directly (new rule)
    w_relationship = trial.suggest_float('relationship_score', 0.15, 0.40)

    total = w_vertical + w_size + w_adoption + w_relationship

    # ----- Feasibility checks -----
    # All weights must be positive and total exactly 1 (within tolerance)
    if include_size:
        # Allow a small tolerance; residual will be absorbed into size later
        if not 0.95 <= total <= 1.05:
            raise optuna.exceptions.TrialPruned()
        # Adjust proportionally so they sum to 1 exactly
        w_vertical /= total
        w_size /= total
        w_adoption /= total
        w_relationship /= total
    else:
        # With size locked to 0, the other three must sum to 1
        if abs(total - 1.0) > 1e-3:
            raise optuna.exceptions.TrialPruned()

    weights_dict = {
        'vertical_score': w_vertical,
        'size_score': w_size,
        'adoption_score': w_adoption,
        'relationship_score': w_relationship,
    }

    # Ensure the weights are in the same order as the columns in the dataframe
    weights = np.array([weights_dict[name] for name in X.columns])

    # 1. Calculate weighted ICP scores
    icp_scores = X.dot(weights)
    
    # 2. Calculate Spearman correlation with revenue (we want to maximize this)
    # The function returns a tuple (correlation, p-value), we only need the correlation
    spearman_corr, _ = spearmanr(icp_scores, y)
    
    # 3. Calculate KL Divergence from the target grade distribution (we want to minimize this)
    actual_grades = calculate_grades(icp_scores)
    actual_distribution = pd.Series(actual_grades).value_counts(normalize=True).reindex(
        ['F', 'D', 'C', 'B', 'A'], fill_value=0
    )
    
    # Target distribution as array in same order as actual_distribution
    target_distribution = np.array([
        TARGET_GRADE_DISTRIBUTION['F'],
        TARGET_GRADE_DISTRIBUTION['D'], 
        TARGET_GRADE_DISTRIBUTION['C'],
        TARGET_GRADE_DISTRIBUTION['B'],
        TARGET_GRADE_DISTRIBUTION['A']
    ])
    
    # Add a small epsilon to prevent division by zero in KL divergence
    epsilon = 1e-10
    kl_divergence = np.sum(
        actual_distribution * np.log((actual_distribution + epsilon) / (target_distribution + epsilon))
    )

    # Combine the two objectives into a single value to minimize
    # We use -spearman_corr because Optuna minimizes, and we want to maximize correlation
    combined_objective = (lambda_param * kl_divergence) + ((1 - lambda_param) * -spearman_corr)
    
    return combined_objective 