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

def objective(trial, X, y, lambda_param, weight_names):
    """
    Objective function for Optuna to minimize.
    It balances two goals:
    1. Maximize Spearman correlation between ICP score and revenue.
    2. Minimize the difference between the resulting score distribution and a target bell curve.
    """
    # Define constrained search space for weights based on business rules
    # Minimum weight of 0.10 (10%) for all criteria - no criterion can be ignored
    w_vertical = trial.suggest_float('vertical_score', 0.10, 0.40)
    w_size = trial.suggest_float('size_score', 0.10, 0.40)
    w_adoption = trial.suggest_float('adoption_score', 0.10, 0.40)
    
    # The last weight is what's left over to ensure they sum to 1
    w_relationship = 1.0 - (w_vertical + w_size + w_adoption)

    # Prune trials that don't satisfy the constraints for the calculated weight
    # Ensure relationship weight also meets the minimum 0.10 requirement
    if not (0.10 <= w_relationship <= 0.40):
        raise optuna.exceptions.TrialPruned()

    weights_dict = {
        'vertical_score': w_vertical,
        'size_score': w_size,
        'adoption_score': w_adoption,
        'relationship_score': w_relationship
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