import optuna
import numpy as np
from scipy.stats import spearmanr, kstest, norm
import pandas as pd

# Define the target distribution for A-F grades, which the optimizer will try to match.
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
    """
    Assigns A-F grades based on percentile cutoffs defined in the target distribution.

    Args:
        scores (pd.Series): A series of calculated ICP scores.

    Returns:
        np.ndarray: An array of corresponding letter grades ('A' through 'F').
    """
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
    Objective function for the Optuna optimization study.

    This function defines the value that Optuna will attempt to minimize. It balances
    two competing goals:
    1.  **Maximize Revenue Correlation**: Find weights that make the ICP score highly
        correlated with historical customer revenue (Spearman correlation).
    2.  **Match Target Distribution**: Shape the final ICP scores so that the
        resulting A-F grade distribution matches a predefined target (e.g., 10% 'A's),
        measured by KL Divergence.

    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters (the weights).
        X (pd.DataFrame): DataFrame of the input features (component scores).
        y (pd.Series): Series of the target variable (historical revenue).
        lambda_param (float): A hyperparameter (0.0 to 1.0) that controls the trade-off
                              between the two optimization goals.
                              - 0.0 = Purely maximize revenue correlation.
                              - 1.0 = Purely match the target grade distribution.
        weight_names (list): The names of the weight columns in the correct order.
        include_size (bool): If False, the 'size_score' weight is locked to 0 and the other weights are normalized to sum to 1.0.

    Returns:
        float: The combined objective value to be minimized.
    """
    # Define the search space for each weight using the trial object.
    # These ranges are based on business rules and prior analysis.
    w_vertical = trial.suggest_float('vertical_score', 0.10, 0.40)

    if include_size:
        w_size = trial.suggest_float('size_score', 0.10, 0.40)
    else:
        w_size = 0.0  # Lock the size weight to 0 if specified.

    w_adoption = trial.suggest_float('adoption_score', 0.10, 0.40)
    w_relationship = trial.suggest_float('relationship_score', 0.15, 0.40)

    total = w_vertical + w_size + w_adoption + w_relationship

    # --- Feasibility Checks (Constraints) ---
    # Prune (discard) trials that don't meet the basic constraint that weights
    # form a valid set. When size is included we require the weights to sum
    # to approximately 1.0; otherwise we simply normalize the remaining
    # weights, pruning only if they are all zero.
    if include_size:
        # Allow a small tolerance for floating point inaccuracies.
        if not 0.95 <= total <= 1.05:
            raise optuna.exceptions.TrialPruned()
        # Normalize the weights to ensure they sum to exactly 1.0.
        w_vertical /= total
        w_size /= total
        w_adoption /= total
        w_relationship /= total
    else:
        # Normalize the non-size weights so they sum to 1.0. If their
        # combined value is zero, prune the trial because normalization
        # would be impossible.
        total_no_size = w_vertical + w_adoption + w_relationship
        if abs(total_no_size) < 1e-12:
            raise optuna.exceptions.TrialPruned()
        w_vertical /= total_no_size
        w_adoption /= total_no_size
        w_relationship /= total_no_size

    weights_dict = {
        'vertical_score': w_vertical,
        'size_score': w_size,
        'adoption_score': w_adoption,
        'relationship_score': w_relationship,
    }

    # Ensure the weights are in the same order as the columns in the feature DataFrame.
    weights = np.array([weights_dict[name] for name in X.columns])

    # --- Calculate Objective Components ---

    # 1. Calculate the weighted ICP scores for this trial's weights.
    icp_scores = X.dot(weights)
    
    # 2. Calculate Spearman correlation between the ICP scores and the target revenue.
    #    We want to MAXIMIZE this value. The function returns (correlation, p-value).
    spearman_corr, _ = spearmanr(icp_scores, y)
    
    # 3. Calculate the difference between the actual grade distribution produced by these
    #    scores and the predefined target distribution. We use Kullback-Leibler (KL)
    #    Divergence, which measures how one probability distribution is different from
    #    a second, reference distribution. We want to MINIMIZE this value.
    actual_grades = calculate_grades(icp_scores)
    actual_distribution = pd.Series(actual_grades).value_counts(normalize=True).reindex(
        ['F', 'D', 'C', 'B', 'A'], fill_value=0
    )
    
    # Target distribution as an array, in the same order as the actual distribution.
    target_distribution = np.array([
        TARGET_GRADE_DISTRIBUTION['F'],
        TARGET_GRADE_DISTRIBUTION['D'], 
        TARGET_GRADE_DISTRIBUTION['C'],
        TARGET_GRADE_DISTRIBUTION['B'],
        TARGET_GRADE_DISTRIBUTION['A']
    ])
    
    # Add a small epsilon to prevent division by zero in the KL divergence calculation.
    epsilon = 1e-10
    kl_divergence = np.sum(
        actual_distribution * np.log((actual_distribution + epsilon) / (target_distribution + epsilon))
    )

    # --- Combine Objectives ---
    # Create a single value for Optuna to minimize.
    # - We use -spearman_corr because Optuna minimizes, and we want to maximize correlation.
    # - The lambda_param controls the balance between the two goals.
    combined_objective = (lambda_param * kl_divergence) + ((1 - lambda_param) * -spearman_corr)
    
    return combined_objective 