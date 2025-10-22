import pandas as pd
import optuna
import json
from functools import partial
from pathlib import Path
from icp.optimization import objective

def run_optimization(n_trials=5000, lambda_param=0.50, include_size=True):
    """
    Main function to run the weight optimization process using Optuna.

    This script loads the scored customer data, prepares it for optimization,
    and then runs an Optuna study to find the best set of weights that
    balance revenue prediction and score distribution, as defined in the
    `objective` function.

    Args:
        n_trials (int): The number of optimization trials to run.
        lambda_param (float): The hyperparameter controlling the trade-off between
                              revenue correlation and distribution matching.
        include_size (bool): Whether to include the 'size_score' in the optimization.
                             If False, its weight is locked to 0.
    """
    ROOT = Path(__file__).resolve().parents[3]
    print("Loading scored accounts data...")
    try:
        df = pd.read_csv(ROOT / 'data' / 'processed' / 'icp_scored_accounts.csv')
    except FileNotFoundError:
        print("Error: `data/processed/icp_scored_accounts.csv` not found.")
        print("Please run `python -m icp.cli.score_accounts` first to generate the necessary file.")
        return

    print(f"Loaded {len(df)} total accounts.")

    # We only train on "engaged" customers who have historical revenue data.
    # This ensures the weights are optimized to predict real-world success.
    # The target variable is the total hardware and consumable revenue.
    engaged_customers = df[df['Total Hardware + Consumable Revenue'] > 0].copy()
    print(f"Found {len(engaged_customers)} engaged customers with revenue > $0 for training.")

    if len(engaged_customers) < 50:
        print("Error: Too few engaged customers to run optimization. Need at least 50.")
        return

    # Define the four criteria (features) to be weighted.
    weight_names = [
        'vertical_score', 
        'size_score', 
        'adoption_score',
        'relationship_score'
    ]

    # Check if all required score columns exist in the DataFrame.
    for col in weight_names:
        if col not in df.columns:
            print(f"Error: Score column '{col}' not found in the data. Exiting.")
            return

    # Prepare the training data (X) and target variable (y).
    X = engaged_customers[weight_names]
    y = engaged_customers['Total Hardware + Consumable Revenue']

    print(f"Optimizing weights for: {weight_names}")
    print(f"Using lambda = {lambda_param} (Balance between revenue prediction and distribution shape)")

    # Use functools.partial to create a version of the objective function
    # with the data arguments (X, y, etc.) already "baked in".
    obj_func = partial(
        objective,
        X=X,
        y=y,
        lambda_param=lambda_param,
        weight_names=weight_names,
        include_size=include_size,
    )
    
    # Create and run the Optuna study.
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_func, n_trials=n_trials, show_progress_bar=True)

    # --- Process and save results ---
    best_weights = study.best_params
    
    print("\n--- Optimization Complete ---")
    print(f"Best objective value: {study.best_value:.4f}")
    
    print("Best weights (normalized):")
    
    # Ensure the size score weight is explicitly set if it was excluded.
    if include_size is False:
        best_weights['size_score'] = 0.0

    # Re-normalize the final weights to ensure they sum to exactly 1.0.
    total = sum(best_weights.values())
    normalized_weights = {k: v / total for k, v in best_weights.items()}

    for name, weight in normalized_weights.items():
        print(f"  - {name}: {weight:.4f}")

    # Save the results to a JSON file for the dashboard to use.
    output_data = {
        'weights': normalized_weights,
        'lambda_param': lambda_param,
        'n_trials': n_trials,
        'best_objective_value': study.best_value,
        'include_size': include_size,
    }

    out_path = ROOT / 'artifacts' / 'weights' / 'optimized_weights.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nSaved optimized weights to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Optimize ICP weights with Optuna")
    parser.add_argument("--n-trials", type=int, default=5000, help="Number of trials")
    parser.add_argument("--lambda", dest="lambda_param", type=float, default=0.25, help="Trade-off between corr and distribution")
    parser.add_argument("--include-size", action="store_true", help="Include size_score in optimization (default False)")
    args = parser.parse_args()

    run_optimization(n_trials=args.n_trials, lambda_param=args.lambda_param, include_size=args.include_size)
