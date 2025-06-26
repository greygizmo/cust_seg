import pandas as pd
import optuna
import json
from functools import partial
from optimize_weights import objective

def run_optimization(n_trials=5000, lambda_param=0.25):
    """
    Main function to run the weight optimization process.
    """
    print("Loading scored accounts data...")
    try:
        df = pd.read_csv('icp_scored_accounts.csv')
    except FileNotFoundError:
        print("Error: `icp_scored_accounts.csv` not found.")
        print("Please run `goe_icp_scoring.py` first to generate the necessary file.")
        return

    print(f"Loaded {len(df)} total accounts.")

    # We only train on "engaged" customers who have historical revenue data
    # This ensures the weights are optimized to predict real-world success
    engaged_customers = df[df['Total Hardware + Consumable Revenue'] > 0].copy()
    print(f"Found {len(engaged_customers)} engaged customers with revenue > $0 for training.")

    if len(engaged_customers) < 50:
        print("Error: Too few engaged customers to run optimization. Need at least 50.")
        return

    # Define the four criteria to be weighted, as per the constrained optimization strategy
    weight_names = [
        'vertical_score', 
        'size_score', 
        'adoption_score',
        'relationship_score'
    ]

    # Check if all score columns exist
    for col in weight_names:
        if col not in df.columns:
            print(f"Error: Score column '{col}' not found in the data. Exiting.")
            return

    X = engaged_customers[weight_names]
    y = engaged_customers['Total Hardware + Consumable Revenue']

    print(f"Optimizing weights for: {weight_names}")
    print(f"Using lambda = {lambda_param} (Balance between revenue prediction and distribution shape)")

    # Define the objective function with the data baked in
    # Use functools.partial to pass the dataframes and lambda to the objective function
    obj_func = partial(objective, X=X, y=y, lambda_param=lambda_param, weight_names=weight_names)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_func, n_trials=n_trials, show_progress_bar=True)

    # --- Process and save results ---
    best_weights = study.best_params
    
    print("\n--- Optimization Complete ---")
    print(f"Best objective value: {study.best_value:.4f}")
    
    print("Best weights (already normalized):")
    # The weights are already normalized to sum to 1 by the objective function
    # but we will re-normalize just in case of float precision issues
    total_weight = sum(best_weights.values())
    normalized_weights = {name: w / total_weight for name, w in best_weights.items()}

    for name, weight in normalized_weights.items():
        print(f"  - {name}: {weight:.4f}")

    # Save the results to a file for the dashboard to use
    output_data = {
        'weights': normalized_weights,
        'lambda_param': lambda_param,
        'n_trials': n_trials,
        'best_objective_value': study.best_value
    }

    with open('optimized_weights.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print("\n✅ Saved optimized weights to `optimized_weights.json`")

if __name__ == '__main__':
    # Configuration
    N_TRIALS = 5000  # Number of optimization rounds
    LAMBDA = 0.25   # 0.0 = pure revenue prediction, 1.0 = pure distribution matching

    run_optimization(n_trials=N_TRIALS, lambda_param=LAMBDA) 