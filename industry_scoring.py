"""
industry_scoring.py
------------------
Data-driven industry scoring based on historical sales performance.

This module calculates industry vertical scores using Empirical-Bayes shrinkage
to balance between observed performance and global averages, handling small
sample sizes gracefully while ensuring robust scoring for all industries.
"""

import pandas as pd
import numpy as np
import json
import os


def calculate_industry_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total sales performance per customer from hardware, consumable, and service revenue.
    
    Args:
        df (pd.DataFrame): Customer dataframe with revenue columns
        
    Returns:
        pd.DataFrame: Original dataframe with added 'total_performance' column
    """
    # Revenue columns to sum for performance metric
    revenue_cols = ['Total Hardware Revenue', 'Total Consumable Revenue', 'Total Service Bureau Revenue']
    
    # Ensure all revenue columns exist, fill missing with 0
    for col in revenue_cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found, treating as 0")
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate total performance per customer
    df['total_performance'] = df[revenue_cols].sum(axis=1)
    
    print(f"[INFO] Calculated performance for {len(df)} customers")
    print(f"[INFO] Total performance range: ${df['total_performance'].min():,.0f} - ${df['total_performance'].max():,.0f}")
    print(f"[INFO] Mean performance per customer: ${df['total_performance'].mean():,.0f}")
    
    return df


def aggregate_by_industry(df: pd.DataFrame, min_sample: int = 10) -> pd.DataFrame:
    """
    Aggregate performance metrics by industry using adoption-adjusted success metric.
    
    Success = (adoption_rate) × (mean_revenue_among_adopters)
    This accounts for both the likelihood of hardware adoption and the value when it happens.
    
    Args:
        df (pd.DataFrame): Customer dataframe with 'Industry' and 'total_performance' columns
        min_sample (int): Minimum number of customers required for an industry to be scored
        
    Returns:
        pd.DataFrame: Industry-level aggregated metrics with adoption-adjusted success
    """
    # Handle missing/blank industries
    df['Industry_clean'] = df['Industry'].fillna('Unknown').str.strip()
    df.loc[df['Industry_clean'] == '', 'Industry_clean'] = 'Unknown'
    
    # Calculate adoption-adjusted metrics by industry
    def calc_adoption_metrics(group):
        total_customers = len(group)
        adopters = group[group['total_performance'] > 0]
        adopter_count = len(adopters)
        
        if adopter_count > 0:
            adoption_rate = adopter_count / total_customers
            mean_among_adopters = adopters['total_performance'].mean()
            success_metric = adoption_rate * mean_among_adopters
        else:
            adoption_rate = 0.0
            mean_among_adopters = 0.0
            success_metric = 0.0
        
        return pd.Series({
            'customer_count': total_customers,
            'adopter_count': adopter_count,
            'adoption_rate': adoption_rate,
            'mean_among_adopters': mean_among_adopters,
            'success_metric': success_metric,
            'mean_performance': group['total_performance'].mean()  # Keep for reference
        })
    
    industry_stats = df.groupby('Industry_clean').apply(calc_adoption_metrics).reset_index()
    
    # Filter out industries with insufficient sample size
    sufficient_sample = industry_stats['customer_count'] >= min_sample
    small_industries = industry_stats[~sufficient_sample]
    industry_stats = industry_stats[sufficient_sample]
    
    print(f"[INFO] Found {len(industry_stats)} industries with >= {min_sample} customers")
    print(f"[INFO] Using adoption-adjusted success metric: adoption_rate × mean_revenue_among_adopters")
    
    if len(small_industries) > 0:
        print(f"[INFO] {len(small_industries)} industries have < {min_sample} customers and will be treated as 'Unknown'")
        print(f"[INFO] Small industries: {small_industries['Industry_clean'].tolist()}")
    
    # Store success metric in 'shrunk_mean' column so downstream code works unchanged
    industry_stats['shrunk_mean'] = industry_stats['success_metric']
    
    return industry_stats


def apply_empirical_bayes_shrinkage(industry_stats: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Apply Empirical-Bayes shrinkage to industry success metrics to handle small sample sizes.
    
    Args:
        industry_stats (pd.DataFrame): Industry aggregated statistics with success_metric
        k (int): Shrinkage parameter (effective sample size of prior)
        
    Returns:
        pd.DataFrame: Industry stats with shrunk success metrics
    """
    # Calculate global success metric (weighted by customer count)
    total_customers = industry_stats['customer_count'].sum()
    global_success = (industry_stats['success_metric'] * industry_stats['customer_count']).sum() / total_customers
    
    print(f"[INFO] Global success metric: {global_success:,.0f}")
    print(f"[INFO] Applying Empirical-Bayes shrinkage with k={k}")
    
    # Apply shrinkage formula to success metric: (n_i * success_i + k * global_success) / (n_i + k)
    industry_stats['shrunk_mean'] = (
        (industry_stats['customer_count'] * industry_stats['success_metric'] + k * global_success) /
        (industry_stats['customer_count'] + k)
    )
    
    # Calculate shrinkage effect
    industry_stats['shrinkage_factor'] = k / (industry_stats['customer_count'] + k)
    
    print(f"[INFO] Shrinkage applied - range: {industry_stats['shrinkage_factor'].min():.2f} to {industry_stats['shrinkage_factor'].max():.2f}")
    
    return industry_stats


def build_industry_weights(df: pd.DataFrame, min_sample: int = 10, k: int = 20, neutral_score: float = 0.3) -> dict:
    """
    Main function to build data-driven industry weights from customer performance data.
    """
    print(f"[INFO] Building hybrid industry weights with min_sample={min_sample}, k={k}, neutral={neutral_score}")

    # Step 1: Calculate historical performance (data-driven score)
    df_with_performance = calculate_industry_performance(df.copy())
    industry_stats = aggregate_by_industry(df_with_performance, min_sample)
    if len(industry_stats) > 0:
        industry_stats = apply_empirical_bayes_shrinkage(industry_stats, k)
    
        # Normalize data-driven score to 0-1 range
        min_shrunk = industry_stats['shrunk_mean'].min()
        max_shrunk = industry_stats['shrunk_mean'].max()
        if max_shrunk > min_shrunk:
            industry_stats['data_driven_score'] = (industry_stats['shrunk_mean'] - min_shrunk) / (max_shrunk - min_shrunk)
        else:
            industry_stats['data_driven_score'] = 0.5 # Neutral if all same
    else:
        # Handle case with no industries meeting sample size
        return {'unknown': neutral_score, '': neutral_score, np.nan: neutral_score}


    # Step 2: Load strategic scores from config
    print("[INFO] Loading strategic tiers from strategic_industry_tiers.json")
    with open('strategic_industry_tiers.json', 'r') as f:
        strategic_config = json.load(f)
    tier_scores = strategic_config['tier_scores']
    blend_weights = strategic_config['blend_weight']
    industry_to_tier = {industry: tier for tier, industries in strategic_config['industry_tiers'].items() for industry in industries}

    def get_strategic_score(industry_name):
        tier = industry_to_tier.get(industry_name, 'tier_3') # Default to tier_3
        return tier_scores.get(tier, 0.4)

    industry_stats['strategic_score'] = industry_stats['Industry_clean'].apply(get_strategic_score)

    # Step 3: Blend data-driven and strategic scores
    print(f"[INFO] Blending scores with weights: Data-Driven({blend_weights['data_driven']}), Strategic({blend_weights['strategic']})")
    industry_stats['blended_score'] = (
        blend_weights['data_driven'] * industry_stats['data_driven_score'] +
        blend_weights['strategic'] * industry_stats['strategic_score']
    )

    # Step 4: Apply final bucketing
    bucketed_scores = (np.round(industry_stats['blended_score'] / 0.05) * 0.05).clip(0.0, 1.0)
    industry_stats['final_score'] = np.maximum(neutral_score, bucketed_scores)
    print(f"[INFO] Applied final bucketing: 0.05 increments, minimum {neutral_score:.2f}")

    # Step 5: Create final weights dictionary
    weights = dict(zip(
        industry_stats['Industry_clean'].str.lower().str.strip(),
        industry_stats['final_score']
    ))
    weights['unknown'] = neutral_score
    weights[''] = neutral_score
    weights[np.nan] = neutral_score

    print(f"[INFO] Generated scores for {len(weights)} industry categories")
    return weights


def save_industry_weights(weights: dict, filepath: str = "industry_weights.json"):
    """
    Save industry weights to JSON file with metadata.
    
    Args:
        weights (dict): Industry weights dictionary
        filepath (str): Output file path
    """
    output_data = {
        'weights': weights,
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'method': 'empirical_bayes_shrinkage',
            'total_industries': len(weights),
            'score_range': [min(weights.values()), max(weights.values())]
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"[INFO] Saved industry weights to {filepath}")


def load_industry_weights(filepath: str = "industry_weights.json") -> dict:
    """
    Load industry weights from JSON file.
    
    Args:
        filepath (str): Input file path
        
    Returns:
        dict: Industry weights, or empty dict if file not found
    """
    if not os.path.exists(filepath):
        print(f"[WARN] Industry weights file {filepath} not found")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        weights = data.get('weights', {})
        metadata = data.get('metadata', {})
        
        print(f"[INFO] Loaded industry weights from {filepath}")
        print(f"[INFO] Generated: {metadata.get('generated_at', 'Unknown')}")
        print(f"[INFO] Total industries: {metadata.get('total_industries', len(weights))}")
        
        return weights
    except Exception as e:
        print(f"[ERROR] Failed to load industry weights: {e}")
        return {} 