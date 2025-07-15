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
    Aggregate performance metrics by industry with sample size filtering.
    
    Args:
        df (pd.DataFrame): Customer dataframe with 'Industry' and 'total_performance' columns
        min_sample (int): Minimum number of customers required for an industry to be scored
        
    Returns:
        pd.DataFrame: Industry-level aggregated metrics
    """
    # Handle missing/blank industries
    df['Industry_clean'] = df['Industry'].fillna('Unknown').str.strip()
    df.loc[df['Industry_clean'] == '', 'Industry_clean'] = 'Unknown'
    
    # Aggregate by industry
    industry_stats = df.groupby('Industry_clean').agg({
        'total_performance': ['count', 'mean', 'sum', 'std']
    }).round(2)
    
    # Flatten column names
    industry_stats.columns = ['customer_count', 'mean_performance', 'total_performance', 'std_performance']
    industry_stats = industry_stats.reset_index()
    
    # Filter out industries with insufficient sample size
    sufficient_sample = industry_stats['customer_count'] >= min_sample
    small_industries = industry_stats[~sufficient_sample]
    industry_stats = industry_stats[sufficient_sample]
    
    print(f"[INFO] Found {len(industry_stats)} industries with >= {min_sample} customers")
    if len(small_industries) > 0:
        print(f"[INFO] {len(small_industries)} industries have < {min_sample} customers and will be treated as 'Unknown'")
        print(f"[INFO] Small industries: {small_industries['Industry_clean'].tolist()}")
    
    return industry_stats


def apply_empirical_bayes_shrinkage(industry_stats: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Apply Empirical-Bayes shrinkage to industry means to handle small sample sizes.
    
    Args:
        industry_stats (pd.DataFrame): Industry aggregated statistics
        k (int): Shrinkage parameter (effective sample size of prior)
        
    Returns:
        pd.DataFrame: Industry stats with shrunk means
    """
    # Calculate global mean across all customers (weighted by customer count)
    total_customers = industry_stats['customer_count'].sum()
    global_mean = (industry_stats['mean_performance'] * industry_stats['customer_count']).sum() / total_customers
    
    print(f"[INFO] Global mean performance: ${global_mean:,.0f}")
    print(f"[INFO] Applying Empirical-Bayes shrinkage with k={k}")
    
    # Apply shrinkage formula: (n_i * μ_i + k * μ_global) / (n_i + k)
    industry_stats['shrunk_mean'] = (
        (industry_stats['customer_count'] * industry_stats['mean_performance'] + k * global_mean) /
        (industry_stats['customer_count'] + k)
    )
    
    # Calculate shrinkage effect
    industry_stats['shrinkage_factor'] = k / (industry_stats['customer_count'] + k)
    
    print(f"[INFO] Shrinkage applied - range: {industry_stats['shrinkage_factor'].min():.2f} to {industry_stats['shrinkage_factor'].max():.2f}")
    
    return industry_stats


def normalize_to_scores(industry_stats: pd.DataFrame, neutral_score: float = 0.3) -> dict:
    """
    Normalize shrunk means to 0-1 scores with special handling for neutral categories.
    
    Args:
        industry_stats (pd.DataFrame): Industry stats with shrunk means
        neutral_score (float): Score to assign to unknown/blank industries
        
    Returns:
        dict: Mapping of industry names to scores (0-1)
    """
    if len(industry_stats) == 0:
        print("[WARN] No industries to score, returning empty weights")
        return {}
    
    # Min-max normalization of shrunk means
    min_shrunk = industry_stats['shrunk_mean'].min()
    max_shrunk = industry_stats['shrunk_mean'].max()
    
    if max_shrunk == min_shrunk:
        # Edge case: all industries have same performance
        print("[WARN] All industries have identical performance, assigning neutral scores")
        industry_stats['final_score'] = neutral_score
    else:
        industry_stats['final_score'] = (
            (industry_stats['shrunk_mean'] - min_shrunk) / (max_shrunk - min_shrunk)
        )
    
    # Create weights dictionary
    weights = dict(zip(
        industry_stats['Industry_clean'].str.lower().str.strip(),
        industry_stats['final_score']
    ))
    
    # Add neutral score for unknown/blank industries
    weights['unknown'] = neutral_score
    weights[''] = neutral_score
    weights[np.nan] = neutral_score
    
    print(f"[INFO] Generated scores for {len(weights)} industry categories")
    print(f"[INFO] Score range: {min(weights.values()):.3f} to {max(weights.values()):.3f}")
    
    # Show top and bottom performers
    sorted_industries = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print(f"[INFO] Top 5 performing industries:")
    for industry, score in sorted_industries[:5]:
        if industry not in ['unknown', '', np.nan]:
            print(f"  - {industry}: {score:.3f}")
    
    print(f"[INFO] Bottom 5 performing industries:")
    for industry, score in sorted_industries[-5:]:
        if industry not in ['unknown', '', np.nan]:
            print(f"  - {industry}: {score:.3f}")
    
    return weights


def build_industry_weights(df: pd.DataFrame, min_sample: int = 10, k: int = 20, neutral_score: float = 0.3) -> dict:
    """
    Main function to build data-driven industry weights from customer performance data.
    
    Args:
        df (pd.DataFrame): Customer dataframe with industry and revenue data
        min_sample (int): Minimum customers required for an industry to be scored individually
        k (int): Empirical-Bayes shrinkage parameter
        neutral_score (float): Score for unknown/insufficient sample industries
        
    Returns:
        dict: Industry name -> score (0-1) mapping
    """
    print(f"[INFO] Building industry weights with min_sample={min_sample}, k={k}, neutral={neutral_score}")
    
    # Step 1: Calculate performance per customer
    df_with_performance = calculate_industry_performance(df.copy())
    
    # Step 2: Aggregate by industry
    industry_stats = aggregate_by_industry(df_with_performance, min_sample)
    
    # Step 3: Apply Empirical-Bayes shrinkage
    if len(industry_stats) > 0:
        industry_stats = apply_empirical_bayes_shrinkage(industry_stats, k)
    
    # Step 4: Normalize to 0-1 scores
    weights = normalize_to_scores(industry_stats, neutral_score)
    
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