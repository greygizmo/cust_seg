"""
Centralized scoring logic for the ICP project.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# --- Constants and Configurations ---
LICENSE_COL = "Total Software License Revenue"

# Default weights for ICP scoring
DEFAULT_WEIGHTS = {
    "vertical": 0.3,
    "size": 0.3,
    "adoption": 0.2,
    "relationship": 0.2,
}

# Target grade distribution for A-F grades
TARGET_GRADE_DISTRIBUTION = {
    'A': 0.10,  # Top 10%
    'B': 0.20,  # Next 20%
    'C': 0.40,  # Middle 40%
    'D': 0.20,  # Next 20%
    'F': 0.10   # Bottom 10%
}
TARGET_CUMULATIVE_DISTRIBUTION = np.cumsum([
    TARGET_GRADE_DISTRIBUTION['F'],
    TARGET_GRADE_DISTRIBUTION['D'],
    TARGET_GRADE_DISTRIBUTION['C'],
    TARGET_GRADE_DISTRIBUTION['B'],
    TARGET_GRADE_DISTRIBUTION['A']
])

# Data-driven vertical weights based on actual Total Hardware + Consumable Revenue performance from JY spreadsheet
PERFORMANCE_VERTICAL_WEIGHTS = {
    "aerospace & defense": 1.0,
    "automotive & transportation": 1.0,
    "consumer goods": 1.0,
    "high tech": 1.0,
    "medical devices & life sciences": 1.0,
    "engineering services": 0.8,
    "heavy equip & ind. components": 0.8,
    "industrial machinery": 0.8,
    "mold, tool & die": 0.8,
    "other": 0.8,
    "building & construction": 0.6,
    "chemicals & related products": 0.6,
    "dental": 0.6,
    "manufactured products": 0.6,
    "services": 0.6,
    "education & research": 0.4,
    "electromagnetic": 0.4,
    "energy": 0.4,
    "packaging": 0.4,
    "plant & process": 0.4,
    "shipbuilding": 0.4,
}

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

def calculate_scores(df, weights, size_config=None):
    """
    Calculate all scores based on weights and configurations.
    
    Note: The size_config parameter is kept for compatibility but the logic
    is now hardcoded based on the data-driven analysis from the original
    goe_icp_scoring.py script.
    """
    df_clean = df.copy()

    # --- Pre-emptive Cleanup ---
    # Drop any pre-existing score columns to prevent duplicates. This is the root cause fix.
    score_cols_to_drop = [
        'vertical_score', 'size_score', 'adoption_score', 'relationship_score', 
        'relationship_feature', 'ICP_score_raw', 'ICP_score'
    ]
    for col in score_cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=col)
    
    # 1. New Data-Driven Vertical Score
    v_lower = df_clean["Industry"].astype(str).str.lower().str.strip()
    df_clean["vertical_score"] = v_lower.map(PERFORMANCE_VERTICAL_WEIGHTS).fillna(0.3)

    # 2. Data-Driven Size Score (based on empirical analysis)
    revenue_values = df_clean['revenue_estimate'].fillna(0)
    has_reliable_revenue = revenue_values > 0
    
    df_clean["size_score"] = 0.5  # Neutral default score
    
    conditions = [
        (revenue_values >= 250_000_000) & (revenue_values < 1_000_000_000),
        (revenue_values >= 1_000_000_000),
        (revenue_values >= 50_000_000),
        (revenue_values >= 10_000_000),
        (revenue_values > 0)
    ]
    scores = [1.0, 0.9, 0.6, 0.4, 0.4]
    
    for condition, score in zip(conditions, scores):
        mask = has_reliable_revenue & condition
        df_clean.loc[mask, "size_score"] = score

    # 3. New Adoption Score (Printer Count + Consumable Revenue)
    def min_max_scale(series):
        min_val, max_val = series.min(), series.max()
        if max_val - min_val == 0:
            return pd.Series(0.0, index=series.index)
        return (series - min_val) / (max_val - min_val)

    if 'Total Consumable Revenue' not in df_clean.columns:
        df_clean['Total Consumable Revenue'] = 0

    printer_count_safe = np.maximum(df_clean['printer_count'].fillna(0), 0)
    consumable_revenue_safe = np.maximum(df_clean['Total Consumable Revenue'].fillna(0), 0)
    
    printer_score = min_max_scale(np.log1p(printer_count_safe))
    consumable_score = min_max_scale(np.log1p(consumable_revenue_safe))
    df_clean['adoption_score'] = 0.5 * printer_score + 0.5 * consumable_score

    # 4. New Relationship Score (All Software-related Revenue)
    relationship_cols = ['Total Software License Revenue', 'Total SaaS Revenue', 'Total Maintenance Revenue']
    for col in relationship_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_clean['relationship_feature'] = df_clean[relationship_cols].fillna(0).sum(axis=1)
    relationship_feature_safe = np.maximum(df_clean['relationship_feature'], 0)
    df_clean['relationship_score'] = min_max_scale(np.log1p(relationship_feature_safe))
    
    if LICENSE_COL in df_clean.columns:
        license_revenue = pd.to_numeric(df_clean[LICENSE_COL], errors='coerce').fillna(0)
        bins = [-1, 5000, 25000, 100000, np.inf]
        labels = ["Bronze", "Silver", "Gold", "Platinum"]
        df_clean['cad_tier'] = pd.cut(license_revenue, bins=bins, labels=labels)
    else:
        df_clean['cad_tier'] = 'Bronze'
        df_clean['cad_tier'] = pd.Categorical(df_clean['cad_tier'], categories=["Bronze", "Silver", "Gold", "Platinum"])

    # Calculate RAW ICP score
    df_clean['ICP_score_raw'] = (
        weights["vertical"] * df_clean['vertical_score'] +
        weights["size"] * df_clean['size_score'] +
        weights["adoption"] * df_clean['adoption_score'] +
        weights["relationship"] * df_clean['relationship_score']
    ) * 100

    # Monotonic normalization for bell-curve shape
    ranks = df_clean['ICP_score_raw'].rank(method='first')
    n = len(ranks)
    p = (ranks - 0.5) / n
    z = norm.ppf(p)

    df_clean['ICP_score'] = (50 + 15 * z).clip(0, 100)
    
    # Assign letter grades based on ICP score percentiles
    df_clean['ICP_grade'] = calculate_grades(df_clean['ICP_score'])
    
    # The original 'ICP_score_new' is no longer needed, nor is the rename.
    # df.rename(columns={'ICP_score_new': 'ICP_score'}, inplace=True)

    return df_clean
