"""Division-aware scoring logic for the ICP project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from icp.divisions import DivisionConfig, get_division_config
from icp.schema import (
    COL_INDUSTRY,
    COL_BIG_BOX,
    COL_SMALL_BOX,
    COL_HW_REV,
    COL_CONS_REV,
    COL_REL_LICENSE,
    COL_REL_SAAS,
    COL_REL_MAINT,
    COL_RELATIONSHIP_PROFIT,
    COL_ADOPTION_ASSETS,
    COL_ADOPTION_PROFIT,
)

# --- Constants and Configurations ---
LICENSE_COL = COL_REL_LICENSE

# Default division key used by historical scripts/tests.
DEFAULT_DIVISION = "hardware"

# Default weights for ICP scoring (hardware defaults unless overridden at runtime).
# Note: Size is no longer a scoring component; only vertical, adoption, and
# relationship contribute to the final ICP score.
DEFAULT_WEIGHTS = get_division_config(DEFAULT_DIVISION).weight_dict()

# Defines the target distribution for the final A-F grades.
# For example, the top 10% of customers should receive an 'A' grade.
TARGET_GRADE_DISTRIBUTION = {
    'A': 0.10,  # Top 10%
    'B': 0.20,  # Next 20%
    'C': 0.40,  # Middle 40%
    'D': 0.20,  # Next 20%
    'F': 0.10   # Bottom 10%
}
# Cumulative distribution for easier grade assignment based on percentile ranks.
TARGET_CUMULATIVE_DISTRIBUTION = np.cumsum([
    TARGET_GRADE_DISTRIBUTION['F'],
    TARGET_GRADE_DISTRIBUTION['D'],
    TARGET_GRADE_DISTRIBUTION['C'],
    TARGET_GRADE_DISTRIBUTION['B'],
    TARGET_GRADE_DISTRIBUTION['A']
])

# Data-driven vertical weights based on the historical revenue performance of each industry.
# Higher values indicate better historical performance.
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


def _percentile_scale(series: pd.Series) -> pd.Series:
    """Convert numeric values to percentile ranks, ignoring pure-zero cohorts."""

    series = pd.to_numeric(series, errors="coerce").fillna(0)
    if series.empty:
        return pd.Series(dtype=float)
    if series.nunique(dropna=False) == 1:
        return pd.Series(0.5, index=series.index)

    result = pd.Series(0.0, index=series.index, dtype=float)
    non_zero_mask = series > 0
    if non_zero_mask.any():
        result.loc[non_zero_mask] = series[non_zero_mask].rank(method="average", pct=True)
    return result


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale to the 0-1 range using min/max with zero safeguard."""

    series = pd.to_numeric(series, errors="coerce").fillna(0)
    if series.empty:
        return pd.Series(dtype=float)
    min_val, max_val = series.min(), series.max()
    if max_val - min_val == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - min_val) / (max_val - min_val)


def _sum_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Aggregate numeric columns, ignoring missing ones."""

    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index, dtype=float)
    numeric = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    return numeric.sum(axis=1)


def _compute_adoption_scores(df: pd.DataFrame, config: DivisionConfig) -> pd.Series:
    index = df.index

    assets_series = None
    if config.adoption.asset_column and config.adoption.asset_column in df.columns:
        assets_series = pd.to_numeric(df[config.adoption.asset_column], errors="coerce").fillna(0)
    elif config.adoption.asset_goals:
        assets_series = _sum_columns(df, config.adoption.asset_goals)

    profit_series = None
    if config.adoption.profit_column and config.adoption.profit_column in df.columns:
        profit_series = pd.to_numeric(df[config.adoption.profit_column], errors="coerce").fillna(0)
    elif config.adoption.profit_goals:
        profit_series = _sum_columns(df, config.adoption.profit_goals)
    elif config.adoption.fallback_revenue_columns:
        profit_series = _sum_columns(df, config.adoption.fallback_revenue_columns)

    if assets_series is not None or profit_series is not None:
        if assets_series is None:
            assets_series = pd.Series(0.0, index=index, dtype=float)
        if profit_series is None:
            profit_series = pd.Series(0.0, index=index, dtype=float)

        P = _percentile_scale(assets_series)
        R = _percentile_scale(profit_series)

        adoption_scores = np.zeros(len(df), dtype=float)
        zero_assets = assets_series == 0
        zero_profit = profit_series == 0

        profit_only_mask = zero_assets & ~zero_profit
        adoption_scores[profit_only_mask] = 0.5 * np.sqrt(R[profit_only_mask])

        with_assets_mask = ~zero_assets
        if with_assets_mask.any():
            adoption_scores[with_assets_mask] = (0.6 * P + 0.4 * R)[with_assets_mask]

        return pd.Series(adoption_scores, index=index)

    # No legacy fallbacks: if neither assets nor profit are available, treat adoption as 0.0
    return pd.Series(0.0, index=index, dtype=float)


def _compute_relationship_scores(
    df: pd.DataFrame, config: DivisionConfig
) -> tuple[pd.Series, pd.Series | None]:
    index = df.index
    feature_series: pd.Series | None = None

    if config.relationship.profit_column and config.relationship.profit_column in df.columns:
        rel_safe = pd.to_numeric(df[config.relationship.profit_column], errors="coerce").fillna(0)
        return _min_max_scale(np.log1p(rel_safe)), rel_safe

    if config.relationship.profit_goals:
        rel_series = _sum_columns(df, config.relationship.profit_goals)
        if rel_series.any():
            return _min_max_scale(np.log1p(rel_series)), rel_series

    if config.relationship.revenue_fallback_columns:
        feature_series = _sum_columns(df, config.relationship.revenue_fallback_columns)
        if feature_series.any():
            return _min_max_scale(np.log1p(feature_series)), feature_series

    return pd.Series(0.0, index=index, dtype=float), feature_series

def load_dynamic_industry_weights(division: str | DivisionConfig | None = None) -> dict[str, float]:
    """Load industry weights for a division, falling back to static defaults."""

    if isinstance(division, DivisionConfig):
        config = division
    else:
        config = get_division_config(division or DEFAULT_DIVISION)

    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    if config.industry_weights_file:
        try:
            candidates.append(Path(config.industry_weights_file))
        except TypeError:
            # Gracefully skip invalid/None-like paths
            pass

    candidates.extend(
        [
            root / "artifacts" / "weights" / "industry_weights.json",
            Path.cwd() / "industry_weights.json",
        ]
    )

    for industry_weights_file in candidates:
        if not industry_weights_file:
            continue

        path = Path(industry_weights_file)
        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            weights = data.get("weights", PERFORMANCE_VERTICAL_WEIGHTS)
            print(
                f"[INFO] Loaded {len(weights)} dynamic industry weights from {path}"
            )
            return weights
        except Exception as exc:  # pragma: no cover - defensive log
            print(f"[WARN] Error loading dynamic industry weights from {path}: {exc}")
            print("[WARN] Trying next candidate or falling back to static weights")
            continue

    fallback = dict(PERFORMANCE_VERTICAL_WEIGHTS)
    neutral = config.neutral_vertical_score
    fallback.setdefault("unknown", neutral)
    fallback.setdefault("", neutral)
    fallback.setdefault(None, neutral)  # type: ignore[key-type]
    print("[INFO] No dynamic industry weights found, using static weights")
    return fallback

def calculate_grades(scores):
    """
    Assigns A-F grades based on the percentile rank of the final scores.

    Args:
        scores (pd.Series): A series of final, normalized ICP scores.

    Returns:
        np.ndarray: An array of corresponding letter grades ('A' through 'F').
    """
    ranks = scores.rank(pct=True)
    grades = np.select(
        [
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[0], # F
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[1], # D
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[2], # C
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[3], # B
            ranks > TARGET_CUMULATIVE_DISTRIBUTION[3]  # A
        ],
        ['F', 'D', 'C', 'B', 'A'],
        default='C'
    )
    return grades

def calculate_scores(
    df: pd.DataFrame,
    weights: Mapping[str, float] | None,
    size_config: Mapping[str, float] | None = None,
    division: str | DivisionConfig | None = None,
) -> pd.DataFrame:
    """Compute component and final ICP scores for the requested division."""

    config = division if isinstance(division, DivisionConfig) else get_division_config(division)
    df_clean = df.copy()
    _ = size_config  # legacy compatibility

    score_cols_to_drop = [
        "vertical_score",
        "size_score",
        "adoption_score",
        "relationship_score",
        "relationship_feature",
        "ICP_score_raw",
        "ICP_score",
    ]
    df_clean = df_clean.drop(columns=[c for c in score_cols_to_drop if c in df_clean.columns])

    weight_bundle = config.weight_dict()
    if weights:
        weight_bundle.update(dict(weights))
    # Only vertical/adoption/relationship are used; size has been retired.
    for key in ("vertical", "adoption", "relationship"):
        weight_bundle.setdefault(key, 0.0)

    industry_weights = load_dynamic_industry_weights(config)
    industry_series = df_clean.get(COL_INDUSTRY, pd.Series("", index=df_clean.index))
    v_lower = industry_series.astype(str).str.lower().str.strip()
    df_clean["vertical_score"] = v_lower.map(industry_weights).fillna(config.neutral_vertical_score)

    df_clean["adoption_score"] = _compute_adoption_scores(df_clean, config)

    relationship_scores, relationship_feature = _compute_relationship_scores(df_clean, config)
    df_clean["relationship_score"] = relationship_scores
    if relationship_feature is not None:
        df_clean["relationship_feature"] = relationship_feature

    df_clean["ICP_score_raw"] = (
        weight_bundle["vertical"] * df_clean["vertical_score"]
        + weight_bundle["adoption"] * df_clean["adoption_score"]
        + weight_bundle["relationship"] * df_clean["relationship_score"]
    ) * 100

    ranks = df_clean["ICP_score_raw"].rank(method="first")
    n = len(ranks)
    p = (ranks - 0.5) / n
    z = norm.ppf(p)
    df_clean["ICP_score"] = (50 + 15 * z).clip(0, 100)
    df_clean["ICP_grade"] = calculate_grades(df_clean["ICP_score"])

    return df_clean
