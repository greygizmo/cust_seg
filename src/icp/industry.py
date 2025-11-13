"""Division-aware industry scoring helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from icp.divisions import DivisionConfig, get_division_config


def _sum_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index, dtype=float)
    numeric = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    return numeric.sum(axis=1)


def calculate_industry_performance(
    df: pd.DataFrame, division: str | DivisionConfig | None = None
) -> pd.DataFrame:
    """Annotate a customer dataframe with division-specific performance totals."""

    config = division if isinstance(division, DivisionConfig) else get_division_config(division)
    df = df.copy()

    if config.performance_columns:
        perf = _sum_numeric(df, config.performance_columns)
    elif config.adoption.profit_column and config.adoption.profit_column in df.columns:
        perf = pd.to_numeric(df[config.adoption.profit_column], errors="coerce").fillna(0)
    elif "Profit_Since_2023_Total" in df.columns:
        perf = pd.to_numeric(df["Profit_Since_2023_Total"], errors="coerce").fillna(0)
    else:
        fallback_cols = [
            "Total Hardware Revenue",
            "Total Consumable Revenue",
            "Total Service Bureau Revenue",
        ]
        perf = _sum_numeric(df, fallback_cols)

    df["total_performance"] = perf

    print(f"[INFO] Calculated performance for {len(df)} customers ({config.label})")
    if not perf.empty:
        print(f"[INFO] Total performance range: ${perf.min():,.0f} - ${perf.max():,.0f}")
        print(f"[INFO] Mean performance per customer: ${perf.mean():,.0f}")

    return df


def aggregate_by_industry(df: pd.DataFrame, min_sample: int = 10) -> pd.DataFrame:
    """Aggregate performance metrics by industry using adoption-adjusted success."""

    df = df.copy()
    df["Industry_clean"] = df["Industry"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["Industry_clean"] == "", "Industry_clean"] = "Unknown"

    def calc_adoption_metrics(group: pd.DataFrame) -> pd.Series:
        total_customers = len(group)
        adopters = group[group["total_performance"] > 0]
        adopter_count = len(adopters)
        if adopter_count > 0:
            adoption_rate = adopter_count / total_customers
            mean_among_adopters = adopters["total_performance"].mean()
            success_metric = adoption_rate * mean_among_adopters
        else:
            adoption_rate = 0.0
            mean_among_adopters = 0.0
            success_metric = 0.0
        return pd.Series(
            {
                "customer_count": total_customers,
                "adopter_count": adopter_count,
                "adoption_rate": adoption_rate,
                "mean_among_adopters": mean_among_adopters,
                "success_metric": success_metric,
                "mean_performance": group["total_performance"].mean(),
            }
        )

    industry_stats = df.groupby("Industry_clean").apply(calc_adoption_metrics).reset_index()

    sufficient_sample = industry_stats["customer_count"] >= min_sample
    small_industries = industry_stats[~sufficient_sample]
    industry_stats = industry_stats[sufficient_sample]

    print(f"[INFO] Found {len(industry_stats)} industries with >= {min_sample} customers")
    print("[INFO] Using adoption-adjusted success metric: adoption_rate × mean_performance")

    if len(small_industries) > 0:
        print(
            f"[INFO] {len(small_industries)} industries have < {min_sample} customers and will be treated as 'Unknown'"
        )
        print(f"[INFO] Small industries: {small_industries['Industry_clean'].tolist()}")

    industry_stats["shrunk_mean"] = industry_stats["success_metric"]
    return industry_stats


def apply_empirical_bayes_shrinkage(industry_stats: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Apply Empirical-Bayes shrinkage to industry success metrics."""

    total_customers = industry_stats["customer_count"].sum()
    if total_customers == 0:
        return industry_stats

    global_success = (
        industry_stats["success_metric"] * industry_stats["customer_count"]
    ).sum() / total_customers

    print(f"[INFO] Global success metric: {global_success:,.0f}")
    print(f"[INFO] Applying Empirical-Bayes shrinkage with k={k}")

    industry_stats = industry_stats.copy()
    industry_stats["shrunk_mean"] = (
        (industry_stats["customer_count"] * industry_stats["success_metric"] + k * global_success)
        / (industry_stats["customer_count"] + k)
    )
    industry_stats["shrinkage_factor"] = k / (industry_stats["customer_count"] + k)

    return industry_stats


def build_industry_weights(
    df: pd.DataFrame,
    division: str | DivisionConfig | None = None,
    min_sample: int = 10,
    k: int = 20,
    neutral_score: float | None = None,
) -> dict[str, float]:
    """Build data-driven industry weights for the requested division."""

    config = division if isinstance(division, DivisionConfig) else get_division_config(division)
    neutral = neutral_score if neutral_score is not None else config.neutral_vertical_score

    print(
        f"[INFO] Building hybrid industry weights for {config.label} "
        f"with min_sample={min_sample}, k={k}, neutral={neutral:.2f}"
    )

    df_with_performance = calculate_industry_performance(df.copy(), config)
    industry_stats = aggregate_by_industry(df_with_performance, min_sample)
    if industry_stats.empty:
        return {"unknown": neutral, "": neutral, None: neutral}  # type: ignore[key-type]

    industry_stats = apply_empirical_bayes_shrinkage(industry_stats, k)

    min_shrunk = industry_stats["shrunk_mean"].min()
    max_shrunk = industry_stats["shrunk_mean"].max()
    if max_shrunk > min_shrunk:
        industry_stats["data_driven_score"] = (
            industry_stats["shrunk_mean"] - min_shrunk
        ) / (max_shrunk - min_shrunk)
    else:
        industry_stats["data_driven_score"] = 0.5

    root = Path(__file__).resolve().parents[2]
    strategic_path = root / "artifacts" / "industry" / "strategic_industry_tiers.json"
    with strategic_path.open("r", encoding="utf-8") as handle:
        strategic_config = json.load(handle)

    tier_scores = strategic_config["tier_scores"]
    blend_weights = strategic_config["blend_weight"]
    industry_to_tier = {
        industry: tier
        for tier, industries in strategic_config["industry_tiers"].items()
        for industry in industries
    }

    def get_strategic_score(industry_name: str) -> float:
        tier = industry_to_tier.get(industry_name, "tier_3")
        return tier_scores.get(tier, neutral)

    industry_stats["strategic_score"] = industry_stats["Industry_clean"].apply(get_strategic_score)

    industry_stats["blended_score"] = (
        blend_weights["data_driven"] * industry_stats["data_driven_score"]
        + blend_weights["strategic"] * industry_stats["strategic_score"]
    )

    bucketed_scores = (np.round(industry_stats["blended_score"] / 0.05) * 0.05).clip(neutral, 1.0)
    industry_stats["final_score"] = bucketed_scores

    weights = dict(
        zip(industry_stats["Industry_clean"].str.lower().str.strip(), industry_stats["final_score"])
    )
    weights["unknown"] = neutral
    weights[""] = neutral
    weights[None] = neutral  # type: ignore[key-type]

    print(f"[INFO] Generated scores for {len(weights)} industry categories")
    return weights


def save_industry_weights(
    weights: dict[str, float],
    division: str | DivisionConfig | None = None,
    filepath: str | Path | None = None,
) -> None:
    """Persist industry weights with metadata."""

    config = division if isinstance(division, DivisionConfig) else get_division_config(division)
    if filepath is None:
        filepath = config.industry_weights_file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "weights": weights,
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "division": config.key,
            "method": "empirical_bayes_shrinkage",
            "total_industries": len(weights),
            "score_range": [min(weights.values()), max(weights.values())],
        },
    }

    with filepath.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=4)

    print(f"[INFO] Saved industry weights to {filepath}")


def load_industry_weights(
    division: str | DivisionConfig | None = None,
    filepath: str | Path | None = None,
) -> dict[str, float]:
    """Load previously persisted industry weights."""

    config = division if isinstance(division, DivisionConfig) else get_division_config(division)
    if filepath is None:
        filepath = config.industry_weights_file
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[WARN] Industry weights file {filepath} not found")
        return {}

    try:
        with filepath.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        weights = data.get("weights", {})
        metadata = data.get("metadata", {})
        print(f"[INFO] Loaded industry weights from {filepath}")
        if metadata:
            print(
                f"[INFO] Generated: {metadata.get('generated_at', 'Unknown')} | "
                f"Division: {metadata.get('division', config.key)}"
            )
        return weights
    except Exception as exc:  # pragma: no cover - defensive log
        print(f"[ERROR] Failed to load industry weights: {exc}")
        return {}
