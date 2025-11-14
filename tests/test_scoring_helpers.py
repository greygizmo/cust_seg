from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icp import scoring as scoring_module
from icp.divisions import get_division_config
from icp.scoring import (
    _compute_adoption_scores,
    _compute_relationship_scores,
    _min_max_scale,
    _percentile_scale,
    load_dynamic_industry_weights,
)


def test_percentile_scale_handles_constant_and_sparse_inputs():
    constant = pd.Series([0, 0, 0], dtype=float)
    assert (_percentile_scale(constant) == 0.5).all()

    mixed = pd.Series([0, 10, 20, 0], dtype=float)
    result = _percentile_scale(mixed)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.5)
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(0.0)


def test_min_max_scale_clamps_when_range_missing():
    constant = pd.Series([5, 5, 5], dtype=float)
    assert (_min_max_scale(constant) == 0.0).all()

    values = pd.Series([0.0, 5.0, 10.0], dtype=float)
    scaled = _min_max_scale(values)
    assert scaled.iloc[0] == pytest.approx(0.0)
    assert scaled.iloc[1] == pytest.approx(0.5)
    assert scaled.iloc[2] == pytest.approx(1.0)


def test_compute_adoption_scores_from_columns_and_fallbacks():
    config = get_division_config('hardware')

    # Direct asset/profit columns present
    df_direct = pd.DataFrame({
        'adoption_assets': [10, 20, 0],
        'adoption_profit': [100, 50, 0],
    })
    direct_scores = _compute_adoption_scores(df_direct, config)
    assert direct_scores.tolist() == pytest.approx([0.7, 0.8, 0.0])

    # Profit-only path (zero assets but positive profit)
    df_profit_only = pd.DataFrame({
        'adoption_assets': [0, 0, 0],
        'adoption_profit': [100, 50, 0],
    })
    profit_scores = _compute_adoption_scores(df_profit_only, config)
    expected = 0.5 * np.sqrt(pd.Series([1.0, 0.5, 0.0]))
    assert profit_scores.tolist() == pytest.approx(expected.tolist())

    # Fall back to goal aggregations when direct columns are missing
    df_fallback = pd.DataFrame({
        'Printers': [1, 0],
        'Printer Accessorials': [0, 2],
        'Scanners': [0, 0],
        'Geomagic': [0, 0],
        'Training/Services': [0, 5],
    })
    fallback_scores = _compute_adoption_scores(df_fallback, config)
    assert fallback_scores.tolist() == pytest.approx([0.5, 1.0])


def test_compute_relationship_scores_uses_profit_and_fallbacks():
    config = get_division_config('hardware')

    df_profit = pd.DataFrame({'relationship_profit': [0, 10, 110]})
    scores, feature = _compute_relationship_scores(df_profit, config)
    logs = np.log1p(df_profit['relationship_profit'])
    expected = (logs - logs.min()) / (logs.max() - logs.min())
    assert scores.tolist() == pytest.approx(expected.tolist())
    assert feature is not None
    assert feature.tolist() == pytest.approx(df_profit['relationship_profit'].tolist())

    # Remove profit column to trigger fallback revenue usage
    df_fallback = pd.DataFrame({
        'Total Software License Revenue': [0.0, 1_000.0],
        'Total SaaS Revenue': [0.0, 500.0],
        'Total Maintenance Revenue': [0.0, 0.0],
    })
    scores, feature = _compute_relationship_scores(df_fallback, config)
    assert feature is not None
    assert feature.tolist() == pytest.approx([0.0, 1_500.0])
    assert scores.tolist() == pytest.approx([0.0, 1.0])


def test_load_dynamic_industry_weights_prefers_division_file():
    config = get_division_config('hardware')
    weights = load_dynamic_industry_weights(config)
    assert weights['high tech'] == pytest.approx(0.8500000000000001)


def test_load_dynamic_industry_weights_falls_back_to_static(monkeypatch):
    config = get_division_config('hardware')
    fake_config = replace(
        config,
        industry_weights_file=Path('/definitely/missing.json'),
        neutral_vertical_score=0.42,
    )

    original_exists = Path.exists
    repo_root = Path(scoring_module.__file__).resolve().parents[2]
    candidate_strings = {
        str(fake_config.industry_weights_file),
        str(repo_root / 'artifacts' / 'weights' / 'industry_weights.json'),
        str(Path.cwd() / 'industry_weights.json'),
    }

    def fake_exists(self):  # type: ignore[override]
        if str(self) in candidate_strings:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, 'exists', fake_exists)

    weights = load_dynamic_industry_weights(fake_config)
    assert weights['unknown'] == pytest.approx(fake_config.neutral_vertical_score)
    assert weights[''] == pytest.approx(fake_config.neutral_vertical_score)
    assert weights[None] == pytest.approx(fake_config.neutral_vertical_score)
