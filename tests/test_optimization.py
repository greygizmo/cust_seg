import math

import numpy as np
import pandas as pd
import pytest

from icp import optimization


class DummyTrial:
    """Minimal Optuna-like trial for deterministic testing."""

    def __init__(self, suggestions: dict[str, float]):
        self._suggestions = suggestions

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = self._suggestions[name]
        # Mirror optuna's bounds contract to catch regressions quickly.
        if not (low <= value <= high):  # pragma: no cover - defensive
            raise AssertionError(f"{name} suggestion {value} outside [{low}, {high}]")
        return value


def test_calculate_grades_matches_target_distribution():
    scores = pd.Series(range(100), dtype=float)
    grades = pd.Series(optimization.calculate_grades(scores))
    counts = grades.value_counts().reindex(['A', 'B', 'C', 'D', 'F']).fillna(0).astype(int)
    assert counts.to_dict() == {'A': 10, 'B': 20, 'C': 40, 'D': 20, 'F': 10}


def test_lift_by_grade_matches_expected_concentration():
    scores = pd.Series(range(100), dtype=float)
    # Concentrate all positives in the top decile (grade A)
    y = pd.Series([0] * 90 + [1] * 10, dtype=float)

    lift = optimization._lift_by_grade(scores, y)

    assert lift['lift@A'] == pytest.approx(10.0)
    for grade in ['B', 'C', 'D', 'F']:
        assert lift[f'lift@{grade}'] == pytest.approx(0.0)


def test_stability_penalty_uses_within_group_correlation_std():
    scores = pd.Series(np.arange(120, dtype=float))
    y = scores.copy()
    # Force opposite correlation for the second group
    y.iloc[60:] = -y.iloc[60:]
    groups = pd.Series(['g1'] * 60 + ['g2'] * 60)

    penalty = optimization._stability_penalty(scores, y, groups)

    assert penalty == pytest.approx(1.0)


def test_cumulative_lift_produces_expected_points_and_auc():
    scores = pd.Series(np.arange(100, 0, -1), dtype=float)
    y = pd.Series([1] * 10 + [0] * 90, dtype=float)

    result = optimization.cumulative_lift(scores, y)

    points = {pt['pop_share']: pt for pt in result['points']}
    assert points[0.01]['cum_y_share'] == pytest.approx(0.1)
    assert points[0.10]['cum_y_share'] == pytest.approx(1.0)
    assert points[0.50]['cum_y_share'] == pytest.approx(1.0)
    assert points[0.01]['lift'] == pytest.approx(10.0)
    assert result['auc'] == pytest.approx(0.9495, rel=1e-4)


def test_objective_matches_manual_combination():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        'vertical_score': rng.uniform(0.2, 1.0, 300),
        'size_score': np.zeros(300),
        'adoption_score': rng.uniform(0.0, 1.0, 300),
        'relationship_score': rng.uniform(0.0, 1.0, 300),
    })
    y = pd.Series(rng.uniform(0.0, 1.0, 300))
    groups = pd.Series(['g1'] * 150 + ['g2'] * 150)

    suggestions = {
        'vertical_score': 0.30,
        'adoption_score': 0.40,
        'relationship_score': 0.30,
    }
    trial = DummyTrial(suggestions)

    lambda_param = 0.55
    lift_weight = 0.10
    stability_weight = 0.25

    actual = optimization.objective(
        trial,
        X,
        y,
        lambda_param=lambda_param,
        weight_names=list(X.columns),
        include_size=False,
        groups=groups,
        lift_weight=lift_weight,
        stability_weight=stability_weight,
    )

    weights = np.array([
        suggestions['vertical_score'],
        0.0,
        suggestions['adoption_score'],
        suggestions['relationship_score'],
    ])
    weights /= weights.sum()
    icp_scores = X.to_numpy().dot(weights)

    spearman_corr, _ = optimization.spearmanr(icp_scores, y)
    spearman_corr = 0.0 if not math.isfinite(spearman_corr) else spearman_corr

    actual_distribution = pd.Series(
        optimization.calculate_grades(pd.Series(icp_scores))
    ).value_counts(normalize=True).reindex(['F', 'D', 'C', 'B', 'A'], fill_value=0.0)
    target_distribution = np.array([
        optimization.TARGET_GRADE_DISTRIBUTION['F'],
        optimization.TARGET_GRADE_DISTRIBUTION['D'],
        optimization.TARGET_GRADE_DISTRIBUTION['C'],
        optimization.TARGET_GRADE_DISTRIBUTION['B'],
        optimization.TARGET_GRADE_DISTRIBUTION['A'],
    ])
    epsilon = 1e-10
    kl_divergence = float(np.sum(
        actual_distribution * np.log((actual_distribution + epsilon) / (target_distribution + epsilon))
    ))

    lift_a = float(optimization._lift_by_grade(pd.Series(icp_scores), y).get('lift@A'))
    stability = optimization._stability_penalty(pd.Series(icp_scores), y, groups)

    expected = (
        lambda_param * kl_divergence
        + (1 - lambda_param) * -spearman_corr
        + stability_weight * stability
        + (-lift_weight) * (lift_a - 1.0)
    )

    assert actual == pytest.approx(expected)


def test_objective_prunes_when_weights_violate_post_normalization_bounds():
    X = pd.DataFrame({
        'vertical_score': [0.1, 0.2],
        'size_score': [0.0, 0.0],
        'adoption_score': [0.3, 0.4],
        'relationship_score': [0.2, 0.1],
    })
    y = pd.Series([0.1, 0.2])
    trial = DummyTrial({
        'vertical_score': 0.45,
        'adoption_score': 0.20,
        'relationship_score': 0.20,
    })

    with pytest.raises(optimization.optuna.exceptions.TrialPruned):
        optimization.objective(
            trial,
            X,
            y,
            lambda_param=0.5,
            weight_names=list(X.columns),
        )
