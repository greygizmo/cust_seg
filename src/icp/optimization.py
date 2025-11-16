import optuna
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

# Target A–F distribution
TARGET_GRADE_DISTRIBUTION = {
    'A': 0.10,
    'B': 0.20,
    'C': 0.40,
    'D': 0.20,
    'F': 0.10,
}
TARGET_CUMULATIVE_DISTRIBUTION = np.cumsum([
    TARGET_GRADE_DISTRIBUTION['F'],
    TARGET_GRADE_DISTRIBUTION['D'],
    TARGET_GRADE_DISTRIBUTION['C'],
    TARGET_GRADE_DISTRIBUTION['B'],
    TARGET_GRADE_DISTRIBUTION['A'],
])


def calculate_grades(scores: pd.Series):
    ranks = scores.rank(pct=True)
    grades = np.select(
        [
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[0],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[1],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[2],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[3],
            ranks > TARGET_CUMULATIVE_DISTRIBUTION[3],
        ],
        ['F', 'D', 'C', 'B', 'A'],
        default='C',
    )
    return grades


def _assign_grades(scores: pd.Series) -> pd.Series:
    ranks = scores.rank(pct=True)
    return pd.Series(np.select(
        [
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[0],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[1],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[2],
            ranks <= TARGET_CUMULATIVE_DISTRIBUTION[3],
            ranks > TARGET_CUMULATIVE_DISTRIBUTION[3],
        ],
        ['F','D','C','B','A'],
        default='C'
    ), index=scores.index)


def _lift_by_grade(scores: pd.Series, y: pd.Series) -> dict:
    """Per-grade lift relative to baseline share (e.g., lift@A)."""
    grades = _assign_grades(scores)
    total = float(pd.to_numeric(y, errors='coerce').fillna(0).sum()) or 1.0
    out = {}
    for g, frac in [('A',0.10),('B',0.20),('C',0.40),('D',0.20),('F',0.10)]:
        yy = pd.to_numeric(y[grades==g], errors='coerce').fillna(0).sum()
        denom = max(frac, 1e-9) * total
        out[f'lift@{g}'] = float(yy) / denom if denom > 0 else 1.0
    return out


def _stability_penalty(scores: pd.Series, y: pd.Series, groups: pd.Series | None, min_n: int = 50) -> float:
    """Std of within-group Spearman correlations (lower is better)."""
    if groups is None:
        return 0.0
    df = pd.DataFrame({'s': scores, 'y': y, 'g': groups}).dropna(subset=['s','y'])
    stats = []
    for g, sub in df.groupby('g'):
        if len(sub) >= min_n and sub['y'].abs().sum() > 0:
            try:
                r, _ = spearmanr(sub['s'], sub['y'])
                if np.isfinite(r):
                    stats.append(float(r))
            except Exception:
                continue
    if len(stats) < 2:
        return 0.0
    return float(np.nanstd(stats))


def cumulative_lift(scores: pd.Series, y: pd.Series, steps: list[float] | None = None) -> dict:
    """Compute cumulative lift curve data.

    Returns a dict with keys:
      - points: list of {"pop_share": p, "cum_y_share": s, "lift": s / p}
      - auc: trapezoidal area under cum_y_share vs pop_share curve (baseline=0.5)
    """
    if steps is None:
        steps = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    df = pd.DataFrame({"s": scores, "y": pd.to_numeric(y, errors='coerce').fillna(0)})
    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    total_y = float(df["y"].sum()) or 1.0
    n = len(df)
    cum = df["y"].cumsum()
    points = []
    for p in steps:
        k = min(n - 1, max(0, int(np.floor(p * n)) - 1))
        cum_y_share = float(cum.iloc[k]) / total_y
        lift = (cum_y_share / max(p, 1e-9)) if p > 0 else 1.0
        points.append({"pop_share": round(p, 4), "cum_y_share": round(cum_y_share, 6), "lift": round(lift, 6)})
    # AUC via trapezoid rule
    xs = [pt["pop_share"] for pt in points]
    ys = [pt["cum_y_share"] for pt in points]
    auc = 0.0
    for i in range(1, len(xs)):
        auc += 0.5 * (ys[i] + ys[i-1]) * (xs[i] - xs[i-1])
    return {"points": points, "auc": round(auc, 6)}


def objective(
    trial,
    X,
    y,
    lambda_param,
    weight_names,
    include_size: bool = False,
    groups=None,
    lift_weight: float = 0.05,
    stability_weight: float = 0.05,
):
    """Optuna objective for weight tuning (no ML).

    - Weights: vertical, adoption, relationship. Size locked to 0 when include_size=False.
    - Bounds (common for all divisions) to avoid overfit:
        vertical in [0.15, 0.45], adoption in [0.20, 0.55], relationship in [0.20, 0.55].
    - Objective: lambda*KL(grade||target) - (1-lambda)*Spearman + w_stab*stability - w_lift*(lift@A-1).
    """
    # Common conservative bounds (size component removed)
    v_min, v_max = 0.15, 0.45
    a_min, a_max = 0.20, 0.55
    r_min, r_max = 0.20, 0.55

    w_vertical = trial.suggest_float('vertical_score', v_min, v_max)
    w_adoption = trial.suggest_float('adoption_score', a_min, a_max)
    w_relationship = trial.suggest_float('relationship_score', r_min, r_max)

    total = w_vertical + w_adoption + w_relationship
    if total <= 0:
        raise optuna.exceptions.TrialPruned()
    w_vertical /= total
    w_adoption /= total
    w_relationship /= total
    if not (v_min <= w_vertical <= v_max and a_min <= w_adoption <= a_max and r_min <= w_relationship <= r_max):
        raise optuna.exceptions.TrialPruned()

    weights_dict = {
        'vertical_score': w_vertical,
        'adoption_score': w_adoption,
        'relationship_score': w_relationship,
    }
    # Any unknown components (e.g., legacy size_score) get weight 0.0
    weights = np.array([weights_dict.get(name, 0.0) for name in X.columns])

    # Score with linear blend (no ML)
    icp_scores = X.dot(weights)

    # Predictive strength: Spearman corr with outcome
    spearman_corr, _ = spearmanr(icp_scores, y)
    if not np.isfinite(spearman_corr):
        spearman_corr = 0.0

    # KL divergence to target grade distribution
    actual_grades = calculate_grades(icp_scores)
    actual_distribution = pd.Series(actual_grades).value_counts(normalize=True).reindex(
        ['F','D','C','B','A'], fill_value=0
    )
    target_distribution = np.array([
        TARGET_GRADE_DISTRIBUTION['F'],
        TARGET_GRADE_DISTRIBUTION['D'],
        TARGET_GRADE_DISTRIBUTION['C'],
        TARGET_GRADE_DISTRIBUTION['B'],
        TARGET_GRADE_DISTRIBUTION['A'],
    ])
    epsilon = 1e-10
    kl_divergence = float(np.sum(
        actual_distribution * np.log((actual_distribution + epsilon) / (target_distribution + epsilon))
    ))

    # Per-grade lift and stability
    liftA = float(_lift_by_grade(icp_scores, y).get('lift@A', 1.0))
    stab = _stability_penalty(icp_scores, y, groups)

    combined = (
        (lambda_param * kl_divergence)
        + ((1 - lambda_param) * -spearman_corr)
        + stability_weight * stab
        + (-lift_weight) * (liftA - 1.0)
    )
    return combined
