import pandas as pd
import pytest

from icp.industry import apply_empirical_bayes_shrinkage, build_industry_weights
from icp.schema import COL_INDUSTRY


def test_apply_empirical_bayes_shrinkage_shrinks_small_samples():
    stats = pd.DataFrame(
        {
            "Industry_clean": ["A", "B"],
            "customer_count": [5, 100],
            "success_metric": [100.0, 50.0],
        }
    )

    result = apply_empirical_bayes_shrinkage(stats, k=20)

    small = result.loc[result["Industry_clean"] == "A", "shrunk_mean"].iloc[0]
    large = result.loc[result["Industry_clean"] == "B", "shrunk_mean"].iloc[0]

    # The small sample should move substantially toward the pooled mean,
    # whereas the large sample should remain close to its original value.
    assert small < 100.0
    assert pytest.approx(50.0, abs=1) == large
    assert abs(100.0 - small) > abs(50.0 - large)


def test_build_industry_weights_blends_data_and_strategic(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            COL_INDUSTRY: ["High Tech"] * 12 + ["Services"] * 12,
            "Printers": [10] * 24,
            "Printer Accessorials": [0] * 24,
            "Scanners": [0] * 24,
            "Geomagic": [0] * 24,
            "Training/Services": [0] * 24,
        }
    )

    weights = build_industry_weights(df, division="hardware", min_sample=5, k=10)

    assert "high tech" in weights
    assert "services" in weights
    assert weights["unknown"] == pytest.approx(0.30, abs=0.05)
    assert weights["high tech"] >= weights["services"]


def test_build_industry_weights_ignores_small_samples():
    df = pd.DataFrame(
        {
            COL_INDUSTRY: ["Tiny"] * 3 + ["High Tech"] * 15,
            "Printers": [0] * 18,
            "Printer Accessorials": [0] * 18,
            "Scanners": [0] * 18,
            "Geomagic": [0] * 18,
            "Training/Services": [0] * 18,
            "Profit_Since_2023_Total": [1000] * 18,
        }
    )

    weights = build_industry_weights(df, division="hardware", min_sample=10, k=5)

    assert "tiny" not in weights
    assert weights["unknown"] == pytest.approx(0.30, abs=0.05)
