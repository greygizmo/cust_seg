import pandas as pd
import pytest

from features.cross_division_signals import compute_cross_division_signals


def test_compute_cross_division_signals_generates_balanced_and_opportunity_scores():
    tx = pd.DataFrame(
        [
            {
                "account_id": "1001",
                "date": "2024-01-15",
                "super_division": "Hardware",
                "division": "Printers",
                "net_revenue": 100.0,
            },
            {
                "account_id": "1001",
                "date": "2024-01-10",
                "super_division": "Software",
                "division": "CAD",
                "net_revenue": 50.0,
            },
            {
                "account_id": "1001",
                "date": "2023-12-10",
                "super_division": "Hardware",
                "division": "Training/Services",
                "net_revenue": 20.0,
            },
            {
                "account_id": "1001",
                "date": "2023-10-01",
                "super_division": "Hardware",
                "division": "Printers",
                "net_revenue": 60.0,
            },
            {
                "account_id": "2001",
                "date": "2024-01-12",
                "super_division": "Software",
                "division": "CPE",
                "net_revenue": 80.0,
            },
        ]
    )

    out = compute_cross_division_signals(tx, pd.Timestamp("2024-01-31"))

    row_1001 = out[out["account_id"] == "1001"].iloc[0]
    row_2001 = out[out["account_id"] == "2001"].iloc[0]

    assert row_1001["hw_spend_13w"] == 120.0
    assert row_1001["hw_spend_13w_prior"] == 60.0
    assert row_1001["sw_spend_13w"] == 50.0
    assert row_1001["training_to_hw_ratio"] == pytest.approx(20.0 / 180.0)
    assert row_1001["training_to_cre_ratio"] == pytest.approx(20.0 / 50.0)
    assert row_1001["division_breadth_12m"] == 3
    assert row_1001["super_division_breadth_12m"] == 2
    assert row_1001["software_division_breadth_12m"] == 1
    assert row_1001["hw_to_sw_cross_sell_score"] > 0
    assert row_1001["sw_to_hw_cross_sell_score"] == 0

    assert row_2001["hw_spend_13w"] == 0
    assert row_2001["sw_spend_13w"] == 80.0
    assert row_2001["sw_to_hw_cross_sell_score"] > 0

