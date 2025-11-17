import pandas as pd
import pytest

from goe_icp_scoring import engineer_features


def test_engineer_features_training_filter_and_printer_count():
    # Minimal master frame with one customer
    master = pd.DataFrame({'Customer ID': ['1001'], 'Industry': ['High Tech']})

    # Assets: include a Printer asset_count to derive printer_count
    assets = pd.DataFrame([
        {
            'Customer ID': '1001',
            'Goal': 'Printer',
            'item_rollup': 'FDM',
            'asset_count': 2,
            'seats_sum': 0,
        }
    ])

    # Profit by rollup: include multiple goals with Training/Services specific rollups
    profit_roll = pd.DataFrame([
        {'Customer ID': '1001', 'Goal': 'Training/Services', 'item_rollup': '3DP Training', 'Profit_Since_2023': 500},
        {'Customer ID': '1001', 'Goal': 'Training/Services', 'item_rollup': 'Service', 'Profit_Since_2023': 300},  # should be excluded
        {'Customer ID': '1001', 'Goal': 'Printer Accessorials', 'item_rollup': 'Consumables', 'Profit_Since_2023': 200},
    ])

    # Attach raw attributes as expected by engineer_features
    master.attrs['_assets_raw'] = assets
    master.attrs['_profit_rollup_raw'] = profit_roll

    # Minimal asset weights (defaults inside function handle missing weights)
    asset_weights = {"focus_goals": ["Printer", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"], "weights": {}}

    out = engineer_features(master, asset_weights)

    row = out.iloc[0]
    # Printer_count should sum Printer asset_count
    assert row['printer_count'] == 2
    # adoption_profit should include 3DP Training (500) and Printer Accessorials (200), but exclude generic Service (300)
    assert row['adoption_profit'] == 700
    # CRE columns should exist; small single-row fixtures yield a percentile of 1.0
    # for the blended relationship metric because all ranks collapse to 100%.
    assert 'cre_adoption_assets' in out.columns
    assert 'cre_adoption_profit' in out.columns
    assert 'cre_relationship_profit' in out.columns
    assert row['cre_adoption_assets'] == 0
    assert row['cre_adoption_profit'] == 0
    assert row['cre_relationship_profit'] == pytest.approx(1.0)


def test_engineer_features_relationship_profit_normalizes_goal_case():
    master = pd.DataFrame({'Customer ID': ['C-001'], 'Industry': ['Healthcare']})

    profit_roll = pd.DataFrame([
        {'Customer ID': 'C-001', 'Goal': 'cad', 'item_rollup': 'Software', 'Profit_Since_2023': 150},
        {'Customer ID': 'C-001', 'Goal': 'specialty software', 'item_rollup': 'Nesting', 'Profit_Since_2023': 250},
        {'Customer ID': 'C-001', 'Goal': 'printers', 'item_rollup': 'FDM', 'Profit_Since_2023': 50},
    ])

    master.attrs['_profit_rollup_raw'] = profit_roll

    asset_weights = {"focus_goals": ["Printers", "CAD", "Specialty Software"], "weights": {}}

    out = engineer_features(master, asset_weights)

    row = out.iloc[0]
    assert row['relationship_profit'] == 400


def test_engineer_features_uses_gp_history_from_attrs():
    master = pd.DataFrame({'Customer ID': ['cust-99'], 'Industry': ['Manufacturing']})

    gp_last90 = pd.DataFrame([
        {'Customer ID': 'cust-99', 'GP_Last_ND': 120},
        {'Customer ID': 'cust-99', 'GP_Last_ND': 30},
    ])

    gp_monthly12 = pd.DataFrame([
        {'Customer ID': 'cust-99', 'Year': 2024, 'Month': 1, 'Profit': 0},
        {'Customer ID': 'cust-99', 'Year': 2024, 'Month': 2, 'Profit': 50},
        {'Customer ID': 'cust-99', 'Year': 2024, 'Month': 3, 'Profit': 150},
    ])

    master.attrs['_gp_last90'] = gp_last90
    master.attrs['_gp_monthly12'] = gp_monthly12

    out = engineer_features(master, {"focus_goals": [], "weights": {}})

    row = out.iloc[0]
    assert row['GP_Last_90D'] == pytest.approx(150)
    assert row['Months_Active_12M'] == 2
    assert row['GP_Trend_Slope_12M'] == pytest.approx(75.0)
