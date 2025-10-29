import pandas as pd

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
    setattr(master, '_assets_raw', assets)
    setattr(master, '_profit_rollup_raw', profit_roll)

    # Minimal asset weights (defaults inside function handle missing weights)
    asset_weights = {"focus_goals": ["Printer", "Printer Accessorials", "Scanners", "Geomagic", "Training/Services"], "weights": {}}

    out = engineer_features(master, asset_weights)

    row = out.iloc[0]
    # Printer_count should sum Printer asset_count
    assert row['printer_count'] == 2
    # adoption_profit should include 3DP Training (500) and Printer Accessorials (200), but exclude generic Service (300)
    assert row['adoption_profit'] == 700
    # CRE columns should exist and default to zero for this hardware-only fixture
    assert 'cre_adoption_assets' in out.columns
    assert 'cre_adoption_profit' in out.columns
    assert 'cre_relationship_profit' in out.columns
    assert row['cre_adoption_assets'] == 0
    assert row['cre_adoption_profit'] == 0
    assert row['cre_relationship_profit'] == 0
