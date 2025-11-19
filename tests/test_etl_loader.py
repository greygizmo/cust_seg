import pandas as pd
from unittest.mock import MagicMock, patch
from icp.etl.loader import assemble_master_from_db

@patch('icp.etl.loader.da')
@patch('icp.etl.loader.load_industry_enrichment')
def test_assemble_master_from_db(mock_load_enrichment, mock_da):
    # Mock database returns
    mock_engine = MagicMock()
    mock_da.get_engine.return_value = mock_engine
    
    # Customers
    mock_da.get_customers_since_2023.return_value = pd.DataFrame({
        'Customer ID': ['1001', '1002'],
        'CRM Full Name': ['1001 Acme Corp', '1002 Beta Inc'],
        'Industry': ['Manufacturing', 'High Tech']
    })
    
    # Headers
    mock_da.get_customer_headers.return_value = pd.DataFrame({
        'Customer ID': ['1001', '1002'],
        'am_sales_rep': ['Alice', 'Bob'],
        'AM_Territory': ['West', 'East']
    })
    
    # Assets
    mock_da.get_assets_and_seats.return_value = pd.DataFrame({
        'Customer ID': ['1001'],
        'Goal': 'Printers',
        'asset_count': 5,
        'active_assets': 5,
        'seats_sum': 0,
        'first_purchase_date': '2023-01-01'
    })
    
    # Profit by Goal
    mock_da.get_profit_since_2023_by_goal.return_value = pd.DataFrame({
        'Customer ID': ['1001'],
        'Goal': 'Printers',
        'Profit_Since_2023': 1000
    })
    
    # Profit by Rollup
    mock_da.get_profit_since_2023_by_rollup.return_value = pd.DataFrame({
        'Customer ID': ['1001'],
        'Goal': 'Printers',
        'item_rollup': 'FDM',
        'Profit_Since_2023': 1000
    })

    # Profit by Customer Rollup (for Total)
    mock_da.get_profit_since_2023_by_customer_rollup.return_value = pd.DataFrame({
        'Customer ID': ['1001'],
        'item_rollup': 'FDM',
        'Profit_Since_2023': 1000
    })
    
    # Quarterly Profit (Total)
    mock_da.get_quarterly_profit_total.return_value = pd.DataFrame({
        'Customer ID': ['1001'],
        'Quarter': '2024Q1',
        'Profit': 500
    })
    
    # Mock empty for others to avoid errors
    mock_da.get_primary_contacts.return_value = pd.DataFrame()
    mock_da.get_account_primary_contacts.return_value = pd.DataFrame()
    mock_da.get_customer_shipping.return_value = pd.DataFrame()
    mock_da.get_quarterly_profit_by_goal.return_value = pd.DataFrame() # Fallback to total
    
    # Mock enrichment
    mock_load_enrichment.return_value = pd.DataFrame()
    
    # Run function
    master = assemble_master_from_db()
    
    # Assertions
    assert len(master) == 2
    assert 'Company Name' in master.columns
    assert master.loc[master['Customer ID'] == '1001', 'Company Name'].values[0] == 'Acme Corp'
    assert 'am_sales_rep' in master.columns
    assert master.loc[master['Customer ID'] == '1001', 'active_assets_total'].values[0] == 5
    assert master.loc[master['Customer ID'] == '1001', 'Profit_Since_2023_Total'].values[0] == 1000.0
