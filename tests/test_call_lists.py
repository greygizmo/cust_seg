import pandas as pd
from icp.reporting.call_lists import normalize_portfolio, preset_top_ab_by_segment

def test_normalize_portfolio_defaults():
    df = pd.DataFrame({
        "Customer ID": ["1"],
        "Company Name": ["Acme"],
        "profit_since_2023": [1000]
    })
    
    norm = normalize_portfolio(df)
    
    assert "grade" in norm.columns
    assert norm.iloc[0]["grade"] == "C" # Default
    assert "score" in norm.columns
    assert norm.iloc[0]["score"] == 50.0 # Default
    assert "customer_segment" in norm.columns
    # With only 1 row, quantile logic might be weird but it should assign something
    assert norm.iloc[0]["customer_segment"] != "Unassigned"

def test_preset_top_ab_by_segment():
    df = pd.DataFrame({
        "Customer ID": ["1", "2", "3"],
        "Company Name": ["A", "B", "C"],
        "grade": ["A", "B", "C"],
        "score": [90, 80, 70],
        "profit_since_2023": [1000, 500, 100],
        "customer_segment": ["Core", "Core", "Core"],
        "printer_count": [0, 0, 0],
        "adoption_score": [0.5, 0.5, 0.5],
        "relationship_score": [0.5, 0.5, 0.5],
        "territory": ["West", "East", "West"],
        "sales_rep": ["Alice", "Bob", "Charlie"],
        "call_to_action": ["Call", "Call", "Call"]
    })
    
    # Should filter to A and B only
    table, meta = preset_top_ab_by_segment(df)
    
    assert len(table) == 2
    assert "1" in table["Customer ID"].values
    assert "2" in table["Customer ID"].values
    assert "3" not in table["Customer ID"].values
