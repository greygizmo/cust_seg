import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
from icp.cli.generate_weights import generate_weights

@pytest.fixture
def mock_engine():
    with patch("icp.data_access.get_engine") as mock:
        yield mock

def test_generate_weights_creates_file(mock_engine, tmp_path):
    # Setup mock data
    mock_df = pd.DataFrame({
        "item_rollup": ["Rollup A", "Rollup B"],
        "Goal": ["Printer", "Printer"]
    })
    
    # Mock pd.read_sql to return our dataframe
    with patch("pandas.read_sql", return_value=mock_df):
        out_file = tmp_path / "weights.json"
        generate_weights(out_file)
        
        assert out_file.exists()
        assert "Printer" in out_file.read_text()
        assert "Rollup A" in out_file.read_text()

def test_generate_weights_handles_empty_db(mock_engine, tmp_path):
    # Setup empty mock data
    mock_df = pd.DataFrame(columns=["item_rollup", "Goal"])
    
    with patch("pandas.read_sql", return_value=mock_df):
        out_file = tmp_path / "weights.json"
        generate_weights(out_file)
        
        assert out_file.exists()
        # Should still have keys for focus goals
        assert "Printer" in out_file.read_text()
