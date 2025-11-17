import pandas as pd

from icp.cli import export_call_lists as call_lists
from icp.schema import COL_COMPANY_NAME, COL_CUSTOMER_ID


def sample_scored_accounts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COL_CUSTOMER_ID: "1001.0",
                COL_COMPANY_NAME: "1001 ACME Corp",
                "ICP_grade_hardware": "A",
                "ICP_score_hardware": 88.0,
                "Hardware_score": 0.72,
                "Software_score": 0.61,
                "GP_Since_2023_Total": 250_000,
                "printer_count": 15,
                "customer_segment": "Strategic",
                "territory": "West",
                "am_sales_rep": "A. Johnson",
                "Industry": "High Tech",
                "call_to_action": "Expand hardware fleet",
            },
            {
                COL_CUSTOMER_ID: "1002",
                COL_COMPANY_NAME: "Beta LLC",
                "ICP_grade_hardware": "B",
                "ICP_score_hardware": 81.0,
                "Hardware_score": 0.65,
                "Software_score": 0.82,
                "GP_Since_2023_Total": 95_000,
                "printer_count": 0,
                "customer_segment": "Growth",
                "territory": "Central",
                "am_sales_rep": "B. Singh",
                "Industry": "Medical Devices",
            },
            {
                COL_CUSTOMER_ID: "1003",
                COL_COMPANY_NAME: "Gamma Inc",
                "ICP_grade_hardware": "C",
                "ICP_score_hardware": 60.0,
                "Hardware_score": 0.35,
                "Software_score": 0.42,
                "GP_Since_2023_Total": 40_000,
                "printer_count": 5,
                "customer_segment": "Core",
                "territory": "East",
                "am_sales_rep": "C. Zhu",
                "Industry": "Industrial Machinery",
            },
        ]
    )


def test_normalize_portfolio_generates_expected_columns():
    raw = sample_scored_accounts()
    normalized = call_lists._normalize_portfolio(raw)

    assert normalized[COL_CUSTOMER_ID].tolist()[0] == "1001"
    required = {
        "adoption_band",
        "relationship_band",
        "revenue_only_flag",
        "heavy_fleet_flag",
        "call_to_action",
    }
    assert required.issubset(normalized.columns)


def test_presets_generate_non_empty_tables():
    normalized = call_lists._normalize_portfolio(sample_scored_accounts())

    for preset in (
        call_lists._preset_top_ab_by_segment,
        call_lists._preset_revenue_only_high_relationship,
        call_lists._preset_heavy_fleet_expansion,
    ):
        table, meta = preset(normalized)
        assert "preset" in meta
        assert meta["rows"] == len(table)
        assert isinstance(table, pd.DataFrame)
