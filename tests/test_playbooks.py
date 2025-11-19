import pandas as pd
from icp.playbooks.logic import flag_heroes, compute_neighbor_flags, derive_playbooks

def test_flag_heroes():
    df = pd.DataFrame({
        "ICP_grade_hardware": ["A", "B", "A", "C"],
        "ICP_grade_cre": ["C", "A", "B", "A"],
        "GP_Since_2023_Total": [1000, 1000, 100, 1000]
    })
    # Mock quantile to return 500
    # But we can't easily mock pandas quantile inside the function without patching
    # So let's just rely on the logic: 80th percentile of [1000, 1000, 100, 1000] is 1000.
    
    heroes = flag_heroes(df)
    # Row 0: HW A, GP 1000 >= 1000 -> Hero
    # Row 1: CRE A, GP 1000 >= 1000 -> Hero
    # Row 2: HW A, GP 100 < 1000 -> Not Hero
    # Row 3: CRE A, GP 1000 >= 1000 -> Hero
    
    assert heroes.tolist() == [True, True, False, True]

def test_compute_neighbor_flags_empty_neighbors():
    scored = pd.DataFrame({
        "customer_id": ["1", "2"],
        "ICP_grade_hardware": ["A", "C"],
        "GP_Since_2023_Total": [1000, 100]
    })
    neighbors = pd.DataFrame()
    
    result = compute_neighbor_flags(scored, neighbors)
    assert "is_hero" in result.columns
    assert not result["is_hero_neighbor"].any()

def test_derive_playbooks():
    scored = pd.DataFrame({
        "customer_id": ["1"],
        "Customer ID": ["1"],
        "Company Name": ["Acme"],
        "ICP_grade_hardware": ["A"],
        "ICP_grade_cre": ["C"],
        "GP_Since_2023_Total": [1000],
        "whitespace_score": [0.5],
        "momentum_score": [0.6],
        "is_hero": [True],
        "is_hero_neighbor": [False],
        "is_hero_orphan_neighbor": [False],
        "hero_neighbor_count": [0]
    })
    
    # This should trigger HW Expansion Sprint
    # mask_hw_expansion = (hw_grade is A/B) & (whitespace >= 0.45) & (momentum >= 0.5) & (gp >= median)
    
    playbooks = derive_playbooks(scored)
    assert playbooks.iloc[0]["playbook_primary"] == "HW Expansion Sprint"
