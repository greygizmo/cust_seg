import pandas as pd

from goe_icp_scoring import apply_industry_enrichment


def test_enrichment_matches_by_id_then_name_then_cleaned_name():
    # Base customers as produced from SQL layer
    customers = pd.DataFrame({
        'Customer ID': ['1001.0', '1002', '1003'],
        'CRM Full Name': ['1001 ACME Corp', '1002 Beta LLC', '1003 Gamma Inc'],
        'Company Name': ['ACME Corp', 'Beta LLC', 'Gamma Inc'],
    })

    # Enrichment data with different matching possibilities
    enrichment = pd.DataFrame({
        # ID-based match for 1001 (note: canonicalization must handle string vs float-like)
        'Customer ID': ['1001', None, None],
        'Industry': ['High Tech', 'Aerospace & Defense', 'Industrial Machinery'],
        'Industry Sub List': ['07.6 Semiconductors', '02.1 Aircraft', '04.5 Other Industrial Machinery'],
        # Name-based match for 1002 using CRM Full Name
        'CRM Full Name': [None, '1002 Beta LLC', '1003 Gamma Inc'],
        # Also include a cleaned field to emulate dataset completeness
        'Cleaned Customer Name': [None, None, 'gamma inc'],
    })

    out = apply_industry_enrichment(customers.copy(), enrichment.copy())

    # All three should be populated via their respective strategies
    row1 = out[out['Customer ID'] == '1001'].iloc[0]
    row2 = out[out['Customer ID'] == '1002'].iloc[0]
    row3 = out[out['Customer ID'] == '1003'].iloc[0]

    assert row1['Industry'] == 'High Tech'
    assert row2['Industry'] == 'Aerospace & Defense'
    assert row3['Industry'] == 'Industrial Machinery'
