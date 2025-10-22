import pandas as pd

from icp.schema import canonicalize_customer_id


def test_canonicalize_customer_id_preserves_leading_zeros():
    raw = pd.Series(["00123", "00123.0", "ABC", "123.0 ", " 0456", "789.10"])
    cleaned = canonicalize_customer_id(raw)
    assert cleaned.tolist() == ["00123", "00123", "ABC", "123", "0456", "789.10"]


def test_canonicalize_customer_id_handles_numeric_series():
    raw = pd.Series([123.0, 456.5, 7000.0])
    cleaned = canonicalize_customer_id(raw)
    assert cleaned.tolist() == ["123", "456.5", "7000"]
