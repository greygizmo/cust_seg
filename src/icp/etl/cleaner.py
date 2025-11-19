"""Data cleaning utilities."""
import pandas as pd
import re

def clean_name(x: str) -> str:
    """
    Aggressively normalizes a company name for matching purposes.
    - Converts to lowercase.
    - Removes leading customer ID numbers.
    - Removes common punctuation and extra spaces.
    """
    if pd.isna(x):
        return ""
    x = str(x).lower()
    
    # Remove leading customer ID numbers (e.g., "123456 Company Name" -> "Company Name") 
    x = re.sub(r'^\d+\s+', '', x)
    
    junk = {",", ".", "&", "  "}
    for j in junk:
        x = x.replace(j, " ")
    return " ".join(x.split())
