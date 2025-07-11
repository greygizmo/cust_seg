"""
Company name normalization utilities
"""
import re
import unicodedata
from typing import Optional


def clean_company_name(name: str) -> str:
    """
    Clean and normalize a company name for consistent matching.
    
    Args:
        name: Raw company name string
        
    Returns:
        Cleaned lowercase company name
    """
    if not name or name.strip() == "":
        return ""
    
    # Normalize unicode characters
    n = unicodedata.normalize("NFKD", name)
    
    # Remove Customer ID prefix if present (e.g., "10005 Santa Rosa Junior College" → "Santa Rosa Junior College")
    # Pattern: one or more digits followed by space at the beginning
    n = re.sub(r"^\d+\s+", "", n)
    
    # Remove punctuation and special characters
    n = re.sub(r"[^\w\s]", "", n)
    
    # Remove common company suffixes
    suffixes = [
        r"\binc\b", r"\bincorporated\b", r"\bllc\b", r"\bltd\b", 
        r"\bcorp\b", r"\bcorporation\b", r"\bcompany\b", r"\bco\b",
        r"\blimited\b", r"\bltd\b", r"\bp\.?c\.?\b", r"\bl\.?p\.?\b"
    ]
    
    for suffix in suffixes:
        n = re.sub(suffix, "", n, flags=re.IGNORECASE)
    
    # Collapse multiple spaces and strip
    n = re.sub(r"\s{2,}", " ", n).strip()
    
    return n.lower()


def extract_domain_from_url(url: str) -> str:
    """
    Extract a clean domain from a URL.
    
    Args:
        url: Raw URL string
        
    Returns:
        Clean domain string
    """
    if not url or url.strip() == "":
        return ""
    
    # Remove protocol
    domain = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    
    # Remove www prefix
    domain = re.sub(r"^www\.", "", domain, flags=re.IGNORECASE)
    
    # Remove path and query parameters
    domain = re.sub(r"/.*$", "", domain)
    
    # Remove port numbers
    domain = re.sub(r":\d+$", "", domain)
    
    return domain.lower().strip()


def is_valid_domain(domain: str) -> bool:
    """
    Check if a domain looks valid.
    
    Args:
        domain: Domain string to validate
        
    Returns:
        True if domain appears valid
    """
    if not domain or len(domain) < 3:
        return False
    
    # Basic domain pattern check
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}$"
    
    return bool(re.match(pattern, domain))


def normalize_name_for_matching(name: str) -> str:
    """
    Advanced normalization specifically for fuzzy matching.
    Strips location details, legal suffixes, and punctuation.
    """
    if not isinstance(name, str):
        return ""

    # Convert to lowercase
    n = name.lower()

    # Remove location details often separated by a hyphen or other delimiters
    # e.g., "Apple Inc - HQ Cupertino" -> "apple inc"
    n = re.split(r'\s-\s|\s–\s|–', n)[0]

    # Remove common legal suffixes and abbreviations
    # More comprehensive list
    suffixes = [
        'incorporated', 'corporation', 'company', 'limited', 'association',
        'inc', 'corp', 'co', 'ltd', 'llc', 'lp', 'llp', 'pllc', 'pc',
        'gmbh', 'ag', 'sas', 'sarl', 'bv', 'nv', 'pty'
    ]
    # This regex ensures we only match whole words
    suffix_pattern = r'\b(' + r'|'.join(suffixes) + r')\b'
    n = re.sub(suffix_pattern, '', n)

    # Remove all non-alphanumeric characters (except spaces)
    n = re.sub(r'[^\w\s]', '', n)

    # Collapse multiple spaces into a single space and strip whitespace
    n = re.sub(r'\s+', ' ', n).strip()

    return n 