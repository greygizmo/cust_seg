"""
cleanup_industry_data.py
------------------------
Single-use script to clean up industry data in "TR - Industry Enrichment.csv"
using fuzzy matching to standardize categories according to industry_fields.txt.

This script will:
1. Parse target categories from industry_fields.txt
2. Apply multi-level fuzzy matching to current data
3. Generate mapping report for review
4. Update the CSV with standardized categories
5. Create backup and detailed logs
"""

import pandas as pd
import re
import json
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict
import shutil
import os
import argparse


def parse_industry_fields(filename="industry_fields.txt"):
    """
    Parse the target industry categories from industry_fields.txt
    
    Returns:
        tuple: (industries_list, industry_sublist_mapping)
    """
    print(f"[INFO] Parsing target categories from {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract main industries
    industries_section = re.search(r'## Industries\s*\n(.*?)## Industry Sub Lists', content, re.DOTALL)
    industries = []
    if industries_section:
        industry_lines = industries_section.group(1).strip().split('\n')
        for line in industry_lines:
            line = line.strip()
            if line.startswith('- ') and not line.startswith('- **'):
                industry = line[2:].strip()
                industries.append(industry)
    
    # Extract industry sub lists with their parent mapping
    sublist_mapping = {}
    sublist_section = re.search(r'## Industry Sub Lists.*', content, re.DOTALL)
    if sublist_section:
        current_industry = None
        lines = sublist_section.group(0).split('\n')
        
        for line in lines:
            line = line.strip()
            # Parent industry
            if line.startswith('- **') and line.endswith('**'):
                current_industry = line[4:-2].strip()  # Remove - ** and **
            # Sub-category
            elif line.startswith('- ') and current_industry:
                subcat = line[2:].strip()  # Remove "- "
                sublist_mapping[subcat] = current_industry
    
    print(f"[INFO] Found {len(industries)} target industries")
    print(f"[INFO] Found {len(sublist_mapping)} target sub-categories")
    
    return industries, sublist_mapping


def normalize_string(s):
    """Normalize string for better matching"""
    if pd.isna(s):
        return ""
    
    # Convert to lowercase and strip
    s = str(s).lower().strip()
    
    # Remove common variations
    s = re.sub(r'[&\-\(\)\[\],\.]+', ' ', s)  # Replace punctuation with spaces
    s = re.sub(r'\s+', ' ', s)  # Multiple spaces to single
    s = s.strip()
    
    return s


def calculate_similarity(s1, s2):
    """Calculate similarity between two strings"""
    s1_norm = normalize_string(s1)
    s2_norm = normalize_string(s2)
    
    if not s1_norm or not s2_norm:
        return 0.0
    
    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, s1_norm, s2_norm).ratio()


def keyword_match_score(current, target):
    """Calculate keyword-based matching score"""
    current_norm = normalize_string(current)
    target_norm = normalize_string(target)
    
    if not current_norm or not target_norm:
        return 0.0
    
    current_words = set(current_norm.split())
    target_words = set(target_norm.split())
    
    if not current_words or not target_words:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(current_words.intersection(target_words))
    union = len(current_words.union(target_words))
    
    return intersection / union if union > 0 else 0.0


def find_best_match(current_value, target_list, min_confidence=0.6):
    """
    Find the best match for a current value in the target list
    
    Args:
        current_value: The value to match
        target_list: List of target values to match against
        min_confidence: Minimum confidence threshold
        
    Returns:
        tuple: (best_match, confidence_score, method)
    """
    if pd.isna(current_value) or not current_value.strip():
        return None, 0.0, "empty"
    
    current_clean = current_value.strip()
    best_match = None
    best_score = 0.0
    best_method = "none"
    
    for target in target_list:
        # Method 1: Direct match (case-insensitive)
        if current_clean.lower() == target.lower():
            return target, 1.0, "exact"
        
        # Method 2: Fuzzy string similarity
        fuzzy_score = calculate_similarity(current_clean, target)
        if fuzzy_score > best_score:
            best_score = fuzzy_score
            best_match = target
            best_method = "fuzzy"
        
        # Method 3: Keyword matching
        keyword_score = keyword_match_score(current_clean, target)
        if keyword_score > best_score:
            best_score = keyword_score
            best_match = target
            best_method = "keyword"
    
    # Return result only if confidence is above threshold
    if best_score >= min_confidence:
        return best_match, best_score, best_method
    
    return None, best_score, "low_confidence"


def create_custom_mappings():
    """
    Create custom mappings for known problematic cases
    
    Returns:
        dict: Manual mappings for industry names
    """
    return {
        # General & High-Level
        "manufacturing": "Manufactured Products",
        "services": "Services",
        "tech": "High Tech",
        "technology": "High Tech",
        "industrial": "Industrial Machinery",
        "industrials": "Industrial Machinery",
        
        # Specific Keywords
        "automotive": "Automotive & Transportation",
        "auto": "Automotive & Transportation",
        "transport": "Automotive & Transportation",
        "logistics": "Automotive & Transportation",
        "construction": "Building & Construction",
        "building": "Building & Construction",
        "real estate": "Building & Construction",
        "defense": "Aerospace & Defense",
        "aerospace": "Aerospace & Defense",
        "medical": "Medical Devices & Life Sciences",
        "pharmaceutical": "Medical Devices & Life Sciences",
        "pharma": "Medical Devices & Life Sciences",
        "chemicals": "Chemicals & Related Products",
        "chemical": "Chemicals & Related Products",
        "machinery": "Industrial Machinery",
        "equipment": "Heavy Equip & Ind. Components",
        "tooling": "Mold, Tool & Die",
        "die": "Mold, Tool & Die",
        "mold": "Mold, Tool & Die",
        "plant": "Plant & Process",
        "process": "Plant & Process",
        "energy": "Energy",
        "oil": "Energy",
        "gas": "Energy",
        "petroleum": "Energy",
        "oil & gas": "Energy",
        "mining": "Plant & Process",
        "agriculture": "Plant & Process",
        "forestry": "Plant & Process",
        "farming": "Plant & Process",
        "fishing": "Plant & Process",
        "hunting": "Plant & Process",
        "food": "Consumer Goods",
        "beverage": "Consumer Goods",
        "consumer": "Consumer Goods",
        "sports": "Consumer Goods",
        "packaging": "Packaging",
        "containers": "Packaging",

        # Service-based industries
        "advertising": "Services",
        "marketing": "Services",
        "media": "Services",
        "entertainment": "Services",
        "finance": "Services",
        "financial": "Services",
        "financials": "Services",
        "financial services": "Services",
        "legal": "Services",
        "professional services": "Services",
        "commercial services": "Services",
        "government": "Services",
        "retail": "Services",
        "wholesale": "Services",
        "merchant wholesalers": "Services",
        
        # Catch-alls
        "conglomerate": "Manufactured Products",
        
        # Specific failing phrases
        "advertising & marketing": "Services",
        "agriculture & forestry": "Plant & Process",
        "agriculture, forestry, fishing and hunting": "Plant & Process",
        "food & beverage": "Consumer Goods",
        "industrial machinery and equipment merchant wholesalers": "Services",
        "manufacturing, transport & logistics": "Automotive & Transportation",
        "media & entertainment": "Services",
        "oil & gas": "Energy",
        "professional & commercial services": "Services",
        "retail & wholesale trade": "Services",
    }


def analyze_current_data(df, target_industries, target_sublists):
    """
    Analyze current data and create mapping suggestions
    
    Returns:
        dict: Analysis results with mapping suggestions
    """
    print("[INFO] Analyzing current industry data...")
    
    # Get unique values
    current_industries = df['Industry'].dropna().unique()
    current_sublists = df['Industry Sub List'].dropna().unique()
    
    print(f"[INFO] Current data has {len(current_industries)} unique industries")
    print(f"[INFO] Current data has {len(current_sublists)} unique sub-lists")
    
    # Create mappings
    industry_mappings = {}
    sublist_mappings = {}
    
    # Custom mappings
    custom_mappings = create_custom_mappings()
    
    # Map industries
    print("[INFO] Mapping industries...")
    for current in current_industries:
        # Check custom mappings first - both exact and keyword-based
        custom_match = None
        current_normalized = normalize_string(current)
        
        # Method 1: Check for exact normalized match
        if current_normalized in custom_mappings:
            custom_match = custom_mappings[current_normalized]
        
        # Method 2: Check for keyword-based match (any word in the industry name)
        if not custom_match:
            current_words = current_normalized.split()
            for word in current_words:
                if word in custom_mappings:
                    custom_match = custom_mappings[word]
                    break
        
        # Method 3: Check for partial phrase matches (useful for "oil & gas" etc.)
        if not custom_match:
            for keyword, target in custom_mappings.items():
                if keyword in current_normalized:
                    custom_match = target
                    break
        
        if custom_match:
            industry_mappings[current] = (custom_match, 1.0, "custom")
        else:
            match, score, method = find_best_match(current, target_industries)
            industry_mappings[current] = (match, score, method)
    
    # Map sub-lists
    print("[INFO] Mapping sub-lists...")
    all_target_sublists = list(target_sublists.keys())
    for current in current_sublists:
        match, score, method = find_best_match(current, all_target_sublists)
        sublist_mappings[current] = (match, score, method)
    
    return {
        'industry_mappings': industry_mappings,
        'sublist_mappings': sublist_mappings,
        'target_industries': target_industries,
        'target_sublists': target_sublists
    }


def generate_mapping_report(analysis_results, output_file="mapping_report.txt"):
    """Generate a detailed mapping report for review"""
    print(f"[INFO] Generating mapping report: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INDUSTRY DATA CLEANUP MAPPING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Industry mappings
        f.write("INDUSTRY MAPPINGS\n")
        f.write("-" * 20 + "\n\n")
        
        industry_mappings = analysis_results['industry_mappings']
        
        # High confidence mappings
        f.write("HIGH CONFIDENCE MAPPINGS (≥0.8):\n")
        high_conf = [(k, v) for k, v in industry_mappings.items() if v[1] >= 0.8]
        for current, (target, score, method) in sorted(high_conf):
            f.write(f"  '{current}' → '{target}' (confidence: {score:.2f}, method: {method})\n")
        
        # Medium confidence mappings
        f.write(f"\nMEDIUM CONFIDENCE MAPPINGS (0.6-0.8):\n")
        med_conf = [(k, v) for k, v in industry_mappings.items() if 0.6 <= v[1] < 0.8]
        for current, (target, score, method) in sorted(med_conf):
            f.write(f"  '{current}' → '{target}' (confidence: {score:.2f}, method: {method})\n")
        
        # Low confidence / unmapped
        f.write(f"\nLOW CONFIDENCE / UNMAPPED:\n")
        low_conf = [(k, v) for k, v in industry_mappings.items() if v[1] < 0.6]
        for current, (target, score, method) in sorted(low_conf):
            f.write(f"  '{current}' → NO MATCH (best: '{target}', confidence: {score:.2f})\n")
        
        # Sub-list mappings
        f.write(f"\n\nSUB-LIST MAPPINGS\n")
        f.write("-" * 20 + "\n\n")
        
        sublist_mappings = analysis_results['sublist_mappings']
        
        # Show sample mappings (first 20)
        f.write("SAMPLE SUB-LIST MAPPINGS (first 20):\n")
        sample_sublists = list(sublist_mappings.items())[:20]
        for current, (target, score, method) in sample_sublists:
            status = "✓" if score >= 0.6 else "✗"
            f.write(f"  {status} '{current}' → '{target}' (confidence: {score:.2f})\n")
        
        # Summary statistics
        f.write(f"\n\nSUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        
        total_industries = len(industry_mappings)
        mapped_industries = len([v for v in industry_mappings.values() if v[1] >= 0.6])
        
        total_sublists = len(sublist_mappings)
        mapped_sublists = len([v for v in sublist_mappings.values() if v[1] >= 0.6])
        
        f.write(f"Industries: {mapped_industries}/{total_industries} mapped ({mapped_industries/total_industries*100:.1f}%)\n")
        f.write(f"Sub-lists: {mapped_sublists}/{total_sublists} mapped ({mapped_sublists/total_sublists*100:.1f}%)\n")
    
    print(f"[INFO] Report saved to {output_file}")


def apply_mappings(df, analysis_results, min_confidence=0.6, backup=True):
    """
    Apply the mappings to the dataframe
    
    Args:
        df: Input dataframe
        analysis_results: Results from analyze_current_data
        min_confidence: Minimum confidence to apply mapping
        backup: Whether to create backup file
        
    Returns:
        pd.DataFrame: Updated dataframe
    """
    if backup:
        backup_file = f"TR - Industry Enrichment_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2("TR - Industry Enrichment.csv", backup_file)
        print(f"[INFO] Created backup: {backup_file}")
    
    df_updated = df.copy()
    
    # Apply industry mappings
    print("[INFO] Applying industry mappings...")
    industry_mappings = analysis_results['industry_mappings']
    industry_changes = 0
    
    for current, (target, confidence, method) in industry_mappings.items():
        if confidence >= min_confidence and target:
            mask = df_updated['Industry'] == current
            count = mask.sum()
            df_updated.loc[mask, 'Industry'] = target
            if count > 0:
                industry_changes += count
                print(f"  Updated {count} rows: '{current}' → '{target}' (conf: {confidence:.2f})")
    
    # Apply sub-list mappings
    print("[INFO] Applying sub-list mappings...")
    sublist_mappings = analysis_results['sublist_mappings']
    sublist_changes = 0
    
    for current, (target, confidence, method) in sublist_mappings.items():
        if confidence >= min_confidence and target:
            mask = df_updated['Industry Sub List'] == current
            count = mask.sum()
            df_updated.loc[mask, 'Industry Sub List'] = target
            if count > 0:
                sublist_changes += count
                print(f"  Updated {count} rows: '{current}' → '{target}' (conf: {confidence:.2f})")
    
    print(f"[INFO] Total changes: {industry_changes} industry updates, {sublist_changes} sub-list updates")
    
    # Update reasoning ONLY for rows that were changed AND have a blank/NaN Reasoning value
    total_changes = industry_changes + sublist_changes
    if total_changes > 0 and 'Reasoning' in df_updated.columns:
        changed_mask = df_updated['Industry'].isin([
            target for current, (target, conf, method) in industry_mappings.items()
            if conf >= min_confidence and target
        ])

        # Identify rows where Reasoning is empty or NaN to avoid overwriting existing notes
        blank_reason_mask = df_updated['Reasoning'].isna() | (df_updated['Reasoning'].astype(str).str.strip() == '')
        update_mask = changed_mask & blank_reason_mask
        df_updated.loc[update_mask, 'Reasoning'] = (
            'Industry categories standardized via fuzzy matching cleanup'
        )
    
    return df_updated


def main():
    """Main function to run the cleanup process"""
    parser = argparse.ArgumentParser(description="Clean up industry data using fuzzy matching.")
    parser.add_argument(
        '--apply-mappings',
        action='store_true',
        help="Apply mappings non-interactively without asking for confirmation."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("INDUSTRY DATA CLEANUP SCRIPT")
    print("=" * 60)
    
    # Check required files
    required_files = ["TR - Industry Enrichment.csv", "industry_fields.txt"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"[ERROR] Required file not found: {file}")
            return
    
    try:
        # Step 1: Parse target categories
        target_industries, target_sublists = parse_industry_fields()
        
        # Step 2: Load current data
        print("[INFO] Loading current industry data...")
        df = pd.read_csv("TR - Industry Enrichment.csv")
        print(f"[INFO] Loaded {len(df)} records")
        
        # Step 3: Analyze and create mappings
        analysis_results = analyze_current_data(df, target_industries, target_sublists)
        
        # Step 4: Generate report
        generate_mapping_report(analysis_results)
        
        # Step 5: Ask user to proceed or apply non-interactively
        if args.apply_mappings:
            proceed = 'y'
        else:
            print("\n" + "=" * 60)
            print("MAPPING ANALYSIS COMPLETE")
            print("=" * 60)
            print("Please review 'mapping_report.txt' before proceeding.")
            print("This script will update the CSV file with standardized categories.")
            
            proceed = input("\nDo you want to apply the mappings? (y/N): ").strip().lower()
        
        if proceed == 'y':
            # Step 6: Apply mappings
            print("\n[INFO] Applying mappings...")
            df_updated = apply_mappings(df, analysis_results, min_confidence=0.6, backup=True)
            
            # Step 7: Save updated file
            output_file = "TR - Industry Enrichment.csv"
            df_updated.to_csv(output_file, index=False)
            print(f"[INFO] Updated file saved: {output_file}")
            
            # Step 8: Generate summary
            print("\n" + "=" * 60)
            print("CLEANUP COMPLETE")
            print("=" * 60)
            print(f"✓ Original file backed up")
            print(f"✓ {output_file} updated with standardized categories")
            print(f"✓ Mapping report available in 'mapping_report.txt'")
            print("\nNext steps:")
            print("1. Delete the old industry_weights.json to force regeneration")
            print("2. Re-run goe_icp_scoring.py to update scores with cleaned data")
            
        else:
            print("[INFO] Cleanup cancelled. No changes made.")
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise


if __name__ == "__main__":
    main() 