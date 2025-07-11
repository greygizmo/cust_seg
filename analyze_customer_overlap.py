"""
Customer Data Overlap Analysis
============================
Analyzes the overlap between enrichment_progress.csv and icp_scored_accounts.csv
and identifies multi-location company issues.
"""

import pandas as pd
import numpy as np

def analyze_customer_overlap():
    """Analyze overlap between revenue estimates and ICP scoring datasets"""
    
    print("🔍 Loading datasets...")
    
    # Load both datasets
    try:
        enrichment_df = pd.read_csv('enrichment_progress.csv')
        print(f"   ✅ Loaded enrichment_progress.csv: {len(enrichment_df):,} rows")
    except FileNotFoundError:
        print("   ❌ enrichment_progress.csv not found")
        return
    
    try:
        icp_df = pd.read_csv('icp_scored_accounts.csv')
        print(f"   ✅ Loaded icp_scored_accounts.csv: {len(icp_df):,} rows")
    except FileNotFoundError:
        print("   ❌ icp_scored_accounts.csv not found")
        return
    
    print("\n📊 DATASET OVERVIEW")
    print("=" * 50)
    
    # Basic stats
    enrichment_customers = set(enrichment_df['customer_id'].dropna())
    icp_customers = set(icp_df['Customer ID'].dropna())
    
    print(f"Unique customers in enrichment_progress.csv: {len(enrichment_customers):,}")
    print(f"Unique customers in icp_scored_accounts.csv:  {len(icp_customers):,}")
    
    # Overlap analysis
    overlap = enrichment_customers.intersection(icp_customers)
    enrichment_only = enrichment_customers - icp_customers
    icp_only = icp_customers - enrichment_customers
    
    print(f"\n🔗 CUSTOMER ID OVERLAP ANALYSIS")
    print("=" * 50)
    print(f"Customers in BOTH datasets:        {len(overlap):,}")
    print(f"Customers ONLY in enrichment:      {len(enrichment_only):,}")
    print(f"Customers ONLY in ICP scoring:     {len(icp_only):,}")
    
    # Coverage percentages
    enrichment_coverage = (len(overlap) / len(enrichment_customers) * 100) if enrichment_customers else 0
    icp_coverage = (len(overlap) / len(icp_customers) * 100) if icp_customers else 0
    
    print(f"\n📈 COVERAGE RATES")
    print("=" * 30)
    print(f"% of enrichment customers in ICP scoring: {enrichment_coverage:.1f}%")
    print(f"% of ICP customers with revenue data:     {icp_coverage:.1f}%")
    
    # Multi-location company analysis
    print(f"\n🏢 MULTI-LOCATION COMPANY ANALYSIS")
    print("=" * 50)
    
    # Find companies with multiple customer IDs in ICP data
    icp_company_counts = icp_df['Company Name'].value_counts()
    multi_location_companies = icp_company_counts[icp_company_counts > 1]
    
    print(f"Companies with multiple locations in ICP data: {len(multi_location_companies)}")
    
    if len(multi_location_companies) > 0:
        print(f"\nTop 10 companies with most locations:")
        for company, count in multi_location_companies.head(10).items():
            customer_ids = icp_df[icp_df['Company Name'] == company]['Customer ID'].tolist()
            print(f"  • {company}: {count} locations (IDs: {customer_ids})")
    
    # Check which multi-location companies have revenue data
    print(f"\n💰 REVENUE DATA FOR MULTI-LOCATION COMPANIES")
    print("=" * 55)
    
    multi_location_with_revenue = []
    multi_location_without_revenue = []
    
    for company_name in multi_location_companies.index:
        company_customer_ids = icp_df[icp_df['Company Name'] == company_name]['Customer ID'].tolist()
        has_revenue = any(cid in enrichment_customers for cid in company_customer_ids)
        
        if has_revenue:
            # Find which specific customer IDs have revenue
            revenue_ids = [cid for cid in company_customer_ids if cid in enrichment_customers]
            multi_location_with_revenue.append((company_name, len(company_customer_ids), revenue_ids))
        else:
            multi_location_without_revenue.append((company_name, len(company_customer_ids), company_customer_ids))
    
    print(f"Multi-location companies WITH revenue data: {len(multi_location_with_revenue)}")
    print(f"Multi-location companies WITHOUT revenue data: {len(multi_location_without_revenue)}")
    
    if multi_location_with_revenue:
        print(f"\n✅ Companies with revenue data (showing top 10):")
        for company, total_locations, revenue_ids in multi_location_with_revenue[:10]:
            print(f"  • {company}: {len(revenue_ids)}/{total_locations} locations have revenue (IDs: {revenue_ids})")
    
    # Revenue source analysis
    if 'source' in enrichment_df.columns:
        print(f"\n📋 REVENUE SOURCE BREAKDOWN")
        print("=" * 35)
        source_counts = enrichment_df['source'].value_counts()
        total_enrichment = len(enrichment_df)
        
        for source, count in source_counts.items():
            pct = (count / total_enrichment * 100)
            print(f"  • {source}: {count:,} customers ({pct:.1f}%)")
    
    # Specific examples
    print(f"\n🔍 DETAILED EXAMPLES")
    print("=" * 30)
    
    # Ball Corporation example
    ball_in_enrichment = enrichment_df[enrichment_df['company_name'].str.contains('Ball', case=False, na=False)]
    ball_in_icp = icp_df[icp_df['Company Name'].str.contains('Ball', case=False, na=False)]
    
    if not ball_in_enrichment.empty:
        print(f"Ball Corporation in enrichment: {len(ball_in_enrichment)} entries")
        for _, row in ball_in_enrichment.iterrows():
            print(f"  • ID {row['customer_id']}: {row['company_name']} (Revenue: ${row['revenue_estimate']:,.0f})")
    
    if not ball_in_icp.empty:
        print(f"Ball Corporation in ICP data: {len(ball_in_icp)} entries")
        for _, row in ball_in_icp.iterrows():
            print(f"  • ID {row['Customer ID']}: {row['Company Name']}")
    
    # Stratasys example
    stratasys_in_enrichment = enrichment_df[enrichment_df['company_name'].str.contains('Stratasys', case=False, na=False)]
    stratasys_in_icp = icp_df[icp_df['Company Name'].str.contains('Stratasys', case=False, na=False)]
    
    if not stratasys_in_enrichment.empty:
        print(f"\nStratasys in enrichment: {len(stratasys_in_enrichment)} entries")
        for _, row in stratasys_in_enrichment.iterrows():
            print(f"  • ID {row['customer_id']}: {row['company_name']} (Revenue: ${row['revenue_estimate']:,.0f})")
    
    if not stratasys_in_icp.empty:
        print(f"Stratasys in ICP data: {len(stratasys_in_icp)} entries")
        for _, row in stratasys_in_icp.iterrows():
            print(f"  • ID {row['Customer ID']}: {row['Company Name']}")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 25)
    print("1. OVERLAP ISSUE:")
    print(f"   • {len(enrichment_only):,} customers have revenue estimates but no ICP scores")
    print("   • Consider expanding ICP scoring to cover more customers")
    print("   • This represents untapped revenue potential")
    
    print("\n2. MULTI-LOCATION ISSUE:")
    print("   • Companies like Stratasys have multiple customer IDs for different locations")
    print("   • Solution options:")
    print("     a) Use company name matching as fallback when customer ID fails")
    print("     b) Create parent company mapping table")
    print("     c) Consolidate locations under parent company ID")
    
    return {
        'enrichment_total': len(enrichment_customers),
        'icp_total': len(icp_customers),
        'overlap': len(overlap),
        'enrichment_only': len(enrichment_only),
        'icp_only': len(icp_only),
        'multi_location_companies': len(multi_location_companies)
    }

def suggest_company_name_matching():
    """Suggest implementation for hybrid matching approach"""
    
    print(f"\n🔧 SUGGESTED SOLUTION: HYBRID MATCHING")
    print("=" * 45)
    
    print("""
The best approach for handling multi-location companies is hybrid matching:

1. PRIMARY: Customer ID matching (fast, accurate)
2. FALLBACK: Company name matching (catches multi-location cases)

This would work as follows:
- First, match customers by exact Customer ID
- For unmatched customers, try fuzzy company name matching
- This catches cases where Stratasys ID 108 gets revenue data
  but Stratasys IDs 129642, 317785, etc. don't

Implementation steps:
1. Update dashboard matching logic to use hybrid approach
2. Add company name normalization function
3. Add matching success reporting
4. Optionally create parent company mapping table
""")

if __name__ == "__main__":
    results = analyze_customer_overlap()
    
    # Ask user about implementing solution
    print(f"\n❓ Would you like me to implement the hybrid matching solution?")
    print("   This would update the dashboard to handle multi-location companies better.")
    
    suggest_company_name_matching() 