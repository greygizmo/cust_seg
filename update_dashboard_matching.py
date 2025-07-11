"""
Updates the dashboard to use hybrid Customer ID + Company Name matching
to handle multi-location companies like Stratasys better.
"""

def create_hybrid_matching_solution():
    """Create the updated load_data function with hybrid matching"""
    
    updated_load_data_function = '''
@st.cache_data
def load_data():
    """Load the scored accounts data and merge with revenue data using hybrid matching"""
    try:
        # Load main scored accounts data
        df = pd.read_csv('icp_scored_accounts.csv')
        
        # Load revenue data
        try:
            revenue_df = pd.read_csv('enrichment_progress.csv')
            
            # HYBRID MATCHING APPROACH
            # Step 1: Direct Customer ID matching (primary method)
            if 'customer_id' in revenue_df.columns and 'Customer ID' in df.columns:
                df = df.merge(
                    revenue_df[['customer_id', 'revenue_estimate', 'company_name', 'source']], 
                    left_on='Customer ID',
                    right_on='customer_id',
                    how='left',
                    suffixes=('', '_revenue')
                )
                
                # Track matching success
                primary_matches = len(df[df['revenue_estimate'].notna()])
                total_customers = len(df)
                
                # Step 2: Company name matching for unmatched customers (fallback)
                unmatched_mask = df['revenue_estimate'].isna()
                unmatched_count = unmatched_mask.sum()
                
                if unmatched_count > 0:
                    # Company name normalization function
                    def normalize_company_name(name):
                        """Normalize company names for matching"""
                        if pd.isna(name):
                            return ""
                        
                        import re
                        name = str(name).lower().strip()
                        
                        # Remove common suffixes and prefixes
                        name = re.sub(r'\\b(inc|corp|corporation|llc|ltd|limited|co|company)\\b', '', name)
                        # Remove customer ID prefix (e.g., "12345 Company Name")
                        name = re.sub(r'^\\d+\\s+', '', name)
                        # Remove special characters and normalize spaces
                        name = re.sub(r'[^\\w\\s]', ' ', name)
                        name = re.sub(r'\\s+', ' ', name).strip()
                        
                        return name
                    
                    # Create normalized company names
                    df['normalized_company'] = df['Company Name'].apply(normalize_company_name)
                    revenue_df['normalized_company'] = revenue_df['company_name'].apply(normalize_company_name)
                    
                    # For unmatched customers, try company name matching
                    for idx in df[unmatched_mask].index:
                        company_normalized = df.loc[idx, 'normalized_company']
                        
                        if company_normalized:
                            # Find matching company in revenue data
                            revenue_matches = revenue_df[
                                revenue_df['normalized_company'] == company_normalized
                            ]
                            
                            if not revenue_matches.empty:
                                # Use the first match (could prioritize by revenue source)
                                best_match = revenue_matches.iloc[0]
                                df.loc[idx, 'revenue_estimate'] = best_match['revenue_estimate']
                                df.loc[idx, 'company_name_revenue'] = best_match['company_name']
                                df.loc[idx, 'source'] = best_match['source']
                    
                    # Calculate final matching statistics
                    secondary_matches = len(df[df['revenue_estimate'].notna()]) - primary_matches
                    total_matches = primary_matches + secondary_matches
                    
                    # Display matching results
                    st.sidebar.markdown("### 🔗 Data Matching Results")
                    st.sidebar.markdown(f"**Primary (Customer ID):** {primary_matches:,} matches")
                    st.sidebar.markdown(f"**Secondary (Company Name):** {secondary_matches:,} matches")
                    st.sidebar.markdown(f"**Total Matched:** {total_matches:,} / {total_customers:,} ({total_matches/total_customers*100:.1f}%)")
                    st.sidebar.markdown(f"**Unmatched:** {total_customers - total_matches:,} customers")
                
                # Fill missing revenue with 0 for segmentation purposes
                df['revenue_estimate'] = df['revenue_estimate'].fillna(0)
                
            else:
                st.error("Required columns not found for customer matching")
                df['revenue_estimate'] = 0
            
        except FileNotFoundError:
            st.warning("⚠️ Revenue analysis file not found. Using fallback segmentation based on printer count.")
            df['revenue_estimate'] = df['printer_count'] * 10000000
        except Exception as e:
            st.error(f"❌ Error loading revenue data: {str(e)}. Using fallback segmentation.")
            df['revenue_estimate'] = df['printer_count'] * 10000000
        
        # Rest of the function remains the same...
        # Handle different column name variations for industry data
        if 'Industry' not in df.columns:
            df['Industry'] = 'Unknown'
            
        # Ensure numeric columns are properly typed
        numeric_cols = ['Big Box Count', 'Small Box Count', 'printer_count', 'revenue_estimate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ Could not find 'icp_scored_accounts.csv'. Please run the main scoring script first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
'''
    
    return updated_load_data_function

if __name__ == "__main__":
    solution = create_hybrid_matching_solution()
    print("Updated load_data function created!")
    print("This function provides:")
    print("1. Primary Customer ID matching")
    print("2. Fallback company name matching for multi-location companies")
    print("3. Detailed matching statistics in sidebar")
    print("4. Handles companies like Stratasys with multiple customer IDs") 