"""
Interactive ICP Scoring and Segmentation Dashboard
=================================================
This Streamlit application provides a real-time, interactive dashboard for
customer segmentation and Ideal Customer Profile (ICP) analysis. It allows users
to dynamically adjust scoring weights, configure customer segments, and visualize
the impact of these changes on various metrics and charts.

The dashboard is designed to be data-driven, using an underlying scoring model
that can be optimized based on historical data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import json
from scipy.stats import norm

# Import the centralized scoring logic and constants
from scoring_logic import calculate_scores, LICENSE_COL, DEFAULT_WEIGHTS
from normalize_names import normalize_name_for_matching

def load_optimized_weights():
    """
    Loads the optimized weights from the `optimized_weights.json` file.

    If the file doesn't exist or contains invalid JSON, it falls back to the
    `DEFAULT_WEIGHTS` from `scoring_logic.py`. It also returns the raw
    optimization metadata (number of trials, lambda parameter) for display.

    Returns:
        tuple: A tuple containing the weights dictionary and the optimization data dictionary.
    """
    try:
        with open('optimized_weights.json', 'r') as f:
            data = json.load(f)
            weights = data.get('weights', {})
            
            # Convert the weights to the format expected by the dashboard's session state.
            dashboard_weights = {
                "vertical_score": weights.get('vertical_score', 0.25),
                "size_score": weights.get('size_score', 0.25),
                "adoption_score": weights.get('adoption_score', 0.25),
                "relationship_score": weights.get('relationship_score', 0.25),
            }
            
            return dashboard_weights, data
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to default weights if the file is missing or corrupt.
        return DEFAULT_WEIGHTS, None

# Load optimized weights and metadata at the start of the script.
optimized_weights, optimization_data = load_optimized_weights()

# --- Page Configuration ---
st.set_page_config(
    page_title="ICP Dashboard - GoEngineer",
    page_icon="🎯",
    layout="wide"
)

# --- Custom CSS for Styling ---
# This section injects custom CSS to enhance the visual appearance of the dashboard,
# including custom metric cards, headers, and responsive design adjustments.
st.markdown("""...""", unsafe_allow_html=True) # CSS code omitted for brevity

# Default thresholds for customer segmentation based on annual revenue.
DEFAULT_SEGMENT_THRESHOLDS = {
    'small_business_max': 100000000,      # Up to $100M
    'mid_market_max': 1000000000,         # $100M to $1B
    # Customers with revenue > mid_market_max are considered Large Enterprise.
}

def determine_customer_segment(revenue, thresholds):
    """
    Determines a customer's segment based on their annual revenue.

    Args:
        revenue (float): The customer's annual revenue.
        thresholds (dict): A dictionary with 'small_business_max' and 'mid_market_max'.

    Returns:
        str: The name of the customer segment ('Small Business', 'Mid-Market', or 'Large Enterprise').
    """
    if pd.isna(revenue) or revenue <= thresholds['small_business_max']:
        return 'Small Business'
    elif revenue <= thresholds['mid_market_max']:
        return 'Mid-Market'
    else:
        return 'Large Enterprise'

# ... (Chart and metric functions are omitted here for brevity, but are commented in the full code)

@st.cache_data
def load_data():
    """
    Loads the base `icp_scored_accounts.csv` data.
    
    It then attempts to load the `enrichment_progress.csv` file and performs a
    hybrid matching process to merge the revenue data:
    1.  **Direct Match**: Merges based on `Customer ID`.
    2.  **Fuzzy Match**: For any customers not matched in step 1, it uses a normalized
        company name to find a match.
    
    This cached function ensures data is loaded only once.
    """
    try:
        df = pd.read_csv('icp_scored_accounts.csv')
        
        # Attempt to load and merge enriched revenue data.
        try:
            revenue_df = pd.read_csv('enrichment_progress.csv')
            # ... (Hybrid matching logic is omitted for brevity)
            
        except FileNotFoundError:
            st.warning("⚠️ Revenue analysis file not found. Segmentation will be based on printer count as a fallback.")
            df['revenue_estimate'] = df['printer_count'] * 10000000
        
        # ... (Data cleaning and type conversion)
        return df
        
    except FileNotFoundError:
        st.error("❌ Could not find 'icp_scored_accounts.csv'. Please run `goe_icp_scoring.py` first.")
        st.stop()

def main():
    """Main function to render the Streamlit dashboard."""
    st.markdown('<h1 class="main-header">🎯 ICP SCORING DASHBOARD</h1>', unsafe_allow_html=True)
    
    df_loaded = load_data()
    
    # --- CUSTOMER SEGMENTATION CONTROLS ---
    st.markdown("## 🏢 Customer Segmentation")
    
    # Initialize segment configuration in session state if it doesn't exist.
    if 'segment_config' not in st.session_state:
        st.session_state.segment_config = DEFAULT_SEGMENT_THRESHOLDS.copy()
    
    # Create an expandable section for users to configure the revenue thresholds.
    with st.expander("⚙️ Configure Customer Segments", expanded=False):
        # ... (UI elements for setting segment thresholds)
    
    # Apply the current segmentation configuration.
    current_segment_config = st.session_state.segment_config.copy()
    df_loaded['customer_segment'] = df_loaded['revenue_estimate'].apply(
        lambda x: determine_customer_segment(x, current_segment_config)
    )
    
    # Create the main segment selector dropdown.
    selected_segment = st.selectbox(
        "Select Customer Segment to Analyze:",
        ['All Segments', 'Small Business', 'Mid-Market', 'Large Enterprise'],
        # ...
    )
    st.session_state.selected_segment = selected_segment
    
    # Filter the main DataFrame based on the selected segment.
    if selected_segment == 'All Segments':
        df_filtered = df_loaded.copy()
    else:
        df_filtered = df_loaded[df_loaded['customer_segment'] == selected_segment].copy()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("🎯 ICP Scoring Controls")
        
        # Display information about whether optimized or default weights are being used.
        if optimization_data:
            st.success("🤖 **Optimized Weights Active**")
            # ... (Details of the optimization)
        else:
            st.warning("⚠️ **Using Default Weights**")
        
        # Create sliders for adjusting the four main scoring weights.
        st.subheader("⚖️ Adjust Scoring Weights")
        # ... (Weight sliders)

    # --- SCORE RECALCULATION ---
    # Recalculate all scores for the filtered data based on the current weights from the sidebar.
    df_scored = calculate_scores(df_filtered.copy(), current_main_weights, current_size_config)
    
    # --- MAIN DASHBOARD DISPLAY ---
    st.markdown(f"## 📈 Key Metrics - {dashboard_title_suffix}")
    # ... (Display custom metric cards)
    
    # --- SEGMENT ANALYSIS SECTION ---
    # This section is only displayed when viewing "All Segments".
    if selected_segment == 'All Segments':
        st.markdown("## 🏢 Customer Segment Analysis")
        # ... (Display segment comparison charts and summary table)
    
    # --- REAL-TIME ANALYTICS ---
    st.markdown(f"## 📊 Real-time Analytics")
    # ... (Display score distribution, vertical analysis, and other charts)
    
    # --- DATA TABLE ---
    st.markdown(f"## 📋 Top Scoring Customers")
    # ... (Display a formatted table of the top 100 customers)
    
    # --- DOWNLOAD BUTTON ---
    st.download_button(
        label=f"📥 Download {dashboard_title_suffix} Scores (CSV)",
        # ...
    )

if __name__ == "__main__":
    main() 
