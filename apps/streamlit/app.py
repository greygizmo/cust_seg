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
import sys
from pathlib import Path
import re
import json
from scipy.stats import norm

# Ensure `src` is on the path for imports
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import the centralized scoring logic and constants
from icp.scoring import calculate_scores, LICENSE_COL, DEFAULT_WEIGHTS
from icp.utils.normalize import normalize_name_for_matching

# Page configuration
st.set_page_config(
    page_title="ICP Dashboard - GoEngineer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        weights_path = ROOT / 'artifacts' / 'weights' / 'optimized_weights.json'
        with open(weights_path, 'r') as f:
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
        mapped = {
            "vertical_score": DEFAULT_WEIGHTS.get("vertical", 0.25),
            "size_score": DEFAULT_WEIGHTS.get("size", 0.0),
            "adoption_score": DEFAULT_WEIGHTS.get("adoption", 0.5),
            "relationship_score": DEFAULT_WEIGHTS.get("relationship", 0.25),
        }
        return mapped, None

# Load optimized weights and metadata at the start of the script.
optimized_weights, optimization_data = load_optimized_weights()

# --- Navigation ---
def main():
    # Sidebar navigation
    st.sidebar.title("ICP Dashboard Navigation")

    page = st.sidebar.radio("Navigate to:", ["Main Dashboard", "Call List Builder", "System Documentation", "Scoring Details"]) 

    if page == "Main Dashboard":
        show_main_dashboard()
    elif page == "Call List Builder":
        show_call_list_builder()
    elif page == "System Documentation":
        show_documentation()
    elif page == "Scoring Details":
        show_scoring_details()

def show_main_dashboard():
    """Main dashboard page with metrics, charts, and analysis"""

    # --- Custom CSS for Styling ---
    # Beautiful styling with custom metric cards, animations, and professional appearance
    st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Hide default Streamlit metric containers completely */
    div[data-testid="metric-container"] {
        display: none !important;
    }
    
    /* Custom Metric Display */
    .custom-metric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        cursor: pointer;
        animation: fadeInUp 0.6s ease-out;
        animation-fill-mode: both;
    }
    
    /* Stagger animation for each card */
    .custom-metric:nth-child(1) { animation-delay: 0.1s; }
    .custom-metric:nth-child(2) { animation-delay: 0.2s; }
    .custom-metric:nth-child(3) { animation-delay: 0.3s; }
    .custom-metric:nth-child(4) { animation-delay: 0.4s; }
    .custom-metric:nth-child(5) { animation-delay: 0.5s; }
    .custom-metric:nth-child(6) { animation-delay: 0.6s; }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 30px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }
    
    .custom-metric:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border-color: rgba(31, 119, 180, 0.3);
    }
    
    .custom-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        border-radius: 16px 16px 0 0;
    }
    
    .custom-metric::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
    }
    
    .metric-customers::before {
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    
    .metric-score::before {
        background: linear-gradient(90deg, #1f77b4, #4dabf7);
    }
    
    .metric-high-value::before {
        background: linear-gradient(90deg, #ffc107, #fd7e14);
    }
    
    .metric-gp::before {
        background: linear-gradient(90deg, #6f42c1, #e83e8c);
    }
    
    .metric-hv-gp::before {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
    }
    
    .metric-percentage::before {
        background: linear-gradient(90deg, #17a2b8, #6610f2);
    }
    
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6c757d;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.2;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #212529;
        margin-bottom: 0.5rem;
        line-height: 1;
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
    
    .metric-subtitle {
        font-size: 0.7rem;
        color: #8e9196;
        font-weight: 500;
        line-height: 1.3;
    }
    
    .metric-icon {
        font-size: 1.1rem;
        margin-right: 0.4rem;
        opacity: 0.9;
    }
    
    /* Enhanced metric card variants for better visual distinction */
    .metric-customers {
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);
        border-color: #c3e6cb;
    }
    
    .metric-score {
        background: linear-gradient(135deg, #f8fbff 0%, #e3f2fd 100%);
        border-color: #b3d9ff;
    }
    
    .metric-high-value {
        background: linear-gradient(135deg, #fffdf8 0%, #fff3cd 100%);
        border-color: #ffeaa7;
    }
    
    .metric-gp {
        background: linear-gradient(135deg, #fdf8ff 0%, #f3e5f5 100%);
        border-color: #e1bee7;
    }
    
    .metric-hv-gp {
        background: linear-gradient(135deg, #fff8f8 0%, #ffebee 100%);
        border-color: #ffcdd2;
    }
    
    .metric-percentage {
        background: linear-gradient(135deg, #f8fcff 0%, #e0f2f1 100%);
        border-color: #b2dfdb;
    }
    
    /* Additional responsive adjustments */
    @media (max-width: 768px) {
        .custom-metric {
            height: 120px;
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.6rem;
        }
        
        .metric-title {
            font-size: 0.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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

def create_score_components_radar(weights, segment_name="Overall"):
    """Create radar chart showing weight distribution"""
    categories = list(weights.keys())
    # Capitalize categories for better display
    theta_labels = [cat.replace('_', ' ').capitalize() for cat in categories]
    values = list(weights.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], # Close the loop
        theta=theta_labels + [theta_labels[0]], # Close the loop
        fill='toself',
        name='Current Weights',
        line_color='#1f77b4',
        marker=dict(color='#1f77b4', size=8) # Add markers
    ))
    
    chart_title = f"ICP Scoring Weight Distribution"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(0.5, max(values) + 0.05)] # Dynamic range based on max weight, at least 0.5
            ),
            angularaxis=dict(
                tickfont_size=10 # Adjust tick font size if labels are too long
            )
        ),
        title=chart_title,
        showlegend=False,
        height=500 # Adjust height as needed
    )
    return fig

def create_score_by_vertical(df):
    """Create average score by vertical chart"""
    # Find the ICP_score column
    icp_score_col = 'ICP_score'
    if df.empty or 'Industry' not in df.columns or icp_score_col not in df.columns:
        return go.Figure().update_layout(title_text="No data available for Industry Score Analysis")
    
    vertical_scores = df.groupby('Industry')[icp_score_col].agg(['mean', 'count']).reset_index()
    vertical_scores = vertical_scores[vertical_scores['count'] >= 1].nlargest(15, 'mean') # Show top 15, min 1 customer
    
    fig = px.bar(
        vertical_scores, 
        x='mean', 
        y='Industry',
        text=[f" ({int(x)} cust.)" for x in vertical_scores['count'].tolist()], # Add count text
        title="Average ICP Score by Industry (Top 15)",
        labels={'mean': 'Average ICP Score', 'Industry': 'Industry'},
        color='mean',
        color_continuous_scale=px.colors.sequential.Viridis_r, # Reversed Viridis for better high-score emphasis
        height=max(400, len(vertical_scores) * 35) # Dynamic height
    )
    fig.update_traces(texttemplate='%{x:.1f}%{text}', textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'}, # This will be overridden by sort_values typically
        xaxis_title="Average ICP Score (Customer Count in Parentheses)",
        yaxis_title="Industry",
        coloraxis_colorbar_title='Avg. Score'
    )
    return fig

def create_profit_by_vertical(df, profit_col: str):
    """Create total profit by industry chart for selected scope"""
    if df.empty or 'Industry' not in df.columns or profit_col not in df.columns:
        return go.Figure().update_layout(title_text="No data available for Profit by Industry")
    by_industry = df.groupby('Industry')[profit_col].sum().reset_index()
    by_industry = by_industry.sort_values(profit_col, ascending=False).head(15)
    fig = px.bar(
        by_industry,
        x=profit_col,
        y='Industry',
        title=f"Total Profit by Industry ({profit_col})",
        labels={profit_col: 'Profit', 'Industry': 'Industry'},
        color=profit_col,
        color_continuous_scale=px.colors.sequential.Blues,
        height=max(400, len(by_industry) * 35)
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

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
        df = pd.read_csv(ROOT / 'data' / 'processed' / 'icp_scored_accounts.csv')
        
        # Attempt to load and merge enriched revenue data.
        try:
            revenue_df = pd.read_csv(ROOT / 'data' / 'raw' / 'enrichment_progress.csv')
            # ... (Hybrid matching logic is omitted for brevity)
            
        except FileNotFoundError:
            st.warning(" Revenue analysis file not found. Using existing revenue data from the dataset.")
            # Use the existing revenue column if available, otherwise fall back to printer count estimation
            if 'Total Hardware + Consumable Revenue' not in df.columns:
                df['Total Hardware + Consumable Revenue'] = df['printer_count'] * 10000000
        
        # ... (Data cleaning and type conversion)
        return df
        
    except FileNotFoundError:
        st.error(" Could not find 'data/processed/icp_scored_accounts.csv'. Please run `python -m icp.cli.score_accounts` first.")
        st.stop()

def main():
    """Main function to render the Streamlit dashboard."""
    df_loaded = load_data()
    
    # --- CUSTOMER SEGMENTATION CONTROLS ---
    st.markdown("##  Customer Segmentation")
    
    # Initialize segment configuration in session state if it doesn't exist.
    if 'segment_config' not in st.session_state:
        st.session_state.segment_config = DEFAULT_SEGMENT_THRESHOLDS.copy()
    
    # Create an expandable section for users to configure the revenue thresholds.
    with st.expander(" Configure Customer Segments", expanded=False):
        pass  # TODO: Add UI elements for setting segment thresholds
    
    # Apply the current segmentation configuration.
    current_segment_config = st.session_state.segment_config.copy()
    # Use Profit Since 2023 by default for segmentation
    revenue_col = 'Profit_Since_2023_Total'
    if revenue_col not in df_loaded.columns:
        # fallback to previous column if profit not present
        revenue_col = 'Total Hardware + Consumable Revenue' if 'Total Hardware + Consumable Revenue' in df_loaded.columns else revenue_col
    df_loaded['customer_segment'] = df_loaded[revenue_col].apply(
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
        st.title(" ICP Scoring Controls")
        
        # Display information about whether optimized or default weights are being used.
        if optimization_data:
            st.success(" **Optimized Weights Active**")
            # ... (Details of the optimization)
        else:
            st.warning(" **Using Default Weights**")
        
        # Create sliders for adjusting the four main scoring weights.
        st.subheader(" Adjust Scoring Weights")
        
        # Weight sliders using the optimized weights as defaults
        vertical_weight = st.slider("Vertical Score Weight", 0.0, 1.0, optimized_weights["vertical_score"], 0.01)
        size_weight = st.slider("Size Score Weight", 0.0, 1.0, optimized_weights["size_score"], 0.01)
        adoption_weight = st.slider("Adoption Score Weight", 0.0, 1.0, optimized_weights["adoption_score"], 0.01)
        relationship_weight = st.slider("Relationship Score Weight", 0.0, 1.0, optimized_weights["relationship_score"], 0.01)
        
        # Normalize weights to sum to 1.0
        total_weight = vertical_weight + size_weight + adoption_weight + relationship_weight
        if total_weight > 0:
            current_main_weights = {
                "vertical": vertical_weight / total_weight,
                "size": size_weight / total_weight,
                "adoption": adoption_weight / total_weight,
                "relationship": relationship_weight / total_weight
            }
        else:
            current_main_weights = optimized_weights
        
        current_size_config = {"enabled": size_weight > 0}
        current_size_config = {"enabled": size_weight > 0}
        st.subheader("Profit & Filters")
        selected_scope = st.selectbox("Profit Scope", ["Since 2023", "Trailing 4Q", "Last Quarter"], index=0)
        st.session_state.profit_scope = selected_scope
        active_only = st.checkbox("Active assets only", value=False)
        min_cad_seats = st.number_input("Min CAD seats", min_value=0, value=0, step=1)
        min_cpe_seats = st.number_input("Min CPE seats", min_value=0, value=0, step=1)

    # --- SCORE RECALCULATION ---
    # Recalculate all scores for the filtered data based on the current weights from the sidebar.
    df_scored = calculate_scores(df_filtered.copy(), current_main_weights, current_size_config)
    
    # Apply sidebar filters and profit scope to the dataframe
    scope_to_col = {
        "Since 2023": "Profit_Since_2023_Total",
        "Trailing 4Q": "Profit_T4Q_Total",
        "Last Quarter": "Profit_LastQ_Total",
    }
    profit_col = scope_to_col.get(selected_scope, "Profit_Since_2023_Total") if "selected_scope" in locals() else "Profit_Since_2023_Total"
    if "active_only" in locals() and active_only and "active_assets_total" in df_scored.columns:
        df_scored = df_scored[df_scored["active_assets_total"] > 0]
    if "min_cad_seats" in locals() and "Seats_CAD" in df_scored.columns:
        df_scored = df_scored[df_scored["Seats_CAD"] >= min_cad_seats]
    if "min_cpe_seats" in locals() and "Seats_CPE" in df_scored.columns:
        df_scored = df_scored[df_scored["Seats_CPE"] >= min_cpe_seats]
    # Set dashboard title suffix
    dashboard_title_suffix = selected_segment if selected_segment != 'All Segments' else 'All Customers'
    
    # --- MAIN DASHBOARD DISPLAY ---
    st.markdown('<h1 class="main-header"> ICP SCORING DASHBOARD</h1>', unsafe_allow_html=True)
    st.markdown(f"##  Key Metrics - {dashboard_title_suffix}")
    
    # Calculate key metrics
    total_customers = len(df_scored)
    avg_score = df_scored['ICP_score'].mean()
    high_score_count = len(df_scored[df_scored['ICP_score'] >= 70])
    total_revenue = df_scored[profit_col].sum() if "profit_col" in locals() and profit_col in df_scored.columns else df_scored[revenue_col].sum()
    hv_rate = (high_score_count / total_customers * 100) if total_customers > 0 else 0
    
    # Display key metrics with beautiful custom cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_title = "Total Customers" if selected_segment == 'All Segments' else f"{selected_segment} Customers"
        st.markdown(f"""
        <div class="custom-metric metric-customers">
            <div class="metric-title">
                <span class="metric-icon"></span>{metric_title}
            </div>
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-subtitle">Active customer accounts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score_color = "#28a745" if avg_score >= 70 else "#ffc107" if avg_score >= 50 else "#dc3545"
        st.markdown(f"""
        <div class="custom-metric metric-score">
            <div class="metric-title">
                <span class="metric-icon"></span>Average ICP Score
            </div>
            <div class="metric-value" style="color: {score_color};">{avg_score:.1f}</div>
            <div class="metric-subtitle">Out of 100 points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="custom-metric metric-high-value">
            <div class="metric-title">
                <span class="metric-icon"></span>High-Value Customers
            </div>
            <div class="metric-value">{high_score_count:,}</div>
            <div class="metric-subtitle">{hv_rate:.1f}% of total (70 score)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="custom-metric metric-gp">
            <div class="metric-title">
                <span class="metric-icon"></span>Total Profit
            </div>
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div class="metric-subtitle">Profit (selected scope)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- SEGMENT ANALYSIS SECTION ---
    # This section is only displayed when viewing "All Segments".
    if selected_segment == 'All Segments':
        st.markdown("##  Customer Segment Analysis")
        
        # Segment summary table
        segment_summary = df_scored.groupby('customer_segment').agg({
            'ICP_score': ['count', 'mean'],
            revenue_col: 'sum',
            'ICP_grade': lambda x: (x == 'A').sum()
        }).round(2)
        
        segment_summary.columns = ['Customer Count', 'Avg ICP Score', 'Total Profit', 'A-Grade Count']
        st.dataframe(segment_summary, use_container_width=True)
    
    # --- REAL-TIME ANALYTICS ---
    st.markdown(f"##  Real-time Analytics")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced ICP Score Distribution
        fig_hist = px.histogram(df_scored, x='ICP_score', nbins=30, 
                               title='Distribution of ICP Scores',
                               color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(
            xaxis_title="ICP Score",
            yaxis_title="Number of Customers",
            showlegend=False,
            bargap=0.1
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Grade Distribution
        grade_counts = df_scored['ICP_grade'].value_counts()
        colors = {'A': '#28a745', 'B': '#1f77b4', 'C': '#ffc107', 'D': '#fd7e14', 'F': '#dc3545'}
        grade_colors = [colors.get(grade, '#999999') for grade in grade_counts.index]
        
        fig_pie = px.pie(values=grade_counts.values, names=grade_counts.index,
                        title='Customer Grade Distribution',
                        color_discrete_sequence=grade_colors)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Weight Distribution Radar Chart
        fig_radar = create_score_components_radar(current_main_weights)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col4:
        # Profit by Industry (selected scope)
        # Determine profit column from selected scope
        scope_to_col = {
            "Since 2023": "Profit_Since_2023_Total",
            "Trailing 4Q": "Profit_T4Q_Total",
            "Last Quarter": "Profit_LastQ_Total",
        }
        profit_col = scope_to_col.get(selected_scope, "Profit_Since_2023_Total") if 'selected_scope' in locals() else "Profit_Since_2023_Total"
        fig_profit_industry = create_profit_by_vertical(df_scored, profit_col)
        st.plotly_chart(fig_profit_industry, use_container_width=True)

    # Third row of charts: Profit distribution and Profit vs Install Base
    col5, col6 = st.columns(2)
    with col5:
        fig_profit_hist = px.histogram(
            df_scored, x=profit_col,
            nbins=30,
            title=f'Distribution of Profit ({selected_scope})',
            color_discrete_sequence=['#2ca02c']
        )
        fig_profit_hist.update_layout(
            xaxis_title="Profit",
            yaxis_title="Number of Customers",
            showlegend=False,
            bargap=0.1
        )
        st.plotly_chart(fig_profit_hist, use_container_width=True)

    with col6:
        # Choose install base proxy: printer_count if available, else seats_sum_total
        base_col = 'printer_count' if 'printer_count' in df_scored.columns else 'seats_sum_total'
        fig_profit_vs_base = px.scatter(
            df_scored, x=base_col, y=profit_col,
            title=f'Profit ({selected_scope}) vs Install Base ({base_col})',
            trendline='ols'
        )
        fig_profit_vs_base.update_layout(
            xaxis_title=base_col,
            yaxis_title="Profit"
        )
        st.plotly_chart(fig_profit_vs_base, use_container_width=True)

    # Fourth row of charts: QoQ Profit Growth and Renewal Horizon Bands
    col7, col8 = st.columns(2)
    with col7:
        # QoQ Profit Growth (LastQ vs PrevQ)
        if 'Profit_QoQ_Growth' in df_scored.columns:
            growth = df_scored['Profit_QoQ_Growth']
        elif 'Profit_LastQ_Total' in df_scored.columns and 'Profit_PrevQ_Total' in df_scored.columns:
            prev = df_scored['Profit_PrevQ_Total'].replace(0, pd.NA)
            growth = ((df_scored['Profit_LastQ_Total'] - df_scored['Profit_PrevQ_Total']) / prev).fillna(0)
        else:
            growth = pd.Series([0])
        fig_qoq = px.histogram(
            growth, x=growth,
            nbins=30,
            title='QoQ Profit Growth (LastQ vs PrevQ)',
            color_discrete_sequence=['#9467bd']
        )
        fig_qoq.update_layout(xaxis_title='Growth (ratio)', yaxis_title='Customers', showlegend=False)
        st.plotly_chart(fig_qoq, use_container_width=True)

    with col8:
        # Renewal horizon bands using LatestExpirationDate
        if 'LatestExpirationDate' in df_scored.columns:
            exp = pd.to_datetime(df_scored['LatestExpirationDate'], errors='coerce')
            days = (exp - pd.Timestamp.today()).dt.days
            def band(d):
                if pd.isna(d):
                    return 'No Expiry'
                if d < 0:
                    return 'Expired'
                if d <= 90:
                    return '0-90 days'
                if d <= 180:
                    return '91-180 days'
                if d <= 365:
                    return '181-365 days'
                return '> 365 days'
            bands = days.map(band)
            order = ['Expired','0-90 days','91-180 days','181-365 days','> 365 days','No Expiry']
            band_counts = bands.value_counts().reindex(order).fillna(0)
            fig_ren = px.bar(
                x=band_counts.index, y=band_counts.values,
                title='Customers by Renewal Horizon (Latest Asset Expiry)',
                labels={'x':'Horizon','y':'Customers'},
                color=band_counts.values, color_continuous_scale=px.colors.sequential.Sunset
            )
            fig_ren.update_layout(showlegend=False)
            st.plotly_chart(fig_ren, use_container_width=True)
        else:
            st.info('No expiration data available for renewal horizon chart.')

    # --- DATA TABLE ---
    st.markdown(f"##  Top Scoring Customers")

    # Display top 100 customers sorted by ICP score
    top_customers = df_scored.nlargest(100, 'ICP_score')

    # Select key columns to display
    display_columns = ['Company Name', 'Industry', 'ICP_score', 'ICP_grade',
                      'vertical_score', 'adoption_score', 'relationship_score', revenue_col]

    # Only show columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in top_customers.columns]

    st.dataframe(
        top_customers[available_columns],
        use_container_width=True,
        height=400
    )

    # --- DOWNLOAD BUTTON ---
    csv_data = df_scored.to_csv(index=False)
    scope_key = selected_scope.lower().replace(' ', '_') if 'selected_scope' in locals() else 'since_2023'
    st.download_button(
        label=f"Download {dashboard_title_suffix} Scores ({selected_scope}) (CSV)" if 'selected_scope' in locals() else f"Download {dashboard_title_suffix} Scores (CSV)",
        data=csv_data,
        file_name=f"icp_scores_{scope_key}_{dashboard_title_suffix.lower().replace(' ', '_')}.csv" if 'selected_scope' in locals() else f"icp_scores_{dashboard_title_suffix.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

def show_documentation():
    """System documentation page with Mermaid charts"""
    st.markdown('<h1 class="main-header">System Documentation</h1>', unsafe_allow_html=True)
    st.markdown("### Complete technical documentation for the ICP Scoring System")

    # Chart categories with all charts
    chart_categories = {
        " System Architecture": [
            ("01-overall-architecture.md", "Overall System Architecture", "Complete system overview and data flow"),
            ("07-data-flow-dependencies.md", "Data Flow & Dependencies", "File relationships and dependencies"),
            ("08-component-interaction.md", "Component Interaction", "Python module interactions"),
            ("09-file-relationships.md", "File Relationships", "Import/export relationships")
        ],
        " Data Processing": [
            ("02-data-processing-pipeline.md", "Data Processing Pipeline", "8-stage data processing workflow")
        ],
        " Scoring System": [
            ("03-scoring-methodology.md", "Scoring Methodology", "4-component ICP scoring system"),
            ("04-weight-optimization.md", "Weight Optimization", "ML optimization with Optuna"),
            ("06-industry-scoring.md", "Industry Scoring", "Data-driven industry weights")
        ],
        " User Interface": [
            ("05-dashboard-workflow.md", "Dashboard Workflow", "User interaction and experience")
        ]
    }

    # Create a flat list of all charts for the dropdown
    all_charts = []
    for category, charts in chart_categories.items():
        for chart_file, title, description in charts:
            all_charts.append((chart_file, title, description, category))

    # Create options list for dropdown
    chart_options = [f"{title} ({category})" for _, title, _, category in all_charts]

    # Chart selection dropdown
    selected_chart_option = st.selectbox(
        "Select a diagram to view:",
        chart_options,
        help="Choose from any of the 9 Mermaid diagrams across all categories"
    )

    st.markdown("---")

    # Find the selected chart
    selected_index = chart_options.index(selected_chart_option)
    chart_file, title, description, category = all_charts[selected_index]
    chart_path = ROOT / 'docs' / 'mermaid-charts' / chart_file

    # Display the selected chart
    st.markdown(f"##  {title}")
    st.markdown(f"**Category:** {category}")
    st.markdown(f"**Description:** {description}")
    st.markdown("---")

    try:
        with open(chart_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Mermaid diagram (between ```mermaid and ```)
        import re
        mermaid_match = re.search(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
        if mermaid_match:
            mermaid_code = mermaid_match.group(1)

            # Display Mermaid chart
            st.markdown("###  Mermaid Diagram:")
            st.markdown("**Copy this code to any Mermaid renderer:**")
            st.code(mermaid_code, language="mermaid")

            # Try to render Mermaid in Streamlit (if supported)
            try:
                st.markdown("###  Visual Diagram:")
                st.markdown(f"```mermaid\n{mermaid_code}\n```")
            except Exception as render_error:
                st.warning(" Mermaid rendering may not be supported in this environment. Use the code above in a Mermaid-compatible viewer.")

            # Show full documentation
            st.markdown("###  Full Documentation:")
            st.markdown(content.replace(f"```mermaid\n{mermaid_code}\n```", ""))
        else:
            st.markdown("###  Documentation Content:")
            st.markdown(content)

    except FileNotFoundError:
        st.error(f" Documentation file not found: {chart_file}")
        st.info("Please ensure Mermaid chart files are present in the `docs/mermaid-charts/` directory.")
    except Exception as e:
        st.error(f" Error loading documentation: {str(e)}")
        st.info("Check the file format and encoding. The file should be a valid Markdown file with Mermaid diagram.")

def show_scoring_details():
    """Scoring methodology details page"""
    st.markdown('<h1 class="main-header">Scoring Details</h1>', unsafe_allow_html=True)
    st.markdown("### Detailed breakdown of the ICP scoring methodology")

    # Load current weights
    weights = optimized_weights if optimization_data else DEFAULT_WEIGHTS

    # Component explanations
    st.markdown("## ICP Scoring Components")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vertical Score (Industry)")
        st.info(f"**Current Weight: {weights['vertical_score']:.3f}**")
        st.markdown("""
        - **Purpose**: Measures industry performance potential
        - **Method**: Data-driven weights from historical revenue
        - **Scale**: 0.0 - 1.0 (higher = better industry)
        - **Example**: Aerospace & Defense = 1.0, Education = 0.4
        """)

        st.markdown("### Adoption Score (Hardware Engagement)")
        st.info(f"**Current Weight: {weights['adoption_score']:.3f}**")
        st.markdown("""
        - **Purpose**: Measures hardware investment level
        - **Method**: Weighted printer counts + revenue percentiles
        - **Business Rules**:
          - No printers + no revenue = 0.0
          - Revenue only = 0.0-0.5
          - Printer customers = 0.0-1.0
          - 10+ printers = +0.05 bonus
        """)

    with col2:
        st.markdown("### Size Score (Revenue)")
        st.info(f"**Current Weight: {weights['size_score']:.3f}**")
        st.markdown("""
        - **Purpose**: Evaluates company size and scale
        - **Tiers**:
          - $250M-$1B: 1.0 (Enterprise)
          - $50M-$250M: 0.6 (Large)
          - $10M-$50M: 0.4 (Medium)
          - $0-$10M: 0.4 (Small)
        - **Fallback**: 0.5 for missing data
        """)

        st.markdown("### Relationship Score (Software)")
        st.info(f"**Current Weight: {weights['relationship_score']:.3f}**")
        st.markdown("""
        - **Purpose**: Measures software engagement strength
        - **Method**: Log transformation of software revenue
        - **Revenue Types**: License, SaaS, Maintenance
        - **Scale**: 0.0 - 1.0 (higher = more software revenue)
        """)

    st.markdown("---")

    # Final scoring explanation
    st.markdown("##  Final Score Calculation")

    st.markdown("""
    ### Raw Score
    ```
    Raw Score = (Vertical  W_v) + (Size  W_s) + (Adoption  W_a) + (Relationship  W_r)
    ```
    Where weights sum to 1.0

    ### Normalization Process
    1. **Percentile Conversion**: Convert to percentile ranks
    2. **Normal Distribution**: Apply inverse CDF (bell curve)
    3. **Scaling**: Transform to 0-100 scale (mean=50, std dev=15)

    ### Grade Assignment
    - **A Grade**: Top 10% (90-100 percentile)
    - **B Grade**: Next 20% (70-90 percentile)
    - **C Grade**: Middle 40% (30-70 percentile)
    - **D Grade**: Next 20% (10-30 percentile)
    - **F Grade**: Bottom 10% (0-10 percentile)
    """)

    # Weight optimization info
    if optimization_data:
        st.markdown("---")
        st.markdown("##  Optimization Status")
        st.success(" ML-Optimized weights are active!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimization Trials", optimization_data.get('n_trials', 'N/A'))
        with col2:
            st.metric("Best Objective", f"{optimization_data.get('best_objective_value', 0):.4f}")
        with col3:
            st.metric("Lambda Parameter", optimization_data.get('lambda_param', 'N/A'))
    else:
        st.warning(" Using default weights. Run optimization for better results.")

# ---------------------------
# Call List Builder Page
# ---------------------------

def show_call_list_builder():
    st.markdown('<h1 class="main-header">Call List Builder</h1>', unsafe_allow_html=True)

    try:
        df = pd.read_csv(ROOT / 'data' / 'processed' / 'icp_scored_accounts.csv')
    except Exception:
        st.error("Could not load scored data. Run the scoring pipeline first.")
        return

    segment_options = ["All", "Small Business", "Mid-Market", "Large Enterprise"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        segment = st.selectbox("Segment", segment_options, index=0)
    with col2:
        industry = st.text_input("Industry contains", "")
    with col3:
        adoption_min = st.slider("Min Adoption Score", 0.0, 1.0, 0.5, 0.05)
    with col4:
        relationship_min = st.slider("Min Relationship Score", 0.0, 1.0, 0.3, 0.05)

    col5, col6, col7 = st.columns(3)
    with col5:
        revenue_only = st.checkbox("Revenue-only (no printers)", value=False)
    with col6:
        heavy_fleet = st.checkbox("Heavy fleet (≥10 weighted)", value=False)
    with col7:
        a_b_only = st.checkbox("A/B grades only", value=True)

    flt = df.copy()
    if segment != "All" and 'customer_segment' in flt.columns:
        flt = flt[flt['customer_segment'] == segment]
    if industry and 'Industry' in flt.columns:
        flt = flt[flt['Industry'].astype(str).str.contains(industry, case=False, na=False)]
    if 'adoption_score' in flt.columns:
        flt = flt[flt['adoption_score'] >= adoption_min]
    if 'relationship_score' in flt.columns:
        flt = flt[flt['relationship_score'] >= relationship_min]
    if revenue_only and {'Big Box Count','Small Box Count'}.issubset(flt.columns):
        weighted = (2.0 * flt['Big Box Count'].fillna(0)) + (1.0 * flt['Small Box Count'].fillna(0))
        flt = flt[weighted == 0]
    if heavy_fleet and {'Big Box Count','Small Box Count'}.issubset(flt.columns):
        weighted = (2.0 * flt['Big Box Count'].fillna(0)) + (1.0 * flt['Small Box Count'].fillna(0))
        flt = flt[weighted >= 10]
    if a_b_only and 'ICP_grade' in flt.columns:
        flt = flt[flt['ICP_grade'].isin(['A','B'])]

    sort_cols = []
    if 'ICP_score' in flt.columns:
        sort_cols.append(('ICP_score', False))
    if 'adoption_score' in flt.columns:
        sort_cols.append(('adoption_score', False))
    for c, asc in reversed(sort_cols):
        flt = flt.sort_values(by=c, ascending=asc)

    display_cols = [c for c in [
        'Company Name','customer_segment','Industry','ICP_grade','ICP_score','adoption_score','relationship_score',
        'Profit_Since_2023_Total'
    ] if c in flt.columns]

    st.subheader(f"Results ({len(flt)} accounts)")
    st.dataframe(flt[display_cols].head(500), use_container_width=True)

    if st.button("Export CSV"):
        out_dir = ROOT / 'reports' / 'call_lists' / pd.Timestamp.now().strftime('%Y%m%d')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"call_list_{pd.Timestamp.now().strftime('%H%M%S')}.csv"
        flt.to_csv(out_path, index=False)
        st.success(f"Exported to {out_path}")

if __name__ == "__main__":
    main() 





