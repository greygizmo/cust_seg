"""
Interactive ICP Scoring Dashboard
=================================
Real-time customer segmentation with adjustable weights for customer account analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="ICP Scoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    /* Old metric card styling - keep for other sections */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e3e6ea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border-color: #1f77b4;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1f77b4, #4dabf7);
    }
    
    .weight-section {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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

# Constants from original script
VERTICAL_WEIGHTS = {
    "aerospace & defense": 1.00,
    "aerospace": 1.00,
    "automotive & transportation": 0.90,
    "automotive": 0.90,
    "medical devices & life sci.": 0.85,
    "medical devices": 0.85,
    "high tech": 0.80,
    "consumer / cpg": 0.80,
    "consumer goods": 0.80,
    "industrial machinery": 0.70,
    "government": 0.75,
}

HIGH_PAIN_VERTICALS = {
    "aerospace & defense",
    "aerospace",
    "automotive & transportation",
    "automotive",
    "industrial machinery",
}

LICENSE_COL = "Total Software License Revenue"

# === CUSTOMER SEGMENTATION CONFIGURATION ===
DEFAULT_SEGMENT_THRESHOLDS = {
    'small_business_max': 1,      # 0-1 printers = Small Business
    'mid_market_max': 4,          # 2-4 printers = Mid-Market  
    # 5+ printers = Large Enterprise
}

def determine_customer_segment(printer_count, thresholds):
    """Determine customer segment based on printer count and configurable thresholds"""
    if printer_count <= thresholds['small_business_max']:
        return 'Small Business'
    elif printer_count <= thresholds['mid_market_max']:
        return 'Mid-Market'
    else:
        return 'Large Enterprise'

def get_segment_metrics(df, segment_name, segment_thresholds):
    """Calculate metrics for a specific customer segment"""
    # Add segment column if not exists
    if 'customer_segment' not in df.columns:
        df['customer_segment'] = df['printer_count'].apply(
            lambda x: determine_customer_segment(x, segment_thresholds)
        )
    
    segment_df = df[df['customer_segment'] == segment_name]
    
    metrics = {
        'count': len(segment_df),
        'avg_score': segment_df['ICP_score_new'].mean() if len(segment_df) > 0 else 0,
        'high_value_count': len(segment_df[segment_df['ICP_score_new'] >= 70]),
        'total_gp': segment_df['GP24'].sum() if 'GP24' in segment_df.columns else 0,
        'high_value_gp': segment_df[segment_df['ICP_score_new'] >= 70]['GP24'].sum() if 'GP24' in segment_df.columns else 0,
        'avg_printer_count': segment_df['printer_count'].mean() if len(segment_df) > 0 else 0
    }
    
    return metrics, segment_df

def create_segment_comparison_chart(df, segment_thresholds):
    """Create a comparison chart across customer segments"""
    # Ensure a fresh copy of the DataFrame for this chart to avoid side effects
    df_chart = df.copy()

    # Add segment column
    df_chart['customer_segment'] = df_chart['printer_count'].apply(
        lambda x: determine_customer_segment(x, segment_thresholds)
    )
    
    # Calculate metrics by segment
    agg_funcs = {
        'ICP_score_new': ['mean', 'count'],
        'printer_count': 'mean'
    }
    if 'GP24' in df_chart.columns:
        agg_funcs['GP24'] = 'sum'
    else: # Create a dummy GP24 column if it doesn't exist to prevent key errors
        df_chart['GP24'] = 0 
        agg_funcs['GP24'] = 'sum'
        
    segment_summary = df_chart.groupby('customer_segment').agg(agg_funcs).round(2)
    
    # Flatten column names
    segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns.values]
    segment_summary = segment_summary.reset_index()
    segment_summary = segment_summary.rename(columns={
        'ICP_score_new_mean': 'Avg_ICP_Score',
        'ICP_score_new_count': 'Customer_Count',
        'printer_count_mean': 'Avg_Printer_Count',
        'GP24_sum': 'Total_GP24'
    })

    # Calculate total customer count and total GP for percentage calculations
    total_customers = segment_summary['Customer_Count'].sum()
    total_gp_all = segment_summary['Total_GP24'].sum()

    # Calculate percentages
    segment_summary['Customer_Count_Pct'] = (segment_summary['Customer_Count'] / total_customers * 100) if total_customers > 0 else 0
    segment_summary['Total_GP24_Pct'] = (segment_summary['Total_GP24'] / total_gp_all * 100) if total_gp_all > 0 else 0
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Avg ICP Score by Segment', '% Customer Count by Segment', 
                       'Avg Printer Count by Segment', '% Total 24mo GP by Segment'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = {'Small Business': '#FF6B6B', 'Mid-Market': '#4ECDC4', 'Large Enterprise': '#45B7D1'}
    segment_order = ['Small Business', 'Mid-Market', 'Large Enterprise']
    segment_summary['customer_segment'] = pd.Categorical(segment_summary['customer_segment'], categories=segment_order, ordered=True)
    segment_summary = segment_summary.sort_values('customer_segment')

    fig.add_trace(
        go.Bar(x=segment_summary['customer_segment'], y=segment_summary['Avg_ICP_Score'],
               name='Avg ICP Score', marker_color=[colors.get(seg, '#999') for seg in segment_summary['customer_segment']],
               text=segment_summary['Avg_ICP_Score'].apply(lambda x: f'{x:.1f}'), textposition='auto'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=segment_summary['customer_segment'], y=segment_summary['Customer_Count_Pct'],
               name='% Customer Count', marker_color=[colors.get(seg, '#999') for seg in segment_summary['customer_segment']],
               text=segment_summary.apply(lambda row: f"{row['Customer_Count_Pct']:.1f}% ({row['Customer_Count']:,})", axis=1), 
               textposition='auto', hovertemplate='% Customer Count: %{y:.1f}%<br>Absolute Count: %{customdata[0]:,}<extra></extra>',
               customdata=segment_summary[['Customer_Count']]),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=segment_summary['customer_segment'], y=segment_summary['Avg_Printer_Count'],
               name='Avg Printer Count', marker_color=[colors.get(seg, '#999') for seg in segment_summary['customer_segment']],
               text=segment_summary['Avg_Printer_Count'].apply(lambda x: f'{x:.1f}'), textposition='auto'),
        row=2, col=1
    )
    
    if 'GP24' in df.columns: # Check original df, not df_chart which might have dummy GP24
        fig.add_trace(
            go.Bar(x=segment_summary['customer_segment'], y=segment_summary['Total_GP24_Pct'],
                   name='% Total 24mo GP', marker_color=[colors.get(seg, '#999') for seg in segment_summary['customer_segment']],
                   text=segment_summary.apply(lambda row: f"{row['Total_GP24_Pct']:.1f}% (${row['Total_GP24']:,.0f})", axis=1), 
                   textposition='auto', hovertemplate='% Total 24mo GP: %{y:.1f}%<br>Absolute GP: $%{customdata[0]:,.0f}<extra></extra>',
                   customdata=segment_summary[['Total_GP24']]),
            row=2, col=2
        )
    else:
        fig.add_trace(go.Bar(x=[],y=[]), row=2, col=2) # Empty trace if no GP24 data
        fig.layout.annotations[3].update(text="Total 24mo GP by Segment (No Data)")

    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    fig.update_yaxes(ticksuffix="%", row=2, col=2)
    
    fig.update_layout(
        title_text="Customer Segment Comparison (Metrics by Segment)",
        showlegend=False,
        height=700, # Increased height for better text visibility
        bargap=0.2
    )
    
    return fig

def create_segment_distribution_chart(df, segment_thresholds):
    """Create a distribution chart showing ICP scores within each segment"""
    # Add segment column
    df['customer_segment'] = df['printer_count'].apply(
        lambda x: determine_customer_segment(x, segment_thresholds)
    )
    
    fig = px.box(
        df, 
        x='customer_segment', 
        y='ICP_score_new',
        color='customer_segment',
        title="ICP Score Distribution by Customer Segment",
        labels={'customer_segment': 'Customer Segment', 'ICP_score_new': 'ICP Score'},
        color_discrete_map={'Small Business': '#FF6B6B', 'Mid-Market': '#4ECDC4', 'Large Enterprise': '#45B7D1'}
    )
    
    fig.update_layout(showlegend=False)
    return fig

@st.cache_data
def load_data():
    """Load the scored accounts data"""
    try:
        df = pd.read_csv('icp_scored_accounts.csv')
        
        # Handle different column name variations
        if 'vertical' in df.columns and 'Industry' not in df.columns:
            df['Industry'] = df['vertical']
        elif 'Industry' not in df.columns:
            # If neither exists, create a default Industry column
            df['Industry'] = 'Unknown'
            
        # Ensure numeric columns are properly typed
        numeric_cols = ['Big Box Count', 'Small Box Count', 'printer_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle the license column if it has a different name
        license_variations = [
            'Total Software License Revenue',
            'Total Software License Revenue',
            'cad_tier',
            'GP24',
            'Revenue24'
        ]
        
        # Find the license column
        for col_name in license_variations:
            if col_name in df.columns:
                break
        else:
            # If no license column found, create a default one
            df[LICENSE_COL] = 0
            
        return df
        
    except FileNotFoundError:
        st.error("❌ Could not find 'icp_scored_accounts.csv'. Please run the main scoring script first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def calculate_scores(df, weights, pain_config, size_config, cad_config):
    """Recalculate ICP scores with new weights and configurable logic"""
    df = df.copy()
    
    # Ensure we have the necessary columns
    if 'printer_count' not in df.columns:
        big_box = df.get('Big Box Count', 0)
        small_box = df.get('Small Box Count', 0)
        df['printer_count'] = pd.to_numeric(big_box, errors='coerce').fillna(0) + pd.to_numeric(small_box, errors='coerce').fillna(0)
    
    if 'scaling_flag' not in df.columns:
        df['scaling_flag'] = (df['printer_count'] >= 4).astype(int) # Default, can be made configurable later
    
    # --- Calculate individual scores --- 

    # 1. Vertical Score (using existing global VERTICAL_WEIGHTS for now)
    if 'Industry' in df.columns:
        v_lower = df['Industry'].astype(str).str.lower()
        df['vertical_score'] = v_lower.map(VERTICAL_WEIGHTS).fillna(0.5) # Default for unknown
    else:
        df['vertical_score'] = 0.5

    # 2. Size Score (configurable)
    min_printers = size_config['min_printers_sweet_spot']
    max_printers = size_config['max_printers_sweet_spot']
    score_in_sweet_spot = size_config['score_in_sweet_spot']
    score_outside_sweet_spot = size_config['score_outside_sweet_spot']
    df['size_score'] = np.where(
        df['printer_count'].between(min_printers, max_printers), 
        score_in_sweet_spot, 
        score_outside_sweet_spot
    )

    # 3. Adoption Score (using existing logic for now)
    df['adoption_score'] = df['scaling_flag'].astype(float)

    # 4. Relationship Score (configurable CAD tiers)
    if LICENSE_COL in df.columns:
        license_revenue = pd.to_numeric(df[LICENSE_COL], errors='coerce').fillna(0)
        
        # Use configurable thresholds
        bins = [-1, cad_config['bronze_max'], cad_config['silver_max'], cad_config['gold_max'], np.inf]
        labels = ["Bronze", "Silver", "Gold", "Platinum"]
        cad_tier_series = pd.cut(license_revenue, bins=bins, labels=labels)
        
        # Use configurable scores
        tier_map_local = {
            "Bronze": cad_config['bronze_score'], 
            "Silver": cad_config['silver_score'], 
            "Gold": cad_config['gold_score'], 
            "Platinum": cad_config['platinum_score']
        }
        
        relationship_scores_list = []
        for tier_val in cad_tier_series:
            if pd.isna(tier_val):
                relationship_scores_list.append(cad_config['missing_data_score'])
            else:
                relationship_scores_list.append(tier_map_local.get(str(tier_val), cad_config['missing_data_score']))
        df['relationship_score'] = relationship_scores_list
        df['cad_tier'] = cad_tier_series
    else:
        df['relationship_score'] = cad_config['bronze_score']  # Default to bronze score
        df['cad_tier'] = 'Bronze'
        df['cad_tier'] = pd.Categorical(df['cad_tier'], categories=["Bronze", "Silver", "Gold", "Platinum"])
    
    # 5. Pain Score (configurable - now requires both industry AND printer count)
    if 'Industry' in df.columns:
        v_lower_pain = df['Industry'].astype(str).str.lower()
        high_pain_list = [ind.lower() for ind in pain_config['high_pain_verticals']]
        min_printer_threshold = pain_config['min_printer_count_for_pain']
        
        # Pain score requires BOTH high-pain industry AND minimum printer count
        is_high_pain_industry = v_lower_pain.isin(high_pain_list)
        has_enough_printers = df['printer_count'] >= min_printer_threshold
        
        df['pain_score'] = np.where(
            is_high_pain_industry & has_enough_printers,
            pain_config['score_is_high_pain'],
            pain_config['score_is_not_high_pain']
        )
    else:
        df['pain_score'] = pain_config['score_is_not_high_pain']
    
    # Calculate new ICP score with current weights
    df['ICP_score_new'] = (
        df['vertical_score'] * weights['vertical'] +
        df['size_score'] * weights['size'] +
        df['adoption_score'] * weights['adoption'] +
        df['relationship_score'] * weights['relationship'] +
        df['pain_score'] * weights['pain']
    ) * 100
    
    return df

def create_score_distribution(df):
    """Create ICP score distribution chart"""
    fig = px.histogram(
        df, 
        x='ICP_score_new', 
        nbins=30,
        title="Distribution of ICP Scores with High-Value Threshold",
        labels={'ICP_score_new': 'ICP Score', 'count': 'Number of Customers'},
        color_discrete_sequence=['#1f77b4'],
        histnorm='probability density' # To better overlay KDE
    )

    # Add a vertical line for the high-value threshold
    fig.add_vline(x=70, line_dash="dash", line_color="red", 
                  annotation_text="High-Value (70+)", 
                  annotation_position="top right")
    
    fig.update_layout(
        xaxis_title="ICP Score",
        yaxis_title="Density / Number of Customers", # Adjusted Y-axis title
        showlegend=False,
        bargap=0.1 # Add some gap between bars
    )
    return fig

def create_score_by_vertical(df):
    """Create average score by vertical chart"""
    if df.empty or 'Industry' not in df.columns or 'ICP_score_new' not in df.columns:
        return go.Figure().update_layout(title_text="No data available for Industry Score Analysis")
        
    vertical_scores = df.groupby('Industry')['ICP_score_new'].agg(['mean', 'count']).reset_index()
    vertical_scores = vertical_scores[vertical_scores['count'] >= 1].nlargest(15, 'mean') # Show top 15, min 1 customer
    vertical_scores = vertical_scores.sort_values(by='mean', ascending=True) # For horizontal bar chart
    
    fig = px.bar(
        vertical_scores, 
        x='mean', 
        y='Industry',
        text=vertical_scores['count'].apply(lambda x: f" ({x} cust.)"), # Add count text
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
    
    chart_title = f"ICP Scoring Weight Distribution" # Weights are global, not segment-specific
    
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

def create_scatter_matrix(df):
    """Create scatter plot matrix of key metrics"""
    numeric_cols = ['vertical_score', 'size_score', 'adoption_score', 'relationship_score', 'pain_score']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        fig = px.scatter_matrix(
            df[available_cols + ['ICP_score_new']].sample(min(500, len(df))),
            dimensions=available_cols,
            color='ICP_score_new',
            title="Component Score Relationships",
            color_continuous_scale='viridis'
        )
        return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 ICP Scoring Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Customer Segmentation & Analysis**")
    
    # Load data
    df_loaded = load_data()
    
    all_industries = sorted(df_loaded['Industry'].astype(str).str.lower().unique().tolist())

    # === CUSTOMER SEGMENT CONFIGURATION ===
    st.markdown("## 🏢 Customer Segmentation")
    
    # Initialize segment configuration in session state
    if 'segment_config' not in st.session_state:
        st.session_state.segment_config = DEFAULT_SEGMENT_THRESHOLDS.copy()
    
    # Segment configuration in expandable section
    with st.expander("⚙️ Configure Customer Segments", expanded=False):
        st.markdown("**Define customer segments based on printer count thresholds:**")
        
        col_seg1, col_seg2 = st.columns(2)
        with col_seg1:
            small_max = st.number_input(
                "Small Business Max Printers", 
                min_value=0, 
                value=st.session_state.segment_config['small_business_max'], 
                step=1,
                help="Maximum printer count for Small Business segment"
            )
        with col_seg2:
            mid_max = st.number_input(
                "Mid-Market Max Printers", 
                min_value=small_max + 1, 
                value=max(st.session_state.segment_config['mid_market_max'], small_max + 1), 
                step=1,
                help="Maximum printer count for Mid-Market segment (Large Enterprise is above this)"
            )
        
        st.session_state.segment_config['small_business_max'] = small_max
        st.session_state.segment_config['mid_market_max'] = mid_max
        
        st.markdown(f"""
        **Current Segmentation:**
        - 🏪 **Small Business**: 0 - {small_max} printers
        - 🏢 **Mid-Market**: {small_max + 1} - {mid_max} printers  
        - 🏭 **Large Enterprise**: {mid_max + 1}+ printers
        """)
        
        if st.button("🔄 Reset Segment Thresholds", key="reset_segments"):
            st.session_state.segment_config = DEFAULT_SEGMENT_THRESHOLDS.copy()
            st.rerun()
    
    current_segment_config = st.session_state.segment_config.copy()
    
    # Add segment column to data
    df_loaded['customer_segment'] = df_loaded['printer_count'].apply(
        lambda x: determine_customer_segment(x, current_segment_config)
    )
    
    # Update selected segment from session state if it exists
    segment_options = ['All Segments', 'Small Business', 'Mid-Market', 'Large Enterprise']
    if 'selected_segment' in st.session_state:
        segment_index = segment_options.index(st.session_state.selected_segment) if st.session_state.selected_segment in segment_options else 0
    else:
        segment_index = 0
    
    # Segment selector
    st.markdown("### 📊 View by Customer Segment")
    selected_segment = st.selectbox(
        "Select Customer Segment to Analyze:",
        segment_options,
        index=segment_index,
        help="Filter dashboard view by customer segment"
    )
    
    # Update session state
    st.session_state.selected_segment = selected_segment
    
    # Filter data based on segment selection
    if selected_segment == 'All Segments':
        df_filtered = df_loaded.copy()
        dashboard_title_suffix = "All Customer Segments"
    else:
        df_filtered = df_loaded[df_loaded['customer_segment'] == selected_segment].copy()
        dashboard_title_suffix = f"{selected_segment} Customers"
    
    st.markdown(f"**Currently viewing: {dashboard_title_suffix}** ({len(df_filtered):,} customers)")

    # Show data info in an expander for debugging
    with st.expander("📋 Data Information", expanded=False):
        st.write(f"**Loaded {len(df_loaded):,} total customers, viewing {len(df_filtered):,} in current segment**")
        
        # Show segment breakdown
        if selected_segment == 'All Segments':
            segment_counts = df_loaded['customer_segment'].value_counts()
            st.write("**Customer Segment Breakdown:**")
            for segment, count in segment_counts.items():
                percentage = (count / len(df_loaded)) * 100
                st.write(f"  - {segment}: {count:,} customers ({percentage:.1f}%)")
        
        st.write("**Available columns:**")
        st.write(df_filtered.columns.tolist())
        st.write("**Sample data:**")
        st.dataframe(df_filtered.head(3))
        
        st.write("**Industry Analysis:**")
        st.write(f"**Unique Industries Found ({len(all_industries)}):**")
        st.write(all_industries)
        
        # Show the mapping between HIGH_PAIN_VERTICALS and actual data
        st.write("**High Pain Verticals Mapping:**")
        st.write("*Original HIGH_PAIN_VERTICALS constants:*")
        st.write(list(HIGH_PAIN_VERTICALS))
        
        # Find matches
        default_high_pain_debug = []
        for hpv in HIGH_PAIN_VERTICALS:
            hpv_lower = hpv.lower()
            matches = []
            for industry in all_industries:
                if hpv_lower == industry or hpv_lower in industry or industry in hpv_lower:
                    matches.append(industry)
            if matches:
                default_high_pain_debug.extend(matches)
                st.write(f"  - '{hpv}' → {matches}")
            else:
                st.write(f"  - '{hpv}' → ❌ No matches found")
        
        st.write("*Mapped high pain industries from your data:*")
        st.write(list(set(default_high_pain_debug)))
    
    # === Sidebar for weight controls ===
    st.sidebar.markdown("## 📊 Adjust Scoring Weights")
    st.sidebar.markdown("*Main category weights must sum to 1.0*")
    
    # Main Weight sliders
    with st.sidebar.container():
        st.markdown('<div class="weight-section">', unsafe_allow_html=True)
        # Initialize main weights in session state if not present
        if 'main_weights' not in st.session_state:
            st.session_state.main_weights = {
                'vertical': 0.30,
                'size': 0.20,
                'adoption': 0.25,
                'relationship': 0.15,
                'pain': 0.10
            }

        vertical_weight = st.slider(
            "🏭 Vertical Weight", 0.0, 1.0, st.session_state.main_weights['vertical'], 0.05, key="v_w"
        )
        size_weight = st.slider(
            "📏 Size Weight", 0.0, 1.0, st.session_state.main_weights['size'], 0.05, key="s_w"
        )
        adoption_weight = st.slider(
            "📈 Adoption Weight", 0.0, 1.0, st.session_state.main_weights['adoption'], 0.05, key="a_w"
        )
        relationship_weight = st.slider(
            "🤝 Relationship Weight", 0.0, 1.0, st.session_state.main_weights['relationship'], 0.05, key="r_w"
        )
        pain_weight = st.slider(
            "⚡ Pain Weight", 0.0, 1.0, st.session_state.main_weights['pain'], 0.05, key="p_w"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Update session state on change (necessary for reset)
    st.session_state.main_weights['vertical'] = vertical_weight
    st.session_state.main_weights['size'] = size_weight
    st.session_state.main_weights['adoption'] = adoption_weight
    st.session_state.main_weights['relationship'] = relationship_weight
    st.session_state.main_weights['pain'] = pain_weight
    
    current_main_weights = st.session_state.main_weights.copy()

    # Check if main weights sum to 1.0
    main_weight_sum = sum(current_main_weights.values())
    if abs(main_weight_sum - 1.0) > 0.01:
        st.sidebar.error(f"⚠️ Main weights sum to {main_weight_sum:.2f}, not 1.0")
    else:
        st.sidebar.success(f"✅ Main weights sum to {main_weight_sum:.2f}")
    
    if st.sidebar.button("🔄 Reset Main Weights to Defaults", key="reset_main_weights"):
        st.session_state.main_weights = {
            'vertical': 0.30, 'size': 0.20, 'adoption': 0.25, 'relationship': 0.15, 'pain': 0.10
        }
        st.rerun()

    # === QUICK SEGMENT SWITCHER ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🏢 Quick Segment Switch")
    
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.button("🏪 Small Business", key="switch_small", use_container_width=True):
            st.session_state.selected_segment = 'Small Business'
            st.rerun()
        if st.button("🏢 Mid-Market", key="switch_mid", use_container_width=True):
            st.session_state.selected_segment = 'Mid-Market'
            st.rerun()
    
    with col_sb2:
        if st.button("🏭 Large Enterprise", key="switch_large", use_container_width=True):
            st.session_state.selected_segment = 'Large Enterprise'
            st.rerun()
        if st.button("🌐 All Segments", key="switch_all", use_container_width=True):
            st.session_state.selected_segment = 'All Segments'
            st.rerun()
    
    # === Sidebar for Advanced Scoring Logic ===
    with st.sidebar.expander("🔧 Customize Criterion Scoring Logic", expanded=False):
        st.markdown("#### Pain Score Configuration")
        st.markdown("*Pain Score is now a 'kicker' - requires BOTH high-pain industry AND minimum printer count*")
        # Initialize pain_config in session state
        if 'pain_config' not in st.session_state:
            # Find matching industries from the actual data that correspond to HIGH_PAIN_VERTICALS
            default_high_pain = []
            for hpv in HIGH_PAIN_VERTICALS:
                hpv_lower = hpv.lower()
                # Find exact matches or close matches in the actual data
                for industry in all_industries:
                    if hpv_lower == industry or hpv_lower in industry or industry in hpv_lower:
                        if industry not in default_high_pain:
                            default_high_pain.append(industry)
            
            st.session_state.pain_config = {
                'high_pain_verticals': default_high_pain,
                'min_printer_count_for_pain': 4,
                'score_is_high_pain': 1.0,
                'score_is_not_high_pain': 0.0
            }

        # Ensure the current selection is valid (only includes industries that exist in options)
        valid_current_selection = [
            industry for industry in st.session_state.pain_config['high_pain_verticals'] 
            if industry in all_industries
        ]

        hpv_selection = st.multiselect(
            "High Pain Verticals", 
            options=all_industries, 
            default=valid_current_selection,
            key="hpv_ms"
        )
        
        min_printer_pain = st.number_input(
            "Min Printer Count for Pain Score", 
            min_value=0, 
            value=st.session_state.pain_config['min_printer_count_for_pain'], 
            step=1, 
            key="min_printer_pain_ni",
            help="Minimum number of printers required to trigger pain score (makes it a 'kicker' for scaling companies)"
        )
        
        sihp_val = st.slider("Score if High Pain", 0.0, 1.0, st.session_state.pain_config['score_is_high_pain'], 0.05, key="sihp_s")
        sinhp_val = st.slider("Score if Not High Pain", 0.0, 1.0, st.session_state.pain_config['score_is_not_high_pain'], 0.05, key="sinhp_s")

        st.session_state.pain_config['high_pain_verticals'] = hpv_selection
        st.session_state.pain_config['min_printer_count_for_pain'] = min_printer_pain
        st.session_state.pain_config['score_is_high_pain'] = sihp_val
        st.session_state.pain_config['score_is_not_high_pain'] = sinhp_val
        current_pain_config = st.session_state.pain_config.copy()

        st.markdown("---#### Size Score Configuration")
        # Initialize size_config in session state
        if 'size_config' not in st.session_state:
            st.session_state.size_config = {
                'min_printers_sweet_spot': 2,
                'max_printers_sweet_spot': 3,
                'score_in_sweet_spot': 1.0,
                'score_outside_sweet_spot': 0.5
            }

        min_pss_val = st.number_input("Min Printers in Sweet Spot", 
                                      min_value=0, value=st.session_state.size_config['min_printers_sweet_spot'], step=1, key="min_pss_ni")
        max_pss_val = st.number_input("Max Printers in Sweet Spot", 
                                      min_value=0, value=st.session_state.size_config['max_printers_sweet_spot'], step=1, key="max_pss_ni")
        siss_val = st.slider("Score in Sweet Spot", 0.0, 1.0, st.session_state.size_config['score_in_sweet_spot'], 0.05, key="siss_s")
        soss_val = st.slider("Score Outside Sweet Spot", 0.0, 1.0, st.session_state.size_config['score_outside_sweet_spot'], 0.05, key="soss_s")
        
        st.session_state.size_config['min_printers_sweet_spot'] = min_pss_val
        st.session_state.size_config['max_printers_sweet_spot'] = max_pss_val
        st.session_state.size_config['score_in_sweet_spot'] = siss_val
        st.session_state.size_config['score_outside_sweet_spot'] = soss_val
        current_size_config = st.session_state.size_config.copy()

        st.markdown("---#### CAD Tier Configuration")
        st.markdown("*Configure revenue thresholds and scores for relationship tiers*")
        # Initialize cad_config in session state
        if 'cad_config' not in st.session_state:
            st.session_state.cad_config = {
                'bronze_max': 5000,
                'silver_max': 25000,
                'gold_max': 100000,
                'bronze_score': 0.5,
                'silver_score': 0.7,
                'gold_score': 0.9,
                'platinum_score': 1.0,
                'missing_data_score': 0.2
            }

        bronze_max_val = st.number_input("Bronze Tier Max ($)", 
                                        min_value=0, value=st.session_state.cad_config['bronze_max'], step=1000, key="bronze_max_ni")
        silver_max_val = st.number_input("Silver Tier Max ($)", 
                                        min_value=0, value=st.session_state.cad_config['silver_max'], step=1000, key="silver_max_ni")
        gold_max_val = st.number_input("Gold Tier Max ($)", 
                                      min_value=0, value=st.session_state.cad_config['gold_max'], step=1000, key="gold_max_ni")
        
        col_cad1, col_cad2 = st.columns(2)
        with col_cad1:
            bronze_score_val = st.slider("Bronze Score", 0.0, 1.0, st.session_state.cad_config['bronze_score'], 0.05, key="bronze_score_s")
            silver_score_val = st.slider("Silver Score", 0.0, 1.0, st.session_state.cad_config['silver_score'], 0.05, key="silver_score_s")
        with col_cad2:
            gold_score_val = st.slider("Gold Score", 0.0, 1.0, st.session_state.cad_config['gold_score'], 0.05, key="gold_score_s")
            platinum_score_val = st.slider("Platinum Score", 0.0, 1.0, st.session_state.cad_config['platinum_score'], 0.05, key="platinum_score_s")
        
        missing_data_score_val = st.slider("Missing Data Score", 0.0, 1.0, st.session_state.cad_config['missing_data_score'], 0.05, key="missing_data_score_s")
        
        st.session_state.cad_config['bronze_max'] = bronze_max_val
        st.session_state.cad_config['silver_max'] = silver_max_val
        st.session_state.cad_config['gold_max'] = gold_max_val
        st.session_state.cad_config['bronze_score'] = bronze_score_val
        st.session_state.cad_config['silver_score'] = silver_score_val
        st.session_state.cad_config['gold_score'] = gold_score_val
        st.session_state.cad_config['platinum_score'] = platinum_score_val
        st.session_state.cad_config['missing_data_score'] = missing_data_score_val
        current_cad_config = st.session_state.cad_config.copy()

        if st.button("🔄 Reset Scoring Logic to Defaults", key="reset_logic_defaults"):
            # Find matching industries from the actual data that correspond to HIGH_PAIN_VERTICALS
            default_high_pain = []
            for hpv in HIGH_PAIN_VERTICALS:
                hpv_lower = hpv.lower()
                # Find exact matches or close matches in the actual data
                for industry in all_industries:
                    if hpv_lower == industry or hpv_lower in industry or industry in hpv_lower:
                        if industry not in default_high_pain:
                            default_high_pain.append(industry)
            
            st.session_state.pain_config = {
                'high_pain_verticals': default_high_pain,
                'min_printer_count_for_pain': 4,
                'score_is_high_pain': 1.0,
                'score_is_not_high_pain': 0.0
            }
            st.session_state.size_config = {
                'min_printers_sweet_spot': 2,
                'max_printers_sweet_spot': 3,
                'score_in_sweet_spot': 1.0,
                'score_outside_sweet_spot': 0.5
            }
            st.session_state.cad_config = {
                'bronze_max': 5000,
                'silver_max': 25000,
                'gold_max': 100000,
                'bronze_score': 0.5,
                'silver_score': 0.7,
                'gold_score': 0.9,
                'platinum_score': 1.0,
                'missing_data_score': 0.2
            }
            st.rerun()
    
    # Recalculate scores with current configurations
    df_scored = calculate_scores(df_filtered.copy(), current_main_weights, current_pain_config, current_size_config, current_cad_config)
    
    # Main dashboard
    st.markdown(f"## 📈 Key Metrics - {dashboard_title_suffix}")
    
    # Calculate metrics
    avg_score = df_scored['ICP_score_new'].mean()
    high_score_count = len(df_scored[df_scored['ICP_score_new'] >= 70])
    total_gp = df_scored['GP24'].sum() if 'GP24' in df_scored.columns else 0
    high_value_gp = df_scored[df_scored['ICP_score_new'] >= 70]['GP24'].sum() if 'GP24' in df_scored.columns else 0
    hv_percentage = (high_value_gp / total_gp * 100) if total_gp > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        metric_title = "Total Customers" if selected_segment == 'All Segments' else f"{selected_segment} Customers"
        st.markdown(f"""
        <div class="custom-metric metric-customers">
            <div class="metric-title">
                <span class="metric-icon">👥</span>{metric_title}
            </div>
            <div class="metric-value">{len(df_scored):,}</div>
            <div class="metric-subtitle">Active customer accounts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score_color = "#28a745" if avg_score >= 70 else "#ffc107" if avg_score >= 50 else "#dc3545"
        st.markdown(f"""
        <div class="custom-metric metric-score">
            <div class="metric-title">
                <span class="metric-icon">🎯</span>Average ICP Score
            </div>
            <div class="metric-value" style="color: {score_color};">{avg_score:.1f}</div>
            <div class="metric-subtitle">Out of 100 points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        hv_rate = (high_score_count / len(df_scored) * 100) if len(df_scored) > 0 else 0
        st.markdown(f"""
        <div class="custom-metric metric-high-value">
            <div class="metric-title">
                <span class="metric-icon">⭐</span>High-Value Customers
            </div>
            <div class="metric-value">{high_score_count:,}</div>
            <div class="metric-subtitle">{hv_rate:.1f}% of total (≥70 score)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'GP24' in df_scored.columns:
            st.markdown(f"""
            <div class="custom-metric metric-gp">
                <div class="metric-title">
                    <span class="metric-icon">💰</span>Total 24mo GP (HW)
                </div>
                <div class="metric-value">${total_gp:,.0f}</div>
                <div class="metric-subtitle">Hardware division revenue</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-metric metric-gp">
                <div class="metric-title">
                    <span class="metric-icon">💰</span>Total 24mo GP (HW)
                </div>
                <div class="metric-value">N/A</div>
                <div class="metric-subtitle">Data not available</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        if 'GP24' in df_scored.columns:
            st.markdown(f"""
            <div class="custom-metric metric-hv-gp">
                <div class="metric-title">
                    <span class="metric-icon">🚀</span>High-Value 24mo GP (HW)
                </div>
                <div class="metric-value">${high_value_gp:,.0f}</div>
                <div class="metric-subtitle">From top-tier customers</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-metric metric-hv-gp">
                <div class="metric-title">
                    <span class="metric-icon">🚀</span>High-Value 24mo GP (HW)
                </div>
                <div class="metric-value">N/A</div>
                <div class="metric-subtitle">Data not available</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        if 'GP24' in df_scored.columns:
            percentage_color = "#28a745" if hv_percentage >= 50 else "#ffc107" if hv_percentage >= 25 else "#dc3545"
            st.markdown(f"""
            <div class="custom-metric metric-percentage">
                <div class="metric-title">
                    <span class="metric-icon">📊</span>High-Value GP % (HW)
                </div>
                <div class="metric-value" style="color: {percentage_color};">{hv_percentage:.1f}%</div>
                <div class="metric-subtitle">Revenue concentration</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-metric metric-percentage">
                <div class="metric-title">
                    <span class="metric-icon">📊</span>High-Value GP % (HW)
                </div>
                <div class="metric-value">N/A</div>
                <div class="metric-subtitle">Data not available</div>
            </div>
            """, unsafe_allow_html=True)
    
    # === SEGMENT COMPARISON SECTION (only when viewing all segments) ===
    if selected_segment == 'All Segments':
        st.markdown("## 🏢 Customer Segment Analysis")
        st.markdown("*Compare performance across Small Business, Mid-Market, and Large Enterprise segments*")
        
        # Recalculate scores for all data to show segment comparison
        df_all_scored = calculate_scores(df_loaded.copy(), current_main_weights, current_pain_config, current_size_config, current_cad_config)
        
        # Segment comparison charts
        col_seg1, col_seg2 = st.columns(2)
        
        with col_seg1:
            fig_segment_comparison = create_segment_comparison_chart(df_all_scored, current_segment_config)
            st.plotly_chart(fig_segment_comparison, use_container_width=True)
        
        with col_seg2:
            fig_segment_distribution = create_segment_distribution_chart(df_all_scored, current_segment_config)
            st.plotly_chart(fig_segment_distribution, use_container_width=True)
        
        # Segment summary table
        st.markdown("### 📊 Segment Summary Table")
        segment_summary_data = []
        
        for segment in ['Small Business', 'Mid-Market', 'Large Enterprise']:
            metrics, _ = get_segment_metrics(df_all_scored, segment, current_segment_config)
            
            hv_percentage = (metrics['high_value_gp'] / metrics['total_gp'] * 100) if metrics['total_gp'] > 0 else 0
            
            segment_summary_data.append({
                'Customer Segment': segment,
                'Customer Count': f"{metrics['count']:,}",
                'Avg ICP Score': f"{metrics['avg_score']:.1f}",
                'High-Value Customers': f"{metrics['high_value_count']:,}",
                'Avg Printer Count': f"{metrics['avg_printer_count']:.1f}",
                'Total 24mo GP': f"${metrics['total_gp']:,.0f}" if 'GP24' in df_all_scored.columns else "N/A",
                'High-Value 24mo GP': f"${metrics['high_value_gp']:,.0f}" if 'GP24' in df_all_scored.columns else "N/A",
                'High-Value GP %': f"{hv_percentage:.1f}%" if 'GP24' in df_all_scored.columns else "N/A"
            })
        
        segment_summary_df = pd.DataFrame(segment_summary_data)
        st.dataframe(segment_summary_df, use_container_width=True)
    
    # === SEGMENT-SPECIFIC INSIGHTS (when viewing individual segments) ===
    elif selected_segment != 'All Segments':
        st.markdown(f"## 🔍 {selected_segment} Insights")
        
        # Calculate segment-specific metrics
        segment_metrics, _ = get_segment_metrics(df_scored, selected_segment, current_segment_config)
        
        # Show segment insights
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Avg Printer Count", 
                f"{segment_metrics['avg_printer_count']:.1f}",
                help=f"Average number of printers for {selected_segment} customers"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_insight2:
            if segment_metrics['count'] > 0:
                hv_percentage = (segment_metrics['high_value_count'] / segment_metrics['count']) * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "High-Value Rate", 
                    f"{hv_percentage:.1f}%",
                    help=f"Percentage of {selected_segment} customers with ICP score ≥ 70"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High-Value Rate", "0.0%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col_insight3:
            if 'GP24' in df_scored.columns and segment_metrics['total_gp'] > 0:
                avg_gp_per_customer = segment_metrics['total_gp'] / segment_metrics['count']
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Avg 24mo GP/Customer", 
                    f"${avg_gp_per_customer:,.0f}",
                    help=f"Average 24-month gross profit per {selected_segment} customer"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg 24mo GP/Customer", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Segment-specific recommendations
        st.markdown(f"### 💡 {selected_segment} Recommendations")
        
        if selected_segment == 'Small Business':
            st.info("""
            **Small Business Focus Areas:**
            - 🎯 Target customers with 0-1 printers who show growth potential
            - 📈 Focus on adoption and scaling opportunities  
            - 💼 Emphasize cost-effective solutions and ROI
            - 🤝 Build strong relationships for future expansion
            """)
        elif selected_segment == 'Mid-Market':
            st.info("""
            **Mid-Market Focus Areas:**
            - 🏢 Target customers with 2-10 printers showing scaling patterns
            - ⚡ Leverage pain points in high-pain industries
            - 🔄 Focus on workflow optimization and efficiency gains
            - 📊 Provide data-driven ROI demonstrations
            """)
        else:  # Large Enterprise
            st.info("""
            **Large Enterprise Focus Areas:**
            - 🏭 Target customers with 11+ printers for enterprise solutions
            - 🎯 Focus on strategic partnerships and long-term contracts
            - 🔧 Emphasize advanced features and customization
            - 💰 Highlight enterprise-level support and services
            """)
    
    # Charts
    chart_title = f"Real-time Analytics - {dashboard_title_suffix}" if selected_segment != 'All Segments' else "Real-time Analytics"
    st.markdown(f"## 📊 {chart_title}")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = create_score_distribution(df_scored)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_radar = create_score_components_radar(current_main_weights, dashboard_title_suffix)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vertical = create_score_by_vertical(df_scored)
        st.plotly_chart(fig_vertical, use_container_width=True)
    
    with col2:
        if 'printer_count' in df_scored.columns and not df_scored.empty:
            # Ensure 'Industry' column exists, provide a default if not
            df_scatter = df_scored.copy()
            if 'Industry' not in df_scatter.columns:
                df_scatter['Industry'] = 'N/A'
            else:
                df_scatter['Industry'] = df_scatter['Industry'].fillna('N/A')

            fig_scatter_printers = px.scatter(
                df_scatter.sample(min(500, len(df_scatter))),
                x='printer_count',
                y='ICP_score_new',
                color='Industry', # Color by Industry if available and diverse enough
                title=f"Printer Count vs ICP Score - {dashboard_title_suffix}",
                labels={'printer_count': 'Printer Count', 'ICP_score_new': 'ICP Score'},
                hover_data=['Company Name'] # Add company name to tooltip
            )
            fig_scatter_printers.update_layout(
                height=500, # Standard height
                coloraxis_showscale=False if len(df_scatter['Industry'].unique()) > 20 else True # Hide color scale if too many industries
            )
            st.plotly_chart(fig_scatter_printers, use_container_width=True)
        elif df_scored.empty:
            st.info(f"No customers in {dashboard_title_suffix} to display Printer Count vs ICP Score chart.")
        else:
            st.warning("Printer count data not available for scatter plot.")
    
    # Data table
    table_title = f"Top Scoring Customers - {dashboard_title_suffix}" if selected_segment != 'All Segments' else "Top Scoring Customers"
    st.markdown(f"## 📋 {table_title}")
    
    # Select columns to display (removed Customer ID, renamed columns)
    display_cols = ['Company Name', 'Industry', 'ICP_score_new', 'printer_count', 'cad_tier']
    if 'GP24' in df_scored.columns:
        display_cols.append('GP24')
    if 'customer_segment' in df_scored.columns and selected_segment == 'All Segments':
        display_cols.insert(2, 'customer_segment')  # Add segment column when viewing all segments
    
    available_cols = [col for col in display_cols if col in df_scored.columns]
    
    # Get top 100 customers and rename columns for display
    top_customers = df_scored.nlargest(100, 'ICP_score_new')[available_cols].copy()
    
    # Rename columns for better display
    column_renames = {
        'ICP_score_new': 'ICP Score',
        'printer_count': 'Printer Count',
        'cad_tier': 'CAD Tier',
        'GP24': '24mo GP',
        'customer_segment': 'Customer Segment'
    }
    top_customers = top_customers.rename(columns=column_renames)
    
    st.dataframe(top_customers, use_container_width=True)
    
    # Download button
    csv = df_scored.to_csv(index=False)
    
    # Create segment-specific filename
    if selected_segment == 'All Segments':
        filename_segment = "all_segments"
    else:
        filename_segment = selected_segment.lower().replace(' ', '_').replace('-', '_')
    
    download_filename = f"icp_scores_{filename_segment}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    st.download_button(
        label=f"📥 Download {dashboard_title_suffix} Scores (CSV)",
        data=csv,
        file_name=download_filename,
        mime="text/csv",
        help=f"Download ICP scores for {dashboard_title_suffix.lower()}"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("*Customer Segmentation Dashboard*")

if __name__ == "__main__":
    main() 