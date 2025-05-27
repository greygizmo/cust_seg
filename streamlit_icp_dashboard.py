"""
Interactive ICP Scoring Dashboard
=================================
Real-time customer segmentation with adjustable weights for GoEngineer Digital Manufacturing accounts.
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .weight-section {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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
        title="Distribution of ICP Scores",
        labels={'ICP_score_new': 'ICP Score', 'count': 'Number of Customers'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_title="ICP Score",
        yaxis_title="Number of Customers",
        showlegend=False
    )
    return fig

def create_score_by_vertical(df):
    """Create average score by vertical chart"""
    vertical_scores = df.groupby('Industry')['ICP_score_new'].agg(['mean', 'count']).reset_index()
    vertical_scores = vertical_scores[vertical_scores['count'] >= 3].nlargest(10, 'mean')
    
    fig = px.bar(
        vertical_scores, 
        x='mean', 
        y='Industry',
        title="Average ICP Score by Industry (Top 10)",
        labels={'mean': 'Average ICP Score', 'Industry': 'Industry'},
        color='mean',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_score_components_radar(weights):
    """Create radar chart showing weight distribution"""
    categories = list(weights.keys())
    values = list(weights.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Weights',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.5]
            )),
        title="ICP Scoring Weight Distribution",
        showlegend=False
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
    st.markdown("**GoEngineer Digital Manufacturing Customer Segmentation**")
    
    # Load data
    df_loaded = load_data()
    
    all_industries = sorted(df_loaded['Industry'].astype(str).str.lower().unique().tolist())

    # Show data info in an expander for debugging
    with st.expander("📋 Data Information", expanded=False):
        st.write(f"**Loaded {len(df_loaded):,} customers**")
        st.write("**Available columns:**")
        st.write(df_loaded.columns.tolist())
        st.write("**Sample data:**")
        st.dataframe(df_loaded.head(3))
        
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
    df_scored = calculate_scores(df_loaded.copy(), current_main_weights, current_pain_config, current_size_config, current_cad_config)
    
    # Main dashboard
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Customers", 
            f"{len(df_scored):,}",
            help="Total number of customers in dataset"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_score = df_scored['ICP_score_new'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Average ICP Score", 
            f"{avg_score:.1f}",
            help="Average ICP score with current weights"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        high_score_count = len(df_scored[df_scored['ICP_score_new'] >= 70])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "High-Value Customers", 
            f"{high_score_count:,}",
            help="Customers with ICP score ≥ 70"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if 'GP24' in df_scored.columns:
            total_gp = df_scored['GP24'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Total 24mo GP", 
                f"${total_gp:,.0f}",
                help="Total gross profit (24 months)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Total 24mo GP", 
                "N/A",
                help="Total 24mo GP not available"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        if 'GP24' in df_scored.columns:
            high_value_gp = df_scored[df_scored['ICP_score_new'] >= 70]['GP24'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "High-Value 24mo GP", 
                f"${high_value_gp:,.0f}",
                help="24mo GP from customers with ICP score ≥ 70"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "High-Value 24mo GP", 
                "N/A",
                help="High-Value 24mo GP not available"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        if 'GP24' in df_scored.columns:
            total_gp = df_scored['GP24'].sum()
            high_value_gp = df_scored[df_scored['ICP_score_new'] >= 70]['GP24'].sum()
            hv_percentage = (high_value_gp / total_gp * 100) if total_gp > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "High-Value GP %", 
                f"{hv_percentage:.1f}%",
                help="Percentage of total 24mo GP from high-value customers"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "High-Value GP %", 
                "N/A",
                help="High-Value GP percentage not available"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    st.markdown("## 📊 Real-time Analytics")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = create_score_distribution(df_scored)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        fig_radar = create_score_components_radar(current_main_weights)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vertical = create_score_by_vertical(df_scored)
        st.plotly_chart(fig_vertical, use_container_width=True)
    
    with col2:
        if 'printer_count' in df_scored.columns:
            fig_scatter_printers = px.scatter(
                df_scored.sample(min(500, len(df_scored))),
                x='printer_count',
                y='ICP_score_new',
                color='Industry',
                title="Printer Count vs ICP Score",
                labels={'printer_count': 'Printer Count', 'ICP_score_new': 'ICP Score'}
            )
            st.plotly_chart(fig_scatter_printers, use_container_width=True)
        else:
            st.write("Printer count data not available for scatter plot.")
    
    # Data table
    st.markdown("## 📋 Top Scoring Customers")
    
    # Select columns to display (removed Customer ID, renamed columns)
    display_cols = ['Company Name', 'Industry', 'ICP_score_new', 'printer_count', 'cad_tier']
    if 'GP24' in df_scored.columns:
        display_cols.append('GP24')
    
    available_cols = [col for col in display_cols if col in df_scored.columns]
    
    # Get top 100 customers and rename columns for display
    top_customers = df_scored.nlargest(100, 'ICP_score_new')[available_cols].copy()
    
    # Rename columns for better display
    column_renames = {
        'ICP_score_new': 'ICP Score',
        'printer_count': 'Printer Count',
        'cad_tier': 'CAD Tier',
        'GP24': '24mo GP'
    }
    top_customers = top_customers.rename(columns=column_renames)
    
    st.dataframe(top_customers, use_container_width=True)
    
    # Download button
    csv = df_scored.to_csv(index=False)
    st.download_button(
        label="📥 Download Updated Scores (CSV)",
        data=csv,
        file_name=f"icp_scores_interactive_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard built for GoEngineer Digital Manufacturing Customer Segmentation*")

if __name__ == "__main__":
    main() 