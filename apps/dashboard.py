import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add src to path so we can import if needed (though mostly we just read CSVs)
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

st.set_page_config(
    page_title="ICP Scoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load scored accounts and neighbors."""
    data_dir = ROOT / "data" / "processed"
    artifacts_dir = ROOT / "artifacts"
    
    scored_path = data_dir / "icp_scored_accounts.csv"
    neighbors_path = artifacts_dir / "account_neighbors.csv"
    playbooks_path = artifacts_dir / "account_playbooks.csv"
    
    scored = pd.read_csv(scored_path) if scored_path.exists() else None
    neighbors = pd.read_csv(neighbors_path) if neighbors_path.exists() else None
    playbooks = pd.read_csv(playbooks_path) if playbooks_path.exists() else None
    
    return scored, neighbors, playbooks

scored_df, neighbors_df, playbooks_df = load_data()

if scored_df is None:
    st.error("❌ Could not find `data/processed/icp_scored_accounts.csv`. Please run the scoring pipeline first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

# Division / Industry Filters
if "Industry" in scored_df.columns:
    all_industries = sorted(scored_df["Industry"].dropna().unique().tolist())
    selected_industries = st.sidebar.multiselect("Industry", all_industries, default=[])
else:
    selected_industries = []

# Score Range
if "Hardware_score" in scored_df.columns:
    min_score, max_score = st.sidebar.slider("Hardware Score Range", 0.0, 100.0, (0.0, 100.0))
else:
    min_score, max_score = 0.0, 100.0

# Apply Filters
filtered_df = scored_df.copy()
if selected_industries:
    filtered_df = filtered_df[filtered_df["Industry"].isin(selected_industries)]

if "Hardware_score" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["Hardware_score"] >= min_score) & 
        (filtered_df["Hardware_score"] <= max_score)
    ]

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("🚀 ICP Scoring Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Overview", "🔍 Account Explorer", "🤝 Neighbor Visualizer", "📘 Playbooks"])

# --- Tab 1: Portfolio Overview ---
with tab1:
    st.header("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accounts", f"{len(filtered_df):,}")
    
    if "Hardware_score" in filtered_df.columns:
        avg_score = filtered_df["Hardware_score"].mean()
        col2.metric("Avg Hardware Score", f"{avg_score:.2f}")
        
    if "Software_score" in filtered_df.columns:
        avg_sw_score = filtered_df["Software_score"].mean()
        col3.metric("Avg Software Score", f"{avg_sw_score:.2f}")
        
    if "GP_Since_2023_Total" in filtered_df.columns:
        total_gp = filtered_df["GP_Since_2023_Total"].sum()
        col4.metric("Total GP (Since 2023)", f"${total_gp:,.0f}")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        if "Hardware_score" in filtered_df.columns:
            fig = px.histogram(filtered_df, x="Hardware_score", nbins=20, title="Hardware Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        if "Industry" in filtered_df.columns:
            top_inds = filtered_df["Industry"].value_counts().head(10).reset_index()
            top_inds.columns = ["Industry", "Count"]
            fig = px.bar(top_inds, x="Count", y="Industry", orientation='h', title="Top 10 Industries")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Account Explorer ---
with tab2:
    st.header("Account Explorer")
    st.dataframe(filtered_df, use_container_width=True)

# --- Tab 3: Neighbor Visualizer ---
with tab3:
    st.header("Neighbor Visualizer")
    
    if neighbors_df is None:
        st.warning("No neighbors artifact found.")
    else:
        # Account Selector
        # Limit to filtered accounts for performance, but allow searching by ID or Name
        search_opts = filtered_df["Customer ID"].astype(str) + " - " + filtered_df["Company Name"].astype(str)
        selected_account_str = st.selectbox("Select Account", search_opts)
        
        if selected_account_str:
            selected_id = selected_account_str.split(" - ")[0]
            
            # Get Neighbors
            # account_neighbors.csv has columns: account_id, neighbor_account_id, neighbor_rank, sim_overall, ...
            account_neighbors = neighbors_df[neighbors_df["account_id"].astype(str) == selected_id].copy()
            
            if account_neighbors.empty:
                st.info("No neighbors found for this account.")
            else:
                st.subheader(f"Neighbors for {selected_account_str}")
                
                # Join with scored data to get names/details of neighbors
                # Ensure neighbor_account_id is string for merge
                account_neighbors["neighbor_account_id"] = account_neighbors["neighbor_account_id"].astype(str)
                scored_df["Customer ID"] = scored_df["Customer ID"].astype(str)
                
                enriched_neighbors = account_neighbors.merge(
                    scored_df, 
                    left_on="neighbor_account_id", 
                    right_on="Customer ID", 
                    how="left"
                )
                
                # Display
                cols_to_show = ["neighbor_account_id", "Company Name", "sim_overall", "neighbor_rank", "Industry", "Hardware_score"]
                # Filter cols that exist
                cols_to_show = [c for c in cols_to_show if c in enriched_neighbors.columns]
                
                st.dataframe(
                    enriched_neighbors[cols_to_show].sort_values("neighbor_rank"),
                    use_container_width=True
                )

# --- Tab 4: Playbooks ---
with tab4:
    st.header("Account Playbooks")
    
    if playbooks_df is None:
        st.warning("No playbooks artifact found. Run `python -m icp.cli.build_playbooks`.")
    else:
        # Filter playbooks by selected account if one is selected in Neighbor Visualizer, 
        # or just show all / filter by industry
        
        # Join with scored data for context
        # Ensure columns exist for merge
        if "Customer ID" in scored_df.columns:
            # Ensure types match for merge
            playbooks_df["customer_id"] = playbooks_df["customer_id"].astype(str)
            scored_df["Customer ID"] = scored_df["Customer ID"].astype(str)
            
            playbooks_enriched = playbooks_df.merge(
                scored_df[["Customer ID", "Company Name", "Industry"]],
                left_on="customer_id",
                right_on="Customer ID",
                how="left"
            )
            
            # Apply sidebar filters
            if selected_industries:
                playbooks_enriched = playbooks_enriched[playbooks_enriched["Industry"].isin(selected_industries)]
                
            st.dataframe(playbooks_enriched, use_container_width=True)
        else:
            st.dataframe(playbooks_df, use_container_width=True)
