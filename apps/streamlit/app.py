"""Streamlit dashboard for sales leadership to surface ICP portfolio insights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys

# Ensure project source is available for optional imports
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Try to import scoring utilities. The dashboard gracefully handles environments
# where the scoring module is unavailable (e.g., minimal deployments).
try:  # pragma: no cover - optional import for local environments
    from icp.scoring import DEFAULT_WEIGHTS
except Exception:  # pragma: no cover - dashboard falls back to static weights
    DEFAULT_WEIGHTS = {
        "vertical": 0.3,
        "size": 0.2,
        "adoption": 0.3,
        "relationship": 0.2,
    }


st.set_page_config(
    page_title="Revenue Acceleration Command Center",
    page_icon=":bar_chart:",
    layout="wide",
)

GRADE_ORDER = ["A", "B", "C", "D", "F"]
GRADE_COLOR_MAP = {
    "A": "#1b9e77",
    "B": "#66a61e",
    "C": "#e6ab02",
    "D": "#d95f02",
    "F": "#7570b3",
}

PROFIT_LENSES = (
    {"key": "profit_since_2023", "label": "GP Since Jan 2023", "help": "Total gross profit captured since the 2023 reset."},
    {"key": "profit_t4q", "label": "Trailing 4Q GP", "help": "Rolling four-quarter gross profit."},
    {"key": "profit_last_q", "label": "Last Quarter GP", "help": "Most recent quarter gross profit."},
)

COLUMN_CANDIDATES = {
    "profit_since_2023": [
        "GP_Since_2023_Total",
        "Profit_Since_2023_Total",
        "GP_Since_2023",
        "Profit_Since_2023",
    ],
    "profit_t4q": [
        "GP_T4Q_Total",
        "Profit_T4Q_Total",
        "GP_T4Q",
        "Profit_T4Q",
    ],
    "profit_last_q": [
        "GP_LastQ_Total",
        "Profit_LastQ_Total",
        "GP_LastQ",
        "Profit_LastQ",
    ],
    "component_vertical": ["vertical_score", "Vertical_score", "vertical_component"],
    "component_size": ["size_score", "Size_score", "size_component"],
    "component_adoption": [
        "Hardware_score",
        "adoption_score",
        "adoption_assets",
    ],
    "component_relationship": [
        "Software_score",
        "relationship_score",
        "relationship_profit",
    ],
    "score_raw": ["ICP_score_raw"],
    "score": ["ICP_score", "score"],
    "grade": ["ICP_grade", "grade"],
    "activity_segment": ["activity_segment"],
    "customer_segment": ["customer_segment"],
    "territory": ["AM_Territory", "territory"],
    "sales_rep": ["am_sales_rep", "AM_Sales_Rep"],
    "industry": ["Industry", "industry"],
    "company_name": ["Company Name", "company_name", "Account Name"],
    "customer_id": ["Customer ID", "customer_id", "Account ID"],
    "gp_qoq_growth": ["GP_QoQ_Growth", "gp_qoq_growth"],
}

NUMERIC_DEFAULTS = {
    "profit_since_2023": 0.0,
    "profit_t4q": 0.0,
    "profit_last_q": 0.0,
    "component_vertical": 0.5,
    "component_size": 0.5,
    "component_adoption": 0.5,
    "component_relationship": 0.5,
    "score_raw": 0.0,
    "score": 50.0,
    "gp_qoq_growth": 0.0,
}


@dataclass
class FilterState:
    score_range: tuple[float, float]
    grades: Sequence[str]
    segments: Sequence[str]
    industries: Sequence[str]
    territories: Sequence[str]
    reps: Sequence[str]
    activities: Sequence[str]
    adoption_focus: str
    min_profit: float
    search_text: str
    profit_key: str
    profit_label: str


@st.cache_data(show_spinner=False)
def load_portfolio_data() -> tuple[pd.DataFrame, Path | None]:
    """Load the scored portfolio from disk or fall back to a synthetic sample."""

    candidate_files = [
        ROOT / "data" / "processed" / "icp_scored_accounts.parquet",
        ROOT / "data" / "processed" / "icp_scored_accounts.csv",
        ROOT / "artifacts" / "icp_scored_accounts.parquet",
        ROOT / "artifacts" / "icp_scored_accounts.csv",
        ROOT / "reports" / "icp_scored_accounts.parquet",
        ROOT / "reports" / "icp_scored_accounts.csv",
    ]

    for path in candidate_files:
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            return df, path

    return create_sample_portfolio(), None


def create_sample_portfolio(num_accounts: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate a representative synthetic dataset when production data is unavailable."""

    rng = np.random.default_rng(seed)
    industries = [
        "High Tech",
        "Medical Devices",
        "Industrial Machinery",
        "Automotive",
        "Aerospace & Defense",
        "Education",
        "Energy",
    ]
    territories = ["West", "Central", "Southeast", "Northeast", "Canada"]
    reps = [
        "J. Chen",
        "M. Patel",
        "S. Ramirez",
        "T. Williams",
        "A. Johnson",
        "L. Dubois",
    ]
    segments = ["Strategic", "Growth", "Core"]
    activity = ["Warm", "Cold", "Dormant"]

    base_profit = rng.gamma(shape=2.8, scale=45000, size=num_accounts)
    t4q = base_profit * rng.uniform(0.55, 0.9, size=num_accounts)
    last_q = t4q * rng.uniform(0.18, 0.35, size=num_accounts)

    components = np.column_stack([
        rng.beta(3, 2, size=num_accounts),  # vertical
        rng.beta(2.5, 2, size=num_accounts),  # size
        rng.beta(2, 2.5, size=num_accounts),  # adoption
        rng.beta(2.8, 2, size=num_accounts),  # relationship
    ])
    weights = np.array([
        DEFAULT_WEIGHTS.get("vertical", 0.3),
        DEFAULT_WEIGHTS.get("size", 0.2),
        DEFAULT_WEIGHTS.get("adoption", 0.3),
        DEFAULT_WEIGHTS.get("relationship", 0.2),
    ])
    weighted = components @ weights
    score = (weighted - weighted.mean()) / (weighted.std() + 1e-6) * 15 + 50
    score = np.clip(score, 0, 100)

    df = pd.DataFrame(
        {
            "Customer ID": [f"{100000 + i}" for i in range(num_accounts)],
            "Company Name": [
                f"{rng.choice(['Nova', 'Apex', 'Vertex', 'Quantum', 'Blue'])} {rng.choice(['Systems', 'Labs', 'Dynamics', 'Robotics', 'Solutions'])}"
                for _ in range(num_accounts)
            ],
            "Industry": rng.choice(industries, size=num_accounts, p=[0.18, 0.15, 0.16, 0.14, 0.13, 0.12, 0.12]),
            "AM_Territory": rng.choice(territories, size=num_accounts),
            "am_sales_rep": rng.choice(reps, size=num_accounts),
            "activity_segment": rng.choice(activity, size=num_accounts, p=[0.6, 0.25, 0.15]),
            "customer_segment": rng.choice(segments, size=num_accounts, p=[0.25, 0.5, 0.25]),
            "profit_since_2023": base_profit,
            "profit_t4q": t4q,
            "profit_last_q": last_q,
            "component_vertical": components[:, 0],
            "component_size": components[:, 1],
            "component_adoption": components[:, 2],
            "component_relationship": components[:, 3],
            "gp_qoq_growth": rng.normal(loc=0.06, scale=0.18, size=num_accounts),
            "score": score,
        }
    )
    df["grade"] = df["score"].apply(score_to_grade)
    return df


def score_to_grade(score: float) -> str:
    """Map numeric ICP score (0-100) to grade buckets following the 10/20/40/20/10 targets."""

    if pd.isna(score):
        return "F"
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 45:
        return "C"
    if score >= 30:
        return "D"
    return "F"


def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, fill gaps, and compute derived metrics."""

    df = df.copy()

    # Normalize column availability
    for canonical, candidates in COLUMN_CANDIDATES.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df[canonical] = df[candidate]
                break

    for col, default in NUMERIC_DEFAULTS.items():
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(default)

    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(50.0).clip(0, 100)
    if "grade" not in df.columns:
        df["grade"] = df["score"].apply(score_to_grade)
    df["grade"] = df["grade"].astype(str).str.upper().str.strip()
    df.loc[~df["grade"].isin(GRADE_ORDER), "grade"] = "F"

    df["company_name"] = df.get("company_name", df.get("Company Name"))
    if "company_name" not in df or df["company_name"].isna().all():
        df["company_name"] = df.index.map(lambda i: f"Account {i}")

    if "customer_id" not in df.columns:
        df["customer_id"] = df.index.astype(str)
    df["customer_id"] = df["customer_id"].astype(str)

    if "customer_segment" not in df.columns or df["customer_segment"].isna().all():
        quantiles = df["profit_since_2023"].quantile([0.5, 0.85]).tolist()
        bins = [-np.inf, quantiles[0], quantiles[1], np.inf]
        labels = ["Core", "Growth", "Strategic"]
        df["customer_segment"] = pd.cut(df["profit_since_2023"], bins=bins, labels=labels)
        df["customer_segment"] = df["customer_segment"].astype(str)
    else:
        df["customer_segment"] = df["customer_segment"].fillna("Unassigned").astype(str)

    if "activity_segment" not in df.columns:
        df["activity_segment"] = np.where(df["profit_last_q"] > 0, "Warm", "Dormant")
    df["activity_segment"] = df["activity_segment"].fillna("Unknown").astype(str)

    df["territory"] = df.get("territory", df.get("AM_Territory", "Unassigned"))
    df["territory"] = df["territory"].fillna("Unassigned").astype(str)

    df["sales_rep"] = df.get("sales_rep", df.get("am_sales_rep", "Unassigned"))
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned").astype(str)

    df["industry"] = df.get("industry", df.get("Industry", "Unknown"))
    df["industry"] = df["industry"].fillna("Unknown").astype(str)

    df["adoption_maturity"] = df[["component_adoption", "component_relationship"]].mean(axis=1)
    df["whitespace_score"] = (1 - df["adoption_maturity"]).clip(0, 1)
    df["whitespace_value"] = df["profit_since_2023"] * df["whitespace_score"]

    df["growth_flag"] = pd.cut(
        df["gp_qoq_growth"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["Declining", "Stable", "Accelerating"],
    ).astype(str)

    df["score_band"] = pd.cut(
        df["score"],
        bins=[0, 30, 45, 65, 80, 100],
        labels=["<30", "30-45", "45-65", "65-80", "80+"],
        include_lowest=True,
    )
    return df


def render_sidebar(df: pd.DataFrame) -> FilterState:
    """Render sidebar controls and collect filter selections."""

    st.sidebar.title("Portfolio Controls")

    lens_labels = [lens["label"] for lens in PROFIT_LENSES]
    selected_label = st.sidebar.radio(
        "Profit lens",
        lens_labels,
        help="Switch between cumulative, trailing, or most recent gross profit lenses.",
    )
    selected_lens = next(lens for lens in PROFIT_LENSES if lens["label"] == selected_label)
    profit_key = selected_lens["key"]
    profit_label = selected_lens["label"]

    min_score, max_score = float(df["score"].min()), float(df["score"].max())
    score_range = st.sidebar.slider(
        "ICP score range",
        min_value=0.0,
        max_value=100.0,
        value=(min(30.0, min_score), max(85.0, max_score)),
        step=1.0,
    )

    grade_selection = st.sidebar.multiselect(
        "Grades",
        options=GRADE_ORDER,
        default=GRADE_ORDER[:4],
    )

    segment_options = sorted(df["customer_segment"].dropna().unique().tolist())
    segments = st.sidebar.multiselect(
        "Customer segment",
        options=segment_options,
        default=segment_options,
    )

    activity_options = sorted(df["activity_segment"].dropna().unique().tolist())
    activities = st.sidebar.multiselect(
        "Activity",
        options=activity_options,
        default=activity_options,
    )

    industry_options = sorted(df["industry"].dropna().unique().tolist())
    industries = st.sidebar.multiselect(
        "Industry",
        options=industry_options,
        default=industry_options[: min(10, len(industry_options))],
    )

    territory_options = sorted(df["territory"].dropna().unique().tolist())
    territories = st.sidebar.multiselect(
        "Territory",
        options=territory_options,
        default=territory_options,
    )

    rep_options = sorted(df["sales_rep"].dropna().unique().tolist())
    reps = st.sidebar.multiselect(
        "Account owner",
        options=rep_options,
        default=rep_options,
    )

    adoption_focus = st.sidebar.selectbox(
        "Expansion focus",
        options=[
            "All accounts",
            "Expansion whitespace (>40%)",
            "High adoption (>70%)",
        ],
    )

    max_profit = float(df[profit_key].max())
    default_threshold = 0.0 if max_profit <= 0 else 0.05 * max_profit
    min_profit = st.sidebar.number_input(
        f"Minimum {profit_label}",
        min_value=0.0,
        value=float(round(default_threshold, -3)) if max_profit > 20000 else default_threshold,
        step=max(1000.0, max_profit / 50 if max_profit > 0 else 1000.0),
        help="Exclude smaller accounts to sharpen focus on material opportunities.",
    )

    search_text = st.sidebar.text_input(
        "Search",
        placeholder="Company, ID, or territory",
    )

    return FilterState(
        score_range=score_range,
        grades=grade_selection or GRADE_ORDER,
        segments=segments or segment_options,
        industries=industries or industry_options,
        territories=territories or territory_options,
        reps=reps or rep_options,
        activities=activities or activity_options,
        adoption_focus=adoption_focus,
        min_profit=min_profit,
        search_text=search_text.strip(),
        profit_key=profit_key,
        profit_label=profit_label,
    )


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Apply sidebar selections to the portfolio."""

    mask = (
        df["score"].between(filters.score_range[0], filters.score_range[1])
        & df["grade"].isin(filters.grades)
        & df["customer_segment"].isin(filters.segments)
        & df["industry"].isin(filters.industries)
        & df["territory"].isin(filters.territories)
        & df["sales_rep"].isin(filters.reps)
        & df["activity_segment"].isin(filters.activities)
        & (df[filters.profit_key] >= filters.min_profit)
    )

    df_filtered = df[mask].copy()

    if filters.adoption_focus == "Expansion whitespace (>40%)":
        df_filtered = df_filtered[df_filtered["whitespace_score"] >= 0.4]
    elif filters.adoption_focus == "High adoption (>70%)":
        df_filtered = df_filtered[df_filtered["adoption_maturity"] >= 0.7]

    if filters.search_text:
        search_lower = filters.search_text.lower()
        df_filtered = df_filtered[
            df_filtered["company_name"].str.lower().str.contains(search_lower)
            | df_filtered["customer_id"].str.lower().str.contains(search_lower)
            | df_filtered["territory"].str.lower().str.contains(search_lower)
        ]

    return df_filtered


def render_metric_css() -> None:
    """Inject CSS for custom metric cards."""

    st.markdown(
        """
        <style>
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f4f7fb 100%);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(15, 23, 42, 0.04);
        }
        .metric-card h4 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4b5563;
            margin: 0;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.25rem;
        }
        .metric-card .context {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame, filters: FilterState) -> None:
    """Display executive-level KPIs."""

    render_metric_css()

    total_accounts = len(df)
    total_profit = df[filters.profit_key].sum()
    high_grade = df[df["grade"].isin(["A", "B"])]
    high_grade_profit = high_grade[filters.profit_key].sum()
    whitespace_value = df["whitespace_value"].sum()
    avg_score = df["score"].mean() if total_accounts else 0.0
    adoption_avg = df["adoption_maturity"].mean() if total_accounts else 0.0
    accelerating_share = (
        len(df[df["growth_flag"] == "Accelerating"]) / total_accounts if total_accounts else 0.0
    )

    cards = [
        {
            "title": "Accounts in Focus",
            "value": f"{total_accounts:,}",
            "context": f"Filtered from {filters.profit_label.lower()} lens",
        },
        {
            "title": filters.profit_label,
            "value": format_currency(total_profit),
            "context": "Gross profit under management",
        },
        {
            "title": "A/B Coverage",
            "value": f"{(len(high_grade) / total_accounts * 100 if total_accounts else 0):.1f}%",
            "context": f"{len(high_grade):,} accounts • {format_currency(high_grade_profit)} GP",
        },
        {
            "title": "Expansion White Space",
            "value": format_currency(whitespace_value),
            "context": f"Avg adoption {adoption_avg:.0%} • {accelerating_share:.0%} accelerating",
        },
        {
            "title": "Average ICP Score",
            "value": f"{avg_score:.1f}",
            "context": "Portfolio quality across filters",
        },
    ]

    st.markdown('<div class="metric-grid">' + "".join(
        f"<div class='metric-card'><h4>{card['title']}</h4><div class='value'>{card['value']}</div><div class='context'>{card['context']}</div></div>"
        for card in cards
    ) + "</div>", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def render_leadership_headlines(df: pd.DataFrame, filters: FilterState) -> None:
    """Generate narrative highlights for sales leadership."""

    st.markdown("#### Leadership Headlines")
    if df.empty:
        st.info("No accounts match the current filters. Adjust the sidebar to widen the aperture.")
        return

    profit_col = filters.profit_key
    total_profit = df[profit_col].sum()

    grade_summary = (
        df.groupby("grade")[profit_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "profit", "count": "accounts"})
    )
    grade_summary["profit_share"] = grade_summary["profit"] / max(total_profit, 1e-9)
    grade_summary = grade_summary.reindex(GRADE_ORDER).fillna(0)
    top_grade = grade_summary.sort_values("profit", ascending=False).head(1)

    industry_summary = (
        df.groupby("industry")[profit_col]
        .sum()
        .sort_values(ascending=False)
    )
    top_industry = industry_summary.head(1)

    territory_summary = (
        df.groupby("territory")[profit_col]
        .sum()
        .sort_values(ascending=False)
    )
    top_territory = territory_summary.head(1)

    whitespace_segment = (
        df.groupby("customer_segment")["whitespace_value"]
        .sum()
        .sort_values(ascending=False)
    )
    top_whitespace = whitespace_segment.head(1)

    accelerating = df[df["growth_flag"] == "Accelerating"]
    accelerating_profit = accelerating[profit_col].sum()

    bullets = []
    if not top_grade.empty:
        grade = top_grade.index[0]
        profit = top_grade["profit"].iloc[0]
        share = top_grade["profit_share"].iloc[0]
        bullets.append(
            f"**Grade {grade}** accounts are delivering {format_currency(profit)} ({share:.0%}) of {filters.profit_label.lower()} in scope."
        )
    if not top_industry.empty:
        industry = top_industry.index[0]
        share = top_industry.iloc[0] / max(total_profit, 1e-9)
        bullets.append(
            f"**{industry}** leads the portfolio at {share:.0%} of profit; ensure playbooks stay sharp here."
        )
    if not top_territory.empty:
        territory = top_territory.index[0]
        share = top_territory.iloc[0] / max(total_profit, 1e-9)
        bullets.append(
            f"**{territory}** territory contributes {share:.0%} of profit; replicate their coverage motion across regions."
        )
    if not top_whitespace.empty:
        segment = top_whitespace.index[0]
        value = top_whitespace.iloc[0]
        bullets.append(
            f"**{segment}** segment hides {format_currency(value)} in expansion whitespace—prioritize enablement and campaigns."
        )
    if accelerating_profit > 0:
        share = accelerating_profit / max(total_profit, 1e-9)
        bullets.append(
            f"Momentum check: {len(accelerating):,} accelerating accounts control {share:.0%} of the book. Keep success plans active."
        )

    for bullet in bullets:
        st.markdown(f"- {bullet}")


def grade_mix_chart(df: pd.DataFrame, profit_col: str) -> go.Figure:
    summary = (
        df.groupby("grade")
        .agg(accounts=("customer_id", "count"), profit=(profit_col, "sum"), avg_score=("score", "mean"))
        .reindex(GRADE_ORDER)
        .fillna(0)
        .reset_index()
    )
    total_accounts = summary["accounts"].sum()
    total_profit = summary["profit"].sum()
    summary["account_share"] = np.where(total_accounts > 0, summary["accounts"] / total_accounts, 0)
    summary["profit_share"] = np.where(total_profit > 0, summary["profit"] / total_profit, 0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["grade"],
            y=summary["account_share"],
            name="Account share",
            marker_color="#94a3b8",
            hovertemplate="Grade %{x}<br>Account share: %{y:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary["grade"],
            y=summary["profit_share"],
            name="Profit share",
            marker_color="#2563eb",
            hovertemplate="Grade %{x}<br>Profit share: %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Grade mix: volume vs. value",
        yaxis=dict(tickformat=".0%"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def score_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="score",
        color="grade",
        nbins=25,
        color_discrete_map=GRADE_COLOR_MAP,
        title="ICP score distribution",
    )
    fig.update_layout(
        bargap=0.05,
        margin=dict(l=10, r=10, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="ICP score")
    fig.update_yaxes(title="Accounts")
    return fig


def component_radar_chart(df: pd.DataFrame) -> go.Figure:
    components = {
        "Vertical fit": df["component_vertical"].mean(),
        "Firmographics": df["component_size"].mean(),
        "Adoption": df["component_adoption"].mean(),
        "Relationship": df["component_relationship"].mean(),
    }
    labels = list(components.keys())
    values = [float(np.nan_to_num(v, nan=0.0)) for v in components.values()]
    if not values:
        values = [0.0] * len(labels)
    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line_color="#1d4ed8",
            name="Component average",
        )
    )
    fig.update_layout(
        title="Score driver profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def industry_performance_chart(df: pd.DataFrame, profit_col: str) -> go.Figure:
    summary = (
        df.groupby("industry", as_index=False)[profit_col]
        .sum()
        .rename(columns={profit_col: "profit"})
        .sort_values("profit", ascending=False)
        .head(12)
    )
    fig = px.bar(
        summary,
        x="profit",
        y="industry",
        orientation="h",
        title="Top industries by profit",
        labels={"industry": "Industry", "profit": "Gross profit"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=20))
    fig.update_traces(marker_color="#0ea5e9")
    return fig


def territory_heatmap(df: pd.DataFrame, profit_col: str) -> go.Figure:
    pivot = (
        df.pivot_table(index="territory", columns="grade", values=profit_col, aggfunc="sum")
        .reindex(index=sorted(df["territory"].unique()), columns=GRADE_ORDER)
        .fillna(0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            hovertemplate="Territory %{y}<br>Grade %{x}<br>Profit %{z:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Territory x grade heatmap",
        xaxis_nticks=len(GRADE_ORDER),
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return fig


def whitespace_scatter(df: pd.DataFrame, profit_col: str) -> go.Figure:
    fig = px.scatter(
        df,
        x="component_adoption",
        y="component_relationship",
        size=profit_col,
        color="grade",
        color_discrete_map=GRADE_COLOR_MAP,
        hover_data={
            "company_name": True,
            "territory": True,
            profit_col: ":$,.0f",
            "whitespace_score": ":.0%",
        },
        labels={
            "component_adoption": "Adoption score",
            "component_relationship": "Relationship score",
        },
        title="White space map: adoption vs. relationship",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def whitespace_by_segment(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.groupby("customer_segment")["whitespace_value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        summary,
        x="customer_segment",
        y="whitespace_value",
        title="Expansion white space by segment",
        labels={"customer_segment": "Segment", "whitespace_value": "White space value"},
    )
    fig.update_traces(marker_color="#f97316")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=20))
    return fig


def opportunity_watchlist(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    cols = [
        "customer_id",
        "company_name",
        "grade",
        "score",
        filters.profit_key,
        "whitespace_value",
        "whitespace_score",
        "territory",
        "sales_rep",
        "industry",
        "growth_flag",
    ]
    available_cols = [c for c in cols if c in df.columns]
    watchlist = df[available_cols].copy()
    watchlist = watchlist.sort_values(["whitespace_value", filters.profit_key], ascending=False)
    watchlist = watchlist.head(30)

    rename_map = {
        "customer_id": "Customer ID",
        "company_name": "Company",
        "grade": "Grade",
        "score": "ICP Score",
        filters.profit_key: filters.profit_label,
        "whitespace_value": "Expansion White Space",
        "whitespace_score": "Whitespace %",
        "territory": "Territory",
        "sales_rep": "Account Owner",
        "industry": "Industry",
        "growth_flag": "Momentum",
    }
    watchlist = watchlist.rename(columns=rename_map)
    if "Whitespace %" in watchlist.columns:
        watchlist["Whitespace %"] = (watchlist["Whitespace %"] * 100).map(lambda x: f"{x:.0f}%")
    if filters.profit_label in watchlist.columns:
        watchlist[filters.profit_label] = watchlist[filters.profit_label].map(format_currency)
    if "Expansion White Space" in watchlist.columns:
        watchlist["Expansion White Space"] = watchlist["Expansion White Space"].map(format_currency)
    return watchlist


def render_account_table(df: pd.DataFrame, filters: FilterState) -> None:
    watchlist = opportunity_watchlist(df, filters)
    st.markdown("#### Opportunity Watchlist (Top 30)")
    if watchlist.empty:
        st.info("No accounts surfaced. Relax the filters or lower the minimum profit threshold.")
        return

    st.dataframe(watchlist, use_container_width=True, hide_index=True)
    csv_bytes = watchlist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download watchlist",
        data=csv_bytes,
        file_name="icp_watchlist.csv",
        mime="text/csv",
        use_container_width=False,
    )


def render_dashboard(df: pd.DataFrame, filters: FilterState) -> None:
    filtered = apply_filters(df, filters)

    st.title("Revenue Acceleration Command Center")
    st.caption(
        "Track ICP portfolio quality, surface whitespace, and align coverage motions around the data-backed truth."
    )

    st.markdown(
        f"**{len(filtered):,}** accounts match the current filters (from **{len(df):,}** total in the loaded portfolio)."
    )

    render_kpis(filtered, filters)
    render_leadership_headlines(filtered, filters)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Grade & Score Dynamics",
        "Industry & Territory",
        "White Space & Momentum",
        "Account Drilldown",
    ])

    with tab1:
        col1, col2 = st.columns((2, 1))
        with col1:
            st.plotly_chart(grade_mix_chart(filtered, filters.profit_key), use_container_width=True)
            st.plotly_chart(score_distribution_chart(filtered), use_container_width=True)
        with col2:
            st.plotly_chart(component_radar_chart(filtered), use_container_width=True)

    with tab2:
        col1, col2 = st.columns((1, 1))
        with col1:
            st.plotly_chart(industry_performance_chart(filtered, filters.profit_key), use_container_width=True)
        with col2:
            st.plotly_chart(territory_heatmap(filtered, filters.profit_key), use_container_width=True)

    with tab3:
        col1, col2 = st.columns((1.2, 0.8))
        with col1:
            st.plotly_chart(whitespace_scatter(filtered, filters.profit_key), use_container_width=True)
        with col2:
            st.plotly_chart(whitespace_by_segment(filtered), use_container_width=True)

    with tab4:
        render_account_table(filtered, filters)


def main() -> None:
    raw_df, source_path = load_portfolio_data()
    df = prepare_portfolio(raw_df)

    filters = render_sidebar(df)

    if source_path is None:
        st.info(
            "No scored portfolio file found. Displaying a representative sample dataset to illustrate the dashboard."
        )
    else:
        st.caption(f"Loaded portfolio from **{source_path.relative_to(ROOT)}**")

    render_dashboard(df, filters)


if __name__ == "__main__":
    main()
