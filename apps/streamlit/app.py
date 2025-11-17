"""Streamlit dashboard for sales leadership to surface ICP portfolio insights."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
import re

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

LOG_DIR = ROOT / "reports" / "logs"
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

RECENCY_COLOR_MAP = {
    "Engaged": "#22c55e",
    "Watch": "#0ea5e9",
    "At Risk": "#facc15",
    "Dormant": "#ef4444",
    "Unknown": "#94a3b8",
}

MOMENTUM_COLOR_MAP = {
    "Surging": "#22c55e",
    "Steady": "#38bdf8",
    "Cooling": "#f97316",
    "Unknown": "#94a3b8",
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
    "profit_prev_q": [
        "GP_PrevQ_Total",
        "Profit_PrevQ_Total",
        "GP_PrevQ",
        "Profit_PrevQ",
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
    "score": ["ICP_score_hardware", "ICP_score_cre", "score"],
    "grade": ["ICP_grade_hardware", "ICP_grade_cre", "grade"],
    "activity_segment": ["activity_segment"],
    "customer_segment": ["customer_segment"],
    "territory": ["AM_Territory", "territory"],
    "sales_rep": ["am_sales_rep", "AM_Sales_Rep"],
    "industry": ["Industry", "industry"],
    "company_name": ["Company Name", "company_name", "Account Name"],
    "customer_id": ["Customer ID", "customer_id", "Account ID"],
    "gp_qoq_growth": ["GP_QoQ_Growth", "gp_qoq_growth"],
    "delta_13w": ["delta_13w", "Delta_13W"],
    "delta_13w_pct": ["delta_13w_pct", "Delta_13W_Pct", "delta_13w_percent"],
    "spend_13w": ["spend_13w", "Spend_13W"],
    "spend_13w_prior": ["spend_13w_prior", "Spend_13W_Prior"],
    "spend_12m": ["spend_12m", "Spend_12M"],
    "spend_52w": ["spend_52w", "Spend_52W"],
    "yoy_13w_pct": ["yoy_13w_pct", "YoY_13W_Pct"],
    "momentum_score": ["momentum_score", "Momentum_Score"],
    "trend_score": ["trend_score", "Trend_Score"],
    "recency_score": ["recency_score", "Recency_Score"],
    "magnitude_score": ["magnitude_score", "Magnitude_Score"],
    "cadence_score": ["cadence_score", "Cadence_Score"],
    "days_since_last_order": ["days_since_last_order", "Days_Since_Last_Order"],
    "purchase_streak_months": ["purchase_streak_months", "Purchase_Streak_Months"],
    "active_weeks_13w": ["active_weeks_13w", "Active_Weeks_13W"],
    "hw_share_12m": ["hw_share_12m", "HW_Share_12M"],
    "sw_share_12m": ["sw_share_12m", "SW_Share_12M"],
    "sw_to_hw_whitespace_score": [
        "sw_to_hw_whitespace_score",
        "SW_to_HW_Whitespace_Score",
    ],
    "training_to_hw_ratio": ["training_to_hw_ratio"],
    "training_to_cre_ratio": ["training_to_cre_ratio"],
    "discount_pct": ["discount_pct", "Discount_Pct"],
    "cross_division_balance_score": ["cross_division_balance_score"],
    "hw_to_sw_cross_sell_score": ["hw_to_sw_cross_sell_score"],
    "sw_to_hw_cross_sell_score": ["sw_to_hw_cross_sell_score"],
    "printer_count": ["printer_count", "Printers", "Qty_Printers"],
    "contact_email": ["Primary_Contact_Email", "primary_contact_email", "RP_Primary_Email", "RP_Primary_email"],
}

NUMERIC_DEFAULTS = {
    "profit_since_2023": 0.0,
    "profit_t4q": 0.0,
    "profit_last_q": 0.0,
    "profit_prev_q": 0.0,
    "component_vertical": 0.5,
    "component_size": 0.5,
    "component_adoption": 0.5,
    "component_relationship": 0.5,
    "score_raw": 0.0,
    "score": 50.0,
    "gp_qoq_growth": 0.0,
    "delta_13w": 0.0,
    "delta_13w_pct": 0.0,
    "spend_13w": 0.0,
    "spend_13w_prior": 0.0,
    "spend_12m": 0.0,
    "spend_52w": 0.0,
    "yoy_13w_pct": 0.0,
    "momentum_score": 0.0,
    "trend_score": 0.0,
    "recency_score": 0.0,
    "magnitude_score": 0.0,
    "cadence_score": 0.0,
    "days_since_last_order": np.nan,
    "purchase_streak_months": np.nan,
    "active_weeks_13w": 0.0,
    "hw_share_12m": np.nan,
    "sw_share_12m": np.nan,
    "sw_to_hw_whitespace_score": 0.0,
    "training_to_hw_ratio": 0.0,
    "training_to_cre_ratio": 0.0,
    "discount_pct": np.nan,
    "cross_division_balance_score": 0.0,
    "hw_to_sw_cross_sell_score": 0.0,
    "sw_to_hw_cross_sell_score": 0.0,
    "training_to_hw_ratio": 0.0,
    "printer_count": 0.0,
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
    division_mode: str
    hw_grades: Sequence[str]
    cre_grades: Sequence[str]
    am_territories: Sequence[str]
    cad_territories: Sequence[str]
    hw_reps: Sequence[str]
    cre_reps: Sequence[str]


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


@st.cache_data(show_spinner=False)
def load_neighbors_data() -> pd.DataFrame:
    """Load neighbor artifact if available."""

    candidate_files = [
        ROOT / "artifacts" / "account_neighbors.csv",
        ROOT / "data" / "processed" / "account_neighbors.csv",
        ROOT / "reports" / "account_neighbors.csv",
    ]
    for path in candidate_files:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


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
    bins = [-np.inf, 0.33, 0.66, np.inf]
    band_labels = ["Low", "Medium", "High"]

    def _score_band(series: pd.Series) -> pd.Series:
        cats = pd.cut(series, bins=bins, labels=band_labels, include_lowest=True)
        return cats.cat.add_categories(["Unknown"]).fillna("Unknown").astype(str)

    # Normalize column availability
    for canonical, candidates in COLUMN_CANDIDATES.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                df[canonical] = df[candidate]
                break

    for col, default in NUMERIC_DEFAULTS.items():
        series = pd.to_numeric(
            df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        if pd.isna(default):
            df[col] = series
        else:
            df[col] = series.fillna(default)

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

    df["am_territory"] = df.get("AM_Territory", df.get("territory", "Unassigned"))
    df["am_territory"] = df["am_territory"].fillna("Unassigned").astype(str)
    df["cad_territory"] = df.get("CAD_Territory", df["am_territory"])
    df["cad_territory"] = df["cad_territory"].fillna(df["am_territory"]).astype(str)
    df["territory"] = df.get("territory", df["am_territory"])
    df["territory"] = df["territory"].fillna("Unassigned").astype(str)

    df["hw_owner"] = df.get("am_sales_rep", df.get("sales_rep", "Unassigned"))
    df["hw_owner"] = df["hw_owner"].fillna("Unassigned").astype(str)
    df["cre_owner"] = df.get("cre_sales_rep", df["hw_owner"])
    df["cre_owner"] = df["cre_owner"].fillna(df["hw_owner"]).astype(str)
    df["sales_rep"] = df["hw_owner"]
    if "cre_sales_rep" not in df.columns:
        df["cre_sales_rep"] = df["cre_owner"]

    df["industry"] = df.get("industry", df.get("Industry", "Unknown"))
    df["industry"] = df["industry"].fillna("Unknown").astype(str)

    df["adoption_maturity"] = df[["component_adoption", "component_relationship"]].mean(axis=1)
    df["whitespace_score"] = (1 - df["adoption_maturity"]).clip(0, 1)
    df["whitespace_value"] = df["profit_since_2023"] * df["whitespace_score"]
    df["adoption_band"] = _score_band(df["component_adoption"])
    df["relationship_band"] = _score_band(df["component_relationship"])
    df["printer_count"] = pd.to_numeric(df.get("printer_count"), errors="coerce").fillna(0.0)
    df["revenue_only_flag"] = (df["printer_count"] <= 0).astype(bool)
    df["heavy_fleet_flag"] = (df["printer_count"] >= 10).astype(bool)

    df["delta_13w_pct"] = (
        df["delta_13w_pct"].replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    )
    df["delta_13w"] = df["delta_13w"].replace([np.inf, -np.inf], np.nan)
    df["momentum_score"] = df["momentum_score"].clip(0, 1)
    df["trend_score"] = df["trend_score"].clip(0, 1)
    df["recency_score"] = df["recency_score"].clip(0, 1)
    df["magnitude_score"] = df["magnitude_score"].clip(0, 1)
    df["cadence_score"] = df["cadence_score"].clip(0, 1)
    df["days_since_last_order"] = df["days_since_last_order"].clip(lower=0)

    recency_bins = [-np.inf, 45, 120, 210, np.inf]
    recency_labels = ["Engaged", "Watch", "At Risk", "Dormant"]
    recency_cats = pd.cut(
        df["days_since_last_order"],
        bins=recency_bins,
        labels=recency_labels,
        include_lowest=True,
    )
    df["recency_bucket"] = (
        recency_cats.cat.add_categories(["Unknown"]).fillna("Unknown").astype(str)
    )

    df["momentum_segment"] = "Unknown"
    df.loc[df["momentum_score"].notna(), "momentum_segment"] = "Steady"
    df.loc[df["momentum_score"] >= 0.6, "momentum_segment"] = "Surging"
    df.loc[df["momentum_score"] < 0.4, "momentum_segment"] = "Cooling"
    df.loc[df["delta_13w_pct"] >= 0.15, "momentum_segment"] = "Surging"
    df.loc[df["delta_13w_pct"] <= -0.1, "momentum_segment"] = "Cooling"

    whitespace_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    whitespace_labels = ["<20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    whitespace_cats = pd.cut(
        df["whitespace_score"],
        bins=whitespace_bins,
        labels=whitespace_labels,
        include_lowest=True,
    )
    df["whitespace_bucket"] = (
        whitespace_cats.cat.add_categories(["Unknown"]).fillna("Unknown").astype(str)
    )

    df["call_to_action"] = np.select(
        [
            (df["whitespace_score"] >= 0.45)
            & df["momentum_segment"].isin(["Surging", "Steady"]),
            (df["momentum_segment"] == "Cooling") & df["grade"].isin(["A", "B"]),
            df["recency_bucket"] == "Dormant",
            df["recency_bucket"] == "At Risk",
        ],
        [
            "Launch expansion sprint",
            "Protect share with executive outreach",
            "Run reactivation sequence",
            "Schedule 90-day engagement",
        ],
        default="Maintain success cadence",
    )

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
    df["score_band"] = df["score_band"].astype(str)
    df["score_improved_flag"] = (
        df["delta_13w_pct"].fillna(0) >= 0.1
    ) | (df["momentum_segment"] == "Surging")

    contact_email = None
    for candidate in COLUMN_CANDIDATES.get("contact_email", []):
        if candidate in df.columns:
            contact_email = df[candidate]
            break
    if contact_email is None:
        df["contact_email"] = ""
    else:
        df["contact_email"] = contact_email.fillna("").astype(str).str.strip()

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

    division_options = ["Portfolio", "Hardware", "CRE", "Dual"]
    division_mode = st.sidebar.radio(
        "Division focus",
        options=division_options,
        index=min(division_options.index("Dual"), len(division_options) - 1),
        help="Choose which division's metrics drive the visuals. Dual shows both.",
        horizontal=True,
    )

    min_score, max_score = float(df["score"].min()), float(df["score"].max())
    default_lower = max(0.0, min_score)
    default_upper = min(100.0, max_score)
    score_range = st.sidebar.slider(
        "ICP score range",
        min_value=0.0,
        max_value=100.0,
        value=(default_lower, default_upper),
        step=1.0,
    )

    grade_selection = st.sidebar.multiselect(
        "Grades",
        options=GRADE_ORDER,
        default=GRADE_ORDER,
    )

    am_territory_options = sorted(df["am_territory"].dropna().unique().tolist())
    cad_territory_options = sorted(df["cad_territory"].dropna().unique().tolist())
    hw_owner_options = sorted(df["hw_owner"].dropna().unique().tolist())
    cre_owner_options = sorted(df["cre_owner"].dropna().unique().tolist())

    with st.sidebar.expander("Hardware focus", expanded=division_mode in ("Hardware", "Dual")):
        hw_grades = st.multiselect(
            "Hardware grades",
            options=GRADE_ORDER,
            default=["A", "B"],
            help="Used for HW-specific KPIs and call lists.",
        )
        am_territories = st.multiselect(
            "AM territories",
            options=am_territory_options,
            default=am_territory_options,
        )
        hw_reps = st.multiselect(
            "HW account owners",
            options=hw_owner_options,
            default=hw_owner_options,
        )

    with st.sidebar.expander("CRE focus", expanded=division_mode in ("CRE", "Dual")):
        cre_grades = st.multiselect(
            "CRE grades",
            options=GRADE_ORDER,
            default=["A", "B"],
            help="Used for CRE-specific KPIs and call lists.",
        )
        cad_territories = st.multiselect(
            "CAD territories",
            options=cad_territory_options,
            default=cad_territory_options or am_territory_options,
        )
        cre_reps = st.multiselect(
            "CRE account owners",
            options=cre_owner_options,
            default=cre_owner_options or hw_owner_options,
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
        default=industry_options,
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
        division_mode=division_mode,
        hw_grades=hw_grades or GRADE_ORDER,
        cre_grades=cre_grades or GRADE_ORDER,
        am_territories=am_territories or am_territory_options,
        cad_territories=cad_territories or cad_territory_options or am_territory_options,
        hw_reps=hw_reps or hw_owner_options,
        cre_reps=cre_reps or cre_owner_options or hw_owner_options,
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
        & df["am_territory"].isin(filters.am_territories)
        & df["cad_territory"].isin(filters.cad_territories)
        & df["hw_owner"].isin(filters.hw_reps)
        & df["cre_owner"].isin(filters.cre_reps)
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

    hw_df = df[df["ICP_grade_hardware"].notna()]
    cre_df = df[df["ICP_grade_cre"].notna()]
    if not hw_df.empty or not cre_df.empty:
        st.markdown("#### Division snapshots")
        col_hw, col_cre = st.columns(2)
        with col_hw:
            render_division_snapshot(
                hw_df,
                "Hardware",
                grade_col="ICP_grade_hardware",
                selected_grades=filters.hw_grades,
                profit_col=filters.profit_key,
            )
        with col_cre:
            render_division_snapshot(
                cre_df,
                "CRE",
                grade_col="ICP_grade_cre",
                selected_grades=filters.cre_grades,
                profit_col=filters.profit_key,
            )


def render_division_snapshot(
    division_df: pd.DataFrame,
    label: str,
    grade_col: str,
    selected_grades: Sequence[str],
    profit_col: str,
) -> None:
    if division_df.empty:
        st.info(f"No {label.lower()} accounts available in the current view.")
        return
    total = len(division_df)
    ab_df = division_df[division_df[grade_col].isin(selected_grades)]
    ab_count = len(ab_df)
    total_profit = division_df[profit_col].sum()
    ab_profit = ab_df[profit_col].sum()
    share = (ab_count / total * 100) if total else 0.0
    profit_share = (ab_profit / total_profit * 100) if total_profit else 0.0
    st.metric(
        f"{label} A/B Accounts",
        f"{ab_count:,}",
        f"{share:.1f}% of {total:,}",
    )
    st.caption(f"{label} GP: {format_currency(ab_profit)} ({profit_share:.1f}% of {format_currency(total_profit)})")


def format_currency(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def format_percent(value: float | None, decimals: int = 0, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    scaled = value * 100
    formatted = f"{scaled:.{decimals}f}%"
    if signed and scaled > 0:
        return f"+{formatted}"
    return formatted


def format_days(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(round(float(value)))}d"


def safe_mean(series: pd.Series | None) -> float:
    if series is None or series.empty:
        return np.nan
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def share_true(mask: pd.Series | None) -> float:
    if mask is None or mask.empty:
        return np.nan
    valid = mask.dropna()
    if valid.empty:
        return np.nan
    return float(valid.mean())


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
    surging = df[df["momentum_segment"] == "Surging"]
    surging_profit = surging[profit_col].sum()
    dormant = df[df["recency_bucket"] == "Dormant"]
    dormant_profit = dormant[profit_col].sum()

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
    if surging_profit > 0:
        share = surging_profit / max(total_profit, 1e-9)
        bullets.append(
            f"**{len(surging):,} surging** accounts ({share:.0%} of profit) show 13W momentum—capture upsell before competitors do."
        )
    if dormant_profit > 0:
        share = dormant_profit / max(total_profit, 1e-9)
        bullets.append(
            f"Dormancy alert: {len(dormant):,} accounts past 7 months inactivity hold {format_currency(dormant_profit)} of value. Trigger save motions."
        )

    for bullet in bullets:
        st.markdown(f"- {bullet}")


def render_operating_pulse(df: pd.DataFrame) -> None:
    """Surface operating signals around momentum, engagement, and whitespace."""

    st.markdown("#### Operating Pulse")
    if df.empty:
        st.info("Add or broaden filters to surface operating signals.")
        return

    render_metric_css()

    total_accounts = len(df)
    delta_series = df.get("delta_13w_pct", pd.Series(dtype=float))
    momentum_mean = safe_mean(delta_series)
    delta_valid = delta_series.dropna()
    positive_count = int((delta_valid > 0).sum())
    positive_base = int(len(delta_valid))

    momentum_avg = safe_mean(df.get("momentum_score", pd.Series(dtype=float)))

    dormant_mask = df.get("recency_bucket", pd.Series(dtype=object)) == "Dormant"
    dormant_count = int(dormant_mask.sum())
    dormant_share = dormant_count / total_accounts if total_accounts else np.nan

    whitespace_ready_mask = df.get("whitespace_score", pd.Series(dtype=float)) >= 0.4
    whitespace_ready = int(whitespace_ready_mask.sum())
    whitespace_share = whitespace_ready / total_accounts if total_accounts else np.nan

    active_series = df.get("active_weeks_13w", pd.Series(dtype=float))
    active_count = int((active_series.fillna(0) > 0).sum()) if not active_series.empty else 0
    active_share = active_count / total_accounts if total_accounts else np.nan

    cards = [
        {
            "title": "13W Momentum",
            "value": format_percent(momentum_mean, decimals=1, signed=True),
            "context": (
                f"{positive_count}/{positive_base} accounts growing vs. prior 13W"
                if positive_base
                else "No recent momentum history"
            ),
        },
        {
            "title": "Composite Momentum",
            "value": format_percent(momentum_avg, decimals=0),
            "context": "Trend • Recency • Magnitude • Cadence blend",
        },
        {
            "title": "Dormancy Risk",
            "value": format_percent(dormant_share, decimals=0),
            "context": (
                f"{dormant_count:,} accounts >210 days since order" if dormant_count else "Engagement within thresholds"
            ),
        },
        {
            "title": "Expansion Ready",
            "value": format_percent(whitespace_share, decimals=0),
            "context": (
                f"{whitespace_ready:,} accounts with >40% whitespace"
                if whitespace_ready
                else "Limited whitespace under current filters"
            ),
        },
        {
            "title": "Active in 13W",
            "value": format_percent(active_share, decimals=0),
            "context": (
                f"{active_count:,} accounts transacted recently" if active_count else "No recent transactions"
            ),
        },
    ]

    st.markdown(
        '<div class="metric-grid">'
        + "".join(
            f"<div class='metric-card'><h4>{card['title']}</h4><div class='value'>{card['value']}</div><div class='context'>{card['context']}</div></div>"
            for card in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )


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


def momentum_scatter(df: pd.DataFrame, profit_col: str) -> go.Figure:
    data = df.copy()
    data["delta_13w_pct"] = pd.to_numeric(data.get("delta_13w_pct"), errors="coerce")
    data["momentum_score"] = pd.to_numeric(data.get("momentum_score"), errors="coerce")
    data["whitespace_value"] = pd.to_numeric(data.get("whitespace_value"), errors="coerce").fillna(0)
    data["momentum_segment"] = data.get("momentum_segment", "Unknown")

    has_signal = (
        data["delta_13w_pct"].notna().any() or data["momentum_score"].notna().any()
    )

    if not has_signal:
        fig = go.Figure()
        fig.update_layout(
            title="Momentum vs. profit (insufficient 13W history)",
            margin=dict(l=10, r=10, t=60, b=20),
        )
        fig.add_annotation(
            text="No 13W momentum fields detected. Add momentum features to the dataset to unlock this view.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return fig

    data["bubble_whitespace"] = data["whitespace_value"].clip(lower=1.0)

    fig = px.scatter(
        data,
        x="delta_13w_pct",
        y=profit_col,
        color="momentum_segment",
        size="bubble_whitespace",
        hover_data={
            "company_name": True,
            profit_col: ":$,.0f",
            "delta_13w_pct": ":.0%",
            "momentum_segment": True,
            "whitespace_value": ":$,.0f",
        },
        size_max=50,
        color_discrete_map=MOMENTUM_COLOR_MAP,
        labels={
            "delta_13w_pct": "13W Δ vs prior",
            profit_col: "Gross profit",
            "momentum_segment": "Momentum",
            "bubble_whitespace": "Expansion whitespace",
        },
        title="Momentum vs. profit (bubble ≈ whitespace)",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickprefix="$")
    return fig


def recency_distribution_chart(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["days_since_last_order"] = pd.to_numeric(
        data.get("days_since_last_order"), errors="coerce"
    )
    data = data[data["days_since_last_order"].notna()]

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Engagement distribution (no recency data)",
            margin=dict(l=10, r=10, t=60, b=20),
        )
        fig.add_annotation(
            text="Recency fields missing; provide `days_since_last_order` to enable this view.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        return fig

    fig = px.histogram(
        data,
        x="days_since_last_order",
        color="recency_bucket",
        nbins=25,
        color_discrete_map=RECENCY_COLOR_MAP,
        labels={
            "days_since_last_order": "Days since last order",
            "recency_bucket": "Engagement status",
        },
        title="Engagement recency distribution",
    )
    fig.update_layout(
        bargap=0.05,
        margin=dict(l=10, r=10, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Days since last order")
    fig.update_yaxes(title="Accounts")
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


def format_account_subset(df_subset: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    rename_map = {
        "customer_id": "Customer ID",
        "company_name": "Company",
        "grade": "Grade",
        "score": "ICP Score",
        filters.profit_key: filters.profit_label,
        "whitespace_value": "Expansion White Space",
        "whitespace_score": "Whitespace %",
        "delta_13w_pct": "13W Δ%",
        "momentum_segment": "Momentum",
        "territory": "Territory",
        "sales_rep": "Account Owner",
        "industry": "Industry",
        "growth_flag": "Momentum Flag",
        "recency_bucket": "Recency",
        "days_since_last_order": "Days Since Last Order",
        "activity_segment": "Activity",
        "call_to_action": "Next Best Action",
        "cross_division_balance_score": "Cross-Division Balance Score",
        "hw_to_sw_cross_sell_score": "HW->SW Cross-sell Score",
        "sw_to_hw_cross_sell_score": "SW->HW Cross-sell Score",
        "training_to_hw_ratio": "Training to HW Ratio",
        "training_to_cre_ratio": "Training to CRE Ratio",
    }

    formatted = df_subset.rename(columns=rename_map)
    if "Whitespace %" in formatted.columns:
        formatted["Whitespace %"] = (
            formatted["Whitespace %"] * 100
        ).map(lambda x: f"{x:.0f}%" if not pd.isna(x) else "n/a")
    if filters.profit_label in formatted.columns:
        formatted[filters.profit_label] = formatted[filters.profit_label].map(format_currency)
    if "Expansion White Space" in formatted.columns:
        formatted["Expansion White Space"] = formatted["Expansion White Space"].map(format_currency)
    if "13W Δ%" in formatted.columns:
        formatted["13W Δ%"] = formatted["13W Δ%"].map(lambda x: format_percent(x, decimals=0, signed=True))
    if "Days Since Last Order" in formatted.columns:
        formatted["Days Since Last Order"] = formatted["Days Since Last Order"].map(format_days)
    preferred_order = [
        "Customer ID",
        "Company",
        "Grade",
        "ICP Score",
        filters.profit_label,
        "Expansion White Space",
        "Whitespace %",
        "13W Δ%",
        "Cross-Division Balance Score",
        "HW->SW Cross-sell Score",
        "SW->HW Cross-sell Score",
        "Training to HW Ratio",
        "Training to CRE Ratio",
        "Momentum",
        "Momentum Flag",
        "Recency",
        "Days Since Last Order",
        "Activity",
        "Next Best Action",
        "Account Owner",
        "Territory",
        "Industry",
    ]
    ordered = [col for col in preferred_order if col in formatted.columns]
    remaining = [col for col in formatted.columns if col not in ordered]
    formatted = formatted[ordered + remaining]
    return formatted


def opportunity_watchlist(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    cols = [
        "customer_id",
        "company_name",
        "grade",
        "score",
        filters.profit_key,
        "whitespace_value",
        "whitespace_score",
        "delta_13w_pct",
        "momentum_segment",
        "territory",
        "sales_rep",
        "industry",
        "growth_flag",
        "recency_bucket",
        "days_since_last_order",
        "activity_segment",
        "call_to_action",
        "cross_division_balance_score",
        "hw_to_sw_cross_sell_score",
        "sw_to_hw_cross_sell_score",
        "training_to_hw_ratio",
        "training_to_cre_ratio",
    ]
    available_cols = [c for c in cols if c in df.columns]
    watchlist = df[available_cols].copy()
    watchlist = watchlist.sort_values(["whitespace_value", filters.profit_key], ascending=False)
    watchlist = watchlist.head(30)

    return format_account_subset(watchlist, filters)


def compute_playbook_segments(
    df: pd.DataFrame, filters: FilterState, top_n: int = 20
) -> OrderedDict[str, pd.DataFrame]:
    segments: OrderedDict[str, pd.DataFrame] = OrderedDict()
    if df.empty:
        return segments

    base = df.copy()

    expansion_mask = (
        base["grade"].isin(["A", "B"])
        & (base["whitespace_score"] >= 0.35)
        & base[filters.profit_key].notna()
    )
    expansion = base[expansion_mask].copy()
    expansion = expansion.sort_values(
        ["whitespace_value", filters.profit_key], ascending=[False, False]
    ).head(top_n)
    if not expansion.empty:
        segments["Expansion Sprint"] = format_account_subset(expansion, filters)

    rescue_mask = (
        base[filters.profit_key] > 0
    ) & (
        base["growth_flag"].eq("Declining")
        | (base["delta_13w_pct"] <= -0.1)
        | (base["momentum_segment"] == "Cooling")
    )
    rescue = base[rescue_mask].copy()
    rescue = rescue.sort_values(filters.profit_key, ascending=False).head(top_n)
    if not rescue.empty:
        segments["Rescue & Protect"] = format_account_subset(rescue, filters)

    activation_mask = (
        base["recency_bucket"].isin(["Dormant", "At Risk"])
        | base["activity_segment"].str.lower().isin(["dormant", "cold", "unknown"])
    )
    activation = base[activation_mask].copy()
    activation = activation.sort_values(
        ["days_since_last_order", filters.profit_key], ascending=[False, False]
    ).head(top_n)
    if not activation.empty:
        segments["Reactivation Cadence"] = format_account_subset(activation, filters)

    return OrderedDict((name, table) for name, table in segments.items() if not table.empty)


def render_playbooks(df: pd.DataFrame, filters: FilterState) -> None:
    st.markdown("#### Sales Playbooks")
    segments = compute_playbook_segments(df, filters)
    if not segments:
        st.info("Current filters do not surface accounts for the standard playbooks.")
        return

    tabs = st.tabs(list(segments.keys()))
    for tab, (name, table) in zip(tabs, segments.items()):
        with tab:
            st.dataframe(table, use_container_width=True, hide_index=True)
            csv_bytes = table.to_csv(index=False).encode("utf-8")
            safe_name = name.lower().replace(" ", "_").replace("&", "and")
            st.download_button(
                f"Download {name} list",
                data=csv_bytes,
                file_name=f"{safe_name}_call_list.csv",
                mime="text/csv",
                use_container_width=False,
            )


def load_validation_log_entries(max_files: int = 3, max_lines: int = 80) -> list[dict]:
    if not LOG_DIR.exists():
        return []
    entries: list[dict] = []
    files = sorted(LOG_DIR.glob("validation_*.log"), reverse=True)
    for path in files[:max_files]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        entries.append({"path": path, "lines": lines[-max_lines:]})
    return entries


def persist_call_list(table: pd.DataFrame, label: str) -> Path:
    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "call_list"
    target_dir = ROOT / "reports" / "call_lists" / "manual"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{safe_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    table.to_csv(path, index=False)
    return path


def render_call_list_builder(df: pd.DataFrame, filters: FilterState) -> None:
    st.markdown("#### Call List Builder")
    if df.empty:
        st.info("No accounts available for the current filters.")
        return

    builder_df = df.copy()
    hw_tab, cre_tab = st.tabs(["Hardware list", "CRE list"])
    with hw_tab:
        render_division_call_list(builder_df, filters, division="hardware")
    with cre_tab:
        render_division_call_list(builder_df, filters, division="cre")


def render_division_call_list(builder_df: pd.DataFrame, filters: FilterState, division: str) -> None:
    if division == "hardware":
        label = "Hardware"
        grade_col = "ICP_grade_hardware"
        score_col = "ICP_score_hardware"
        territory_col = "am_territory"
        owner_col = "hw_owner"
        grade_defaults = filters.hw_grades
        adoption_col = "Hardware_score"
        relationship_col = "Software_score"
    else:
        label = "CRE"
        grade_col = "ICP_grade_cre"
        score_col = "ICP_score_cre"
        territory_col = "cad_territory"
        owner_col = "cre_owner"
        grade_defaults = filters.cre_grades
        adoption_col = "cre_adoption_assets"
        relationship_col = "cre_relationship_profit"

    if grade_col not in builder_df.columns or builder_df[grade_col].dropna().empty:
        st.info(f"No {label.lower()} accounts available for call list generation.")
        return

    div_df = builder_df[builder_df[grade_col].notna()].copy()
    segments = sorted(div_df["customer_segment"].dropna().unique().tolist())
    industries = sorted(div_df["industry"].dropna().unique().tolist())
    territories = sorted(div_df[territory_col].dropna().unique().tolist())
    reps = sorted(div_df[owner_col].dropna().unique().tolist())
    adoption_bands = sorted(div_df["adoption_band"].dropna().unique().tolist())
    relationship_bands = sorted(div_df["relationship_band"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        selected_segments = st.multiselect(f"{label} segments", options=segments, default=segments or [])
        selected_industries = st.multiselect(f"{label} industries", options=industries, default=industries or [])
        selected_territories = st.multiselect(
            f"{label} territories",
            options=territories,
            default=territories or [],
        )
    with col2:
        selected_grades = st.multiselect(
            f"{label} grades",
            options=GRADE_ORDER,
            default=grade_defaults or ["A", "B"],
        )
        selected_reps = st.multiselect(
            f"{label} owners",
            options=reps,
            default=reps or [],
        )
        selected_relationship = st.multiselect(
            "Relationship band",
            options=relationship_bands,
            default=relationship_bands or [],
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        selected_adoption = st.multiselect(
            "Adoption band",
            options=adoption_bands,
            default=adoption_bands or [],
        )
    with col4:
        revenue_only = st.checkbox("Revenue-only (no printers)" if division == "hardware" else "Revenue-only accounts")
        heavy_fleet = st.checkbox("Heavy fleet (≥10 printers)", value=False, disabled=division != "hardware")
    with col5:
        improved_scores = st.checkbox("Newly improved score")
        min_profit = st.number_input(
            f"Min {filters.profit_label} ($)",
            min_value=0.0,
            value=0.0,
            step=5000.0,
        )

    mask = (
        div_df["customer_segment"].isin(selected_segments)
        & div_df["industry"].isin(selected_industries)
        & div_df[territory_col].isin(selected_territories)
        & div_df[owner_col].isin(selected_reps)
        & div_df[grade_col].isin(selected_grades)
        & div_df["adoption_band"].isin(selected_adoption)
        & div_df["relationship_band"].isin(selected_relationship)
        & (div_df[filters.profit_key] >= min_profit)
    )
    if revenue_only and "revenue_only_flag" in div_df.columns:
        mask &= div_df["revenue_only_flag"]
    if heavy_fleet and "heavy_fleet_flag" in div_df.columns:
        mask &= div_df["heavy_fleet_flag"]
    if improved_scores and "score_improved_flag" in div_df.columns:
        mask &= div_df["score_improved_flag"]

    scoped = div_df.loc[mask].copy()
    if scoped.empty:
        st.warning(f"No {label.lower()} accounts match the selected filters.")
        return

    scoped = scoped.sort_values([score_col if score_col in scoped else "score", filters.profit_key], ascending=[False, False])
    scoped = scoped.reset_index(drop=True)
    scoped.insert(0, "Rank", np.arange(1, len(scoped) + 1))

    adoption_series = scoped.get(adoption_col, scoped.get("component_adoption"))
    relationship_series = scoped.get(relationship_col, scoped.get("component_relationship"))

    table = pd.DataFrame(
        {
            "Rank": scoped["Rank"],
            "Customer ID": scoped["customer_id"],
            "Company": scoped["company_name"],
            f"{label} Territory": scoped[territory_col],
            f"{label} Owner": scoped[owner_col],
            f"ICP Grade ({label})": scoped.get(grade_col),
            f"ICP Score ({label})": scoped.get(score_col, scoped["score"]).round(1),
            "Adoption Score": adoption_series.round(3) if adoption_series is not None else np.nan,
            "Relationship Score": relationship_series.round(3) if relationship_series is not None else np.nan,
            filters.profit_label: scoped[filters.profit_key].round(0),
            "Printers": scoped["printer_count"].round().astype(int) if "printer_count" in scoped else 0,
            "Suggested Playbook": scoped.get("call_to_action"),
        }
    )
    st.markdown(f"**{len(table):,}** {label.lower()} accounts match the current criteria.")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {label.lower()} call list (CSV)",
        data=csv_bytes,
        file_name=f"{label.lower()}_call_list.csv",
        mime="text/csv",
        use_container_width=False,
    )

    email_values: list[str] = []
    if "contact_email" in scoped.columns:
        email_values.extend(scoped["contact_email"].dropna().astype(str).str.strip().tolist())
    email_values = sorted({e for e in email_values if e})
    email_blob = ";".join(email_values)

    st.text_area(
        f"{label} email list",
        value=email_blob,
        height=80,
        placeholder="No contact emails available for the current list.",
    )

    list_label = st.text_input(f"{label} list label", value=f"{label.lower()}_call_list")
    save_key = f"call_list_saved_path_{label.lower()}"
    saved_path = st.session_state.get(save_key)
    if st.button(f"Save {label.lower()} call list"):
        path = persist_call_list(table, list_label)
        st.session_state[save_key] = str(path)
        saved_path = str(path)
        rel = path.relative_to(ROOT)
        st.success(f"Saved {label} call list to {rel}")
    if saved_path:
        st.text_input(f"{label} CSV path", value=saved_path, key=f"{label}_call_list_path_copy")


def render_manager_hq(df: pd.DataFrame, filters: FilterState) -> None:
    st.markdown("#### Manager HQ")
    if df.empty:
        st.info("No accounts available for the current filters.")
        return

    profit_col = filters.profit_key
    hw_df = df[df["ICP_grade_hardware"].notna()].copy()
    hw_df["is_ab"] = hw_df["ICP_grade_hardware"].isin(filters.hw_grades)
    cre_df = df[df["ICP_grade_cre"].notna()].copy()
    cre_df["is_ab"] = cre_df["ICP_grade_cre"].isin(filters.cre_grades)

    def summarize(div_df: pd.DataFrame, territory_col: str, owner_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if div_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        territory_summary = (
            div_df.groupby(territory_col)
            .agg(
                Accounts=("customer_id", "nunique"),
                AB_Accounts=("is_ab", "sum"),
                Profit=(profit_col, "sum"),
            )
            .sort_values("Profit", ascending=False)
        )
        territory_summary["AB_Share"] = np.where(
            territory_summary["Accounts"] > 0,
            territory_summary["AB_Accounts"] / territory_summary["Accounts"],
            0.0,
        )
        rep_summary = (
            div_df.groupby(owner_col)
            .agg(
                Accounts=("customer_id", "nunique"),
                AB_Accounts=("is_ab", "sum"),
                Profit=(profit_col, "sum"),
            )
            .sort_values("Profit", ascending=False)
        )
        return territory_summary.reset_index().rename(columns={territory_col: "Territory", owner_col: "Owner"}), rep_summary.reset_index().rename(columns={owner_col: "Owner"})

    hw_territory, hw_reps = summarize(hw_df, "am_territory", "hw_owner")
    cre_territory, cre_reps = summarize(cre_df, "cad_territory", "cre_owner")

    st.markdown("##### Territory coverage")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Hardware")
        if hw_territory.empty:
            st.info("No hardware accounts.")
        else:
            st.dataframe(hw_territory, use_container_width=True, hide_index=True)
            download_dataframe(hw_territory, "hardware_territory_summary.csv")
    with col2:
        st.caption("CRE")
        if cre_territory.empty:
            st.info("No CRE accounts.")
        else:
            st.dataframe(cre_territory, use_container_width=True, hide_index=True)
            download_dataframe(cre_territory, "cre_territory_summary.csv")

    st.markdown("##### Top owners by GP")
    col3, col4 = st.columns(2)
    with col3:
        st.caption("Hardware owners")
        if hw_reps.empty:
            st.info("No hardware owners available.")
        else:
            st.dataframe(hw_reps.head(15), use_container_width=True, hide_index=True)
    with col4:
        st.caption("CRE owners")
        if cre_reps.empty:
            st.info("No CRE owners available.")
        else:
            st.dataframe(cre_reps.head(15), use_container_width=True, hide_index=True)


def download_dataframe(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def render_neighbor_lab(df: pd.DataFrame, neighbors: pd.DataFrame, filters: FilterState) -> None:
    st.markdown("#### Look-alike Lab")
    if neighbors.empty:
        st.info("Neighbor artifact not found. Run the neighbors pipeline to enable this view.")
        return
    if df.empty:
        st.info("No accounts loaded.")
        return

    options_df = (
        df[["customer_id", "company_name", "industry"]]
        .dropna(subset=["customer_id"])
        .sort_values("company_name")
    )
    options = options_df.apply(
        lambda row: f"{row['company_name']} ({row['customer_id']}) • {row['industry']}",
        axis=1,
    ).tolist()
    if not options:
        st.info("No accounts available for selection.")
        return

    selected_label = st.selectbox("Anchor account", options)
    selected_id = selected_label.split("(")[-1].split(")")[0].strip()
    top_n = st.slider("Neighbors to display", min_value=5, max_value=50, value=15, step=5)

    outbound = neighbors[neighbors["account_id"] == selected_id].copy()
    inbound = neighbors[neighbors["neighbor_account_id"] == selected_id].copy()

    cols_to_merge = [c for c in [
        "customer_id",
        "company_name",
        "industry",
        "am_territory",
        "cad_territory",
        "hw_owner",
        "cre_owner",
        "ICP_grade_hardware",
        "ICP_grade_cre",
        "ICP_score_hardware",
        "ICP_score_cre",
        filters.profit_key,
    ] if c in df.columns]

    def _prepare(table: pd.DataFrame, join_col: str) -> pd.DataFrame:
        if table.empty:
            return table
        merged = table.merge(
            df[cols_to_merge],
            left_on=join_col,
            right_on="customer_id",
            how="left",
            suffixes=("", "_neighbor"),
        )
        display_cols = {
            "neighbor_account_id": "Neighbor ID",
            "account_id": "Account ID",
            "neighbor_rank": "Rank",
            "sim_overall": "Similarity",
        }
        rename_map = {col: display_cols[col] for col in display_cols if col in merged.columns}
        merged = merged.rename(columns=rename_map)
        return merged

    outbound_display = _prepare(outbound.sort_values("neighbor_rank").head(top_n), "neighbor_account_id")
    inbound_display = _prepare(inbound.sort_values("neighbor_rank").head(top_n), "account_id")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Accounts similar to the anchor")
        if outbound_display.empty:
            st.info("No outbound neighbors for this account.")
        else:
            st.dataframe(outbound_display, use_container_width=True, hide_index=True)
            download_dataframe(outbound_display, "neighbors_outbound.csv")
    with col2:
        st.caption("Accounts that reference this anchor")
        if inbound_display.empty:
            st.info("No inbound neighbors for this account.")
        else:
            st.dataframe(inbound_display, use_container_width=True, hide_index=True)
            download_dataframe(inbound_display, "neighbors_inbound.csv")


def render_scoring_details(df: pd.DataFrame) -> None:
    st.markdown("#### Scoring Details & Validation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accounts loaded", f"{len(df):,}")
    with col2:
        as_of = df.get("as_of_date")
        as_of_label = str(pd.Series(as_of).dropna().iloc[0]) if as_of is not None and len(pd.Series(as_of).dropna()) else "N/A"
        st.metric("Latest as-of date", as_of_label)

    entries = load_validation_log_entries()
    if not entries:
        st.info("No validation logs captured yet. Run the scoring CLI to generate validation entries.")
        return

    for idx, entry in enumerate(entries):
        with st.expander(entry["path"].name, expanded=(idx == 0)):
            if entry["lines"]:
                st.code("\n".join(entry["lines"]))
            else:
                st.write("No validation entries recorded in this file.")
            st.caption(f"Location: {entry['path']}")


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


def render_dashboard(df: pd.DataFrame, filters: FilterState, neighbors: pd.DataFrame) -> None:
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
    render_operating_pulse(filtered)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Score & Composition",
            "Coverage & Territory",
            "Expansion & Momentum",
            "Execution Hub",
            "Call List Builder",
            "Manager HQ",
            "Look-alike Lab",
            "Scoring Details",
        ]
    )

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
        expansion_tab, momentum_tab = st.tabs([
            "Expansion White Space",
            "Momentum & Engagement",
        ])
        with expansion_tab:
            col1, col2 = st.columns((1.2, 0.8))
            with col1:
                st.plotly_chart(whitespace_scatter(filtered, filters.profit_key), use_container_width=True)
            with col2:
                st.plotly_chart(whitespace_by_segment(filtered), use_container_width=True)
        with momentum_tab:
            col1, col2 = st.columns((1.2, 0.8))
            with col1:
                st.plotly_chart(momentum_scatter(filtered, filters.profit_key), use_container_width=True)
            with col2:
                st.plotly_chart(recency_distribution_chart(filtered), use_container_width=True)

    with tab4:
        watchlist_tab, playbook_tab = st.tabs([
            "Opportunity Watchlist",
            "Playbooks",
        ])
        with watchlist_tab:
            render_account_table(filtered, filters)
        with playbook_tab:
            render_playbooks(filtered, filters)

    with tab5:
        render_call_list_builder(filtered, filters)

    with tab6:
        render_manager_hq(filtered, filters)

    with tab7:
        render_neighbor_lab(filtered, neighbors, filters)

    with tab8:
        render_scoring_details(df)


def main() -> None:
    raw_df, source_path = load_portfolio_data()
    df = prepare_portfolio(raw_df)
    neighbors = load_neighbors_data()

    filters = render_sidebar(df)

    if source_path is None:
        st.info(
            "No scored portfolio file found. Displaying a representative sample dataset to illustrate the dashboard."
        )
    else:
        st.caption(f"Loaded portfolio from **{source_path.relative_to(ROOT)}**")

    render_dashboard(df, filters, neighbors)


if __name__ == "__main__":
    main()
