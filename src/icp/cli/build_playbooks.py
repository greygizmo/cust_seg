"""Generate rule-based playbooks and tags for scored accounts.

This CLI is intended to run *after* the main scoring + neighbors pipeline.
It reads:

- `data/processed/icp_scored_accounts.csv` (or a supplied path)
- `artifacts/account_neighbors.csv` (optional)

and writes a compact artifact with CRO/CFO-friendly playbooks:

- `artifacts/account_playbooks.csv`

The artifact is safe to import into Power BI or join into Streamlit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from icp.schema import COL_CUSTOMER_ID, COL_COMPANY_NAME, canonicalize_customer_id


ROOT = Path(__file__).resolve().parents[3]
SCORED_DEFAULT = ROOT / "data" / "processed" / "icp_scored_accounts.csv"
NEIGHBORS_DEFAULT = ROOT / "artifacts" / "account_neighbors.csv"
OUT_DEFAULT = ROOT / "artifacts" / "account_playbooks.csv"


@dataclass
class PlaybookDefinition:
    name: str
    description: str


def _load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scored accounts file not found: {path}")
    df = pd.read_csv(path)
    if COL_CUSTOMER_ID not in df.columns:
        raise ValueError(f"Expected '{COL_CUSTOMER_ID}' in scored accounts.")
    df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])
    df["customer_id"] = df[COL_CUSTOMER_ID].astype(str)
    if COL_COMPANY_NAME not in df.columns:
        fallback = df.get("company_name", df.index.astype(str))
        df[COL_COMPANY_NAME] = fallback.astype(str)
    df[COL_COMPANY_NAME] = df[COL_COMPANY_NAME].fillna("").astype(str)
    return df


def _load_neighbors(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("account_id", "neighbor_account_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _safe_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _flag_heroes(scored: pd.DataFrame) -> pd.Series:
    """Identify 'hero' accounts (A-grade, high GP) across HW/CRE."""

    gp = _safe_series(scored, "GP_Since_2023_Total", 0.0)
    gp_thresh = float(np.nanquantile(gp, 0.8)) if gp.notna().any() else 0.0

    hw_grade = scored.get("ICP_grade_hardware", pd.Series("", index=scored.index)).astype(str)
    cre_grade = scored.get("ICP_grade_cre", pd.Series("", index=scored.index)).astype(str)

    is_hw_hero = (hw_grade == "A") & (gp >= gp_thresh)
    is_cre_hero = (cre_grade == "A") & (gp >= gp_thresh)
    return (is_hw_hero | is_cre_hero).astype(bool)


def _compute_neighbor_flags(scored: pd.DataFrame, neighbors: pd.DataFrame) -> pd.DataFrame:
    """Add simple neighbor-based features: hero look-alike and orphan look-alike flags."""

    scored = scored.copy()
    scored["is_hero"] = _flag_heroes(scored)

    if neighbors.empty:
        scored["is_hero_neighbor"] = False
        scored["is_hero_orphan_neighbor"] = False
        scored["hero_neighbor_count"] = 0
        return scored

    hero_ids = set(scored.loc[scored["is_hero"], "customer_id"].astype(str))
    if not hero_ids:
        scored["is_hero_neighbor"] = False
        scored["is_hero_orphan_neighbor"] = False
        scored["hero_neighbor_count"] = 0
        return scored

    # Neighbors of hero accounts
    hero_edges = neighbors[neighbors["account_id"].isin(hero_ids)].copy()
    hero_neighbors = hero_edges["neighbor_account_id"].astype(str)
    hero_counts = hero_neighbors.value_counts()

    scored["hero_neighbor_count"] = scored["customer_id"].map(hero_counts).fillna(0).astype(int)
    scored["is_hero_neighbor"] = scored["hero_neighbor_count"] > 0

    # Orphan: hero neighbor that is dormant / long-recency
    recency_bucket = scored.get("recency_bucket", pd.Series("", index=scored.index)).astype(str)
    days_since = _safe_series(scored, "days_since_last_order", np.nan)
    is_dormant = recency_bucket.str.lower().eq("dormant") | (days_since > 210)
    scored["is_hero_orphan_neighbor"] = scored["is_hero_neighbor"] & is_dormant

    return scored


def _build_playbook_masks(scored: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, PlaybookDefinition]]:
    """Construct boolean masks for each playbook/tag and their definitions."""

    gp = _safe_series(scored, "GP_Since_2023_Total", 0.0)
    gp_med = float(np.nanquantile(gp, 0.5)) if gp.notna().any() else 0.0
    gp_hi = float(np.nanquantile(gp, 0.8)) if gp.notna().any() else 0.0

    hw_score = _safe_series(scored, "ICP_score_hardware", np.nan)
    cre_score = _safe_series(scored, "ICP_score_cre", np.nan)
    hw_grade = scored.get("ICP_grade_hardware", pd.Series("", index=scored.index)).astype(str)
    cre_grade = scored.get("ICP_grade_cre", pd.Series("", index=scored.index)).astype(str)

    whitespace = _safe_series(scored, "whitespace_score", 0.0)
    hw_share = _safe_series(scored, "hw_share_12m", np.nan)
    sw_share = _safe_series(scored, "sw_share_12m", np.nan)

    training_hw = _safe_series(scored, "training_to_hw_ratio", np.nan)
    training_cre = _safe_series(scored, "training_to_cre_ratio", np.nan)
    cre_training_gp = _safe_series(scored, "CRE_Training", 0.0)

    momentum = _safe_series(scored, "momentum_score", np.nan)
    delta_13w_pct = _safe_series(scored, "delta_13w_pct", np.nan)
    gp_qoq = _safe_series(scored, "Profit_QoQ_Growth", np.nan)

    recency_bucket = scored.get("recency_bucket", pd.Series("", index=scored.index)).astype(str)
    customer_segment = scored.get("customer_segment", pd.Series("", index=scored.index)).astype(str)

    is_hero = scored.get("is_hero", pd.Series(False, index=scored.index))
    is_hero_neighbor = scored.get("is_hero_neighbor", pd.Series(False, index=scored.index))
    is_hero_orphan_neighbor = scored.get("is_hero_orphan_neighbor", pd.Series(False, index=scored.index))

    playbooks: dict[str, PlaybookDefinition] = {}
    masks: dict[str, pd.Series] = {}

    def add(name: str, description: str, mask: pd.Series) -> None:
        playbooks[name] = PlaybookDefinition(name=name, description=description)
        masks[name] = mask.fillna(False)

    # 1. Hardware expansion sprints (CRO: maximize GP in high-fit HW accounts)
    mask_hw_expansion = (
        hw_grade.isin(["A", "B"])
        & (whitespace >= 0.45)
        & ((momentum >= 0.5) | (delta_13w_pct >= 0.0))
        & (gp >= gp_med)
    )
    add(
        "HW Expansion Sprint",
        "High-fit HW A/B accounts with meaningful whitespace and non-negative momentum. Prioritize expansion campaigns here.",
        mask_hw_expansion,
    )

    # 2. CRE expansion sprints (CRO: deepen software/CRE footprint)
    mask_cre_expansion = (
        cre_grade.isin(["A", "B"])
        & (cre_score >= 60)
        & (cre_training_gp >= 0)
        & (training_cre < 0.4)
        & (gp >= gp_med)
    )
    add(
        "CRE Expansion Sprint",
        "High-fit CRE A/B accounts with under-attached training/services. Run attach and expansion plays.",
        mask_cre_expansion,
    )

    # 3. Cross-sell software into HW-first accounts
    mask_hw_to_sw = (
        (hw_share >= 0.7)
        & (sw_share <= 0.3)
        & (gp >= gp_med)
    )
    add(
        "Cross-sell CAD/CRE",
        "HW-heavy customers with little CRE/software footprint. Lead with CAD/CRE and portfolio expansion.",
        mask_hw_to_sw,
    )

    # 4. Cross-sell hardware into CRE-first accounts
    printer_count = _safe_series(scored, "printer_count", 0.0)
    mask_sw_to_hw = (
        (sw_share >= 0.7)
        & ((hw_share.isna()) | (hw_share <= 0.3))
        & (printer_count <= 0.5)
        & (gp >= gp_med)
    )
    add(
        "Cross-sell Hardware",
        "CRE/software-heavy customers with little or no hardware footprint. Position printers and hardware bundles.",
        mask_sw_to_hw,
    )

    # 5. Training attach / services uplift (CFO: grow high-margin services)
    mask_training_attach = (
        (gp >= gp_hi)
        & ((training_hw < 0.3) | (training_cre < 0.3))
    )
    add(
        "Training & Services Attach",
        "High GP accounts under-indexed on training/services. Attach success plans, training, and adoption packages.",
        mask_training_attach,
    )

    # 6. Retention risk on high GP accounts
    mask_retention_risk = (
        (gp >= gp_hi)
        & (
            recency_bucket.isin(["At Risk", "Dormant"])
            | (momentum < 0.4)
            | (gp_qoq <= -0.05)
        )
    )
    add(
        "High-Value Retention Risk",
        "High GP accounts with deteriorating momentum or recency. Escalate retention and executive coverage.",
        mask_retention_risk,
    )

    # 7. Reactivation targets: decent history, currently dormant
    mask_reactivation = (
        (gp >= gp_med)
        & recency_bucket.eq("Dormant")
        & (momentum.isna() | (momentum < 0.4))
    )
    add(
        "Reactivation Campaign",
        "Historically meaningful accounts that are now dormant. Run structured reactivation outreach.",
        mask_reactivation,
    )

    # 8. Land-and-expand: high fit, low current GP
    mask_land_expand = (
        (gp < gp_med)
        & ((hw_score >= 65) | (cre_score >= 65))
        & (whitespace >= 0.4)
        & customer_segment.isin(["Strategic", "Growth"])
    )
    add(
        "Land & Expand",
        "High-fit Strategic/Growth accounts with low current GP but strong scores and whitespace.",
        mask_land_expand,
    )

    # 9. Hero look-alikes
    mask_hero_neighbor = is_hero_neighbor & (~is_hero)
    add(
        "Hero Look-alike",
        "Accounts that closely resemble your hero A accounts in behavior and portfolio mix. Replicate winning plays here.",
        mask_hero_neighbor,
    )

    # 10. Orphan look-alikes (hero neighbors at risk)
    mask_orphan_hero_neighbor = is_hero_orphan_neighbor
    add(
        "Orphan Hero Look-alike",
        "Hero-like accounts that are dormant or aging. Prioritize save/reactivation motions.",
        mask_orphan_hero_neighbor,
    )

    # 11. Balanced portfolio densification
    balance = _safe_series(scored, "cross_division_balance_score", np.nan)
    mask_balance = (
        (balance.between(0.7, 1.3, inclusive="both"))
        & (whitespace >= 0.3)
        & (gp >= gp_med)
    )
    add(
        "Portfolio Densification",
        "Balanced HW/SW accounts with remaining whitespace. Deepen product penetration and multi-line adoption.",
        mask_balance,
    )

    return masks, playbooks


def _derive_playbooks(scored_with_flags: pd.DataFrame) -> pd.DataFrame:
    masks, defs = _build_playbook_masks(scored_with_flags)
    tag_names: List[str] = list(masks.keys())

    # Precompute boolean frame of tags
    tag_frame = pd.DataFrame({name: masks[name] for name in tag_names}, index=scored_with_flags.index)

    primary_tags: list[str] = []
    all_tags: list[str] = []
    rationale: list[str] = []

    # Priority order for primary tag (CRO/CFO leaning)
    primary_order = [
        "High-Value Retention Risk",
        "HW Expansion Sprint",
        "CRE Expansion Sprint",
        "Cross-sell CAD/CRE",
        "Cross-sell Hardware",
        "Training & Services Attach",
        "Hero Look-alike",
        "Orphan Hero Look-alike",
        "Land & Expand",
        "Reactivation Campaign",
        "Portfolio Densification",
    ]

    desc_map = {name: defs[name].description for name in tag_names}

    for idx in scored_with_flags.index:
        active_tags = [name for name in tag_names if bool(tag_frame.at[idx, name])]
        if active_tags:
            primary = next((name for name in primary_order if name in active_tags), active_tags[0])
        else:
            primary = ""

        primary_tags.append(primary)
        all_tags.append("; ".join(active_tags))

        if active_tags:
            # Keep rationale short: up to 2 tag descriptions
            chosen = active_tags[:2]
            rationale.append(" | ".join(desc_map[n] for n in chosen))
        else:
            rationale.append("")

    out = pd.DataFrame(
        {
            "customer_id": scored_with_flags["customer_id"].astype(str),
            COL_CUSTOMER_ID: scored_with_flags[COL_CUSTOMER_ID].astype(str),
            COL_COMPANY_NAME: scored_with_flags[COL_COMPANY_NAME].astype(str),
            "playbook_primary": primary_tags,
            "playbook_tags": all_tags,
            "playbook_rationale": rationale,
        },
        index=scored_with_flags.index,
    )

    # Include a few numeric fields for BI convenience
    for col in [
        "GP_Since_2023_Total",
        "ICP_score_hardware",
        "ICP_grade_hardware",
        "ICP_score_cre",
        "ICP_grade_cre",
        "whitespace_score",
        "hw_share_12m",
        "sw_share_12m",
        "CRE_Training",
        "days_since_last_order",
        "customer_segment",
        "recency_bucket",
    ]:
        if col in scored_with_flags.columns:
            out[col] = scored_with_flags[col]

    # Neighbor flags
    for col in ["is_hero", "is_hero_neighbor", "is_hero_orphan_neighbor", "hero_neighbor_count"]:
        if col in scored_with_flags.columns:
            out[col] = scored_with_flags[col]

    return out


def build_playbooks(scored_path: Path, neighbors_path: Path, out_path: Path) -> Path:
    scored = _load_scored(scored_path)
    neighbors = _load_neighbors(neighbors_path)
    scored_flags = _compute_neighbor_flags(scored, neighbors)
    artifact = _derive_playbooks(scored_flags)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_csv(out_path, index=False)
    print(f"[INFO] Wrote playbooks artifact: {out_path}")
    return out_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based playbooks and tags from scored accounts.")
    parser.add_argument(
        "--in-scored",
        type=str,
        default=str(SCORED_DEFAULT),
        help=f"Path to scored accounts CSV (default: {SCORED_DEFAULT})",
    )
    parser.add_argument(
        "--neighbors",
        type=str,
        default=str(NEIGHBORS_DEFAULT),
        help=f"Path to neighbors CSV (default: {NEIGHBORS_DEFAULT}, optional).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_DEFAULT),
        help=f"Output CSV path for playbooks (default: {OUT_DEFAULT})",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    scored_path = Path(args.in_scored)
    neighbors_path = Path(args.neighbors)
    out_path = Path(args.out)
    build_playbooks(scored_path, neighbors_path, out_path)


if __name__ == "__main__":
    main()

