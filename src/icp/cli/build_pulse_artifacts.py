"""Build pulse artifacts summarizing ICP portfolio, neighbors, and playbooks.

This script is designed to be run after:
- score_accounts (icp_scored_accounts.csv)
- neighbor build (account_neighbors.csv)
- build_playbooks (account_playbooks.csv)

It writes lightweight CSVs under `artifacts/` that capture snapshot metrics
for trend tracking in Power BI or other tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from icp.schema import COL_CUSTOMER_ID, canonicalize_customer_id

ROOT = Path(__file__).resolve().parents[3]
SCORED_DEFAULT = ROOT / "data" / "processed" / "icp_scored_accounts.csv"
NEIGHBORS_DEFAULT = ROOT / "artifacts" / "account_neighbors.csv"
PLAYBOOKS_DEFAULT = ROOT / "artifacts" / "account_playbooks.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _portfolio_pulse(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()

    df = scored.copy()
    if COL_CUSTOMER_ID in df.columns:
        df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])
        df["customer_id"] = df[COL_CUSTOMER_ID].astype(str)
    else:
        df["customer_id"] = df.index.astype(str)

    gp = _safe_num(df.get("GP_Since_2023_Total", 0.0))
    hw_grade = df.get("ICP_grade_hardware", pd.Series("", index=df.index)).astype(str)
    cre_grade = df.get("ICP_grade_cre", pd.Series("", index=df.index)).astype(str)
    as_of = df.get("as_of_date")
    as_of_date = None
    if as_of is not None:
        try:
            as_of_date = pd.to_datetime(as_of).dropna().max()
        except Exception:
            as_of_date = None

    total_accounts = int(df["customer_id"].nunique())
    total_gp = float(gp.sum())

    ab_hw = df[hw_grade.isin(["A", "B"])]
    ab_cre = df[cre_grade.isin(["A", "B"])]

    row = {
        "snapshot_ts_utc": datetime.utcnow().isoformat(),
        "as_of_date": as_of_date.date().isoformat() if as_of_date is not None else "",
        "accounts_total": total_accounts,
        "gp_total": total_gp,
        "accounts_ab_hw": int(ab_hw["customer_id"].nunique()),
        "gp_ab_hw": float(_safe_num(ab_hw.get("GP_Since_2023_Total", 0.0)).sum()),
        "accounts_ab_cre": int(ab_cre["customer_id"].nunique()),
        "gp_ab_cre": float(_safe_num(ab_cre.get("GP_Since_2023_Total", 0.0)).sum()),
    }

    return pd.DataFrame.from_records([row])


def _neighbors_pulse(scored: pd.DataFrame, neighbors: pd.DataFrame) -> pd.DataFrame:
    if neighbors.empty:
        return pd.DataFrame()

    df = scored.copy()
    if COL_CUSTOMER_ID in df.columns:
        df[COL_CUSTOMER_ID] = canonicalize_customer_id(df[COL_CUSTOMER_ID])
        df["customer_id"] = df[COL_CUSTOMER_ID].astype(str)
    else:
        df["customer_id"] = df.index.astype(str)

    nb = neighbors.copy()
    nb["account_id"] = nb["account_id"].astype(str)
    nb["neighbor_account_id"] = nb["neighbor_account_id"].astype(str)

    num_accounts = int(df["customer_id"].nunique())
    num_edges = len(nb)
    avg_k = float(num_edges / num_accounts) if num_accounts else 0.0

    inbound_counts = nb["neighbor_account_id"].value_counts()
    inbound_mean = float(inbound_counts.mean()) if not inbound_counts.empty else 0.0
    inbound_p95 = float(inbound_counts.quantile(0.95)) if not inbound_counts.empty else 0.0

    # Hero neighbors: reuse the hero definition from build_playbooks (approximate here)
    hw_grade = df.get("ICP_grade_hardware", pd.Series("", index=df.index)).astype(str)
    cre_grade = df.get("ICP_grade_cre", pd.Series("", index=df.index)).astype(str)
    gp = _safe_num(df.get("GP_Since_2023_Total", 0.0))
    gp_hi = float(np.nanquantile(gp, 0.8)) if gp.notna().any() else 0.0
    is_hero = ((hw_grade == "A") | (cre_grade == "A")) & (gp >= gp_hi)
    hero_ids = set(df.loc[is_hero, "customer_id"].astype(str))

    hero_neighbor_edges = nb[nb["account_id"].isin(hero_ids)]
    hero_neighbor_ids = hero_neighbor_edges["neighbor_account_id"].astype(str).unique().tolist()

    recency_bucket = df.get("recency_bucket", pd.Series("", index=df.index)).astype(str)
    days_since = _safe_num(df.get("days_since_last_order", np.nan), np.nan)
    is_dormant = recency_bucket.str.lower().eq("dormant") | (days_since > 210)

    is_hero_neighbor = df["customer_id"].isin(hero_neighbor_ids)
    is_orphan_hero_neighbor = is_hero_neighbor & is_dormant

    hero_neighbor_count = int(is_hero_neighbor.sum())
    orphan_hero_neighbor_count = int(is_orphan_hero_neighbor.sum())

    row = {
        "snapshot_ts_utc": datetime.utcnow().isoformat(),
        "accounts_with_neighbors": num_accounts,
        "neighbor_edges": num_edges,
        "avg_neighbors_per_account": avg_k,
        "inbound_neighbors_avg": inbound_mean,
        "inbound_neighbors_p95": inbound_p95,
        "hero_count": int(is_hero.sum()),
        "hero_neighbor_count": hero_neighbor_count,
        "orphan_hero_neighbor_count": orphan_hero_neighbor_count,
    }
    return pd.DataFrame.from_records([row])


def _playbook_pulse(playbooks: pd.DataFrame) -> pd.DataFrame:
    if playbooks.empty:
        return pd.DataFrame()

    df = playbooks.copy()
    df["playbook_primary"] = df.get("playbook_primary", "").fillna("").astype(str)
    df["customer_id"] = df.get("customer_id", df.get(COL_CUSTOMER_ID, "")).astype(str)
    gp = _safe_num(df.get("GP_Since_2023_Total", 0.0))

    grouped = (
        df.groupby("playbook_primary")
        .agg(
            accounts=("customer_id", "nunique"),
            gp_total=("GP_Since_2023_Total", "sum"),
            hero_neighbor=("is_hero_neighbor", lambda s: bool(getattr(s, "any", lambda: False)())),
        )
        .reset_index()
    )

    total_accounts = int(df["customer_id"].nunique())
    total_gp = float(gp.sum())

    grouped["accounts_pct"] = grouped["accounts"] / total_accounts if total_accounts else 0.0
    grouped["gp_pct"] = grouped["gp_total"] / total_gp if total_gp else 0.0
    grouped.insert(0, "snapshot_ts_utc", datetime.utcnow().isoformat())

    return grouped


def build_pulse_artifacts(
    scored_path: Path,
    neighbors_path: Path,
    playbooks_path: Path,
    out_root: Path,
) -> None:
    scored = _load_csv(scored_path)
    neighbors = _load_csv(neighbors_path)
    playbooks = _load_csv(playbooks_path)

    out_root.mkdir(parents=True, exist_ok=True)

    portfolio_df = _portfolio_pulse(scored)
    if not portfolio_df.empty:
        portfolio_df.to_csv(out_root / "pulse_portfolio.csv", index=False)
        print(f"[INFO] Wrote portfolio pulse: {out_root / 'pulse_portfolio.csv'}")

    neighbors_df = _neighbors_pulse(scored, neighbors)
    if not neighbors_df.empty:
        neighbors_df.to_csv(out_root / "pulse_neighbors.csv", index=False)
        print(f"[INFO] Wrote neighbors pulse: {out_root / 'pulse_neighbors.csv'}")

    playbook_df = _playbook_pulse(playbooks)
    if not playbook_df.empty:
        playbook_df.to_csv(out_root / "pulse_playbooks.csv", index=False)
        print(f"[INFO] Wrote playbooks pulse: {out_root / 'pulse_playbooks.csv'}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build snapshot pulse artifacts for portfolio, neighbors, and playbooks.")
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
        help=f"Path to neighbors CSV (default: {NEIGHBORS_DEFAULT})",
    )
    parser.add_argument(
        "--playbooks",
        type=str,
        default=str(PLAYBOOKS_DEFAULT),
        help=f"Path to playbooks CSV (default: {PLAYBOOKS_DEFAULT})",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "artifacts"),
        help="Directory for pulse CSV outputs (default: artifacts/).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build_pulse_artifacts(
        scored_path=Path(args.in_scored),
        neighbors_path=Path(args.neighbors),
        playbooks_path=Path(args.playbooks),
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()

