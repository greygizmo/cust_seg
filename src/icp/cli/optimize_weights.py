import os
import json
from functools import partial
from pathlib import Path
from typing import Tuple

import optuna
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

from icp.optimization import objective, calculate_grades, TARGET_GRADE_DISTRIBUTION, cumulative_lift
from icp.divisions import available_divisions
from icp import data_access as da
from icp.schema import canonicalize_customer_id, COL_CUSTOMER_ID


ROOT = Path(__file__).resolve().parents[3]


def _quarter_key_from_date(dt) -> int:
    dt = pd.to_datetime(dt, errors='coerce')
    if pd.isna(dt):
        return 0
    return int(dt.year) * 10 + int(((dt.month - 1) // 3) + 1)


def _quarter_key_from_str(qs: str) -> int:
    # Expect format 'YYYYQn'
    try:
        s = str(qs)
        yr = int(s[0:4]); qn = int(s[-1])
        return yr * 10 + qn
    except Exception:
        return 0


def _build_future_outcome(df: pd.DataFrame, division: str, horizon_q: int = 1) -> pd.Series:
    """Build future GP outcome by division and horizon using Azure SQL aggregates."""
    engine = da.get_engine()
    q_goal = da.get_quarterly_profit_by_goal(engine)
    if COL_CUSTOMER_ID in q_goal.columns:
        q_goal[COL_CUSTOMER_ID] = canonicalize_customer_id(q_goal[COL_CUSTOMER_ID])
    # Prepare keys
    q_goal["_qkey"] = q_goal["Quarter"].map(_quarter_key_from_str)
    # Allowed goals by division
    if division == 'hardware':
        goals = {"Printers", "Printer Accessorials", "Scanners", "Geomagic"}
        goals_norm = {g for g in goals}
        q_goal_div = q_goal[q_goal["Goal"].isin(goals_norm)].copy()
        # No rollup filtering needed
        q_roll = None
        # Use goal-level directly for hardware labels
        q_div = q_goal_div[[COL_CUSTOMER_ID,"_qkey","Profit"]].copy()
    else:
        # CRE: CAD + Specialty Software + Training/Services (Success Plan, Training)
        goals = {"CAD", "Specialty Software"}
        q_goal_div = q_goal[q_goal["Goal"].isin(goals)].copy()
        # Training subset from rollups
        q_roll = da.get_quarterly_profit_by_rollup(engine)
        if COL_CUSTOMER_ID in q_roll.columns:
            q_roll[COL_CUSTOMER_ID] = canonicalize_customer_id(q_roll[COL_CUSTOMER_ID])
        q_roll["_qkey"] = q_roll["Quarter"].map(_quarter_key_from_str)
        is_train = q_roll["Goal"] == "Training/Services"
        ir = q_roll["item_rollup"].astype(str).str.strip().str.lower()
        allowed_train = {"success plan", "training"}
        mask_allowed = is_train & ir.isin(allowed_train)
        q_roll = q_roll[mask_allowed]
        # Combine CAD+Specialty (goal-level) with CRE Training subset (rollup-level)
        q_div = pd.concat([
            q_goal_div[[COL_CUSTOMER_ID,"_qkey","Profit"]],
            q_roll[[COL_CUSTOMER_ID,"_qkey","Profit"]]
        ], ignore_index=True)

    # Build as_of quarter per account
    if 'as_of_date' in df.columns:
        as_of_q = df['as_of_date'].map(_quarter_key_from_date)
    else:
        # Fallback to latest completed quarter for all
        now = pd.Timestamp.now(); as_of_q = pd.Series(_quarter_key_from_date(now), index=df.index)

    # Compute future quarter keys with proper year rollover (Q4 -> next year's Q1)
    def _add_quarters_key(qkey: int, add: int) -> int:
        try:
            qkey_int = int(qkey)
        except Exception:
            return 0
        if qkey_int <= 0:
            return 0
        y = qkey_int // 10
        q = qkey_int % 10
        total = (q - 1) + int(add)
        y2 = y + (total // 4)
        q2 = (total % 4) + 1
        return int(y2) * 10 + int(q2)

    future_q = as_of_q.astype(int).map(lambda v: _add_quarters_key(v, horizon_q))
    # Sum outcome for the exact future quarter (or extend across multiple horizons if desired)
    # Here: single next quarter horizon; build mapping series
    m_goal = (
        q_div.groupby([COL_CUSTOMER_ID, "_qkey"])['Profit'].sum()
        .rename('gp')
    )
    ids = canonicalize_customer_id(df[COL_CUSTOMER_ID]) if COL_CUSTOMER_ID in df.columns else df.index.astype(str)
    keys = list(zip(ids.astype(str), future_q))
    y_goal = pd.Series([m_goal.get(k, 0.0) for k in keys], index=df.index)

    if q_roll is not None:
        m_roll = (
            q_roll.set_index([COL_CUSTOMER_ID,"_qkey"])['Profit']
            .rename('gp_roll')
        )
        y_roll = pd.Series([m_roll.get(k, 0.0) for k in keys], index=df.index)
    else:
        y_roll = pd.Series(0.0, index=df.index)

    y_future = (pd.to_numeric(y_goal, errors='coerce').fillna(0) + pd.to_numeric(y_roll, errors='coerce').fillna(0))
    return y_future


def run_optimization(
    division: str,
    n_trials=1000,
    lambda_param=0.25,
    horizon_quarters: int = 1,
    group_col: str = 'Industry',
    horizons: list[int] | None = None,
    append_history: bool = False,
    include_size: bool = False,
    out_path: str | None = None,
):
    print("Loading scored accounts data...")
    path = ROOT / 'data' / 'processed' / 'icp_scored_accounts.csv'
    if not path.exists():
        print("Error: `data/processed/icp_scored_accounts.csv` not found. Run scoring first.")
        return
    df = pd.read_csv(path)
    if COL_CUSTOMER_ID not in df.columns:
        print(f"Error: '{COL_CUSTOMER_ID}' missing in scored CSV.")
        return

    # Component features (size_score deprecated; optimization uses 3 components)
    weight_names = ['vertical_score', 'adoption_score', 'relationship_score']
    cols_ok = True
    missing = []
    for col in weight_names:
        if col not in df.columns:
            missing.append(col)
    # Fallbacks for adoption/relationship from aliased columns in the CSV
    X = pd.DataFrame(index=df.index)
    if 'vertical_score' in df.columns:
        X['vertical_score'] = df['vertical_score']
    # Adoption
    if 'adoption_score' in df.columns:
        X['adoption_score'] = df['adoption_score']
    elif 'Hardware_score' in df.columns:
        X['adoption_score'] = df['Hardware_score']
    else:
        cols_ok = False
        print("Error: neither 'adoption_score' nor 'Hardware_score' present.")
    # Relationship
    if 'relationship_score' in df.columns:
        X['relationship_score'] = df['relationship_score']
    elif 'Software_score' in df.columns:
        X['relationship_score'] = df['Software_score']
    else:
        cols_ok = False
        print("Error: neither 'relationship_score' nor 'Software_score' present.")
    size_column = None
    if include_size:
        for candidate in ("size_score", "Size_score", "size_component"):
            if candidate in df.columns:
                size_column = candidate
                break
        if size_column is None:
            print("[WARN] --include-size requested but no size column found; continuing without size.")
            include_size = False

    if include_size and size_column:
        X['size_score'] = pd.to_numeric(df[size_column], errors='coerce').fillna(0.0)
        weight_names.append('size_score')

    if not cols_ok or 'vertical_score' not in X.columns:
        print("Error: Required score columns missing; cannot proceed.")
        return

    # Build folds by time horizons (average objective across folds)
    if horizons is None:
        horizons = [horizon_quarters] if horizon_quarters else [1, 2]
    folds: list[tuple[pd.DataFrame, pd.Series, pd.Series | None, int]] = []
    label_counts: dict[str, dict[str, int]] = {}
    for hz in horizons:
        print(f"Building future outcome for division={division}, horizon={hz}Q...")
        y_future = _build_future_outcome(df, division=division, horizon_q=hz)
        mask_valid = (~X.isna().any(axis=1)) & y_future.notna()
        Xf = X[mask_valid]
        yf = y_future[mask_valid]
        gf = df.loc[mask_valid, group_col] if group_col in df.columns else None
        try:
            nz = int((pd.to_numeric(yf, errors='coerce').fillna(0) > 0).sum())
            print(f"  - label rows: {len(yf)} | nonzero labels: {nz}")
            label_counts[str(hz)] = {"rows": int(len(yf)), "nonzero": int(nz)}
        except Exception:
            label_counts[str(hz)] = {"rows": int(len(yf)), "nonzero": 0}
        if len(Xf) == 0:
            continue
        folds.append((Xf, yf, gf, hz))
    if not folds:
        print("Error: no valid folds found for optimization (check horizons and data availability).")
        return

    print(f"Optimizing weights on {len(folds)} time folds | lambda={lambda_param}")

    def fold_objective(trial):
        vals = []
        for Xf, yf, gf, _hz in folds:
            val = objective(
                trial,
                X=Xf,
                y=yf,
                lambda_param=lambda_param,
                weight_names=weight_names,
                include_size=include_size,
                groups=gf,
                lift_weight=0.05,
                stability_weight=0.05,
            )
            vals.append(val)
        return float(np.mean(vals)) if vals else 1e9

    study = optuna.create_study(direction='minimize')
    study.optimize(fold_objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    normalized_keys = weight_names.copy()
    weights_subset = {name: best.get(name, 0.0) for name in normalized_keys}
    tot = sum(weights_subset.values())
    if tot <= 0:
        print("Error: optimizer returned invalid weights.")
        return
    weights_subset = {name: (value / tot) for name, value in weights_subset.items()}
    if include_size:
        weights_subset.setdefault('size_score', best.get('size_score', 0.0))
    else:
        weights_subset['size_score'] = 0.0
    best = weights_subset

    # Recompute key metrics for reporting across folds
    weights = np.array([best[name] for name in weight_names])
    # Aggregate reporting across folds
    agg_sp, agg_kl, agg_stab = [], [], []
    lift_curves = {}
    lifts_by_grade = None
    from icp.optimization import _lift_by_grade, _stability_penalty
    for Xf, yf, gf, hz in folds:
        s = Xf.dot(weights)
        sp, _ = spearmanr(s, yf)
        grades = calculate_grades(s)
        actual_distribution = pd.Series(grades).value_counts(normalize=True).reindex(['F','D','C','B','A'], fill_value=0)
        tgt = TARGET_GRADE_DISTRIBUTION
        epsilon = 1e-10
        kl = float(np.sum(actual_distribution * np.log((actual_distribution + epsilon) / (np.array([tgt['F'],tgt['D'],tgt['C'],tgt['B'],tgt['A']]) + epsilon))))
        lifts = _lift_by_grade(s, yf)
        stab = _stability_penalty(s, yf, gf)
        curve = cumulative_lift(s, yf)
        lift_curves[str(hz)] = curve
        if lifts_by_grade is None:
            lifts_by_grade = lifts
        else:
            for k in list(lifts_by_grade.keys()):
                lifts_by_grade[k] = 0.5 * (lifts_by_grade[k] + lifts.get(k, lifts_by_grade[k]))
        agg_sp.append(0.0 if not np.isfinite(sp) else float(sp))
        agg_kl.append(float(kl))
        agg_stab.append(float(stab))

    sp_mean = float(np.mean(agg_sp)) if agg_sp else 0.0
    kl_mean = float(np.mean(agg_kl)) if agg_kl else 0.0
    stab_mean = float(np.mean(agg_stab)) if agg_stab else 0.0

    # Save per-division weights structure
    default_out = ROOT / 'artifacts' / 'weights' / 'optimized_weights.json'
    out_path = Path(out_path) if out_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try to capture the as_of_date used in scoring
    as_of_val = None
    try:
        if 'as_of_date' in df.columns:
            uniq = [str(x) for x in pd.unique(df['as_of_date'].dropna())]
            as_of_val = uniq[0] if uniq else None
    except Exception:
        as_of_val = None

    from datetime import datetime, timezone
    div_meta = {
        'division': division,
        'as_of_date': as_of_val,
        'n_trials': n_trials,
        'lambda_param': lambda_param,
        'horizons': horizons,
        'group_col': group_col,
        'best_objective_value': study.best_value,
        'spearman_mean': sp_mean,
        'kl_divergence_mean': kl_mean,
        'stability_std_spearman_mean': stab_mean,
        'lifts': lifts_by_grade or {},
        'lift_curves': lift_curves,
        'label_counts': label_counts,
        'run_timestamp_utc': datetime.now(tz=timezone.utc).isoformat(timespec='seconds'),
    }

    payload = {
        'weights': {division: best},
        'division': division,
        'meta': {division: div_meta},
    }
    # Merge with existing if present
    try:
        if out_path.exists():
            with out_path.open('r', encoding='utf-8') as f:
                old = json.load(f)
        else:
            old = {}
    except Exception:
        old = {}
    # Merge strategy: merge weights dict by division, and store meta per-division
    if isinstance(old.get('weights'), dict):
        merged_weights = dict(old['weights'])
        merged_weights.update(payload['weights'])
        payload['weights'] = merged_weights
    # Merge meta
    old_meta = old.get('meta')
    meta_by_div: dict = {}
    if isinstance(old_meta, dict) and any(k in old_meta for k in ('hardware','cre')):
        meta_by_div = dict(old_meta)
    elif isinstance(old_meta, dict) and isinstance(old.get('division'), str):
        # Convert legacy single-meta to per-division if possible
        meta_by_div = {old.get('division'): old_meta}
    meta_by_div.update(payload['meta'])
    payload['meta'] = meta_by_div

    # Optional: append to meta history
    if append_history:
        hist = old.get('meta_history', [])
        if not isinstance(hist, list):
            hist = []
        entry = dict(div_meta)
        hist.append(entry)
        payload['meta_history'] = hist

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)

    print("\n--- Optimization Complete ---")
    print(f"Best objective value: {study.best_value:.4f}")
    for k, v in best.items():
        print(f"  - {k}: {v:.4f}")
    print(f"Spearman (future GP, mean across folds): {div_meta['spearman_mean']:.4f}")
    print(f"KL(grade||target) mean: {div_meta['kl_divergence_mean']:.4f}")
    print("Per-grade lifts:")
    for g in ['A','B','C','D','F']:
        print(f"  lift@{g}: {div_meta['lifts'].get(f'lift@{g}', 1.0):.3f}")
    print(f"Stability (std Spearman by {group_col}, mean): {div_meta['stability_std_spearman_mean']:.4f}")
    for hz, curve in div_meta['lift_curves'].items():
        print(f"Lift curve AUC (horizon {hz}Q): {curve.get('auc', 0.0):.4f}")
    print(f"\nSaved optimized weights to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Optimize ICP weights (no-ML) with Optuna on future GP")
    parser.add_argument("--division", type=str, default="hardware", choices=list(available_divisions()), help="Division key to optimize")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of trials")
    parser.add_argument("--lambda", dest="lambda_param", type=float, default=0.25, help="Trade-off between grade calibration and correlation")
    parser.add_argument("--horizon", dest="horizon_quarters", type=int, default=1, help="Future horizon in quarters (1=next quarter)")
    parser.add_argument("--group-col", type=str, default="Industry", help="Group column for stability (optional)")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated future horizons in quarters, e.g. '1,2'")
    parser.add_argument("--append-history", action="store_true", help="Append this run's meta to meta_history in optimized_weights.json")
    parser.add_argument("--include-size", action="store_true", help="Allow the optimization to allocate weight to the legacy size component")
    parser.add_argument("--out", type=str, default=None, help="Custom output path for optimized_weights.json")
    args = parser.parse_args()

    hz = None
    if args.horizons:
        try:
            hz = [int(x.strip()) for x in str(args.horizons).split(',') if x.strip()]
        except Exception:
            hz = None
    run_optimization(
        division=args.division,
        n_trials=args.n_trials,
        lambda_param=args.lambda_param,
        horizon_quarters=args.horizon_quarters,
        group_col=args.group_col,
        horizons=hz,
        append_history=bool(args.append_history),
        include_size=bool(args.include_size),
        out_path=args.out,
    )
