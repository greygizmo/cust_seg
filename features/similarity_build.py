from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .text_embed import embed_text


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if s.isna().all():
        return s
    lower, upper = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower, upper)


def _robust_z(s: pd.Series) -> pd.Series:
    m = s.median()
    mad = (s - m).abs().median()
    scale = 1.4826 * mad if mad and mad > 0 else (s.std(ddof=0) or 1.0)
    return (s - m) / scale


def _log1p_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64").clip(lower=0))
    return df


def _logit_cols(df: pd.DataFrame, cols: List[str], eps: float = 1e-3) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").astype("float64").fillna(0)
            x = x.clip(eps, 1 - eps)
            df[c] = np.log(x / (1 - x))
    return df


def _build_numeric_block(accounts: pd.DataFrame, cfg: Dict) -> np.ndarray:
    drop = {"account_id", "account_name", "as_of_date", "run_timestamp_utc", "industry", "segment", "territory", cfg.get("text_column", "industry_reasoning")}
    candidates = [c for c in accounts.columns if c not in drop and accounts[c].dtype.kind in "fc"]
    if not candidates:
        return np.zeros((len(accounts), 1), dtype="float32")
    df = accounts[candidates].copy()
    df = _log1p_cols(df, cfg.get("log1p_cols", []))
    df = _logit_cols(df, cfg.get("logit_cols", []))
    p = float(cfg.get("winsor_pct", 0.01))
    df = df.apply(lambda s: _robust_z(_winsorize(s, p)), axis=0)
    X = df.replace([np.inf, -np.inf], 0).fillna(0.0).to_numpy(dtype="float32")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _build_categorical_block(accounts: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cats = {}
    for col in ["industry", "segment", "territory", "top_subdivision_12m"]:
        if col in accounts.columns:
            cats[col] = accounts[col].astype("category")
    if not cats:
        return np.zeros((len(accounts), 1), dtype="float32"), []
    matrices = []
    for col, series in cats.items():
        X = pd.get_dummies(series, prefix=col, drop_first=False, dtype="float32")
        matrices.append(X)
    M = pd.concat(matrices, axis=1)
    X = M.to_numpy(dtype="float32")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n, list(M.columns)


def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


def _topk_neighbors_blockwise(V: np.ndarray, k: int, block_rows: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k cosine neighbors for each row of V without materializing
    the full dense N x N similarity matrix.

    Returns (indices, sims) with shape (N, k).
    Assumes rows of V are L2-normalized.
    """
    N = V.shape[0]
    k_eff = max(0, min(int(k), max(0, N - 1)))
    if N == 0 or k_eff == 0:
        return np.empty((N, 0), dtype=np.int32), np.empty((N, 0), dtype=np.float32)

    block = max(1, int(block_rows))
    out_idx = np.empty((N, k_eff), dtype=np.int32)
    out_sim = np.empty((N, k_eff), dtype=np.float32)

    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        S_blk = V[i0:i1] @ V.T  # (B, N)
        # Exclude self-similarity from candidates
        for r in range(i1 - i0):
            S_blk[r, i0 + r] = -1.0
        # Select top-k per row in block
        # Use argpartition for efficiency, then sort those k
        if k_eff < N - 1:
            idx_part = np.argpartition(-S_blk, kth=k_eff, axis=1)[:, :k_eff]
            # Gather sims and sort within the k subset
            sims_part = np.take_along_axis(S_blk, idx_part, axis=1)
            order = np.argsort(-sims_part, axis=1)
            idx_sorted = np.take_along_axis(idx_part, order, axis=1)
            sims_sorted = np.take_along_axis(sims_part, order, axis=1)
        else:
            # k == N-1, sort all
            idx_sorted = np.argsort(-S_blk, axis=1)[:, :k_eff]
            sims_sorted = np.take_along_axis(S_blk, idx_sorted, axis=1)

        out_idx[i0:i1, :] = idx_sorted.astype(np.int32, copy=False)
        out_sim[i0:i1, :] = sims_sorted.astype(np.float32, copy=False)

    return out_idx, out_sim


def _combine_blocks(num: np.ndarray | None, cat: np.ndarray | None, txt: np.ndarray | None, als: np.ndarray | None,
                    w_num: float, w_cat: float, w_txt: float, w_als: float) -> np.ndarray:
    blocks = []
    if num is not None: blocks.append(w_num * num)
    if cat is not None: blocks.append(w_cat * cat)
    if txt is not None: blocks.append(w_txt * txt)
    if als is not None: blocks.append(w_als * als)
    if not blocks:
        return np.zeros((0, 1), dtype="float32")
    V = np.concatenate(blocks, axis=1).astype("float32")
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V / n


def build_neighbors(accounts_df: pd.DataFrame, tx_joined: pd.DataFrame, cfg: Dict, als_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build hybrid embeddings and compute top-K nearest neighbors per account.
    Returns a table suitable for Power BI import.
    """
    # Numeric block
    X_num = _build_numeric_block(accounts_df, cfg)

    # Categorical block
    X_cat, _ = _build_categorical_block(accounts_df)

    # Text block
    text_col = cfg.get("text_column", "industry_reasoning")
    if text_col in accounts_df.columns and cfg.get("use_text", True):
        X_txt = embed_text(accounts_df[text_col])
    else:
        X_txt = np.zeros((len(accounts_df), 1), dtype="float32")

    # ALS block (optional)
    if cfg.get("use_als", False):
        if als_df is None or "als_vec" not in als_df.columns:
            try:
                from .als_embed import als_account_vectors
                als_df = als_account_vectors(tx_joined)
            except Exception:
                als_df = None
        if als_df is not None and not als_df.empty:
            als_map = dict(zip(als_df["account_id"].astype(str), als_df["als_vec"]))
            V_als = np.stack([als_map.get(str(a), np.zeros((64,), dtype="float32")) for a in accounts_df["account_id"]], axis=0)
            V_als = V_als / (np.linalg.norm(V_als, axis=1, keepdims=True) + 1e-12)
        else:
            V_als = np.zeros((len(accounts_df), 1), dtype="float32")
    else:
        V_als = np.zeros((len(accounts_df), 1), dtype="float32")

    V = _combine_blocks(
        X_num, X_cat, X_txt, V_als,
        float(cfg.get("w_numeric", 0.5)), float(cfg.get("w_categorical", 0.15)),
        float(cfg.get("w_text", 0.25)), float(cfg.get("w_als", 0.10))
    )

    # Compute top-K neighbors without allocating full N x N if requested/needed
    k = int(cfg.get("k_neighbors", 25))
    max_dense = int(cfg.get("max_dense_accounts", 5000))
    block_rows = int(cfg.get("row_block_size", 512))

    if V.shape[0] <= max_dense:
        # Safe to use dense similarity
        S = _cosine_sim(V, V)
        np.fill_diagonal(S, -1.0)
        k_eff = min(k, S.shape[1] - 1) if S.shape[1] > 1 else 0
        if k_eff <= 0:
            return pd.DataFrame(columns=[
                "account_id", "neighbor_account_id", "neighbor_rank",
                "sim_overall", "sim_numeric", "sim_categorical", "sim_text", "sim_als",
                "neighbor_account_name", "neighbor_industry", "neighbor_segment", "neighbor_territory",
            ])
        idx_part = np.argpartition(-S, kth=k_eff, axis=1)[:, :k_eff]
        sims_part = np.take_along_axis(S, idx_part, axis=1)
        order = np.argsort(-sims_part, axis=1)
        neighbors_idx = np.take_along_axis(idx_part, order, axis=1)
        neighbors_sim = np.take_along_axis(sims_part, order, axis=1)
    else:
        neighbors_idx, neighbors_sim = _topk_neighbors_blockwise(V, k, block_rows=block_rows)

    acc_ids = accounts_df["account_id"].astype(str).tolist()
    acc_names = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("account_name", accounts_df["account_id"].astype(str))))
    meta_ind = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("industry", "")))
    meta_seg = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("segment", "")))
    meta_ter = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("territory", "")))

    # Prepare output rows; compute per-component similarities only for neighbor pairs
    out_rows = []
    for i, a in enumerate(acc_ids):
        nbrs = neighbors_idx[i].tolist()
        if not nbrs:
            continue
        # Per-component sims via row-by-row dot products (rows are already normalized)
        num_row = X_num[i]
        cat_row = X_cat[i]
        txt_row = X_txt[i]
        als_row = V_als[i]

        num_blk = X_num[nbrs]
        cat_blk = X_cat[nbrs]
        txt_blk = X_txt[nbrs]
        als_blk = V_als[nbrs]

        sims_num = num_blk @ num_row
        sims_cat = cat_blk @ cat_row
        sims_txt = txt_blk @ txt_row
        sims_als = als_blk @ als_row

        for rank, j in enumerate(nbrs, start=1):
            b = acc_ids[j]
            out_rows.append({
                "account_id": a,
                "neighbor_account_id": b,
                "neighbor_rank": rank,
                "sim_overall": float(neighbors_sim[i, rank - 1]),
                "sim_numeric": float(sims_num[rank - 1]),
                "sim_categorical": float(sims_cat[rank - 1]),
                "sim_text": float(sims_txt[rank - 1]),
                "sim_als": float(sims_als[rank - 1]),
                "neighbor_account_name": acc_names.get(b, ""),
                "neighbor_industry": meta_ind.get(b, ""),
                "neighbor_segment": str(meta_seg.get(b, "")),
                "neighbor_territory": meta_ter.get(b, ""),
            })
    return pd.DataFrame(out_rows)
