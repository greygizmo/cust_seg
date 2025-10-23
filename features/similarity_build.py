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

    # Pairwise cosine similarities (as rows are L2-normalized, dot = cosine)
    S = _cosine_sim(V, V)
    np.fill_diagonal(S, -1.0)
    k = int(cfg.get("k_neighbors", 25))
    k_eff = min(k, S.shape[1] - 1) if S.shape[1] > 1 else 0
    if k_eff <= 0:
        return pd.DataFrame(columns=[
            "account_id", "neighbor_account_id", "neighbor_rank",
            "sim_overall", "sim_numeric", "sim_categorical", "sim_text", "sim_als",
            "neighbor_account_name", "neighbor_industry", "neighbor_segment", "neighbor_territory",
        ])

    idx = np.argpartition(-S, kth=k_eff, axis=1)[:, :k_eff]
    rows, sims = [], []
    for i, neighbors in enumerate(idx):
        order = neighbors[np.argsort(-S[i, neighbors])]
        sim = S[i, order]
        rows.append(order)
        sims.append(sim)
    neighbors_idx = np.vstack(rows)
    neighbors_sim = np.vstack(sims)

    acc_ids = accounts_df["account_id"].astype(str).tolist()
    acc_names = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("account_name", accounts_df["account_id"].astype(str))))
    meta_ind = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("industry", "")))
    meta_seg = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("segment", "")))
    meta_ter = dict(zip(accounts_df["account_id"].astype(str), accounts_df.get("territory", "")))

    # Component-wise sims
    def block_sim(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Xn = X / n
        return Xn @ Xn.T

    sim_num = block_sim(X_num)
    sim_cat = block_sim(X_cat)
    sim_txt = block_sim(X_txt)
    sim_als = block_sim(V_als)

    out_rows = []
    for i, a in enumerate(acc_ids):
        for rank, j in enumerate(neighbors_idx[i].tolist(), start=1):
            b = acc_ids[j]
            out_rows.append({
                "account_id": a,
                "neighbor_account_id": b,
                "neighbor_rank": rank,
                "sim_overall": float(neighbors_sim[i, rank - 1]),
                "sim_numeric": float(sim_num[i, j]),
                "sim_categorical": float(sim_cat[i, j]),
                "sim_text": float(sim_txt[i, j]),
                "sim_als": float(sim_als[i, j]),
                "neighbor_account_name": acc_names.get(b, ""),
                "neighbor_industry": meta_ind.get(b, ""),
                "neighbor_segment": str(meta_seg.get(b, "")),
                "neighbor_territory": meta_ter.get(b, ""),
            })
    return pd.DataFrame(out_rows)

