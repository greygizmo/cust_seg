from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def als_account_vectors(
    tx: pd.DataFrame,
    factors: int = 64,
    alpha: float = 40.0,
    reg: float = 0.01,
    iterations: int = 20,
    use_bm25: bool = True,
) -> pd.DataFrame:
    """
    Train an implicit ALS model on an account x product matrix derived from
    transactions and return L2-normalized account vectors.

    tx must contain at least: account_id, product_id, net_revenue
    Requires the 'implicit' package if enabled.
    """
    try:
        import implicit  # type: ignore
        from implicit.nearest_neighbours import bm25_weight, tfidf_weight  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'implicit' to enable ALS embeddings (pip install implicit).") from e

    # Limit BLAS threadpools to avoid oversubscription
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(1, "blas")
    except Exception:
        pass

    # Map ids to contiguous indices
    acc_ids = {a: i for i, a in enumerate(tx["account_id"].astype(str).unique())}
    prod_ids = {p: i for i, p in enumerate(tx["product_id"].astype(str).unique())}

    rows = tx["account_id"].astype(str).map(acc_ids).values
    cols = tx["product_id"].astype(str).map(prod_ids).values
    data = pd.to_numeric(tx["net_revenue"], errors="coerce").fillna(0.0).astype("float32").values
    X = coo_matrix((data, (rows, cols)), shape=(len(acc_ids), len(prod_ids))).tocsr()

    def _fit_with(weight_fn):
        Xw_local = weight_fn(X).tocsr()
        Xc_local = Xw_local * float(alpha)
        model_local = implicit.als.AlternatingLeastSquares(
            factors=int(factors), regularization=float(reg), iterations=int(iterations)
        )
        model_local.fit(Xc_local)
        U_local = model_local.user_factors.astype("float32")
        # Sanitize any non-finite values
        U_local[~np.isfinite(U_local)] = 0.0
        return U_local

    try:
        Xw = bm25_weight if use_bm25 else tfidf_weight
        U = _fit_with(Xw)
    except Exception:
        # Fallback to TF-IDF weighting if BM25 path fails
        U = _fit_with(tfidf_weight)
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    inv_acc = [None] * len(acc_ids)
    for a, i in acc_ids.items():
        inv_acc[i] = a

    return pd.DataFrame({"account_id": inv_acc, "als_vec": list(U)})


def als_concat_account_vectors(
    components: list[tuple[str, pd.DataFrame]],
    accounts: list[str] | None = None,
    factors: dict | None = None,
    alpha: float = 40.0,
    reg: float = 0.01,
    iterations: int = 20,
    use_bm25: bool = True,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Train separate ALS models for multiple components (e.g., rollup, goal) and
    concatenate user factors into a single vector per account.

    components: list of (label, df) where df has columns: account_id, item_id, value
    accounts: optional master list of account_ids to include/order
    factors: per-component factors, e.g., {"rollup": 64, "goal": 32} (default 64 each)
    component_weights: per-component scalar to scale vectors before concatenation
    """
    try:
        import implicit  # type: ignore
        from implicit.nearest_neighbours import bm25_weight, tfidf_weight  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'implicit' for ALS embeddings (pip install implicit).") from e

    # Limit BLAS threadpools to avoid oversubscription
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(1, "blas")
    except Exception:
        pass

    labels = [lbl for lbl, _ in components]
    if not components:
        return pd.DataFrame(columns=["account_id", "als_vec"])  # empty

    # Build master account list
    if accounts is None:
        acc_set = set()
        for _, df in components:
            if not df.empty:
                acc_set.update(df["account_id"].astype(str).unique().tolist())
        accounts = sorted(acc_set)
    acc_index = {a: i for i, a in enumerate(accounts)}

    # Train per-component ALS and place factors into aligned matrix per component
    vec_blocks = []
    for lbl, df in components:
        if df is None or df.empty:
            continue
        f = int((factors or {}).get(lbl, 64))
        w = float((component_weights or {}).get(lbl, 1.0))
        # maps
        acc_ids = {a: i for i, a in enumerate(df["account_id"].astype(str).unique())}
        item_ids = {p: i for i, p in enumerate(df["item_id"].astype(str).unique())}
        rows = df["account_id"].astype(str).map(acc_ids).values
        cols = df["item_id"].astype(str).map(item_ids).values
        data = pd.to_numeric(df["value"], errors="coerce").fillna(0.0).astype("float32").values
        X = coo_matrix((data, (rows, cols)), shape=(len(acc_ids), len(item_ids))).tocsr()
        def _fit_with(weight_fn):
            Xw_local = weight_fn(X).tocsr()
            Xc_local = Xw_local * float(alpha)
            model_local = implicit.als.AlternatingLeastSquares(
                factors=f, regularization=float(reg), iterations=int(iterations)
            )
            model_local.fit(Xc_local)
            U_l = model_local.user_factors.astype("float32")
            U_l[~np.isfinite(U_l)] = 0.0
            return U_l

        try:
            try:
                weight_fn = bm25_weight if use_bm25 else tfidf_weight
                U_local = _fit_with(weight_fn)
            except Exception:
                U_local = _fit_with(tfidf_weight)
        except Exception:
            # Skip this component if both weightings fail
            continue
        U_local /= (np.linalg.norm(U_local, axis=1, keepdims=True) + 1e-12)
        # Align to master account list
        block = np.zeros((len(accounts), f), dtype="float32")
        for a, i_local in acc_ids.items():
            i_global = acc_index.get(a)
            if i_global is not None:
                block[i_global, :] = U_local[i_local, :] * w
        vec_blocks.append(block)

    if not vec_blocks:
        return pd.DataFrame({"account_id": accounts, "als_vec": [np.zeros((1,), dtype="float32")] * len(accounts)})

    V = np.concatenate(vec_blocks, axis=1)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return pd.DataFrame({"account_id": accounts, "als_vec": list(V)})
