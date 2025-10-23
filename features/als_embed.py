from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def als_account_vectors(tx: pd.DataFrame, factors: int = 64, alpha: float = 40.0, use_bm25: bool = True) -> pd.DataFrame:
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

    # Map ids to contiguous indices
    acc_ids = {a: i for i, a in enumerate(tx["account_id"].astype(str).unique())}
    prod_ids = {p: i for i, p in enumerate(tx["product_id"].astype(str).unique())}

    rows = tx["account_id"].astype(str).map(acc_ids).values
    cols = tx["product_id"].astype(str).map(prod_ids).values
    data = pd.to_numeric(tx["net_revenue"], errors="coerce").fillna(0.0).astype("float32").values
    X = coo_matrix((data, (rows, cols)), shape=(len(acc_ids), len(prod_ids))).tocsr()

    Xw = bm25_weight(X).tocsr() if use_bm25 else tfidf_weight(X).tocsr()
    Xc = Xw * float(alpha)

    model = implicit.als.AlternatingLeastSquares(factors=int(factors), regularization=0.01, iterations=20)
    model.fit(Xc)

    U = model.user_factors.astype("float32")
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    inv_acc = [None] * len(acc_ids)
    for a, i in acc_ids.items():
        inv_acc[i] = a

    return pd.DataFrame({"account_id": inv_acc, "als_vec": list(U)})

