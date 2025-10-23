from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def embed_text(sentences: pd.Series, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Return a 2D float32 array of sentence embeddings for the provided Series.

    Tries Sentence-Transformers first; falls back to TF-IDF + TruncatedSVD if
    that model/package is unavailable. Always L2-normalizes the output rows.
    """
    s = sentences.fillna("").astype(str).tolist()
    # Primary: Sentence-Transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        vecs = model.encode(s, normalize_embeddings=False, convert_to_numpy=True)
        return _safe_normalize(vecs.astype("float32"))
    except Exception:
        pass

    # Fallback: TF-IDF + SVD
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore

        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
        X = tfidf.fit_transform(s)
        n_comp = max(2, min(256, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp)
        Z = svd.fit_transform(X)
        return _safe_normalize(Z.astype("float32"))
    except Exception:
        # Last resort: zeros
        return np.zeros((len(s), 1), dtype="float32")

