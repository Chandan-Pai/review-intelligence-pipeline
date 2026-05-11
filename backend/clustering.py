"""Embed texts and run UMAP + HDBSCAN."""

from __future__ import annotations

import numpy as np

try:
    import hdbscan
    import umap
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError("Install backend requirements (sentence-transformers, umap, hdbscan).") from e


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, show_progress_bar=len(texts) > 32))


def cluster_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (2D UMAP coordinates, cluster labels; -1 = noise).
    """
    n = len(embeddings)
    if n < 3:
        return np.zeros((n, 2)), np.full(n, -1)

    nn = min(max(2, n_neighbors), n - 1)
    reducer = umap.UMAP(
        n_neighbors=nn,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    coords = np.asarray(reducer.fit_transform(embeddings))

    min_cs = min(max(2, min_cluster_size), n)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, metric="euclidean")
    labels = np.asarray(clusterer.fit_predict(coords))
    return coords, labels


def embed_and_cluster(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb = embed_texts(texts)
    coords, labels = cluster_embeddings(emb)
    return emb, coords, labels
