"""ICVI battery required by the contest: SW, CH, S_Dbw, AVI, AVU, MQ.

Definitions used (documented so a judge can recompute them):
* SW   - average silhouette width (Rousseeuw 1987), higher is better.
* CH   - Calinski-Harabasz variance-ratio criterion (1974), higher is better.
* S_Dbw - Halketti et al. (2002) density-based validity index: DBW + the average between-cluster
  density term, computed with the global sigma rule sigma_k = (1 / (5k)) * sum of distances of all
  points to their own cluster centroid. Lower is better.
* AVI  - Average Within Index: mean intra-cluster scatter (average squared distance of points to
  their own centroid, averaged over clusters). Lower is better.
* AVU  - Average Uniqueness/between-cluster separation: mean distance between cluster centroids.
  Higher is better.
* MQ   - Modularity Quality (Newman-Girvan modularity of the partition on the weighted network).
  Higher is better; a negative value means the partition is worse than random.

Attribute-space criteria are computed on the feature matrix, MQ on the adjacency matrix, so a
network clustering can be judged both structurally and substantively.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def _valid(labels) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    keep = labels >= 0
    return labels, keep


def silhouette_width(X: np.ndarray, labels: np.ndarray) -> float | None:
    labels, keep = _valid(labels)
    labels = labels[keep]
    X = X[keep]
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return None
    return float(silhouette_score(X, labels))


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float | None:
    labels, keep = _valid(labels)
    labels = labels[keep]
    X = X[keep]
    if len(np.unique(labels)) < 2:
        return None
    return float(calinski_harabasz_score(X, labels))


def avi_avu(X: np.ndarray, labels: np.ndarray):
    """AVI: mean within-cluster scatter; AVU: mean distance between centroids."""
    labels, keep = _valid(labels)
    labels = labels[keep]
    X = X[keep]
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return None, None
    within = []
    centroids = []
    for c in clusters:
        pts = X[labels == c]
        if len(pts) == 0:
            continue
        cen = pts.mean(axis=0)
        centroids.append(cen)
        within.append(float(np.mean(((pts - cen) ** 2).sum(axis=1))))
    if not centroids:
        return None, None
    cent_arr = np.asarray(centroids)
    inter = cdist(cent_arr, cent_arr)
    n = len(inter)
    pairs = inter[np.triu_indices(n, k=1)]
    avi = float(np.mean(within)) if within else None
    avu = float(np.mean(pairs)) if pairs.size else None
    return avi, avu


def _dbw_terms(X, labels, sigma):
    clusters = np.unique(labels)
    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    # Davies-Bouldin within-cluster scatter, normalized by sigma
    scatter = []
    for c in clusters:
        pts = X[labels == c]
        scatter.append(float(np.sqrt(np.mean(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))))
    dbw = 0.0
    for i in range(len(clusters)):
        best = 0.0
        for j in range(len(clusters)):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            best = max(best, (scatter[i] + scatter[j]) / max(d, 1e-12))
        dbw += best
    dbw /= len(clusters)
    # between-cluster density term
    density = 0.0
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i >= j:
                continue
            (centroids[i] + centroids[j]) / 2.0
            dij = float(np.linalg.norm(centroids[i] - centroids[j])) / 2.0
            cij = 0.5 * (scatter[i] + scatter[j])
            density += np.exp(-(dij**2) / (4.0 * sigma**2)) + cij / (sigma**2) - 1.0
    density /= len(clusters) * (len(clusters) - 1) / 2.0
    return dbw, density


def s_dbw(X: np.ndarray, labels: np.ndarray) -> float | None:
    labels, keep = _valid(labels)
    labels = labels[keep]
    X = X[keep]
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return None
    total = len(labels)
    k = len(clusters)
    dists = np.array(
        [np.linalg.norm(X[i] - X[labels == labels[i]].mean(axis=0)) for i in range(total)]
    )
    sigma = float(dists.sum() / (5.0 * k)) if k else 1.0
    sigma = max(sigma, 1e-9)
    dbw, density = _dbw_terms(X, labels, sigma)
    return float(dbw + density / k)


def modularity_quality(W: np.ndarray, labels: np.ndarray) -> float | None:
    """Weighted Newman-Girvan modularity of a partition; self-loops are ignored."""
    W = W.todense() if hasattr(W, "todense") else W
    W = np.asarray(W, dtype="float64")
    W = np.where(np.triu(W, 1) > 0, W, W.T)  # symmetrize if a directed matrix was given
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    m2 = deg.sum()
    if m2 <= 0:
        return None
    labels = np.asarray(labels)
    q = 0.0
    for c in np.unique(labels):
        mask = labels == c
        internal = W[np.ix_(mask, mask)].sum()
        outside = deg[mask].sum()
        q += internal / m2 - (outside / m2) ** 2
    return float(q)


BATTERY = ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]
HIGHER_IS_BETTER = {"SW": True, "CH": True, "S_Dbw": False, "AVI": False, "AVU": True, "MQ": True}


def score(X: np.ndarray, labels: np.ndarray, W: np.ndarray | None = None) -> dict:
    avi, avu = avi_avu(X, labels)
    out = {
        "SW": silhouette_width(X, labels),
        "CH": calinski_harabasz(X, labels),
        "S_Dbw": s_dbw(X, labels),
        "AVI": avi,
        "AVU": avu,
        "k": len(np.unique([lab for lab in labels if lab >= 0])),
    }
    out["MQ"] = modularity_quality(W, labels) if W is not None else None
    return out
