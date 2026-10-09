"""Directed complementarity edges: w_ij = f(d_ij) * supply_i * demand_j.

The novelty claim lives here. Every competitor builds similarity edges (territories that look
alike are linked). We link a producer i to a consumer j when i *can* supply what j needs, damped
by road distance. d_ij comes from Sber's own OSM road-distance matrix, so no external routing
engine is involved.

Sparsification is a modelling decision with a structural effect, so both are reported: complete
graph edge count, kept edge count, density, mean degree, weak components, largest component share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HIGHWAY = "highway"
RAILWAY = "railway"


def load_distances(path=None):
    """Returns {type: symmetric csr matrix} of km distances (missing pairs stay absent)."""
    from .config import CONNECTION as _C

    c = pd.read_parquet(path or _C)
    out = {}
    for kind, sub in c.groupby("type"):
        n = int(max(sub.territory_id_x.max(), sub.territory_id_y.max())) + 1
        rows = sub.territory_id_x.to_numpy() - 1
        cols = sub.territory_id_y.to_numpy() - 1
        vals = sub.distance.to_numpy(dtype="float64")
        m = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
        m = m.maximum(m.T.tocsr())  # highway rows are one-per-pair, railway both ways
        m.setdiag(0.0)
        m.eliminate_zeros()
        out[kind] = m.tocsr()
    return out


def kernel(d: np.ndarray, kind: str = "exp", tau: float = 600.0, alpha: float = 1.0) -> np.ndarray:
    if kind == "exp":
        return np.exp(-d / tau)
    if kind == "power":
        return 1.0 / np.power(1.0 + d, alpha)
    if kind == "trunc":
        return np.clip(1.0 - d / tau, 0.0, None)
    raise ValueError(f"unknown kernel {kind}")


def complementarity(
    dist: sp.csr_matrix,
    supply: np.ndarray,
    demand: np.ndarray,
    kernel_kind: str = "exp",
    tau: float = 600.0,
    alpha: float = 1.0,
    max_distance: float | None = None,
) -> sp.csr_matrix:
    """w_ij over stored (finite) distances only. Zero-distance/self pairs are dropped."""
    dist = dist.tocoo()
    d = dist.data
    keep = d > 0
    if max_distance is not None:
        keep = keep & (d <= max_distance)
    f = kernel(d[keep], kernel_kind, tau, alpha)
    w = f * np.take(supply, dist.row[keep]) * np.take(demand, dist.col[keep])
    m = sp.coo_matrix((w, (dist.row[keep], dist.col[keep])), shape=dist.shape)
    return sp.csr_matrix((m.data, (m.row, m.col)), shape=dist.shape)


def topk(W: sp.csr_matrix, k: int, symmetric: bool = False) -> sp.csr_matrix:
    """Keep the k largest outgoing weights per row (ties: by index order)."""
    W = W.tocsr()
    n = W.shape[0]
    rows, cols, vals = [], [], []
    indptr, indices, data = W.indptr, W.indices, W.data
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if e - s <= k:
            rows.extend([i] * (e - s))
            cols.extend(indices[s:e])
            vals.extend(data[s:e])
            continue
        w = data[s:e]
        sel = np.argpartition(-w, k)[:k]
        rows.extend([i] * len(sel))
        cols.extend(indices[s:e][sel])
        vals.extend(w[sel])
    out = sp.coo_matrix((vals, (rows, cols)), shape=W.shape).tocsr()
    if symmetric:
        out = out.maximum(out.T.tocsr())
    out.eliminate_zeros()
    return out


def reciprocity(W: sp.csr_matrix) -> float:
    """Share of directed edges that are answered in the opposite direction.

    This is the sharpest discriminator between the two families: a complementarity network is
    one-directional by construction (a producer supplies a consumer, not the reverse), while a
    similarity graph is symmetric because it is built from a symmetric feature comparison.
    """
    W = W.tocsr()
    mutual = W.multiply(W.T.tocsr())
    return float(mutual.nnz) / max(1, int(W.nnz))


def clustering_coefficient(W: sp.csr_matrix) -> float:
    """Global clustering coefficient: closed triangles over connected triples (no networkx)."""
    A = W.maximum(W.T.tocsr()).maximum(W.T.tocsr())
    A = (A > 0).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    A2 = (A @ A).tocsr()
    closed = float(A2.multiply(A).sum()) / 6.0  # each triangle is counted 6 times
    triples = float((deg * (deg - 1) / 2.0).sum())  # paths of length 2 centred on each node
    return closed / triples if triples > 0 else 0.0


def degree_gini(deg: np.ndarray) -> float:
    """Gini of the degree distribution: how much of the network's adjacency is held by a few nodes.
    Hub-catchment structure shows up here long before it shows up in the mean degree."""
    d = np.sort(np.asarray(deg, dtype="float64"))
    n = len(d)
    if n < 2 or d.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * d).sum()) / (n * d.sum()) - (n + 1.0) / n)


def label_propagation(W: sp.csr_matrix, seed: int = 0, sweeps: int = 30) -> np.ndarray:
    """Asynchronous label propagation communities (pure numpy/scipy, fixed seed)."""
    rng = np.random.default_rng(seed)
    W = W.maximum(W.T.tocsr())
    A = (W > 0).tocsr()
    n = A.shape[0]
    labels = np.arange(n)
    order = rng.permutation(n)
    for _ in range(sweeps):
        changed = 0
        for i in order:
            sl = A.indices[A.indptr[i] : A.indptr[i + 1]]
            if len(sl) == 0:
                continue
            neigh = labels[sl]
            vals, counts = np.unique(neigh, return_counts=True)
            best = counts.max()
            cand = vals[counts == best]
            new = labels[i] if labels[i] in cand else int(rng.choice(cand))
            if new != labels[i]:
                labels[i] = new
                changed += 1
        if changed == 0:
            break
    return labels


def modularity(W: sp.csr_matrix, labels: np.ndarray, resolution: float = 1.0) -> float:
    """Newman-Girvan modularity of a partition on an unweighted undirected graph."""
    A = W.maximum(W.T.tocsr())
    A = (A > 0).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    m = float(A.nnz) / 2.0
    if m == 0:
        return 0.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    lab = np.asarray(labels)
    # in-community edges, computed blockwise without materialising a dense matrix
    blocks = {}
    for b in np.unique(lab):
        idx = np.flatnonzero(lab == b)
        sub = A[idx][:, idx]
        blocks[b] = (float(sub.nnz) / 2.0, float(deg[idx].sum()))
    q = 0.0
    for in_edges, vol in blocks.values():
        q += in_edges / m - resolution * (vol / (2.0 * m)) ** 2
    return float(q)


def clique_hub(W: sp.csr_matrix, seed: int = 0) -> dict:
    """The hub-catchment versus similarity-clique evidence, computed per edge family."""
    A = W.maximum(W.T.tocsr()).maximum(W.T.tocsr())
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    labels = label_propagation(W, seed=seed)
    n = W.shape[0]
    top = np.sort(deg)[::-1][: max(1, round(0.01 * n))]
    return {
        "reciprocity": round(reciprocity(W.tocsr()), 4),
        "clustering_coefficient": round(clustering_coefficient(W), 4),
        "degree_gini": round(degree_gini(deg), 4),
        "top1pct_degree_share": round(float(top.sum()) / max(1.0, float(deg.sum())), 4),
        "lp_communities": len(np.unique(labels)),
        "lp_modularity": round(modularity(W, labels), 4),
    }


def structure(W: sp.csr_matrix) -> dict:
    W = W.tocsr()
    n = W.shape[0]
    A = W.maximum(W.T.tocsr())
    A = A.maximum(A.T.tocsr())
    ncomp, labels = connected_components(A, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    biggest = int(sizes.max()) if len(sizes) else 0
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    n_edges = int(A.nnz // 2)
    return {
        "nodes": n,
        "edges": n_edges,
        "density": round(2.0 * n_edges / max(n * (n - 1), 1), 6),
        "mean_degree": round(float(deg.mean()), 3),
        "median_degree": int(np.median(deg)),
        "max_degree": int(deg.max()),
        "isolated": int((deg == 0).sum()),
        "weak_components": int(ncomp),
        "largest_component_share": round(biggest / n, 4),
        "out_nnz": int(W.nnz),
        "total_weight": round(float(W.data.sum()), 3),
    }
