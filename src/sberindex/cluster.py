"""Clustering methods + a normalized ICVI comparison table.

Everything runs on scikit-learn / scipy only (no igraph, no networkx), so the entry has no exotic
dependency. Modularity maximization is implemented here (greedy Clauset-Newman-Moore merging) rather
than borrowed from a community-detection library.

Normalization of the ICVI battery: each criterion is min-max scaled across the compared partitions,
oriented so that higher always means better (S_Dbw and AVI are inverted), then averaged. The
per-criterion numbers are always reported next to the average, because an average hides conflicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

from . import icvi

LEVELS = ["total_log_mean"] + [
    f"{c}_log_mean"
    for c in ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
]
SHARES = [
    f"{c}_share"
    for c in ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
]
DYNAMICS = ["total_cv", "h2_over_h1", "dec24_over_mean", "total_slope_rub_per_month"]


# Column prefixes actually produced by the covariate jobs: worldcover uses wc_, GLC-LC100 uses gls_,
# MODIS land-cover uses mcd_, MODIS NDVI uses modis_ndvi_, VIIRS uses viirs_, elevation uses
# elevation_, WorldClim uses worldclim_ and the burn layer is prefixed burn_ (no job-name prefix).
LAND_PREFIXES = (
    "wc_",
    "gls_",
    "mcd_",
    "modis_",
    "dynworld_",
    "viirs_",
    "elevation_",
    "worldclim_",
    "soilgrids_",
    "soil_",
    "burn_",
    "hansen_",
    "tree_",
    "loss_",
    "spending_per_km2",
)


def land_columns(feats: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Covariate columns present in the feature table: land cover, light, vegetation, relief, climate,
    soil and the official-area spending density. Provenance strings (prov_*) are never features.

    A column needs at least `min_coverage` non-null rows to be used at all. Without this gate a
    partially completed covariate run would enter the model as ~95% imputed values, which would look
    like a land effect while being an artefact of the missing data.
    """
    cand = [
        c
        for c in feats.columns
        if c.startswith(LAND_PREFIXES)
        and not c.startswith("prov_")
        and not c.endswith("_unclassified_pct")
    ]
    return [c for c in cand if feats[c].notna().mean() >= min_coverage]


def prepare(feats: pd.DataFrame, groups: str = "levels+shares", model_panel_only: bool = True):
    # `density` is its own block rather than a relabelling of `levels`: it is spending per official
    # square kilometre, so a territory's size enters the attribute space explicitly instead of being
    # absorbed into a logged total. Keeping it as a named group is also what makes the density
    # comparison reproducible - before this, method_comparison_density.csv existed with no command in
    # the repository that produced it, and its provenance record claimed the land run's inputs.
    cols = {
        "levels": LEVELS,
        "shares": SHARES,
        "dynamics": DYNAMICS,
        "land": land_columns(feats),
        "density": [c for c in feats.columns if c.endswith("_per_official_km2")],
    }
    keep = []
    for g in groups.split("+"):
        keep += [c for c in cols[g] if c in feats.columns]
    sub = (
        feats[feats.tid.isin(feats.loc[feats.in_model_panel, "tid"])] if model_panel_only else feats
    )
    X = sub[keep].to_numpy(dtype="float64")
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X)))
    Z = qt.fit_transform(X)
    return sub.tid.to_numpy(), Z, keep


def prepare_columns(feats: pd.DataFrame, columns: list[str], model_panel_only: bool = True):
    """Build the matrix from an explicitly named column list rather than a group shorthand.

    A group string is not safe when the decision is about which indicators belong in the space: the block
    called `land` contains a spending measure, so asking for "land" while claiming to have dropped the
    spending amount keeps the amount inside the input. And filtering a hand-named list down to whatever
    happens to exist turns "eleven features" into ten while the description stays eleven. Both are the
    silent-weighting defect this repository was built to catch, so a named list either resolves entirely
    or refuses with the missing names, and nothing shrinks quietly.
    """
    missing = [c for c in columns if c not in feats.columns]
    if missing:
        raise SystemExit(
            f"named columns absent from the feature table: {missing} - refusing to run an attribute "
            "space that the panel cannot honour"
        )
    sub = (
        feats[feats.tid.isin(feats.loc[feats.in_model_panel, "tid"])] if model_panel_only else feats
    )
    X = sub[list(columns)].to_numpy(dtype="float64")
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    qt = QuantileTransformer(
        output_distribution="normal", n_quantiles=min(1000, len(X))
    )
    Z = qt.fit_transform(X)
    return sub.tid.to_numpy(), Z, list(columns)


def select_partition(asg: pd.DataFrame, shipped_method: str | None) -> pd.DataFrame:
    """Take the one partition every downstream artifact is meant to read.

    Filtering by whichever row happens to come first in the assignments file makes the shipped answer depend on
    the order methods were written in, and pooling all rows - which the fingerprint step did once - averages
    districts across different partitions and still calls the result a cluster profile. Naming the row makes the
    dependency visible, and a missing row is reported rather than quietly replaced by another one.
    """
    if not shipped_method:
        raise SystemExit(
            "no shipped partition named in configs/clustering.yaml under shipped_method - downstream "
            "artifacts cannot know which cluster number to describe"
        )
    if "method" not in asg.columns:
        raise SystemExit("the assignments artifact carries no method column, so the shipped partition cannot be selected")
    sub = asg[asg["method"] == shipped_method]
    if sub.empty:
        raise SystemExit(
            f"the shipped partition '{shipped_method}' is absent from the assignments; run "
            "scripts/run_clustering.py with the k and the space the written decision names"
        )
    return sub


def _kmeans(Z, k, seed=0):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def _gm(Z, k, seed=0):
    return GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed, reg_covar=1e-4
    ).fit_predict(Z)


def _ward(Z, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def _average(Z, k):
    return AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(Z)


def _complete(Z, k):
    return AgglomerativeClustering(n_clusters=k, linkage="complete").fit_predict(Z)


def _spectral(Z, k, W=None, seed: int = 0):
    aff = "nearest_neighbors" if W is None else "precomputed"
    kw = {
        "n_clusters": int(k),
        "affinity": aff,
        "assign_labels": "kmeans",
        "random_state": int(seed),
    }
    if W is None:
        kw["n_neighbors"] = 10  # sklearn >=1.9 rejects n_neighbors=None with a precomputed matrix
    model = SpectralClustering(**kw)
    if W is not None:
        return model.fit_predict(np.asarray(W.todense()))
    return model.fit_predict(Z)


def _dbscan(Z, eps=None, min_samples=5):
    if eps is None:
        from sklearn.metrics.pairwise import euclidean_distances

        D = euclidean_distances(Z)
        eps = float(np.median(np.sort(D, axis=1)[:, min_samples]))
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)


METHODS = {
    "kmeans": _kmeans,
    "gaussian_mixture": _gm,
    "ward": _ward,
    "average_linkage": _average,
    "complete_linkage": _complete,
    "spectral": _spectral,
    "dbscan": _dbscan,
}


def label_propagation(W: sp.csr_matrix, seed: int = 0, max_iter: int = 40) -> np.ndarray:
    """Weighted label propagation (Raghavan et al. 2007): a node takes the community of its
    heaviest neighbour. Deterministic given the seed; runs until no label changes."""
    W = W.tocsr()
    W = W.maximum(W.T.tocsr())
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n, size=n)
    order = rng.permutation(n)
    indptr, indices, data = W.indptr, W.indices, W.data
    for _ in range(max_iter):
        changed = 0
        for i in order:
            s, e = indptr[i], indptr[i + 1]
            if e == s:
                continue
            nb = indices[s:e]
            w = data[s:e]
            lab = labels[nb]
            best = np.bincount(lab, weights=w)
            cands = np.unique(lab)
            score = np.array([best[c] for c in cands])
            new = int(cands[np.argmax(score)])
            if new != labels[i]:
                labels[i] = new
                changed += 1
        if changed == 0:
            break
    # relabel to 0..k-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def compare(
    X: np.ndarray, results: dict[str, np.ndarray], W: sp.csr_matrix | None = None
) -> pd.DataFrame:
    """ICVI battery per partition plus a normalized average, oriented so higher is better."""
    rows = []
    for name, labels in results.items():
        s = icvi.score(X, labels, W)
        s["method"] = name
        rows.append(s)
    df = pd.DataFrame(rows)
    crit = [c for c in icvi.BATTERY if c in df.columns]
    ok = df[crit].notna().all(axis=1)
    for c in crit:
        # Normalize over the rows that carry a full battery, but write the result back aligned to
        # every row: np.where(ok, norm, np.nan) used to receive a vector as long as the passing rows
        # and a mask as long as the whole frame, which broadcast into
        # "operands could not be broadcast together with shapes (45,) (39,)" the moment DBSCAN
        # variants joined the sweep. Rows without a full battery stay NaN rather than being scored.
        col = df.loc[ok, c].astype("float64")
        lo, hi = col.min(), col.max()
        rng = (hi - lo) or 1.0
        scaled = (col - lo) / rng
        if not icvi.HIGHER_IS_BETTER[c]:
            scaled = 1.0 - scaled
        aligned = pd.Series(np.nan, index=df.index, dtype="float64")
        aligned.loc[ok] = scaled.to_numpy()
        df[f"n_{c}"] = aligned.to_numpy()
    df["ICVI_avg"] = df[[f"n_{c}" for c in crit]].mean(axis=1)
    cols = ["method", "k", "ICVI_avg"] + crit + [f"n_{c}" for c in crit]
    return df[cols].sort_values("ICVI_avg", ascending=False).reset_index(drop=True)
