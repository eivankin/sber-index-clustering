#!/usr/bin/env python
"""Build edge families and report the structural effect of sparsification.

Two families, so the novelty claim is testable rather than asserted:
  * complementarity (ours):  w_ij = f(d_ij) * supply_i * demand_j, directed
  * similarity (competitors): w_ij = exp(-||z_i - z_j|| / h) on the spending attribute space, undirected

While GEE covariates are unavailable the supply side is neutral (=1) and demand is Sber's own
market-access index; this is recorded as version V0 in the diagnostics and in configs/edge_rules.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from sberindex import edges
from sberindex.config import (
    ARTIFACTS,
    MARKET_ACCESS,
    PROCESSED,
    ensure_dirs,
)

KERNELS = {"exp": {"tau": 600.0}, "power": {"alpha": 1.0}, "trunc": {"tau": 1500.0}}
KNN = [10, 25]
# Damping for the two time-based families below. 600 km is the same scale the exponential
# complementarity kernel uses, so the families differ in the signal they score, not in how far
# they let the road distance reach.
TAU_LAGGED = 600.0


def active_index(master: pd.DataFrame):
    ids = master.tid.astype("int64").to_numpy()
    return ids


def edge_frame(W: sp.csr_matrix, ids: np.ndarray, kind: str) -> pd.DataFrame:
    W = W.tocoo()
    return pd.DataFrame(
        {"source_id": ids[W.row], "target_id": ids[W.col], "weight": W.data, "family": kind}
    )


def load_supply(n: int, ids: np.ndarray, pos: pd.Series):
    """supply_i, version v1 of the rule in configs/edge_rules.yaml.

    Land-use production potential = cropland + 0.5 x grazing land + 0.25 x forest, as shares of
    *classified* pixels, divided by the classified coverage of the territory (a municipality whose
    land cover is largely unclassified is not credited with production it cannot show), then rescaled
    so its median is 1.0 - the global scale of the edge rule is preserved and only its ordering is
    changed. Territories without covariates keep supply = 1 (version v0) and are counted in the report.
    """
    cov, source = None, None
    for name in ("covariates_gee.parquet", "covariates_fallback.parquet"):
        p = PROCESSED / name
        if p.exists():
            cand = pd.read_parquet(p)
            if cov is None or len(cand) > len(cov):
                cov, source = cand, name
    if cov is None:
        return np.ones(n), "v0 neutral (no covariate table present)"

    cols = list(cov.columns)

    def pick(*cands):
        return next((c for c in cols if c in cands), None)

    crops = pick("gls_fraction_crops", "wc_c40_pct")
    grass = pick("gls_fraction_grass", "wc_c20_pct")
    tree = pick("gls_fraction_tree", "wc_c10_pct")
    unclassified = pick("gls_unclassified_pct", "wc_unclassified_pct", "mcd_unclassified_pct")
    used = [c for c in (crops, grass, tree, unclassified) if c]
    if crops is None and grass is None and tree is None:
        # A table with only unclassified-share diagnostics would give every territory the same raw
        # production potential of zero, and a median rescale of zero is not a supply signal at all. It is
        # better to report the neutral version than to multiply the edge by a fabricated constant.
        return np.ones(n), f"v0 neutral ({source} has no cropland/grazing/forest share columns)"
    if not used:
        return np.ones(n), "v0 neutral (covariate table has no land-cover share columns)"

    def col(name, default=0.0):
        if name is None:
            return np.full(len(cov), default)
        return pd.to_numeric(cov[name], errors="coerce").fillna(default).to_numpy()

    raw = col(crops) + 0.5 * col(grass) + 0.25 * col(tree)
    if unclassified:
        raw = raw * (1.0 - np.clip(col(unclassified) / 100.0, 0.0, 0.9))
    med = float(np.median(raw[raw > 0])) if (raw > 0).any() else 1.0
    supply_cov = np.clip(raw / (med if med > 0 else 1.0), 0.1, 3.0)

    tid = pd.to_numeric(cov["tid"], errors="coerce").astype("Int64")
    supply = np.ones(n)
    mapped = tid.map(pos)
    ok = mapped.notna() & np.isfinite(supply_cov)
    supply[mapped[ok].to_numpy(dtype="int64")] = supply_cov[ok.to_numpy()]
    return supply, f"v1 land-cover production potential from {source} ({', '.join(used)})"


def main() -> int:
    ensure_dirs()
    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    ids = active_index(master)
    dist = edges.load_distances()
    highway = dist[edges.HIGHWAY]
    n_full = highway.shape[0]
    keep = np.isin(np.arange(n_full), ids - 1)
    sub = highway.tocsr()[np.ix_(keep, keep)]
    print(f"distance matrix full n={n_full} active n={sub.shape[0]} stored pairs={sub.nnz}")

    ma = pd.read_parquet(MARKET_ACCESS)
    demand = np.zeros(sub.shape[0])
    pos = pd.Series(np.arange(len(ids)), index=ids)
    j = ma.territory_id.astype("int64").map(pos)
    filled = j.notna() & ma.market_access.notna()
    demand[j[filled].to_numpy(dtype="int64")] = (ma.market_access[filled] / 1000.0).to_numpy()
    supply, supply_source = load_supply(sub.shape[0], ids, pos)
    print(f"demand filled for {int((demand > 0).sum())} of {sub.shape[0]} active territories")
    print(
        f"supply: {supply_source} | supply != 1 for {int((np.abs(supply - 1.0) > 1e-9).sum())} territories"
    )

    report: dict = {
        "active_nodes": int(sub.shape[0]),
        "stored_distance_pairs": int(sub.nnz),
        "complete_graph_pairs": int(sub.shape[0] * (sub.shape[0] - 1)),
        "demand_source": "market_access / 1000 (SberIndex)",
        "supply_source": supply_source,
        "supply_median": float(np.median(supply)),
        "supply_p90": float(np.quantile(supply, 0.9)),
    }

    # --- complementarity family, three damping kernels ---
    for name, kw in KERNELS.items():
        W = edges.complementarity(sub, supply, demand, kernel_kind=name, **kw)
        W10 = edges.topk(W, KNN[0], symmetric=False)
        report[f"complementarity_{name}"] = {
            "params": kw,
            "dense": edges.structure(W),
            "clique_hub10": edges.clique_hub(W10),
            **{f"knn{ k}": edges.structure(edges.topk(W, k)) for k in KNN},
        }
        edge_frame(W10, ids, f"complementarity_{name}").to_parquet(
            PROCESSED / f"edges_complementarity_{name}.parquet", index=False
        )

    # --- similarity family on spending features (the competitor baseline) ---
    feats = pd.read_parquet(PROCESSED / "spending_features.parquet")
    feats = feats[feats.tid.isin(ids)].sort_values("tid")
    cols = [c for c in feats.columns if c.endswith(("_log_mean", "_share"))]
    X = feats[cols].to_numpy(dtype="float64")
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    X = np.nan_to_num(X, nan=0.0)
    D = cdist(X, X)
    h = float(np.median(D))
    S = np.exp(-D / h)
    np.fill_diagonal(S, 0.0)
    S = sp.csr_matrix(S)
    S10 = edges.topk(S, KNN[0], symmetric=True)
    report["similarity"] = {
        "params": {
            "h_median_distance": round(h, 4),
            "n_features": len(cols),
            "territories": len(feats),
        },
        "dense": edges.structure(S),
        "clique_hub10": edges.clique_hub(S10),
        **{f"knn{k}": edges.structure(edges.topk(S, k, symmetric=True)) for k in KNN},
    }
    edge_frame(S10, ids, "similarity").to_parquet(
        PROCESSED / "edges_similarity.parquet", index=False
    )

    # Two families the scan did not find in anyone else's code: a correlation of *growth paths* and a
    # *lagged* relation, where the composition a territory had in 2023 is matched against the
    # composition another territory reached in 2024. Both read the monthly series, so unlike the
    # similarity baseline they are not restricted to a single static attribute vector. Negative
    # correlations are clipped to zero: two territories moving in opposite directions are not
    # exchanging anything, and letting a negative weight into a graph would silently invert it.
    long = pd.read_parquet(PROCESSED / "spending_long.parquet")
    long["territory_id"] = long["territory_id"].astype("int64")
    long = long[long["territory_id"].isin(set(ids.tolist()))]
    piv_total = long.pivot_table(
        index="territory_id", columns="month", values="value", aggfunc="sum"
    )
    months = sorted(piv_total.columns)
    complete = piv_total.notna().sum(axis=1) == len(months)
    panel_ids = np.array(sorted(piv_total.index[complete]), dtype="int64")
    # A map from a row of `sub` (which is ordered by `ids`) to a row of the panel matrices.
    panel_pos = np.full(len(ids), -1, dtype="int64")
    panel_pos[np.searchsorted(ids, panel_ids)] = np.arange(len(panel_ids), dtype="int64")

    logs = np.log(np.where(piv_total.values[complete] > 0, piv_total.values[complete], np.nan))
    growth = np.diff(logs, axis=1)
    growth = growth - np.nanmean(growth, axis=1, keepdims=True)
    growth = growth / np.nanstd(growth, axis=1, keepdims=True)
    growth = np.nan_to_num(growth)
    corr_growth = (growth @ growth.T) / max(growth.shape[1] - 1, 1)
    np.fill_diagonal(corr_growth, 0.0)

    cat_piv = long.pivot_table(
        index="territory_id", columns=["month", "category"], values="value", aggfunc="sum"
    )
    cats = sorted(long["category"].unique())
    m23 = [m for m in months if str(m).startswith("2023")]
    m24 = [m for m in months if str(m).startswith("2024")]

    def half_shares(months_half):
        # pandas 3 removed groupby(axis=1), so summing across the category level goes through a
        # transpose. The month level may or may not survive month selection, hence the layout test.
        sub = cat_piv.loc[:, cat_piv.columns.get_level_values(0).isin(months_half)]
        catvals = (
            sub.columns.get_level_values("category")
            if isinstance(sub.columns, pd.MultiIndex)
            else sub.columns
        )
        num = sub.T.groupby(catvals).sum().T.reindex(cats, axis=1)
        tot = piv_total[months_half].sum(axis=1)
        share = num.div(tot, axis=0)
        out = share.reindex(panel_ids).to_numpy(dtype="float64")
        out = (out - np.nanmean(out, axis=0)) / np.nanstd(out, axis=0)
        out = np.nan_to_num(out)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm > 0, norm, 1.0)

    early, late = half_shares(m23), half_shares(m24)
    corr_lagged = early @ late.T
    np.fill_diagonal(corr_lagged, 0.0)

    coo = sub.tocoo()
    r_all, c_all, d_all = coo.row, coo.col, coo.data
    rp, cp = panel_pos[r_all], panel_pos[c_all]
    keep = (rp >= 0) & (cp >= 0) & (d_all > 0)
    damping = np.exp(-d_all[keep] / TAU_LAGGED)
    src_ids = ids[r_all[keep]]
    dst_ids = ids[c_all[keep]]

    # The pair arrays carry Sber territory ids, which are not matrix coordinates: the largest id in
    # the dictionary is above 3000 while the square matrix is 2660 wide, so using them directly raised
    # "axis 0 index 3013 exceeds matrix dimension 2660". They must be translated through the id to
    # row-position map the complementarity families already use.
    src_pos = pos.reindex(src_ids).to_numpy(dtype="int64")
    dst_pos = pos.reindex(dst_ids).to_numpy(dtype="int64")

    for fam, mat, sym in (
        ("growth_correlation", corr_growth, True),
        ("lagged_leader", corr_lagged, False),
    ):
        w = np.clip(mat[rp[keep], cp[keep]], 0.0, None) * damping
        if sym:
            w_back = np.clip(mat[cp[keep], rp[keep]], 0.0, None) * damping
            Wf = sp.coo_matrix(
                (
                    np.concatenate([w, w_back]),
                    (np.concatenate([src_pos, dst_pos]), np.concatenate([dst_pos, src_pos])),
                ),
                shape=(len(ids), len(ids)),
            ).tocsr()
        else:
            w_rev = np.clip(mat[cp[keep], rp[keep]], 0.0, None) * damping
            Wf = sp.coo_matrix(
                (
                    np.concatenate([w, w_rev]),
                    (np.concatenate([src_pos, dst_pos]), np.concatenate([dst_pos, src_pos])),
                ),
                shape=(len(ids), len(ids)),
            ).tocsr()
        Wf10 = edges.topk(Wf, KNN[0], symmetric=sym)
        report[fam] = {
            "params": {
                "signal": (
                    "month-over-month log growth of the monthly total, Pearson over "
                    f"{growth.shape[1]} diffs"
                    if sym
                    else "cosine between the 2023 category-share vector of i and the 2024 "
                    f"vector of j, {len(cats)} categories"
                ),
                "territories_with_all_months": int(len(panel_ids)),
                "damping": f"exp(-d/{TAU_LAGGED}) on the stored road distance",
                "negative_correlations": "clipped to 0.0",
                "direction": (
                    "undirected" if sym else "both orientations kept with their own weight"
                ),
            },
            "dense": edges.structure(Wf),
            "clique_hub10": edges.clique_hub(Wf10),
            **{f"knn{k}": edges.structure(edges.topk(Wf, k, symmetric=sym)) for k in KNN},
        }
        edge_frame(Wf10, ids, fam).to_parquet(PROCESSED / f"edges_{fam}.parquet", index=False)

    ARTIFACTS.joinpath("edges").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "edges" / "edge_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    def brief(d):
        return (
            f"nodes={d['nodes']} edges={d['edges']} density={d['density']} "
            f"mean_deg={d['mean_degree']} comps={d['weak_components']} "
            f"largest={d['largest_component_share']} isolated={d['isolated']}"
        )

    for fam in [
        "complementarity_exp",
        "complementarity_power",
        "complementarity_trunc",
        "similarity",
    ]:
        r = report[fam]
        print(f"\n{fam} params={r['params']}")
        print(f"  dense : {brief(r['dense'])}")
        print(f"  knn10 : {brief(r['knn10'])}")
        ch = r.get("clique_hub10", {})
        print(
            f"  hub/clique @knn10: reciprocity={ch.get('reciprocity')} "
            f"clustering={ch.get('clustering_coefficient')} gini={ch.get('degree_gini')} "
            f"top1%={ch.get('top1pct_degree_share')} communities={ch.get('lp_communities')} "
            f"modularity={ch.get('lp_modularity')}"
        )
        for k in KNN:
            print(f"  knn{k:<3d}: {brief(r[f'knn{k}'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
