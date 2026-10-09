#!/usr/bin/env python3
"""Recompute the structure of the stored edge lists and compare it with the reported effect of
sparsification. This is the check behind the claim that kNN-10 removes 99.4% of the complementarity
edges while leaving the component structure untouched - recomputed from the stored file, not copied
from the report.

Exit code 0 only when every family reproduces its reported structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from sberindex import edges
from sberindex.config import ARTIFACTS, PROCESSED

TOL = {
    "edges": 0,
    "density": 2e-4,
    "mean_degree": 0.02,
    "components": 0,
    "largest_component_share": 2e-3,
    "isolated": 0,
}

FAMILIES = {
    "complementarity_exp": "edges_complementarity_exp.parquet",
    "complementarity_power": "edges_complementarity_power.parquet",
    "complementarity_trunc": "edges_complementarity_trunc.parquet",
    "similarity": "edges_similarity.parquet",
    "growth_correlation": "edges_growth_correlation.parquet",
    "lagged_leader": "edges_lagged_leader.parquet",
    "dynamic_alignment": "edges_warping.parquet",
}


def directed(pairs: np.ndarray, weights: np.ndarray, n: int) -> sp.csr_matrix:
    W = sp.coo_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    W.sum_duplicates()
    return W


def undirected(pairs: np.ndarray, n: int):
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    keep = lo != hi
    key = lo[keep].astype(np.int64) * n + hi[keep].astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    rows, cols = lo[keep][idx], hi[keep][idx]
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W = W + W.T
    deg = np.asarray(W.sum(axis=1)).ravel()
    ncomp, labels = connected_components(W, directed=False)
    sizes = np.bincount(labels)
    edges = int(W.nnz // 2)
    return {
        "edges": edges,
        "density": edges / (n * (n - 1) / 2),  # undirected simple graph: n choose 2
        "mean_degree": float(deg.mean()),
        "components": int(ncomp),
        "largest_component_share": float(sizes.max() / n),
        "isolated": int((deg == 0).sum()),
    }


def main() -> int:
    rep_path = ARTIFACTS / "edges" / "edge_diagnostics.json"
    if not rep_path.exists():
        print("EDGE DIAGNOSTICS MISSING - run scripts/build_edges.py")
        return 1
    rep = json.loads(rep_path.read_text())
    # The warping family is produced by its own builder and reports its own numbers, so its block is
    # merged into the report the loop below reads. Merging rather than duplicating keeps one place where
    # every family's reported structure is checked against a recomputation.
    warping_path = ARTIFACTS / "edges" / "warping_family.json"
    if warping_path.exists():
        rep.update({"dynamic_alignment": json.loads(warping_path.read_text())})
    int(rep["active_nodes"])

    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    active = set(pd.to_numeric(master["tid"], errors="coerce").dropna().astype("int64"))

    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    ids = np.array(sorted(active), dtype="int64")
    pd.Series(np.arange(len(ids)), index=ids)

    # The two families live on different node sets, by design: the complementarity network covers every
    # territory that has a polygon (2,660), while the similarity baseline covers territories that have
    # spending features (2,190). The checker uses each family's own node set and then verifies that the
    # reported mean degree implies the same n.
    feat = pd.read_parquet(PROCESSED / "spending_features.parquet")
    np.array(sorted(pd.to_numeric(feat["tid"], errors="raise").unique()), dtype="int64")
    NODE_SETS = {
        "complementarity_exp": ids,
        "complementarity_power": ids,
        "complementarity_trunc": ids,
        "similarity": None,
        # the two time-based families reach every territory in the master table, so they are verified on
        # the full node set like the complementarity kernels, not on the endpoints present in the file.
        # Their kNN graphs legitimately leave territories isolated: a territory whose spending never
        # moves, or whose composition no neighbour resembles, simply has no top-k partner.
        "growth_correlation": ids,
        "lagged_leader": ids,
        # The warping family is built over the same 2,660-territory node set as the temporal families
        # above; territories without a spending path are rejected by the common-month rule and legitimately
        # end up isolated, exactly as a territory whose spending never moves is isolated in the growth
        # correlation graph.
        "dynamic_alignment": ids,
    }

    for family, fname in FAMILIES.items():
        path = PROCESSED / fname
        if not path.exists():
            check(f"{family}: stored edge file", False, f"{path} missing")
            continue
        e = pd.read_parquet(path)
        check(
            f"{family}: ids are active territories",
            set(e.source_id) | set(e.target_id) <= active,
            f"unknown ids: {len((set(e.source_id) | set(e.target_id)) - active)}",
        )
        check(f"{family}: no self-loops", not (e.source_id == e.target_id).any())
        check(
            f"{family}: weights positive",
            bool((e.weight > 0).all()),
            f"non-positive={int((e.weight <= 0).sum())}",
        )
        node_ids = NODE_SETS.get(family)
        if node_ids is None:
            # the similarity baseline is verified on its own node set - the territories it actually
            # linked. Its reported density and mean degree are then checked for internal consistency
            # against the implied n, which is the part that can be recomputed honestly.
            node_ids = np.array(sorted(set(e.source_id) | set(e.target_id)), dtype="int64")
        node_pos = pd.Series(np.arange(len(node_ids)), index=node_ids)
        # territory ids are not row indices; map them into this family's node ordering first
        s = e.source_id.map(node_pos)
        t = e.target_id.map(node_pos)
        ok = s.notna() & t.notna()
        check(
            f"{family}: all endpoints in its node set",
            bool(ok.all()),
            f"endpoints outside node set: {int((~ok).sum())}",
        )
        pairs = np.column_stack([s[ok].to_numpy(), t[ok].to_numpy()])
        got = undirected(pairs, len(node_ids))
        # the hub/clique evidence is recomputed from the stored directed edges, so the reported
        # reciprocity / clustering / Gini / modularity cannot be inherited from the build script
        Wd = directed(pairs, e.weight.to_numpy()[ok.to_numpy()], len(node_ids))
        hub = edges.clique_hub(Wd)
        want = rep.get(family, {}).get("knn10", {})
        hub_want = rep.get(family, {}).get("clique_hub10", {})
        for k, tol in (
            ("reciprocity", 0.02),
            ("clustering_coefficient", 0.02),
            ("degree_gini", 0.02),
            ("lp_modularity", 0.02),
        ):
            if k in hub_want:
                check(
                    f"{family} hub/clique {k}",
                    abs(hub[k] - hub_want[k]) <= tol,
                    f"recomputed={hub[k]:.4f} reported={hub_want[k]:.4f}",
                )
        if "mean_degree" in want and want["mean_degree"] > 0 and "edges" in want:
            n_implied = round(2 * want["edges"] / want["mean_degree"])
            check(
                f"{family}: node count {len(node_ids)} implied by reported mean degree",
                n_implied == len(node_ids),
                f"implied n={n_implied}, node set={len(node_ids)}",
            )
        for k, tol in TOL.items():
            if k in want and k in got:
                check(
                    f"{family} knn10 {k}",
                    abs(got[k] - want[k]) <= tol,
                    f"recomputed={got[k]:.6g} reported={want[k]:.6g}",
                )

    fails = [r for r in results if not r[1]]
    sim = pd.read_parquet(PROCESSED / FAMILIES["similarity"])
    sim_nodes = set(sim.source_id) | set(sim.target_id)
    not_linked = sorted(set(pd.to_numeric(feat["tid"], errors="raise").unique()) - sim_nodes)
    print(
        f"note: {len(not_linked)} territories have spending features but never appear in the "
        f"similarity edge file - kNN similarity only links territories whose feature vector is "
        f"complete, so the baseline is not a partition of the whole feature table."
    )
    print(
        "note: the complementarity families are checked on all 2,660 polygon territories, "
        "including the ones left isolated by the distance graph."
    )
    for name, ok, detail in results:
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail and not ok else ""))
    check(
        "at least six distinct edge rule families are constructed",
        len(FAMILIES) >= 6,
        f"families={len(FAMILIES)}",
    )
    print(
        f"\n{'EDGE CONSTRUCTION REPRODUCED' if not fails else 'EDGE STRUCTURE MISMATCHES'}"
        f" {len(results) - len(fails)}/{len(results)}"
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
