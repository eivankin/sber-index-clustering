#!/usr/bin/env python
"""Method comparison on the spending attribute space, with the ICVI battery.

Runs kmeans / gaussian mixture / ward / average / complete linkage / spectral (attribute and graph
variants) / DBSCAN / weighted label propagation, scores every partition with SW, CH, S_Dbw, AVI,
AVU, MQ and writes a normalized comparison table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from sberindex import cluster, edges
from sberindex.config import ARTIFACTS, PROCESSED, ensure_dirs

KS = [3, 4, 5, 6, 7, 8]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "clustering.yaml"


def load_config():
    """The committed clustering recipe. Reading it here means the config is not decoration: the k
    sweep, the graph sparsification, the seed and the exclusion list all come from this file, and the
    values actually used are written out next to the ICVI table."""
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    return cfg, digest


def _call(fn, Z, k, seed, W=None):
    if W is not None:
        return fn(Z, k, W)
    try:
        return fn(Z, k, seed=seed)
    except TypeError:
        return fn(Z, k)


def knn_adjacency(Z: np.ndarray, k: int = 10) -> sp.csr_matrix:
    D = cdist(Z, Z)
    n = Z.shape[0]
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    W = sp.coo_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
    W = W.maximum(W.T.tocsr())
    return W.tocsr()


def _resolve_space_columns(name: str, cfg: dict) -> list[str]:
    """Resolve a named attribute space to its exact column list, or refuse."""
    if name == "decision":
        path = ARTIFACTS / "validation" / "k_decision.json"
        if not path.exists():
            raise SystemExit("no written k decision to read a space from: artifacts/validation/k_decision.json")
        columns = (json.loads(path.read_text(encoding="utf-8")).get("space") or {}).get("columns")
        if not columns:
            raise SystemExit("the written decision names no space columns, so the space cannot be honoured")
        return list(columns)
    spaces = cfg.get("spaces") or {}
    if name in spaces:
        return list(spaces[name])
    raise SystemExit(
        f"unknown attribute space '{name}': name it under spaces in configs/clustering.yaml, or use 'decision'"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        default=None,
        help="feature table under data/processed (default spending_features.parquet; "
        "use feature_space.parquet once covariates exist)",
    )
    ap.add_argument(
        "--groups", default="levels+shares", help="attribute blocks to use, e.g. levels+shares+land"
    )
    ap.add_argument(
        "--space",
        default=None,
        help="named attribute space: 'decision' reads the approved column list from the entrant's written "
        "k decision; any other name must appear under spaces in configs/clustering.yaml. It replaces the "
        "group shorthand, because a group name can carry indicators the decision excluded",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="the shipped k to carry through the sweep; added to the config's k_values when it is missing",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="suffix for the output artifacts, so two attribute spaces can be compared "
        "side by side instead of overwriting each other",
    )
    args = ap.parse_args()
    ensure_dirs()
    cfg, cfg_digest = load_config()
    if args.features is None and args.space:
        # A named space lives in the full feature table, not in the spending-only one. The default below
        # would otherwise run the approved column list against a table that lacks most of it.
        args.features = "feature_space.parquet"
    path = PROCESSED / (args.features or "spending_features.parquet")
    feats = pd.read_parquet(path)
    if args.space:
        tids, Z, cols = cluster.prepare_columns(feats, _resolve_space_columns(args.space, cfg))
        print(f"attribute space: {path.name} space={args.space} n={len(tids)} features={len(cols)}")
    else:
        tids, Z, cols = cluster.prepare(feats, groups=args.groups)
        print(f"attribute space: {path.name} groups={args.groups} n={len(tids)} features={len(cols)}")
    ks = list(cfg.get("k_values") or KS)
    if args.k is not None and args.k not in ks:
        ks.append(args.k)
    ks = sorted(ks)
    # The shipped partition is resolved here, before any work runs, so a configuration that names nothing or names
    # a method the sweep will not produce fails in a second instead of after minutes of clustering.
    shipped = str(cfg.get("shipped_method") or "")
    if not shipped:
        raise SystemExit(
            "no shipped partition named in configs/clustering.yaml under shipped_method; refusing to write "
            "assignments that would have to be guessed from the ICVI ranking"
        )
    knn_k = int(cfg.get("knn_k", 10))
    seed = int(cfg.get("seed", 0))
    skip = {m for m, p in (cfg.get("methods") or {}).items() if p and p.get("run") is False}
    W = knn_adjacency(Z, k=knn_k)
    struct = edges.structure(W)
    print(
        "graph:",
        {
            k: struct[k]
            for k in [
                "nodes",
                "edges",
                "density",
                "mean_degree",
                "weak_components",
                "largest_component_share",
            ]
        },
    )

    results: dict[str, np.ndarray] = {}
    for k in ks:
        for name, fn in cluster.METHODS.items():
            if name in skip or name == "spectral":
                continue
            try:
                results[f"{name}_k{k}"] = _call(fn, Z, k, seed, W=W if name == "spectral" else None)
            except Exception as exc:
                print(f"  ! {name}_k{k}: {type(exc).__name__}: {str(exc)[:80]}")
        try:
            results[f"spectral_graph_k{k}"] = cluster._spectral(Z, k, W=W)
        except Exception as exc:
            print(f"  ! spectral_graph_k{k}: {type(exc).__name__}: {str(exc)[:80]}")
    try:
        results["dbscan_auto"] = cluster._dbscan(Z)
    except Exception as exc:
        print(f"  ! dbscan: {type(exc).__name__}")
    for lp_seed in (seed, seed + 1):
        results[f"label_propagation_s{lp_seed}"] = cluster.label_propagation(W, seed=lp_seed)

    table = cluster.compare(Z, results, W=W)
    ARTIFACTS.joinpath("icvi").mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (ARTIFACTS / "icvi" / f"clustering_config_used{suffix}.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "config_path": str(CONFIG_PATH.relative_to(CONFIG_PATH.parents[1])),
                "config_sha256": cfg_digest,
                "dbscan_note": "the config's dbscan run flag governs the per-k sweep only, where eps has no k "
                "to pair with; DBSCAN is still evaluated outside that sweep and reported as "
                "dbscan_auto, so run: false in the config is not a claim that DBSCAN was skipped",
                "ks_used": ks,
                "knn_k": knn_k,
                "seed": seed,
                "skipped_methods": sorted(skip | {"spectral"}),
                "shipped_method": shipped,
                "groups": args.groups,
                "space": args.space,
                "columns_used": list(cols),
                "feature_table": path.name,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    table.to_csv(ARTIFACTS / "icvi" / f"method_comparison{suffix}.csv", index=False)
    # The assignments are the shipped partition, so they carry the method the configuration names - not the one
    # that happens to top the ICVI average. Writing the scoreboard's favourite would let an index average quietly
    # replace a decision the entrant made in writing, which is the substitution this repository keeps hunting.
    top = str(table.iloc[0]["method"])
    if shipped not in results:
        raise SystemExit(
            f"the shipped partition '{shipped}' was not produced by this sweep - check k_values, the methods "
            "run flags, and whether the method name matches what the configuration declares"
        )
    labels = results[shipped]
    pd.DataFrame({"tid": tids, "cluster": labels, "method": shipped}).to_parquet(
        ARTIFACTS / "icvi" / f"cluster_assignments{suffix}.parquet", index=False
    )
    print(f"shipped partition written: {shipped} | ICVI average would have preferred: {top}")
    print("\nTOP 8 BY NORMALIZED ICVI:")
    print(
        table.head(8)[
            ["method", "k", "ICVI_avg", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]
        ].to_string(index=False)
    )
    print("\nworst:", table.tail(2)[["method", "k", "ICVI_avg", "SW", "MQ"]].to_string(index=False))
    print(
        f"\nshipped partition: {shipped} - the one the written decision names; "
        f"the ICVI average alone would have preferred {top}"
    )
    print("method x k partitions compared:", len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
