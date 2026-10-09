#!/usr/bin/env python3
"""Measure what the edge rule does to the clustering, not only to the graph.

The brief asks for the influence of the chosen edge construction on the structure of the network and on
the clustering result. The first half was measured long ago - reciprocity, clustering coefficient, degree
gini, top-1% adjacency, label-propagation modularity, dense versus kNN component counts. The second half
was not, and could not have been by looking at the existing feature panel: `feature_space.parquet` carries
no edge-derived column at all, so the attribute clustering was built on spending and land while the
network was built beside it and never entered the space.

Two measurements answer the question, and they say different things:

* node features derived from the networks - how many destinations a territory exports to, how many
  neighbours answer it, how much weight it receives - added to the attribute space. If clusters get
  better at separating economic contrasts, the edges carry information the attributes do not.
* the network's own community partition, found by seeded label propagation, compared with the attribute
  clustering by agreement. If the two disagree, then the edge rule and the attribute space are telling
  different stories about the same territories, which is a more interesting finding than a score.

Everything is measured on the same recipe as the shipped clustering: k-means, seed 0, quantile-normalised
input. The pre-registered bootstrap test decides whether two clusters can be told apart, so this does not
introduce a new standard of evidence halfway through the goal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_k_interpretability import _grab_bag_share, _separation_count  # noqa: E402

from sberindex import icvi  # noqa: E402
from sberindex.cluster import _kmeans  # noqa: E402
from sberindex.config import ARTIFACTS, PROCESSED  # noqa: E402

FEATURES = PROCESSED / "feature_space.parquet"
RULE = ARTIFACTS / "validation" / "meaningfulness_rule.json"
OUT = ARTIFACTS / "validation" / "edge_influence.json"

EDGE_FILES = {
    "complementarity_exp": "edges_complementarity_exp.parquet",
    "complementarity_power": "edges_complementarity_power.parquet",
    "complementarity_trunc": "edges_complementarity_trunc.parquet",
    "similarity": "edges_similarity.parquet",
    "growth_correlation": "edges_growth_correlation.parquet",
    "lagged_leader": "edges_lagged_leader.parquet",
    "dynamic_alignment": "edges_warping.parquet",
}
BASE_MEASURES = [
    "total_per_official_km2",
    "total_mean_rub",
    "Продовольствие_share",
    "Здоровье_share",
    "Маркетплейсы_share",
    "gls_fraction_crops",
    "gls_fraction_grass",
    "gls_fraction_urban",
    "official_land_area_km2",
]
SEED = 0
K = 3
BOOTSTRAP = 300
CI = 5.0


def _node_features(position: pd.Series) -> pd.DataFrame:
    """Per-territory summaries of the networks that the attribute space never saw.

    Territory ids are not row positions: the largest id in these files is above 3000 while the panel is
    about 2190 rows wide, so every endpoint has to be mapped before it can index an array - the same
    translation the edge checker performs.
    """
    n = len(position)
    columns: dict = {}
    for family, fname in EDGE_FILES.items():
        path = PROCESSED / fname
        if not path.exists():
            continue
        e = pd.read_parquet(path)
        src = e.source_id.map(position)
        dst = e.target_id.map(position)
        ok = src.notna() & dst.notna()
        source_row = src.to_numpy()[ok.to_numpy()].astype("int64")
        target_row = dst.to_numpy()[ok.to_numpy()].astype("int64")
        weight = e.weight.to_numpy()[ok.to_numpy()]
        exports = np.zeros(n)
        exports[np.unique(source_row)] = 1.0
        received = np.zeros(n)
        np.add.at(received, target_row, weight)
        in_degree = np.zeros(n)
        np.add.at(in_degree, target_row, 1.0)
        columns[f"edge_{family}_exports_to"] = exports
        columns[f"edge_{family}_received_weight"] = received
        columns[f"edge_{family}_in_degree"] = in_degree
    return pd.DataFrame(columns, index=position.index)


def _cluster(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(matrix)))
    normalized = qt.fit_transform(matrix)
    return np.asarray(_kmeans(normalized, K, seed=SEED)), normalized


def _lp_partition(W: sp.csr_matrix, seed: int = 0) -> np.ndarray:
    """Seeded label propagation over the network's own edges, with no attributes involved."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    A = W.maximum(W.T.tocsr()).tocsr()
    labels = rng.integers(0, K, size=n)
    order = np.argsort(-np.asarray((A > 0).sum(axis=1)).ravel())
    for _ in range(20):
        changed = 0
        for node in order:
            neighbours = A.indices[A.indptr[node] : A.indptr[node + 1]]
            if len(neighbours) == 0:
                continue
            votes = np.bincount(labels[neighbours], minlength=K)
            best = int(np.argmax(votes))
            if best != labels[node]:
                labels[node] = best
                changed += 1
        if changed == 0:
            break
    return labels


def main() -> int:
    missing = [p for p in (FEATURES, RULE) if not p.exists()]
    if missing:
        for path in missing:
            print(f"missing input: {path}")
        print("EDGE INFLUENCE NOT MEASURED")
        return 1

    features = pd.read_parquet(FEATURES)
    features["tid"] = features["tid"].astype("int64")
    available = [m for m in BASE_MEASURES if m in features.columns]
    absent = [m for m in BASE_MEASURES if m not in features.columns]
    if absent:
        print(f"base measures absent from the feature table: {absent}")
    frame = features[["tid"] + available].drop_duplicates("tid").set_index("tid")
    n = len(frame)

    position = pd.Series(np.arange(n), index=frame.index)
    edge_features = _node_features(position)
    joined = pd.concat([frame, edge_features], axis=1)
    edge_columns = [c for c in joined.columns if c.startswith("edge_")]

    base = frame.fillna(frame.median()).to_numpy(dtype="float64")
    with_edges = joined.fillna(joined.median()).to_numpy(dtype="float64")
    labels_base, z_base = _cluster(base)
    labels_edges, z_edges = _cluster(with_edges)

    economic = [
        m
        for m in (
            "total_per_official_km2",
            "total_mean_rub",
            "gls_fraction_crops",
            "gls_fraction_urban",
            "official_land_area_km2",
        )
        if m in joined.columns
    ]

    result: dict = {}
    for name, labels, z, space in (
        ("attributes_only", labels_base, z_base, frame),
        ("attributes_plus_edges", labels_edges, z_edges, joined),
    ):
        labelled = space.copy()
        labelled["cluster"] = labels
        usable = [m for m in economic if m in labelled.columns]
        separated, passed = _separation_count(labelled["cluster"].to_numpy(), labelled, usable)
        flagged = _grab_bag_share(labelled["cluster"].to_numpy(), labelled, usable)
        sizes = {int(k): int((labels == k).sum()) for k in sorted(set(labels.tolist()))}
        result[name] = {
            "columns": list(space.columns),
            "cluster_sizes": sizes,
            "measures_tested": usable,
            "pairs_separated": separated,
            "pairs_total": len(usable) * (len(sizes) * (len(sizes) - 1) // 2),
            "measures_separated_for_every_pair": passed,
            "grab_bag_flags": flagged,
            "metrics": icvi.score(z, np.asarray(labels)),
        }

    agreement = {
        "ari_base_vs_with_edges": round(float(adjusted_rand_score(labels_base, labels_edges)), 4),
        "nmi_base_vs_with_edges": round(
            float(normalized_mutual_info_score(labels_base, labels_edges)), 4
        ),
        "churn_pct": round(float(np.mean(labels_base != labels_edges)) * 100.0, 2),
    }

    network_partitions: dict = {}
    for family, fname in EDGE_FILES.items():
        path = PROCESSED / fname
        if not path.exists():
            network_partitions[family] = {"note": "edge file not on disk"}
            continue
        e = pd.read_parquet(path)
        position = pd.Series(np.arange(n), index=frame.index)
        src = e.source_id.map(position)
        dst = e.target_id.map(position)
        ok = src.notna() & dst.notna()
        W = sp.coo_matrix(
            (
                e.weight.to_numpy()[ok.to_numpy()],
                (
                    src.to_numpy()[ok.to_numpy()].astype("int64"),
                    dst.to_numpy()[ok.to_numpy()].astype("int64"),
                ),
            ),
            shape=(n, n),
        ).tocsr()
        lp = _lp_partition(W)
        network_partitions[family] = {
            "communities_found": int(len(set(lp.tolist()))),
            "ari_vs_attribute_clustering": round(float(adjusted_rand_score(labels_base, lp)), 4),
            "nmi_vs_attribute_clustering": round(
                float(normalized_mutual_info_score(labels_base, lp)), 4
            ),
        }

    payload = {
        "generated_by": "scripts/measure_edge_influence.py",
        "what_this_is": (
            "Whether the edge rule changes the clustering: node features taken from the networks added to "
            "the attribute space, and each network's own community partition compared with the attribute "
            "clustering."
        ),
        "why_this_was_not_measured_before": (
            "feature_space.parquet carries no edge-derived column, so the attribute clustering and the "
            "network were separate artifacts; the influence of the edge rule on clustering could not be "
            "read off the existing panel and had to be constructed."
        ),
        "edge_features_built": edge_columns,
        "base_measures": available,
        "agreement": agreement,
        "results": result,
        "network_partitions": network_partitions,
        "rule": "artifacts/validation/meaningfulness_rule.json",
        "recipe": "k-means, seed 0, quantile-normalised input, k = 3, the shipped recipe",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(
        f"territories: {n} | edge features built: {len(edge_columns)} | base measures: {len(available)}"
    )
    print(
        f"adding edge features: ARI {agreement['ari_base_vs_with_edges']} "
        f"NMI {agreement['nmi_base_vs_with_edges']} churn {agreement['churn_pct']}% | "
        f"separated pairs {result['attributes_only']['pairs_separated']}/"
        f"{result['attributes_only']['pairs_total']} -> "
        f"{result['attributes_plus_edges']['pairs_separated']}/"
        f"{result['attributes_plus_edges']['pairs_total']}"
    )
    for family, block in network_partitions.items():
        if "ari_vs_attribute_clustering" in block:
            print(
                f"  {family}: {block['communities_found']} communities, ARI "
                f"{block['ari_vs_attribute_clustering']} against the attribute clustering"
            )
    print(f"written to {OUT.relative_to(ROOT)}")
    print("EDGE INFLUENCE MEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
