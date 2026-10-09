#!/usr/bin/env python3
"""Build the edge family the contest brief names and this entry lacked: alignment of spending paths.

The brief lists lagged correlation and dynamic time warping among the ways an edge may be defined, and
offers Dynamic Time Warping by name. Our families covered distance decay, complementarity, feature
similarity, growth correlation and a one-year cosine lag - but nothing that warps one territory's monthly
path onto another's at an offset chosen by the data. This adds that family.

The scalable form is a bounded-lag alignment: for each candidate pair, the two z-scored monthly paths are
compared at offsets from -3 to +3 months and the best offset is kept. A true dynamic-programming DTW with
the same band is computed on a random sample of complete pairs, and the agreement between the two is
reported, so the shortcut is justified by measurement rather than asserted. The winning offset carries
direction: a positive lag means the first territory is earlier, which is what makes this family able to
say who leads and who follows - something a symmetric similarity edge cannot say.

Candidate pairs are not all pairs. Warping is a local transmission question, so only the fifty nearest
territories by stored road distance are considered for each place; that keeps the work bounded and states
the reason in the artifact. Missing months are not filled: a pair is scored only on the months both
territories actually have, and a pair with too few common months produces no edge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberindex import edges  # noqa: E402
from sberindex.config import ARTIFACTS, PROCESSED  # noqa: E402

SPENDING = PROCESSED / "spending_long.parquet"
OUT_EDGES = PROCESSED / "edges_warping.parquet"
OUT_REPORT = ARTIFACTS / "edges" / "warping_family.json"

TAU_KM = 600.0
NEIGHBOURS = 50
LAGS = tuple(range(-3, 4))
MIN_COMMON_MONTHS = 12
DTW_SAMPLE = 400
SEED = 0
FAMILY = "dynamic_alignment"


def _paths(nodes: np.ndarray) -> np.ndarray:
    """One z-scored log-spending path per territory, months left missing where they are missing."""
    long = pd.read_parquet(SPENDING)
    long["territory_id"] = pd.to_numeric(long["territory_id"], errors="raise").astype("int64")
    long["month"] = long["month"].astype(str)
    # The table is one row per territory, month and spending category, so a monthly path has to be
    # summed across categories first. Pivoting the raw rows would average categories instead.
    monthly = long.groupby(["territory_id", "month"], observed=True)["value"].sum().reset_index()
    wide = monthly.pivot_table(index="territory_id", columns="month", values="value", aggfunc="sum")
    order = sorted(wide.columns)
    wide = wide[order]
    # Rows are placed in the master table's territory order, the same node set the other families are
    # verified on. Territories without spending become an all-missing path and are rejected by the
    # common-month rule rather than being dropped from the matrix.
    wide = wide.reindex(nodes)
    values = np.log1p(wide.to_numpy(dtype="float64"))
    mean = np.nanmean(values, axis=1, keepdims=True)
    spread = np.nanstd(values, axis=1, keepdims=True)
    spread = np.where(spread > 1e-9, spread, np.nan)
    return (values - mean) / spread


def _shift(matrix: np.ndarray, lag: int) -> np.ndarray:
    """Row values moved so that column t holds the original month t + lag."""
    out = np.full_like(matrix, np.nan)
    width = matrix.shape[1]
    if lag >= 0:
        out[:, : width - lag] = matrix[:, lag:]
    else:
        out[:, -lag:] = matrix[:, : width + lag]
    return out


def _candidate_pairs(distance: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fifty nearest stored road distances in each direction, symmetrized and deduplicated."""
    distance = distance.tocsr()
    rows, cols, dists = [], [], []
    for row in range(distance.shape[0]):
        start, stop = distance.indptr[row], distance.indptr[row + 1]
        if stop - start < 2:
            continue
        block = distance.indices[start:stop]
        values = distance.data[start:stop]
        keep = np.argpartition(values, min(NEIGHBOURS, len(values) - 1))[:NEIGHBOURS]
        keep = keep[values[keep] > 0]
        rows.append(np.full(len(keep), row))
        cols.append(block[keep])
        dists.append(values[keep])
    src = np.concatenate(rows)
    dst = np.concatenate(cols)
    km = np.concatenate(dists)
    both = np.concatenate([np.stack([src, dst], axis=1), np.stack([dst, src], axis=1)])
    # The distance list has to be doubled alongside the pair list. Indexing the single-length array by a
    # position from the doubled one ran past its end and raised IndexError.
    km_both = np.concatenate([km, km])
    key = both[:, 0].astype(np.int64) * 10_000 + both[:, 1].astype(np.int64)
    _, unique_position = np.unique(key, return_index=True)
    return both[unique_position, 0], both[unique_position, 1], km_both[unique_position]


def _dtw(left: np.ndarray, right: np.ndarray, band: int = 3) -> float:
    """True dynamic-programming warping on two complete paths, restricted to a band."""
    length = len(left)
    big = float("inf")
    cost = np.full((length, length), big)
    for row in range(length):
        low = max(0, row - band)
        high = min(length, row + band + 1)
        for column in range(low, high):
            local = (left[row] - right[column]) ** 2
            above = cost[row - 1, column] if row > 0 else big
            leftward = cost[row, column - 1] if column > 0 else big
            diagonal = cost[row - 1, column - 1] if row > 0 and column > 0 else 0.0
            best = min(above, leftward, diagonal)
            cost[row, column] = local + (0.0 if best == big and row == column == 0 else best)
    return float(np.sqrt(cost[length - 1, length - 1] / length))


def main() -> int:
    if not SPENDING.exists():
        print(f"missing input: {SPENDING}")
        print("WARPING FAMILY NOT BUILT")
        return 1
    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    nodes = np.array(
        sorted(pd.to_numeric(master["tid"], errors="coerce").dropna().astype("int64")),
        dtype="int64",
    )
    paths = _paths(nodes)
    distance = edges.load_distances()["highway"]
    # The distance matrix and the edge file are indexed by the master's territory ids in sorted order,
    # which is the node set the edge checker verifies against; the spending table's own order is not it.
    position = pd.Series(np.arange(len(nodes)), index=nodes)
    src_ids, dst_ids, km = _candidate_pairs(distance)
    src = position.reindex(src_ids).to_numpy(dtype="float64")
    dst = position.reindex(dst_ids).to_numpy(dtype="float64")
    usable = ~np.isnan(src) & ~np.isnan(dst)
    src, dst, km = src[usable].astype("int64"), dst[usable].astype("int64"), km[usable]

    scored = np.full((len(src), len(LAGS)), np.inf)
    months = np.zeros((len(src), len(LAGS)), dtype="int16")
    for offset, lag in enumerate(LAGS):
        shifted = _shift(paths, lag)
        diff = paths[src] - shifted[dst]
        present = ~np.isnan(diff)
        count = present.sum(axis=1)
        absolute = np.where(present, np.abs(diff), 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            scored[:, offset] = np.where(
                count >= MIN_COMMON_MONTHS, absolute / np.maximum(count, 1), np.inf
            )
        months[:, offset] = count
    best_offset = np.argmin(scored, axis=1)
    warping = scored[np.arange(len(src)), best_offset]
    best_lag = np.asarray(LAGS)[best_offset]
    valid = np.isfinite(warping) & (warping > 0)
    print(
        f"candidate pairs {len(src):,} | scored {int(valid.sum()):,} | "
        f"rejected for too few common months {int((~valid).sum()):,}",
        flush=True,
    )

    h = float(np.median(warping[valid]))
    weight = np.exp(-warping[valid] / h) * np.exp(-km[valid] / TAU_KM)
    W = sp.coo_matrix((weight, (src[valid], dst[valid])), shape=(len(nodes), len(nodes))).tocsr()
    W.sum_duplicates()
    knn = {k: edges.topk(W, k) for k in (10, 25)}
    W10 = knn[10]

    frame = pd.DataFrame(
        {
            "source_id": nodes[W10.tocoo().row],
            "target_id": nodes[W10.tocoo().col],
            "weight": W10.tocoo().data,
            "family": FAMILY,
        }
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_EDGES, index=False)

    leads = best_lag[valid]
    validation: dict = {}
    complete = (~np.isnan(paths)).sum(axis=1) == paths.shape[
        1
    ]  # territories whose every month in the window is observed
    pair_complete = complete[src[valid]] & complete[dst[valid]]
    rng = np.random.default_rng(SEED)
    picks = rng.choice(
        np.flatnonzero(pair_complete), size=min(DTW_SAMPLE, int(pair_complete.sum())), replace=False
    )
    if len(picks):
        fast = warping[valid][picks]
        slow = np.array([_dtw(paths[src[valid][p]], paths[dst[valid][p]]) for p in picks])
        rho = spearmanr(fast, slow)
        validation = {
            "sampled_pairs": int(len(picks)),
            "territories_with_every_month": int(complete.sum()),
            "spearman_r": round(float(rho.statistic), 4),
            "spearman_p": float(rho.pvalue),
            "median_absolute_difference": round(float(np.median(np.abs(fast - slow))), 4),
            "note": (
                "the bounded-lag form is not the same metric as full warping; the rank agreement is "
                "reported so the shortcut can be judged rather than trusted"
            ),
        }

    report = {
        "generated_by": "scripts/build_edge_warping.py",
        "family": FAMILY,
        "params": {
            "signal": "z-scored log monthly spending total, 24 months of the observation window",
            "lag_window": f"{LAGS[0]} to {LAGS[-1]} months",
            "candidate_pairs": f"the {NEIGHBOURS} nearest territories by stored road distance, symmetrized",
            "distance_damping": f"exp(-d/{TAU_KM:.0f}) on the stored road distance",
            "weight": "exp(-warping_distance / h) * exp(-d/tau), h = median warping distance",
            "h_median_warping": round(h, 4),
            "missing_months": "left missing; a pair is scored only on months both territories have",
            "minimum_common_months": MIN_COMMON_MONTHS,
            "direction": "undirected weight, directed lead carried by the winning offset",
            "sparsification": "per-source top-k outgoing edges, k = 10 and 25",
        },
        "dense": edges.structure(W),
        "clique_hub10": edges.clique_hub(W10),
        "knn10": edges.structure(W10),
        "knn25": edges.structure(knn[25]),
        "lead_lag": {
            "pairs_scored": int(valid.sum()),
            "share_with_nonzero_offset": round(float(np.mean(leads != 0)), 4),
            "mean_offset_months": round(float(np.mean(leads)), 4),
            "offset_distribution": {str(lag): int(np.sum(leads == lag)) for lag in LAGS},
            "leaders": [
                str(nodes[p])
                for p in np.argsort(-np.bincount(src[valid][leads > 0], minlength=len(nodes)))[:5]
            ],
            "leaders_note": (
                "a territory counts as a leader on the pairs where it is the earlier one, so the "
                "count is over pairs and not over lag values"
            ),
        },
        "warping_validation": validation,
        "edge_file": str(OUT_EDGES.relative_to(ROOT)),
        "edges_at_knn10": int(len(frame)),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    dense = report["dense"]
    hub = report["clique_hub10"]
    print(
        f"dense: edges={dense.get('edges')} density={dense.get('density')} "
        f"components={dense.get('weak_components')}"
    )
    print(
        f"knn10: edges={report['knn10'].get('edges')} reciprocity={hub.get('reciprocity')} "
        f"gini={hub.get('degree_gini')} modularity={hub.get('lp_modularity')}"
    )
    print(
        f"lead/lag: {report['lead_lag']['share_with_nonzero_offset']:.1%} of pairs are offset, "
        f"mean {report['lead_lag']['mean_offset_months']} months"
    )
    if validation:
        print(
            f"validation against full warping on {validation['sampled_pairs']} pairs: "
            f"Spearman {validation['spearman_r']}"
        )
    print(f"written to {OUT_EDGES.relative_to(ROOT)} and {OUT_REPORT.relative_to(ROOT)}")
    print("WARPING FAMILY BUILT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
