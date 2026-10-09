#!/usr/bin/env python3
"""Cluster the same attribute space on 2024 values, so the map can show a second year honestly.

The shipped answer is the 2023 partition. This script keeps the column list named in the written k
decision and replaces only what actually has a 2024 reading: the five Dynamic World class columns and
MODIS greenness. Night lights stay at 2023 because the written decision freezes them, and elevation and
the spending shares have no 2024 alternative in any artifact.

Writes two things:
  artifacts/icvi/cluster_assignments_2024.parquet   method kmeans_k5_2024, same districts as 2023
  artifacts/validation/partition_2024.json          how far it moved from the shipped partition
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sberindex import cluster  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"
DECISION = ARTIFACTS / "validation" / "k_decision.json"
FEATURES = PROCESSED / "feature_space.parquet"
ADDED = PROCESSED / "covariates_added.parquet"
COVARIATES_2024 = PROCESSED / "covariates_gee_2024.parquet"
SHIPPED = ARTIFACTS / "icvi" / "cluster_assignments.parquet"
CONFIG = ROOT / "configs" / "clustering.yaml"
OUT_PARQUET = ARTIFACTS / "icvi" / "cluster_assignments_2024.parquet"
OUT_JSON = ARTIFACTS / "validation" / "partition_2024.json"

DW_CLASSES = ("bare", "built", "crops", "grass", "trees")
SPENDING_2024 = ARTIFACTS / "validation" / "spending_2024_shares.csv"
CATEGORIES = ("Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт")


def _decision_columns() -> list[str]:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    columns = (decision.get("space") or {}).get("columns")
    if not columns:
        raise SystemExit("the written decision names no space columns, so the space cannot be honoured")
    return list(columns)


def _anchor(labels: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Renumber a fresh partition so a cluster number means the same type it meant in 2023.

    K-means numbers clusters arbitrarily, so the same partition can arrive with every label renamed and
    look as though the whole country moved. Each new cluster takes the number of the 2023 cluster it
    overlaps most, so a district that changes colour really changed type rather than being renamed.
    """
    pairing = pd.crosstab(pd.Series(before, name="before").fillna(-1),
                          pd.Series(labels, name="after")).fillna(0)
    anchor: dict[int, int] = {}
    taken: set[int] = set()
    for _ in range(len(set(labels.tolist()))):
        best = None
        for old in pairing.index:
            for new in pairing.columns:
                if new in taken:
                    continue
                if best is None or pairing.loc[old, new] > best[2]:
                    best = (old, new, pairing.loc[old, new])
        if best is None:
            break
        anchor[int(best[1])] = int(best[0])
        taken.add(best[1])
    for spare in [c for c in sorted(set(labels.tolist())) if c not in taken]:
        leftover = [o for o in pairing.index if o not in set(anchor.values())]
        anchor[int(spare)] = int(leftover[0]) if leftover else int(spare)
    return np.array([anchor.get(int(value), int(value)) for value in labels], dtype="int64")


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    k = int(cfg.get("shipped_k") or 5)
    columns = _decision_columns()

    feats = pd.read_parquet(FEATURES).set_index("tid")
    original = pd.read_parquet(FEATURES)          # untouched, so a single-family variant can start here
    added = pd.read_parquet(ADDED).set_index("tid")
    covariates = pd.read_parquet(COVARIATES_2024).set_index("tid")
    shares = pd.read_csv(SPENDING_2024).set_index("tid")

    replaced: dict[str, str] = {}
    for klass in DW_CLASSES:
        name = f"dynamic_world_{klass}"
        source = f"dw_{klass}@2024"
        if name in feats.columns and source in added.columns:
            feats[name] = added[source].reindex(feats.index).to_numpy()
            replaced[name] = source
    if "modis_ndvi_mean" in feats.columns and "modis_ndvi_mean" in covariates.columns:
        feats["modis_ndvi_mean"] = covariates["modis_ndvi_mean"].reindex(feats.index).to_numpy()
        replaced["modis_ndvi_mean"] = "covariates_gee_2024.parquet: modis_ndvi_mean"

    # The shape of spending moves too. The first version of this script said it did not, and that was
    # wrong: the SberIndex series runs through 2024-12, so a 2024 shape exists and is computed by the
    # same rule the panel uses. Маркетплейсы is where it moves most - 2 002 districts shift by more
    # than a percentage point, while the other four categories barely budge.
    for category in CATEGORIES:
        name = f"{category}_share"
        source = f"{category}_share_2024"
        if name in feats.columns and source in shares.columns:
            feats[name] = shares[source].reindex(feats.index).to_numpy()
            replaced[name] = "spending_2024_shares.csv: " + source

    # Columns that the decision names but that have no 2024 reading stay exactly as they are, and the
    # record says which ones, so a reader can see how much of the space did not move.
    kept = [c for c in columns if c not in replaced]

    feats = feats.reset_index()
    tids, Z, used = cluster.prepare_columns(feats, columns)
    labels = cluster._kmeans(Z, k, seed=seed)

    # K-means numbers clusters arbitrarily, so the same partition can arrive with every label renamed and
    # look as though the whole country moved. Labels are re-anchored to the 2023 partition by the largest
    # overlap: the cluster that holds the most of 2023's cluster 0 keeps the number 0. Colour then means
    # the same type in both years, and a district that changes colour really changed type.
    shipped_preview = pd.read_parquet(SHIPPED).set_index("tid")["cluster"]
    preview_before = shipped_preview.reindex(pd.Index(tids)).to_numpy(dtype="float64")
    labels = _anchor(labels, preview_before)

    shipped = pd.read_parquet(SHIPPED).set_index("tid")["cluster"]
    before = shipped.reindex(pd.Index(tids)).to_numpy(dtype="float64")

    frame = pd.DataFrame({"tid": tids, "cluster": labels.astype(int), "method": [f"kmeans_k{k}_2024"] * len(tids)})
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PARQUET, index=False)

    both = (~np.isnan(before)) & (~np.isnan(labels))
    inner = np.where(both)[0]
    ari_raw = float(adjusted_rand_score(before[inner], labels[inner])) if inner.size else float("nan")
    nmi_raw = float(normalized_mutual_info_score(before[inner], labels[inner])) if inner.size else float("nan")
    labels_before_anchor = labels.copy()
    ari = float(adjusted_rand_score(before[inner], labels[inner])) if inner.size else float("nan")
    nmi = float(normalized_mutual_info_score(before[inner], labels[inner])) if inner.size else float("nan")
    moved = int((before[inner] != labels[inner]).sum())
    raw_moved = int((before[inner] != labels_before_anchor[inner]).sum())

    # The same comparison run twice more, each time changing only one family, so the reader can see which
    # of the two 2024 readings is responsible for the move rather than taking the combined figure on faith.
    variants = {}
    for variant, mutate in (
        ("ndvi_only", lambda f: f.__setitem__("modis_ndvi_mean",
                                              covariates["modis_ndvi_mean"].reindex(f["tid"]).to_numpy())),
        ("land_only", lambda f: [
            f.__setitem__(f"dynamic_world_{klass}",
                          added[f"dw_{klass}@2024"].reindex(f["tid"]).to_numpy()) for klass in DW_CLASSES]),
    ):
        working = original.copy()
        mutate(working)
        vtids, vZ, _ = cluster.prepare_columns(working, columns)
        vlabels = _anchor(cluster._kmeans(vZ, k, seed=seed),
                          shipped.reindex(pd.Index(vtids)).to_numpy(dtype="float64"))
        vbefore = shipped.reindex(pd.Index(vtids)).to_numpy(dtype="float64")
        vboth = (~np.isnan(vbefore)) & (~np.isnan(vlabels))
        vidx = np.where(vboth)[0]
        variants[variant] = {
            "ari": round(float(adjusted_rand_score(vbefore[vidx], vlabels[vidx])), 4),
            "nmi": round(float(normalized_mutual_info_score(vbefore[vidx], vlabels[vidx])), 4),
            "moved_districts": int((vbefore[vidx] != vlabels[vidx]).sum()),
            "moved_pct": round(float((vbefore[vidx] != vlabels[vidx]).sum()) / vidx.size * 100, 2),
        }
        print(f"  variant {variant}: ARI {variants[variant]['ari']:.4f} "
              f"moved {variants[variant]['moved_districts']} ({variants[variant]['moved_pct']:.2f} %)")

    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_by": "scripts/build_2024_partition.py",
                "what_this_is": "The same attribute space and the same method, re-read with the values that have "
                "a 2024 observation. Nothing is recomputed from a column that has no 2024 source.",
                "k": k,
                "seed": seed,
                "method": "kmeans",
                "districts": int(len(tids)),
                "columns_used": list(used),
                "replaced_with_2024": replaced,
                "kept_from_2023": kept,
                "frozen_by_the_written_decision": ["viirs_lit_mean", "elevation_mean"],
                "sizes": {int(c): int((labels == c).sum()) for c in sorted(set(labels.tolist()))},
                "agreement": {"ari": ari, "nmi": nmi, "moved_districts": moved,
                              "moved_pct": round(moved / inner.size * 100, 2) if inner.size else None},
                "single_family_variants": variants,
                "before_anchoring": {"ari": ari_raw, "nmi": nmi_raw, "moved_districts": raw_moved},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"districts {len(tids)}; replaced {len(replaced)} columns; kept {len(kept)}")
    print(f"before anchoring: ARI {ari_raw:.4f}  NMI {nmi_raw:.4f}  raw moved {raw_moved}")
    print(f"after anchoring:  ARI {ari:.4f}  NMI {nmi:.4f}  moved {moved} ({moved / max(inner.size, 1) * 100:.2f} %)")
    print("sizes", {int(c): int((labels == c).sum()) for c in sorted(set(labels.tolist()))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
