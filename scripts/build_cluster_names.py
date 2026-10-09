#!/usr/bin/env python3
"""Give the shipped clusters names a reader can understand, derived only from artifacts.

"кластер 0" tells a reader nothing. The names here are built from the interpretation artifacts:
for each measure in the committed vocabulary, each cluster median is compared with the mean of the others, and
the measures that separate a cluster from the rest the most become the name, in the wording
the vocabulary provides. Nothing is chosen by eye and nothing is typed by hand - the generator,
the vocabulary file and the fingerprints are all committed, and `scripts/check_cluster_names.py`
rejects any rendered cluster label that does not resolve to the generated artifact.

The comparison uses only values present in the artifacts. The direction of the name (low/high)
is the position of this cluster's median relative to the other clusters, so the name is a
statement about the data, not an adjective someone thought sounded right.
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
sys.path.insert(0, str(ROOT / "scripts"))

from sberindex import cluster  # noqa: E402
from inspect_mid_k_range import _intervals, _separated  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "feature_space.parquet"
ASSIGNMENTS = ROOT / "artifacts" / "icvi" / "cluster_assignments.parquet"
CLUSTERING_CONFIG = ROOT / "configs" / "clustering.yaml"
FINGERPRINTS = ROOT / "artifacts" / "interpretation" / "cluster_fingerprints.csv"
VOCAB = ROOT / "configs" / "cluster_name_vocabulary.yaml"
OUT_CSV = ROOT / "artifacts" / "interpretation" / "cluster_names.csv"
OUT_JSON = ROOT / "artifacts" / "interpretation" / "cluster_names.json"
OUT_CSV = ROOT / "artifacts" / "interpretation" / "cluster_names.csv"


def main() -> int:
    if not FINGERPRINTS.exists():
        raise SystemExit(f"missing fingerprints: {FINGERPRINTS}")
    fp = pd.read_csv(FINGERPRINTS)
    fp["statistic"] = fp["statistic"].astype(str)
    vocab = yaml.safe_load(VOCAB.read_text())["measures"]

    use = [m for m in vocab if m in set(fp.statistic)]
    if not use:
        raise SystemExit(
            "no vocabulary measure appears in the fingerprints - name nothing, fix the vocabulary"
        )

    med = fp[fp["stat"] == "median"].pivot_table(
        index="statistic", columns="cluster", values="value"
    )
    med = med.reindex(use).dropna(how="all")
    clusters = sorted(int(c) for c in med.columns)

    # A name is a claim about where this cluster sits, so it has to carry the interval its own median was
    # estimated from. Medians alone let a cluster be called "much forest" on twelve districts; the bootstrap
    # interval says whether that reading holds when the sample is resampled, and how many of the other
    # clusters fall outside it.
    asg = pd.read_parquet(ASSIGNMENTS)
    shipped = yaml.safe_load(CLUSTERING_CONFIG.read_text(encoding="utf-8")).get("shipped_method")
    asg = cluster.select_partition(asg, shipped)
    feats = pd.read_parquet(FEATURES)
    feats["tid"] = feats["tid"].astype("int64")
    feats = feats.drop_duplicates("tid").set_index("tid")
    usable_measures = [m for m in use if m in feats.columns]
    district_values = feats.reindex(asg["tid"].astype("int64").to_numpy())[usable_measures]
    # district_values was reindexed by the assignment order, so the cluster labels line up positionally;
    # indexing them by the feature table's order instead would misalign every interval with its cluster.
    interval_table = district_values.copy()
    interval_table["cluster"] = asg["cluster"].to_numpy()
    intervals = _intervals(
        asg["cluster"].to_numpy(),
        interval_table[usable_measures + ["cluster"]],
        usable_measures,
        np.random.default_rng(0),
    )

    rows = []
    for c in clusters:
        others = [x for x in clusters if x != c]
        scored = []
        for m in use:
            own = float(med.loc[m, c])
            ref = float(med.loc[m, others].mean())
            spread = float(med.loc[m].std())
            if spread == 0 or pd.isna(spread):
                continue
            z = (own - ref) / spread
            direction = "high" if z > 0 else "low"
            scored.append((abs(z), m, direction, round(z, 2), own, ref))
        scored.sort(reverse=True)
        chosen = scored[:3]
        name = ", ".join(vocab[m][direction] for _, m, direction, *_ in chosen)
        name_en = ", ".join(vocab[m]["en_" + direction] for _, m, direction, *_ in chosen)
        rows.append(
            {
                "cluster": c,
                "name": name,
                "name_en": name_en,
                "measures": "; ".join(m for _, m, *_ in chosen),
                "z": "; ".join(str(z) for _, _, _, z, *_ in chosen),
                "own_median": "; ".join(str(v) for *_, v, _ in chosen),
                "others_median": "; ".join(str(v) for *_, _, v in chosen),
                "own_bootstrap_interval": "; ".join(
                    "{}-{}".format(*intervals.get(c, {}).get(m) or (None, None)) for _, m, *_ in chosen
                ),
                "separated_from_clusters": "; ".join(
                    str(
                        sum(
                            1
                            for o in others
                            if m in _separated(c, o, intervals)
                        )
                    )
                    for _, m, *_ in chosen
                ),
                "provenance": f"cluster_fingerprints.csv medians vs the mean of the other {len(clusters) - 1} clusters, divided by the {len(clusters)}-cluster std; bootstrap median intervals over 200 resamples of district values from feature_space.parquet, seed 0; wording from configs/cluster_name_vocabulary.yaml",
            }
        )

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps({str(int(r["cluster"])): r["name"] for r in rows}, ensure_ascii=False, indent=1)
        + "\n"
    )
    # No English name file is written. The entrant asked on 2026-10-09 for a Russian-only
    # submission surface, and the English names beside these were deleted on that instruction.
    for r in rows:
        print(f"cluster {r['cluster']}: {r['name']}")
    print(f"wrote {OUT_CSV.name} and {OUT_JSON.name}; measures used: {len(use)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
