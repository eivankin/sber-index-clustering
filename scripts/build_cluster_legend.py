#!/usr/bin/env python3
"""Build the cluster legend: nine measures per cluster, with a real example territory for each.

The reader asked for the meaning of a cluster colour to be reachable without hunting through pages.
This artifact is that meaning: for every cluster, nine fingerprint measures, official land area,
spending density on official area, months of observation, the cluster's own median and how it
differs from the other two clusters, plus one example territory per measure — the territory whose
value sits closest to that cluster's median.

The medians, coverage, p25/p75 and provenance all come from
artifacts/interpretation/cluster_fingerprints.csv, which is in git. n, months and the spending
median come from artifacts/interpretation/cluster_profile.csv, also in git. The candidate territories
need a per-territory value, and those live in data/processed/feature_space.parquet, which is not in
git. So this builder ships the shortlist it chose from inside the artifact: the checker can then
prove the ordering from the artifact alone, without reaching for an untracked table.

These are contrasts between clusters, not tests for membership. A territory can sit far from a
cluster median on one measure and still belong to that cluster; the legend says so in its own text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINTS = ROOT / "artifacts/interpretation/cluster_fingerprints.csv"
PROFILE = ROOT / "artifacts/interpretation/cluster_profile.csv"
ASSIGNMENTS = ROOT / "artifacts/icvi/cluster_assignments.parquet"
FEATURES = ROOT / "data/processed/feature_space.parquet"
NAMES = ROOT / "artifacts/interpretation/cluster_names.json"
OUT = ROOT / "artifacts/landing/map_cluster_legend.json"

# Nine measures the three-cluster story actually turns on: what the land produces, what the place
# consumes, and the land claims the geometry is about. Every one of them is a row in the
# fingerprints artifact, so none of them is invented here.
HEADLINE = [
    ("gls_fraction_bare", "доля открытой земли, % классифицированных пикселей"),
    ("gls_fraction_crops", "доля пашни, % классифицированных пикселей"),
    ("gls_fraction_urban", "доля застройки, % классифицированных пикселей"),
    ("viirs_lit_mean", "ночное свечение по освещённым пикселям"),
    ("elevation_mean", "средняя высота над уровнем моря, м"),
    ("burn_pct_of_territory", "доля выгоревшей площади от территории, %"),
    ("total_mean_rub", "среднемесячные траты по округу всего, ₽"),
    ("spending_per_km2_log", "траты на км² официальной площади (log1p)"),
    ("official_land_area_km2", "официальная площадь земли Росстата, км²"),
]

CANDIDATES_PER_CELL = 3


def main() -> int:
    missing = [p for p in (FINGERPRINTS, PROFILE, ASSIGNMENTS) if not p.exists()]
    if missing:
        print(
            "FAIL missing committed input: " + ", ".join(str(m.relative_to(ROOT)) for m in missing)
        )
        return 1
    if not FEATURES.exists():
        print(f"FAIL {FEATURES.relative_to(ROOT)} missing — run scripts/build_features.py first")
        return 1

    fp = pd.read_csv(FINGERPRINTS)
    profile = pd.read_csv(PROFILE)
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    assign = pd.read_parquet(ASSIGNMENTS)
    feats = pd.read_parquet(FEATURES)
    feats = feats[feats["tid"].isin(assign["tid"])].copy()
    # Membership is read from the committed assignments, as a plain map, so indexing a block of
    # territories by cluster stays aligned with the frame being indexed.
    tid_cluster = dict(zip(assign["tid"].astype(int), assign["cluster"].astype(int)))

    medians: dict[tuple[int, str], float] = {}
    rows: dict[tuple[int, str], dict] = {}
    for row in fp.itertuples():
        medians[(int(row.cluster), row.statistic)] = float(row.value)
        rows[(int(row.cluster), row.statistic)] = row._asdict()

    prof = {int(r.cluster): r for r in profile.itertuples()}

    clusters: list[dict] = []
    examples: list[dict] = []
    for cluster in sorted(assign["cluster"].unique()):
        cluster = int(cluster)
        measures = []
        for stat, caption in HEADLINE:
            cell = rows.get((cluster, stat))
            if cell is None:
                print(f"FAIL {stat} has no fingerprint row for cluster {cluster}")
                return 1
            value = float(cell["value"])
            others = [
                {"cluster": other, "median": medians[(other, stat)]}
                for other in sorted(prof)
                if other != cluster and (other, stat) in medians
            ]
            measures.append(
                {
                    "statistic": stat,
                    "caption": caption,
                    "median": round(value, 4),
                    "coverage": round(float(cell["coverage"]), 4),
                    "n_members": int(cell["n_members"]),
                    "p25": round(float(cell["p25"]), 4),
                    "p75": round(float(cell["p75"]), 4),
                    "provenance": str(cell["provenance"]),
                    "contrast_with_others": [
                        {
                            "cluster": o["cluster"],
                            "other_median": round(o["median"], 4),
                            "difference": round(value - o["median"], 4),
                        }
                        for o in others
                    ],
                }
            )

            # The example is the territory whose value is closest to this cluster's median.
            block = feats[feats["tid"].map(tid_cluster) == cluster]
            values = pd.to_numeric(block[stat], errors="coerce")
            scored = pd.DataFrame(
                {"tid": block["tid"].to_numpy(), "value": values.to_numpy()}
            ).dropna()
            scored["distance_to_median"] = (scored["value"] - value).abs()
            scored = scored.sort_values(["distance_to_median", "tid"], kind="mergesort").head(
                CANDIDATES_PER_CELL
            )
            candidates = [
                {
                    "tid": int(t.tid),
                    "value": round(float(t.value), 4),
                    "distance_to_median": round(float(t.distance_to_median), 4),
                }
                for t in scored.itertuples()
            ]
            if candidates:
                examples.append(
                    {
                        "cluster": cluster,
                        "statistic": stat,
                        "tid": candidates[0]["tid"],
                        "value": candidates[0]["value"],
                        "distance_to_median": candidates[0]["distance_to_median"],
                        "candidates": candidates,
                    }
                )

        p = prof[cluster]
        density = (
            float(p.median_spending_rub) / float(p.median_official_area_km2)
            if float(p.median_official_area_km2) > 0
            else None
        )
        clusters.append(
            {
                "cluster": cluster,
                "short_name": f"кластер {cluster}",
                "meaning": str(names[str(cluster)]),
                "n": int(p.n),
                "months_of_observation_mean": round(float(p.mean_months), 2),
                "median_official_area_km2": round(float(p.median_official_area_km2), 2),
                "median_spending_rub": round(float(p.median_spending_rub), 2),
                "spending_per_official_km2": round(density, 2) if density is not None else None,
                "spending_per_official_km2_note": (
                    "отношение медиан: медиана среднемесячных трат по округам кластера делённая на "
                    "медиану официальной площади; это не медиана плотности по округам"
                ),
                "measures": measures,
            }
        )

    payload = {
        "generated_by": "scripts/build_cluster_legend.py",
        "what_this_is": (
            "Сравнения между кластерами, а не признаки, по которым округ в кластер определяют. "
            "Округ может сильно отходить от медианы по одной мере и всё равно принадлежать этому "
            "кластеру."
        ),
        "headline_measures": [stat for stat, _ in HEADLINE],
        "provenance": {
            "fingerprints": str(FINGERPRINTS.relative_to(ROOT)),
            "profile": str(PROFILE.relative_to(ROOT)),
            "assignments": str(ASSIGNMENTS.relative_to(ROOT)),
            "names": str(NAMES.relative_to(ROOT)),
            "per_territory_values": str(FEATURES.relative_to(ROOT)),
            "candidates_shipped": True,
        },
        "clusters": clusters,
        "examples": examples,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT}: {OUT.stat().st_size:,} bytes")
    print(
        f"clusters {len(clusters)} | measures per cluster {len(clusters[0]['measures'])} | "
        f"examples {len(examples)}"
    )
    for c in clusters:
        print(
            f"  cluster {c['cluster']}: n={c['n']} months={c['months_of_observation_mean']} "
            f"official area median={c['median_official_area_km2']} km2, "
            f"spend per official km2={c['spending_per_official_km2']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
