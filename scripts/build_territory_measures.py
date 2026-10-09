#!/usr/bin/env python3
"""Write per-territory measure values so a selected district can be compared with its cluster.

A cluster number on the map is a claim about a district: it says the district resembles the districts
grouped with it. The reader who wants to check that claim needs the district's own values next to the
medians of its cluster and of the two others. This artifact carries exactly that, for every district that
was clustered, and for the districts that were not, it carries a reason instead of a number - a territory
with no spending data must not be shown wearing a cluster number it does not have.

The medians are read from the committed legend artifact rather than recomputed here, so the page and the
legend cannot drift apart. The territory values come from the feature panel, which is a build product and
not a committed file; writing them into this artifact is what makes them traceable, because the artifact
itself is committed and a checker can open it. Values that a district does not have stay null: absence is
reported as absence, never as zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberindex import cluster as clustering  # noqa: E402
PROCESSED = ROOT / "data/processed"
ASSIGNMENTS = ROOT / "artifacts/icvi/cluster_assignments.parquet"
LEGEND = ROOT / "artifacts/landing/map_cluster_legend.json"
OUT = ROOT / "artifacts/landing/territory_measures.json"
SPENDING = ROOT / "data/processed/spending_long.parquet"

MEASURES = [
    "gls_fraction_bare",
    "gls_fraction_crops",
    "gls_fraction_urban",
    "viirs_lit_mean",
    "elevation_mean",
    "burn_pct_of_territory",
    "total_mean_rub",
    "spending_per_km2_log",
    "official_land_area_km2",
]

WHY = (
    "Значения каждого округа рядом с медианами его кластера и двух других. "
    "Медианы взяты из артефакта легенды, значения — из панели признаков. "
    "Отсутствие значения осталось отсутствием: нуля вместо него здесь нет."
)


def main() -> int:
    for path, label in (
        (PROCESSED / "feature_space.parquet", "feature panel"),
        (PROCESSED / "territory_master.parquet", "master table"),
        (ASSIGNMENTS, "cluster assignments"),
        (LEGEND, "cluster legend"),
    ):
        if not path.exists():
            print(f"FAIL missing {label}: {path.relative_to(ROOT)}")
            return 1

    legend = json.loads(LEGEND.read_text(encoding="utf-8"))
    medians: dict[str, float] = {}
    for cluster in legend["clusters"]:
        for measure in cluster["measures"]:
            medians[f"{cluster['cluster']}:{measure['statistic']}"] = float(measure["median"])

    fs = pd.read_parquet(PROCESSED / "feature_space.parquet")
    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    asg = pd.read_parquet(ASSIGNMENTS)
    # The shipped partition is named in the configuration, not guessed from row order: the map's per-district
    # comparisons have to describe the same clustering the legend and the names describe.
    shipped = yaml.safe_load((ROOT / "configs" / "clustering.yaml").read_text(encoding="utf-8")).get(
        "shipped_method"
    )
    asg = clustering.select_partition(asg, shipped)

    names = dict(zip(master["tid"].astype(int), master["municipal_district_name"].astype(str)))
    subjects: dict[int, str] = {}
    subject_column = next((c for c in master.columns if "subject" in c or "region" in c), None)
    if subject_column is not None:
        subjects = dict(zip(master["tid"].astype(int), master[subject_column].astype(str)))

    density_column = "total_per_official_km2"
    available = set(fs.columns)

    territories: dict[str, dict] = {}
    missing_measures = 0
    for row in asg.itertuples(index=False):
        tid = int(row.tid)
        record = fs[fs["tid"] == tid]
        if record.empty:
            print(f"FAIL territory {tid} is assigned a cluster but absent from the feature panel")
            return 1
        record = record.iloc[0]
        values: dict[str, float | None] = {}
        for measure in MEASURES:
            if measure not in available:
                missing_measures += 1
                values[measure] = None
                continue
            raw = record[measure]
            values[measure] = None if pd.isna(raw) else round(float(raw), 6)
        territories[str(tid)] = {
            "name": names.get(tid, f"territory {tid}"),
            "subject": subjects.get(tid, ""),
            "cluster": int(row.cluster),
            "values": values,
            "spending_per_official_km2": (
                None
                if density_column not in available or pd.isna(record[density_column])
                else round(float(record[density_column]), 4)
            ),
        }

    clustered = {int(t) for t in territories}
    # Whether a district was left out of the partition and whether it has no spending data are two
    # different facts, and conflating them made the map tell 174 districts that had months of spending
    # that they had none. The split is read from the spending table itself, not inferred from the
    # absence of a cluster label.
    spending_rows: set[int] = set()
    if SPENDING.exists():
        long = pd.read_parquet(SPENDING)
        spending_rows = {int(v) for v in long["territory_id"]}
    no_spending: dict[str, str] = {}
    counts = {"no_spending_rows": 0, "spending_but_excluded": 0}
    for tid in master["tid"].astype(int):
        if int(tid) in clustered:
            continue
        name = names.get(int(tid), f"territory {int(tid)}")
        if int(tid) in spending_rows:
            counts["spending_but_excluded"] += 1
            no_spending[str(int(tid))] = (
                f"{name}: траты в данных есть, но округ не прошёл порог покрытия панели "
                "признаков, поэтому кластер ему не присвоен"
            )
        else:
            counts["no_spending_rows"] += 1
            no_spending[str(int(tid))] = (
                f"{name}: ни одной строки трат в исходных данных - кластеризация по тратам "
                "для этого округа невозможна"
            )

    artifact = {
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "what_this_is": WHY,
        "measures": MEASURES,
        "captions": legend["clusters"][0]["measures"],
        "medians_source": str(LEGEND.relative_to(ROOT)),
        "cluster_medians": {
            str(cluster["cluster"]): {
                measure["statistic"]: float(measure["median"]) for measure in cluster["measures"]
            }
            for cluster in legend["clusters"]
        },
        "cluster_names": {
            str(cluster["cluster"]): cluster["short_name"] for cluster in legend["clusters"]
        },
        "territories": territories,
        "no_spending": no_spending,
        "counts": {
            "with_cluster": len(territories),
            "without_spending": len(no_spending),
            "no_spending_rows": counts["no_spending_rows"],
            "spending_but_excluded": counts["spending_but_excluded"],
        },
        "absence_basis": str(SPENDING.relative_to(ROOT)) + " (build product, not tracked in git)",
        "absence_rule": (
            "округу без строки трат reason говорит, что трат нет; округу со строками трат, но "
            "не вошедшему в панель, reason говорит, что траты есть и не пройден порог покрытия"
        ),
        "cluster_assignment_artifact": str(ASSIGNMENTS.relative_to(ROOT)),
        "rule": (
            "значения округов взяты из панели признаков; медианы — из "
            "artifacts/landing/map_cluster_legend.json; округу без трат кластер не присваивается"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(
        f"territory measures: {len(territories)} districts with a cluster, "
        f"{len(no_spending)} without one ({counts['no_spending_rows']} with no spending rows, "
        f"{counts['spending_but_excluded']} with spending rows but below the panel gate), written "
        f"to {OUT.relative_to(ROOT)} "
        f"({OUT.stat().st_size:,} B)"
    )
    if missing_measures:
        print(f"note: {missing_measures} measure values absent because the column is absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
