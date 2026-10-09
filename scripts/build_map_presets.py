#!/usr/bin/env python3
"""Compute the map presets from artifacts, not from eyesight.

A preset is a view a reader can jump to: the dense districts around a named city, the place where
a cluster concentrates, and the territories whose area is inflated. Nothing here is a coordinate
typed from memory. Cities come from the municipality names in the repository master; extra dense
areas come from a 0.25 degree grid over the centroids of territories that can actually be
measured; cluster centres come from the region with the most members of that cluster in the
interpretation artifact. The defect preset is the flagged territory in the region the geometry
claim names. Every preset carries the artifact it was computed from, so a rendered preset can be
traced, and `scripts/check_map_full.py` shows what a preset actually draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "territory_master.parquet"
CLUSTER_REGIONS = ROOT / "artifacts" / "interpretation" / "cluster_regions.csv"
ASSIGN = ROOT / "artifacts" / "icvi" / "cluster_assignments.parquet"
OUT = ROOT / "artifacts" / "landing" / "map_presets.json"

CITY_STEMS = ("Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск")
GRID = 0.25
EXTRA_LIMIT = 3


def short_label(key: str, label: str) -> str:
    """A button label short enough to sit in a side column without wrapping to four rows.

    Derived from the same fields the long label is built from, never typed per preset. For a
    cluster the short form is its number plus the first word of the region it concentrates in,
    because the surfaces show numbers and the legend carries the meaning. The long label stays in
    the artifact as what the button means on hover.
    """
    head = label.split(" · ")[0]
    if key.startswith("cluster"):
        return f"кластер {key.replace('cluster', '')} · {head.split()[0]}"
    if key.startswith("dense"):
        return head.split()[0]
    return head


def main() -> int:
    m = pd.read_parquet(MASTER)
    m["lat"] = m["centroid_lat"]
    m["lng"] = m["centroid_lon"]
    meas = m[m["centroid_lat"].notna() & m["centroid_lon"].notna()]

    presets: list[dict] = []

    for stem in CITY_STEMS:
        hit = meas[
            meas["municipal_district_name"].str.contains(stem, na=False, regex=False)
            | meas["region_name"].str.contains(stem, na=False, regex=False)
        ]
        if hit.empty:
            print(f"city preset skipped, nothing in the master matches {stem}")
            continue
        presets.append(
            {
                "key": stem.lower().replace("-", ""),
                "label": stem,
                "short": short_label(stem.lower().replace("-", ""), stem),
                "lat": round(float(hit["lat"].mean()), 4),
                "lng": round(float(hit["lng"].mean()), 4),
                "zoom": 10,
                "n_territories": int(len(hit)),
                "provenance": "territory_master.parquet: mean centroid of territories whose name or region contains this city",
            }
        )

    # dense agglomerations found by the grid, not chosen by eye
    cell_lat = (meas["lat"] / GRID).round().astype(int)
    cell_lng = (meas["lng"] / GRID).round().astype(int)
    grid = meas.assign(cl=cell_lat, clg=cell_lng)
    counts = grid.groupby(["cl", "clg"]).size().sort_values(ascending=False)
    chosen = [(p["lat"], p["lng"]) for p in presets]
    extras = 0
    for (cl, clg), n in counts.items():
        if extras >= EXTRA_LIMIT:
            break
        if n < 12:
            break
        sub = grid[(grid["cl"] == cl) & (grid["clg"] == clg)]
        lat = float(sub["lat"].mean())
        lng = float(sub["lng"].mean())
        if any(abs(lat - a) < 1.0 and abs(lng - b) < 1.0 for a, b in chosen):
            continue
        label = sub["region_name"].mode().iloc[0]
        presets.append(
            {
                "key": f"dense{extras}",
                "label": f"{label} · {int(n)} муниципалитетов в сетке {GRID}°",
                "short": short_label(
                    f"dense{extras}", f"{label} · {int(n)} муниципалитетов в сетке {GRID}°"
                ),
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "zoom": 10,
                "n_territories": int(n),
                "provenance": f"territory_master.parquet centroids binned at {GRID} degrees; cell ({cl},{clg}) ranked by count of measurable territories",
            }
        )
        chosen.append((lat, lng))
        extras += 1

    assign = pd.read_parquet(ASSIGN)
    cr = pd.read_csv(CLUSTER_REGIONS)
    joined = m.merge(assign[["tid", "cluster"]], on="tid", how="left")
    for c in sorted(assign["cluster"].unique()):
        rows = cr[cr["cluster"] == c].sort_values("n", ascending=False)
        if rows.empty:
            continue
        region = rows.iloc[0]["region_name"]
        sub = joined[(joined["cluster"] == c) & (joined["region_name"] == region)]
        sub = sub[sub["centroid_lat"].notna()]
        if sub.empty:
            continue
        presets.append(
            {
                "key": f"cluster{int(c)}",
                "label": f"{region} · где скопился кластер",
                "short": short_label(f"cluster{int(c)}", f"{region} · где скопился кластер"),
                "lat": round(float(sub["centroid_lat"].mean()), 4),
                "lng": round(float(sub["centroid_lon"].mean()), 4),
                "zoom": 8,
                "n_territories": int(len(sub)),
                "provenance": "cluster_regions.csv (region with most members of this cluster) + territory_master centroids of those members",
            }
        )

    suspect = joined[joined["area_suspect"]] if "area_suspect" in joined else joined.iloc[0:0]
    red = suspect[suspect["region_name"].str.contains("Красноярск", na=False, regex=False)]
    pool = red if len(red) > 3 else suspect
    if len(pool):
        presets.append(
            {
                "key": "defect",
                "label": "раздутые территории",
                "short": short_label("defect", "раздутые территории"),
                "lat": round(float(pool["centroid_lat"].mean()), 4),
                "lng": round(float(pool["centroid_lon"].mean()), 4),
                "zoom": 6,
                "n_territories": int(len(pool)),
                "provenance": "territory_master.parquet flag area_suspect; Krasnoyarsk territories when there are enough of them",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(presets, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(presets)} presets to {OUT.relative_to(ROOT)}")
    for p in presets:
        print(f"  {p['key']}: {p['lat']}, {p['lng']} z{p['zoom']} n={p['n_territories']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
