#!/usr/bin/env python3
"""Build region and place labels from our own geometry, so no label is ever typed by hand.

A label is two things: a name and a point. The names come out of the master table and out of
artifacts/geo/region_inflation.csv; the points are computed from our own polygons, never typed and
never copied from a basemap. A region's label sits on land that belongs to that region, which is
proven here by point-in-polygon against the territories that make it up.

Why not just take the centroid of a region: a region with a long thin shape, or an enclave, has a
centroid that can fall outside its own territory. So we start from the area-weighted mean of the
territory centroids and only accept it if some territory of that region contains the point. If it
does not, we fall back to a representative point of the region's largest territory, which shapely
guarantees to be inside that geometry.

The zoom schedule is by region size, which makes the number of labels rise with zoom: big regions
are readable at the national view, small ones only once you are close enough to need them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import shapely
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data/processed/territory_master.parquet"
INFLATION = ROOT / "artifacts/geo/region_inflation.csv"
SUMMARY = ROOT / "artifacts/landing/map_summary.json"
OUT = ROOT / "artifacts/landing/map_labels.json"

# The smallest place worth naming gets its label once the reader is close enough to see it.
PLACE_MAX_AREA_KM2 = 500.0
PLACE_LIMIT = 400
PLACE_MINZOOM = 10

# Municipal naming carries a formality that reads badly as a map label. The prefixes below are
# stripped to get a short label, and the rule is written into the artifact so the checker applies
# the artifact's own rule instead of keeping a copy that could drift.
PLACE_PREFIXES = (
    "городской округ город ",
    "городской округ ",
    "муниципальный район ",
    "городское поселение ",
    "сельское поселение ",
)


def shorten(full: str) -> str:
    low = full.lower()
    for prefix in PLACE_PREFIXES:
        if low.startswith(prefix.lower()):
            rest = full[len(prefix) :].strip()
            return rest or full
    return full


def minzoom_for(area_km2: float) -> int:
    """Regions by size: the bigger the region, the earlier its name can be read."""
    if area_km2 >= 500_000:
        return 3
    if area_km2 >= 100_000:
        return 4
    if area_km2 >= 20_000:
        return 5
    if area_km2 >= 5_000:
        return 6
    if area_km2 >= 1_000:
        return 7
    return 8


def main() -> int:
    if not MASTER.exists():
        print(f"FAIL {MASTER} missing — run scripts/build_master.py first")
        return 1

    master = pd.read_parquet(MASTER)
    master["geom"] = master["geometry"].map(shapely.from_wkb)
    master["area_km2"] = master["official_land_area_km2"].where(
        master["official_land_area_km2"].notna(), master["osm_area_km2"]
    )

    inflation = pd.read_csv(INFLATION)
    known_regions = set(inflation["region_name"])

    regions: list[dict] = []
    unlanded = 0
    for name, block in master.groupby("region_name", sort=True):
        lon = block["centroid_lon"].to_numpy()
        lat = block["centroid_lat"].to_numpy()
        w = block["area_km2"].fillna(0.0).to_numpy()
        if w.sum() <= 0:
            cand = (float(lon.mean()), float(lat.mean()))
        else:
            cand = (
                float((lon * w).sum() / w.sum()),
                float((lat * w).sum() / w.sum()),
            )

        geoms = list(block["geom"])
        tree = STRtree(geoms)
        point = shapely.Point(cand[0], cand[1])
        hit = tree.query(point)
        inside = geoms[hit[0]].contains(point) if len(hit) else False

        if inside:
            lng, lat = cand
        else:
            unlanded += 1
            biggest = block.loc[block["area_km2"].idxmax()]
            rp = biggest["geom"].representative_point()
            lng, lat = float(rp.x), float(rp.y)

        regions.append(
            {
                "name": str(name),
                "lng": round(lng, 5),
                "lat": round(lat, 5),
                "minzoom": minzoom_for(float(block["area_km2"].sum())),
                "n_territories": int(len(block)),
                "area_km2": round(float(block["area_km2"].sum()), 2),
                "candidate_landed": bool(inside),
                "in_inflation_table": bool(name in known_regions),
            }
        )

    places: list[dict] = []
    urban = master[
        master["municipal_district_name"].str.contains("городской округ", case=False, na=False)
        & (master["area_km2"] <= PLACE_MAX_AREA_KM2)
    ].sort_values(["area_km2", "tid"], kind="mergesort")
    for row in urban.head(PLACE_LIMIT).itertuples():
        rp = row.geom.representative_point()
        full = str(row.municipal_district_name)
        places.append(
            {
                "name": full,
                "short": shorten(full),
                "lng": round(float(rp.x), 5),
                "lat": round(float(rp.y), 5),
                "minzoom": PLACE_MINZOOM,
                "tid": int(row.tid),
                "region": str(row.region_name),
                "area_km2": round(float(row.area_km2), 2),
            }
        )

    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    payload = {
        "generated_by": "scripts/build_map_labels.py",
        "provenance": {
            "master": str(MASTER.relative_to(ROOT)),
            "regions_cross_checked_with": str(INFLATION.relative_to(ROOT)),
            "display_geometry": summary.get("geometry_file"),
            "territories_in_master": int(len(master)),
        },
        "regions": regions,
        "places": places,
        "place_name_rule": {"prefixes": list(PLACE_PREFIXES), "minzoom": PLACE_MINZOOM},
        "counts": {
            "regions": len(regions),
            "places": len(places),
            "regions_where_weighted_centroid_missed_own_land": unlanded,
        },
        "labels_by_zoom": {
            str(z): sum(1 for r in regions if r["minzoom"] <= z)
            + sum(1 for p in places if p["minzoom"] <= z)
            for z in range(3, 13)
        },
    }

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT}: {OUT.stat().st_size:,} bytes")
    print(f"regions {len(regions)} | places {len(places)} | unlanded centroids {unlanded}")
    print("labels by zoom:", payload["labels_by_zoom"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
