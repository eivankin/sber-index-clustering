#!/usr/bin/env python3
"""Write the artifact that quantifies duplicated municipalities, so the defect is a number, not an anecdote.

Some places arrive from OSM twice: two rows carry the same name in the same region because OSM keeps
version windows, and one of them is the older outline. The analytical master keeps both - dropping a
row there would silently change what the clustering sees - so the duplication is resolved where the
reader meets it: in search and in neighbour lists. This artifact says how many pairs there are, how
they differ, and which of the two rows carries spending data, so the display rule can be checked
against real cases rather than invented for one district.

Land area comparisons use the official Rosstat figure, never the raw OSM polygon area: the two rows
of the same place differ in OSM area by design, which is exactly what the area_suspect column records.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "territory_master.parquet"
OUT = ROOT / "artifacts/geo/duplicate_municipalities.json"


def _overlap_share(first, second) -> float:
    """How much of the smaller of two outlines the other one covers, as a share of that smaller one."""
    if first is None or second is None or first.is_empty or second.is_empty:
        return -1.0
    smaller = first.area if first.area <= second.area else second.area
    if smaller <= 0:
        return -1.0
    return first.intersection(second).area / smaller


def _prefer(part: pd.DataFrame) -> int:
    """The row the display should offer: spending data first, then the longest version window,
    then the larger official area, then the lowest id."""
    ordered = part.assign(
        _window=(part["year_to_max"] - part["year_from_min"]).fillna(-1),
    ).sort_values(
        ["has_consumption", "_window", "official_land_area_km2", "tid"],
        ascending=[False, False, False, True],
    )
    return int(ordered.iloc[0]["tid"])


def main() -> int:
    if not MASTER.exists():
        raise SystemExit(
            f"master missing: {MASTER} (it is a build product and is not tracked in git)"
        )
    m = pd.read_parquet(MASTER)
    geo = gpd.GeoSeries.from_wkb(m.geometry).set_crs("EPSG:4326")
    m = m.assign(_geo=geo)

    grouped = m.groupby(["municipal_district_name", "region_name"], sort=True)
    pairs: list[dict] = []
    for (name, region), part in grouped:
        if len(part) < 2:
            continue
        rows = part.sort_values("tid")
        windows = [(int(r.year_from_min), int(r.year_to_max)) for r in rows.itertuples()]
        areas = [
            None if pd.isna(r.official_land_area_km2) else float(r.official_land_area_km2)
            for r in rows.itertuples()
        ]
        osm_areas = [float(r.osm_area_km2) for r in rows.itertuples()]
        spending = [bool(r.has_consumption) for r in rows.itertuples()]
        overlaps = []
        iou = []
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                first, second = rows.iloc[a]["_geo"], rows.iloc[b]["_geo"]
                overlaps.append(_overlap_share(first, second))
                union = first.union(second)
                if union.area > 0:
                    iou.append(first.intersection(second).area / union.area)
        area_gap = -1.0
        usable = [a for a in areas if a is not None]
        if len(usable) >= 2:
            area_gap = (max(usable) - min(usable)) / min(usable)
        osm_gap = -1.0
        positive = [a for a in osm_areas if a > 0]
        if len(positive) >= 2:
            osm_gap = (max(positive) - min(positive)) / min(positive)
        pairs.append(
            {
                "name": str(name),
                "region": str(region),
                "tids": [int(t) for t in rows["tid"]],
                "oktmo": sorted({str(v) for v in rows["oktmo_norm"]}),
                "version_windows": [{"from": w[0], "to": w[1]} for w in windows],
                "windows_differ": len(set(windows)) == len(windows),
                "official_area_km2": areas,
                "area_difference_share": None if area_gap < 0 else round(area_gap, 4),
                "osm_area_km2": [round(a, 1) for a in osm_areas],
                "osm_area_difference_share": None if osm_gap < 0 else round(osm_gap, 4),
                "overlap_share": [round(o, 4) for o in overlaps],
                "iou": [round(v, 4) for v in iou],
                "with_spending": sum(spending),
                "has_spending": spending,
                "only_one_row_has_spending": sum(spending) == 1,
                "preferred_tid": _prefer(rows),
            }
        )

    payload = {
        "generated_by": "scripts/build_duplicate_territories.py",
        "what_this_is": (
            "Municipalities that appear twice under one name in one region, with how the copies "
            "differ: version windows, official area, overlap, and which copy carries spending data."
        ),
        "pair_rule": "одна и та же (название, регион) встречается в мастере чаще одного раза",
        "resolution_rule": (
            "в поиске и в списках похожих показывается одна запись на (название, регион); "
            "выбирается строка с данными о тратах, затем самое длинное окно версий, "
            " затем большая официальная площадь, затем наименьший id"
        ),
        "master": str(MASTER.relative_to(ROOT)) + " (build product, not tracked in git)",
        "area_note": "площади берутся официальные, от Росстата; у копий одного места они совпадают, "
        "поэтому расхождение площадей считается по площади полигона OSM - именно оно порождает "
        "флаг area_suspect",
        "total_rows": int(len(m)),
        "distinct_places": int(len(m) - sum(len(p["tids"]) - 1 for p in pairs)),
        "pairs": len(pairs),
        "extra_rows": int(sum(len(p["tids"]) - 1 for p in pairs)),
        "pairs_with_different_windows": sum(1 for p in pairs if p["windows_differ"]),
        "pairs_area_over_one_percent": sum(
            1 for p in pairs if (p["area_difference_share"] or 0) > 0.01
        ),
        "pairs_osm_area_over_one_percent": sum(
            1 for p in pairs if (p["osm_area_difference_share"] or 0) > 0.01
        ),
        "pairs_overlapping_over_ninety": sum(1 for p in pairs if max(p["overlap_share"]) > 0.90),
        "pairs_iou_over_ninety": sum(1 for p in pairs if max(p["iou"]) > 0.90),
        "overlap_note": (
            "перекрытие считается двумя способами: доля меньшей копии, покрытой большей, и "
            "пересечение по объединению (IoU). Они отвечают на разные вопросы, и числа по ним "
            "не совпадают намеренно"
        ),
        "pairs_one_row_with_spending": sum(1 for p in pairs if p["only_one_row_has_spending"]),
        "pairs_without_spending": sum(1 for p in pairs if p["with_spending"] == 0),
        "cases": pairs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"duplicated municipalities: {payload['pairs']} pairs, {payload['extra_rows']} extra rows "
        f"of {payload['total_rows']} rows, {payload['distinct_places']} distinct places"
    )
    print(
        f"different version windows {payload['pairs_with_different_windows']} | "
        f"official area over 1% {payload['pairs_area_over_one_percent']} | "
        f"OSM polygon area over 1% {payload['pairs_osm_area_over_one_percent']} | "
        f"overlap over 90% {payload['pairs_overlapping_over_ninety']} | "
        f"IoU over 90% {payload['pairs_iou_over_ninety']} | "
        f"one row with spending {payload['pairs_one_row_with_spending']} | "
        f"no spending {payload['pairs_without_spending']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
