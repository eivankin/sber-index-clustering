#!/usr/bin/env python3
"""Lossless map geometry: every coordinate exactly as the source holds it, nothing simplified.

Two losses were rejected by the reader. The codec rounds degrees onto a metre grid, which shows up
as staircases once you zoom. Douglas-Peucker then removes vertices, which cuts straight across
places the reader knows - and at full zoom a cut is far more visible than a rounding. For the
interactive map neither is acceptable, so this script writes what the master table holds: the
64-bit floats, in the source's own order, with the ring structure kept so a browser can rebuild
the polygons without decoding anything.

    web/map/map_geom.bin     float64 little-endian, longitude then latitude, one run per ring
    web/map/map_index.json   which vertices form which ring, which rings form which territory,
                             the properties, and precomputed bounds

The baked renders keep using the simplified and quantised overview - there the reader cannot zoom,
so a rounding is not something they can find. This file is for the map you can zoom into.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "processed" / "territory_master.parquet"
ASSIGN = ROOT / "artifacts" / "icvi" / "cluster_assignments.parquet"
ASSIGN_2024 = ROOT / "artifacts" / "icvi" / "cluster_assignments_2024.parquet"
FEATURES = ROOT / "data" / "processed" / "feature_space.parquet"
ADDED = ROOT / "data" / "processed" / "covariates_added.parquet"
GEE_2024 = ROOT / "data" / "processed" / "covariates_gee_2024.parquet"
PROFILE = ROOT / "artifacts" / "interpretation" / "cluster_profile.csv"
BIN_OUT = ROOT / "web" / "map" / "map_geom.bin"
INDEX_OUT = ROOT / "web" / "map" / "map_index.json"

FAR_EAST = "Дальневосточный федеральный округ"


def load_geometries(master: pd.DataFrame):
    out = []
    for raw in master.geometry:
        out.append(shapely.from_wkb(raw) if isinstance(raw, (bytes, bytearray)) else raw)
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from sberindex.dateline import shift_east

    master = pd.read_parquet(MASTER)
    geometries = load_geometries(master)
    valid = [g is not None and not g.is_empty and g.is_valid for g in geometries]
    keep = master[[bool(v) for v in valid]]
    geoms = [g for g, v in zip(geometries, valid) if v]

    moved_total = 0
    shifted = []
    from sberindex.regions import district_of
    districts = keep["region_name"].map(district_of).tolist()
    for geom, district in zip(geoms, districts):
        # Qualification is the federal district the region name resolves to, not a guess from the
        # shape: districts lying wholly west of the antimeridian, such as Провиденский at -177,
        # only qualify through the Far Eastern district. Without that they stay off Alaska.
        new_geom, rings_moved, _ = shift_east(geom, district)
        moved_total += rings_moved
        shifted.append(new_geom)
    gdf = gpd.GeoDataFrame(keep.reset_index(drop=True), geometry=gpd.GeoSeries(shifted, crs="EPSG:4326"),
                           crs="EPSG:4326")

    assign = pd.read_parquet(ASSIGN)
    gdf = gdf.merge(assign[["tid", "cluster"]], on="tid", how="left")
    assign_2024 = pd.read_parquet(ASSIGN_2024)
    gdf = gdf.merge(
        assign_2024[["tid", "cluster"]].rename(columns={"cluster": "cluster_2024"}),
        on="tid",
        how="left",
    )
    medians = {}
    if PROFILE.exists():
        profile = pd.read_csv(PROFILE)
        medians = profile.set_index("cluster").median_spending_rub.to_dict()

    # What the reader is offered when they click a district is not only which cluster it belongs to,
    # but the values that put it there. These are the columns the written decision names, taken from
    # the same feature space the clustering ran on, so the bars on the page are the numbers the
    # partition was built from rather than a paraphrase of them.
    space_columns = json.loads((ROOT / "artifacts" / "validation" / "k_decision.json").read_text())[
        "space"
    ]["columns"]
    features = pd.read_parquet(FEATURES)
    values_by_tid = {
        int(tid): {column: None if pd.isna(value) else float(value) for column, value in record.items()}
        for tid, record in features.set_index("tid")[space_columns].to_dict("index").items()
        if tid is not None
    }

    # The 2024 reading of the same district. Only land and greenness moved - the written decision
    # freezes night lights at 2023, elevation has no refresh, and the shape of spending has no 2024
    # source in this build. Carrying the second year makes the switch honest: the bars the reader
    # sees are the values that produced the colour they are looking at.
    added = pd.read_parquet(ADDED)
    gee24 = pd.read_parquet(GEE_2024)
    land_2024 = {c: c.replace("@2024", "") for c in added.columns if c.endswith("@2024")}
    land_by_tid_2024 = {
        int(tid): row for tid, row in added.set_index("tid")[list(land_2024)].to_dict("index").items()
        if tid is not None
    }
    ndvi_by_tid_2024 = {
        int(tid): (None if pd.isna(value) else float(value))
        for tid, value in gee24.set_index("tid")["modis_ndvi_mean"].to_dict().items()
        if tid is not None
    }

    shares_2024 = pd.read_csv(ROOT / "artifacts" / "validation" / "spending_2024_shares.csv")
    shares_by_tid = {
        int(tid): row
        for tid, row in shares_2024.set_index("tid")[
            [f"{category}_share_2024" for category in
             ("Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт")]
        ].to_dict("index").items()
        if tid is not None
    }

    def values_for_2024(tid):
        base = values_by_tid.get(tid)
        if base is None:
            return None
        land = land_by_tid_2024.get(tid, {})
        share = shares_by_tid.get(tid, {})
        out = dict(base)
        for column, source in (
            ("dynamic_world_bare", "dw_bare@2024"),
            ("dynamic_world_built", "dw_built@2024"),
            ("dynamic_world_crops", "dw_crops@2024"),
            ("dynamic_world_grass", "dw_grass@2024"),
            ("dynamic_world_trees", "dw_trees@2024"),
        ):
            value = land.get(source)
            out[column] = None if value is None or pd.isna(value) else float(value)
        ndvi = ndvi_by_tid_2024.get(tid)
        out["modis_ndvi_mean"] = ndvi
        # The shape of spending is a 2024 value too. Only lights and height stay where they were:
        # lights by the written decision, height because the ground does not move.
        for category in ("Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"):
            value = share.get(f"{category}_share_2024")
            out[f"{category}_share"] = None if value is None or pd.isna(value) else float(value)
        return out

    # Which spending median belongs to the 2024 colour. The number quoted next to a cluster is the
    # median of that cluster's territories, so when the cluster changes the median has to change too.
    spending = features.set_index("tid")["total_mean_rub"]
    by_cluster_2024 = pd.DataFrame(
        {
            "spending": spending,
            "cluster_2024": assign_2024.set_index("tid")["cluster"],
        }
    ).dropna()
    medians_2024 = {
        int(cluster): int(round(group["spending"].median()))
        for cluster, group in by_cluster_2024.groupby("cluster_2024")
    }

    flat = []
    features = []
    cursor = 0
    for tid, geom in zip(gdf.tid.tolist(), gdf.geometry.tolist()):
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        polys = []
        for part in parts:
            rings = []
            for ring in [part.exterior, *part.interiors]:
                coords = list(ring.coords)
                start = cursor
                for lon, lat in coords:
                    flat.append(lon)
                    flat.append(lat)
                cursor += len(coords) * 2
                rings.append([start, len(coords)])
            polys.append(rings)
        minx, miny, maxx, maxy = geom.bounds
        features.append({"tid": int(tid), "polys": polys, "bounds": [miny, minx, maxy, maxx]})

    props = []
    for row, cluster, cluster_2024 in zip(
        gdf.itertuples(), gdf.cluster.tolist(), gdf.cluster_2024.tolist()
    ):
        median = medians.get(int(cluster)) if cluster is not None and not pd.isna(cluster) else None
        props.append(
            {
                "name": row.municipal_district_name,
                "region": row.region_name,
                "cluster": None if pd.isna(cluster) else int(cluster),
                "measured": not pd.isna(cluster),
                "cluster_2024": None if pd.isna(cluster_2024) else int(cluster_2024),
                "measured_2024": not pd.isna(cluster_2024),
                "months": None if pd.isna(row.months_present) else int(row.months_present),
                "official_area_km2": None if pd.isna(row.official_land_area_km2)
                else round(float(row.official_land_area_km2), 1),
                "cluster_median_rub_pm": None if median is None else int(round(median)),
                "cluster_median_rub_pm_2024": medians_2024.get(int(cluster_2024))
                if cluster_2024 is not None and not pd.isna(cluster_2024) else None,
                "values": values_by_tid.get(int(row.tid), None),
                "values_2024": values_for_2024(int(row.tid)),
            }
        )

    BIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    BIN_OUT.write_bytes(struct.pack(f"<{len(flat)}d", *flat))
    # The opening view, the quick jumps and the duplicate rows the search must not offer all come
    # from committed artifacts and from the geometry's own bounds, so the page never chooses a
    # place by eye and never decides on its own which copy of a duplicated district wins.
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_map_data import framed_view  # noqa: E402

    bounds = [f["bounds"] for f in features if f["bounds"]]
    view = framed_view(bounds)
    presets = json.loads((ROOT / "artifacts" / "landing" / "map_presets.json").read_text())
    duplicates = json.loads((ROOT / "artifacts" / "geo" / "duplicate_municipalities.json").read_text())
    dup_drop = sorted(
        int(tid)
        for case in duplicates["cases"]
        for tid in case["tids"]
        if int(tid) != int(case["preferred_tid"])
    )
    INDEX_OUT.write_text(json.dumps(
        {
            "format": "float64 little-endian; longitude then latitude; one run per ring",
            "quantisation": "none",
            "simplification": "none",
            "vertex_pairs": len(flat) // 2,
            "features": features,
            "props": props,
            "view": view,
            "presets": presets,
            "dup_drop": dup_drop,
        },
        ensure_ascii=False,
    ))
    print(
        f"lossless geometry: {BIN_OUT.name} {BIN_OUT.stat().st_size / 1e6:.1f} MB · "
        f"{INDEX_OUT.name} {INDEX_OUT.stat().st_size / 1e6:.2f} MB · "
        f"vertices {len(flat) // 2:,} · territories {len(features):,} · "
        f"dateline rings moved {moved_total:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
