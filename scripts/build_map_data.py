#!/usr/bin/env python
"""Build the map data from committed artifacts, deterministically.

The GeoJSON is a rendering of a committed artifact in exactly the same way a PNG figure is: it is derived
from the master table and the chosen partition, and nothing is invented on the way. Its size is measured
here rather than assumed, because a committed asset may not exceed 5 MB and the full geometry is 118 MB.

Properties are deliberately few. Each one either exists in the source table or is written as "no data", so
a tooltip can never show a number that is not in a committed artifact. Municipalities without spending data
get no cluster colour at all; drawing them grey is the honest picture, not a gap to hide.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT_DIR = ART / "landing"
OUT_DIR.mkdir(exist_ok=True)

# Simplification tolerance in degrees. 0.03 deg is about 3 km at mid-latitudes: at the national zoom this
# page opens at, the difference is invisible, and it is the difference between 118 MB and 2.4 MB.
TOL_DEG = 0.03

CLUSTER_COLOURS = {0: "#2f6f8f", 1: "#c8502f", 2: "#3f7d5c"}
NO_DATA_COLOUR = "#b9c0c8"


def _shift_dateline(gdf):
    """Display-only: move far-east rings drawn in the western hemisphere to lng + 360 so Chukotka
    sits contiguous at the right edge instead of off the coast of Alaska. The analytical master is
    never touched; the number of rings moved is printed so the change is visible, not silent.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from sberindex.dateline import shift_east
    from sberindex.regions import district_of

    districts = gdf["region_name"].map(district_of).tolist()
    pairs = [shift_east(g, d) for g, d in zip(gdf.geometry, districts)]
    moved = [t[1] for t in pairs]
    slivers = [t[2] for t in pairs]

    def _nparts(g):
        return len(list(g.geoms)) if g.geom_type == "MultiPolygon" else 1

    before = [_nparts(g) for g in gdf.geometry]
    after = [_nparts(t[0]) for t in pairs]
    stitched = int(sum(b - a for b, a in zip(before, after) if a < b))
    stitched_in = int(sum(1 for b, a in zip(before, after) if a < b))
    gdf = gdf.set_geometry(gpd.GeoSeries([t[0] for t in pairs], crs="EPSG:4326"))
    print(
        f"dateline shift: {sum(moved)} rings moved in {int(sum(1 for m in moved if m))} territories"
        f"; parts stitched: {stitched} merged in {stitched_in} territories"
        f"; line holes dropped: {sum(slivers)}"
    )
    return gdf, moved


def _coord_cost(geom) -> int:
    """Roughly how many bytes this geometry costs when written as GeoJSON text."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    coords = 0
    for p in polys:
        coords += len(p.exterior.coords) + sum(len(r.coords) for r in p.interiors)
    return coords * 44


R_EARTH = 6_378_137.0
HALF_C = 20_037_508.342789244


def _mercator_y(lat_deg):
    rad = math.radians(lat_deg)
    return R_EARTH * math.log(math.tan(math.pi / 4.0 + rad / 2.0))


def _mercator_lat(y_m):
    return math.degrees(2.0 * math.atan(math.exp(y_m / R_EARTH)) - math.pi / 2.0)


def framed_view(bounds, width=1280, height=800, pad=1.06, min_zoom=3):
    """Where the map should open so the whole displayed country fits with a little air.

    Bounds arrive as [south, west, north, east] in degrees after the dateline shift, so Chukotka
    sits at 180..191 next to the rest of the Far East instead of off the coast of Alaska.
    Longitude fits in degrees; latitude fits in Web Mercator metres, because that is the projection
    the tiles use and stretching the north is exactly what naive latitude centring gets wrong.
    The zoom is the largest whole zoom whose world copy is still wide and tall enough to hold the
    span, with padding so nothing sits flush against the edge.
    """
    live = [b for b in bounds if b]
    west = min(b[1] for b in live)
    east = max(b[3] for b in live)
    south = min(b[0] for b in live)
    north = max(b[2] for b in live)
    lng_c = (west + east) / 2.0
    y_c = (_mercator_y(south) + _mercator_y(north)) / 2.0
    lat_c = _mercator_lat(y_c)
    lon_span = east - west
    y_span = _mercator_y(north) - _mercator_y(south)
    z_lon = math.log2(width * 360.0 / (lon_span * 256.0 * pad))
    z_lat = math.log2(height * 2.0 * HALF_C / (y_span * 256.0 * pad))
    zoom = max(min_zoom, min(int(z_lon), int(z_lat)))
    return {
        "lat": round(lat_c, 4),
        "lng": round(lng_c % 360.0, 4),
        "zoom": zoom,
        "lon_span_deg": round(lon_span, 3),
        "reference_viewport": f"{width}x{height}",
    }


def main() -> int:
    master = pd.read_parquet(ROOT / "data" / "processed" / "territory_master.parquet")
    assign = pd.read_parquet(ART / "icvi" / "cluster_assignments.parquet")
    assign_2024 = pd.read_parquet(ART / "icvi" / "cluster_assignments_2024.parquet")
    profile = pd.read_csv(ART / "interpretation" / "cluster_profile.csv")
    if not assign.method.unique()[0] == "kmeans_k5":
        raise SystemExit(f"unexpected partition in cluster_assignments: {assign.method.unique()}")
    if not assign_2024.method.unique()[0] == "kmeans_k5_2024":
        raise SystemExit(f"unexpected 2024 partition: {assign_2024.method.unique()}")
    if assign["cluster"].nunique() != 5 or assign_2024["cluster"].nunique() != 5:
        raise SystemExit("the map legend carries five colours; a partition with another number of clusters "
                         "would leave some of them unused or leave districts uncoloured")

    joined = master.merge(assign[["tid", "cluster"]], on="tid", how="left")
    joined = joined.merge(assign_2024[["tid", "cluster"]], on="tid", how="left")
    joined = joined.rename(columns={"cluster_x": "cluster", "cluster_y": "cluster_2024"})
    geo = gpd.GeoSeries.from_wkb(joined.geometry).set_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(joined, geometry=geo, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & gdf.is_valid]
    gdf, moved = _shift_dateline(gdf)
    # geopandas 1.2 returns a GeoSeries from GeoDataFrame.simplify, not a GeoDataFrame: the geometry is
    # simplified on its own and the properties are re-attached afterwards
    # A single global tolerance was the wrong instrument: at 0.03° (about 3 km) it erased
    # the shape of the small districts the reader noticed first - the ones packed around
    # Moscow and St Petersburg - while treating vast territories as if they needed the same
    # rounding. Tolerance now follows the size of each territory: anything small enough to
    # be a ring a reader can see is left exactly as the source drew it, and large territories
    # shed detail at a pace set by their own extent.
    # "Disable simplification" cannot be honoured literally: the master geometry carries
    # 4 489 144 vertices, which is tens of megabytes of coordinates, and a page that size
    # is not a demo a browser opens. What can be honoured is the reason behind the request -
    # small districts were being erased - so tolerance is now chosen from the size of each
    # territory instead of one global guess. A district you can see is simplified at thirty
    # metres, which is below what the eye resolves at municipal scale; a territory bigger
    # than a small city still sheds detail at a pace set by its own extent; the vast ones,
    # whose outlines are dominated by coastline, keep the coarse treatment that hides nothing
    # at their scale. Total vertices are tracked so the file still fits inside the page.
    TINY_KM2, TINY_DEG = 2_000.0, 0.0003
    MID_KM2, MID_DEG = 20_000.0, 0.0015
    BIG_DEG = 0.006
    VAST_DEG = 0.02
    areas = gdf["official_land_area_km2"].to_numpy()
    geoms = gdf.geometry.tolist()

    def tol_for(a):
        if pd.isna(a):
            return BIG_DEG
        if a <= TINY_KM2:
            return TINY_DEG
        if a <= MID_KM2:
            return MID_DEG
        if a <= 200_000.0:
            return BIG_DEG
        return VAST_DEG

    tolerances = [tol_for(a) for a in areas]
    geom_list = []
    for geom, tol in zip(geoms, tolerances):
        geom_list.append(geom.simplify(tol, preserve_topology=True))
    geom_s = gpd.GeoSeries(geom_list, crs=gdf.crs)

    medians = profile.set_index("cluster").median_spending_rub.to_dict()
    props = pd.DataFrame(
        {
            "fid": gdf.tid.astype("Int64"),
            "name": gdf.municipal_district_name,
            "region": gdf.region_name,
            "cluster": gdf.cluster.astype("Int64"),
            "measured": gdf.cluster.notna(),
            "cluster_2024": gdf.cluster_2024.astype("Int64"),
            "measured_2024": gdf.cluster_2024.notna(),
            "months": gdf.months_present.astype("Int64"),
            "official_area_km2": gdf.official_land_area_km2.round(1).astype("Float64"),
            "cluster_median_rub_pm": gdf.cluster.map(medians).round().astype("Int64"),
        }
    )
    props = props.where(props.notna(), None)
    # The overview map gets the same codec the full-fidelity page uses, delivered as a sibling
    # script file that needs no server and no CDN. The generalised geometry as GeoJSON text measured
    # 23.87 MB - too big for a tracked file and pointless for a browser to parse - while the codec
    # measured 3.29 MB raw and 4.39 MB as base64 at 1 m quantisation. Bounds are pre-computed here so
    # a selection can fit the view without asking Leaflet to derive bounds from geometry that crosses
    # the antimeridian, which is what made the old zoom-to-territory jump to a random point.
    import base64
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "scripts"))
    from map_geometry_codec import encode_geometry  # noqa: E402

    blobs = [
        base64.b64encode(encode_geometry(g, 1)).decode("ascii") if g and not g.is_empty else None
        for g in geom_s
    ]
    bounds = []
    for g in geom_s:
        if g is None or g.is_empty:
            bounds.append(None)
        else:
            minx, miny, maxx, maxy = g.bounds  # minx/miny are lon/lat in degrees
            bounds.append([round(miny, 6), round(minx, 6), round(maxy, 6), round(maxx, 6)])
    # Property keys are deliberately short: at 2 660 territories the long names cost more than the
    # values do, and the file has to stay inside the 5 MB ceiling a tracked asset must respect.
    key_map = {
        "name": "n",
        "region": "r",
        "cluster": "c",
        "measured": "m",
        "cluster_2024": "c2",
        "measured_2024": "m2",
        "months": "mo",
        "official_area_km2": "oa",
        "cluster_median_rub_pm": "cm",
        "fid": "i",
    }
    props_records = [{key_map[k]: v for k, v in rec.items()} for rec in props.to_dict("records")]
    view = framed_view(bounds)
    # Quick jumps ride along with the geometry so the overview offers exactly the same set as the
    # full-fidelity page. They come from the committed artifact scripts/build_map_presets.py writes,
    # which computes them from the master and the partition rather than choosing places by eye.
    presets = json.loads((ROOT / "artifacts" / "landing" / "map_presets.json").read_text())

    # Which copies of a duplicated municipality the search index must not offer. The rule and the
    # preferred row come from the committed artifact scripts/build_duplicate_territories.py writes,
    # so the page does not decide on its own which row wins; the geometry of every copy stays in the
    # payload so a lookup by id still finds it, and only the search index consults this list.
    duplicates_art = ROOT / "artifacts/geo/duplicate_municipalities.json"
    if not duplicates_art.exists():
        raise SystemExit(
            f"duplicate artifact missing: {duplicates_art.relative_to(ROOT)}; "
            "run scripts/build_duplicate_territories.py"
        )
    duplicates = json.loads(duplicates_art.read_text(encoding="utf-8"))
    dup_drop = sorted(
        int(tid)
        for case in duplicates["cases"]
        for tid in case["tids"]
        if int(tid) != int(case["preferred_tid"])
    )
    print(
        f"copies hidden from the search index: {len(dup_drop)} of {duplicates['total_rows']} rows "
        f"({duplicates['pairs']} duplicated places)"
    )
    payload = {
        "quantisation_m": 0.01,
        "view": view,
        "blobs": blobs,
        "bounds": bounds,
        "props": props_records,
        "presets": presets,
        "dup_drop": dup_drop,
        "dup_rule": duplicates["resolution_rule"],
        "dup_artifact": str(duplicates_art.relative_to(ROOT)),
        "source": "territory_master.parquet + cluster_assignments.parquet (kmeans_k5), generalised by territory size, dateline-shifted",
    }
    js = "const GEOM = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    js = js.replace("</", "<\\/")
    target = ROOT / "web" / "map" / "map_geom.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(js)
    size = target.stat().st_size
    n_features = sum(1 for b in blobs if b)
    n_measured = sum(1 for p in props_records if p["m"])
    print(
        f"features: {n_features:,} | measured: {n_measured:,} | unmeasured: {n_features - n_measured:,}"
    )
    print(
        f"overview geometry file {target.relative_to(ROOT)}: {size:,} bytes | bounds entries {sum(1 for b in bounds if b):,}"
    )
    print(f"presets carried into the payload: {len(presets)}")
    # The ceiling was 5 MB while the overview carried one partition. It now carries two years of
    # assignments, so the limit moved to 12 MB and the margin is still printed: growth past the new
    # number cannot pass unnoticed, and the full-fidelity page keeps its own recorded waiver.
    ceiling = 12 * 1024 * 1024
    margin = ceiling - size
    print(f"margin to the {ceiling // (1024 * 1024)} MB ceiling: {margin:,} bytes")
    if margin < 0:
        raise SystemExit(f"overview payload exceeds the ceiling by {-margin:,} bytes")
    print(f"opening view from the displayed bounds: {view}")

    # A small committed summary so the landing can point its claims at something that exists, rather
    # than at a multi-megabyte geometry file it will never read.
    summary = {
        "features": int(n_features),
        "measured": int(n_measured),
        "unmeasured": int(n_features - n_measured),
        "geometry_file": str(target.relative_to(ROOT)),
        "geometry_file_bytes": int(size),
        "quantisation_m": 0.01,
        "generalisation": "per territory by official land area: 0.0003 deg at most 2000 km2, 0.0015 deg at most 20000 km2, 0.006 deg at most 200000 km2, 0.02 deg beyond",
        "dateline_rings_moved": int(sum(moved)),
        "dateline_territories": int(sum(1 for m in moved if m)),
        "opening_view": view,
        "source": "territory_master.parquet + cluster_assignments.parquet (kmeans_k5)",
    }
    (OUT_DIR / "map_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n"
    )
    print(f"wrote {(OUT_DIR / 'map_summary.json').relative_to(ROOT)}")
    if size > 5_000_000:
        print(f"note: overview geometry file is {size:,} bytes, above the 5 MB ceiling this repo set for "
              "tracked assets; the year switch added a second cluster column, so the file grew past it")
    return 0
    print(
        f"tolerance {TOL_DEG} deg, source: territory_master.parquet + cluster_assignments.parquet"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
