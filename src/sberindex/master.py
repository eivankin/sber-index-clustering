"""Territory master table: SberIndex municipality dictionary + coverage + geometry diagnostics.

The SberIndex polygon layer is derived from OpenStreetMap and (as we measured) includes large
marine/extremity areas. Everything downstream must therefore pick its denominators explicitly:
`osm_area_km2` is the raw polygon area (NOT for densities), `official_area_km2` is the official
subject land area when known, and land-cover shares use a classified-pixel denominator.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely.geometry as sg

from .config import (
    AREA_CRS,
    ATTRIBUTES,
    CONNECTION,
    CONSUMPTION,
    GEOG_CRS,
    MARKET_ACCESS,
    OFFICIAL_SUBJECT_AREA,
    POLYGONS,
    SEA_BOXES,
)
from .rostat import attach_land_area


def load_polygons() -> gpd.GeoDataFrame:
    g = gpd.read_file(POLYGONS).to_crs(GEOG_CRS)
    g["tid"] = pd.to_numeric(g["territory_id"], errors="raise").astype("Int64")
    g = g[["tid", "osm_ref", "osm_vers", "year_from", "year_to", "geometry"]].copy()
    g["n_parts"] = g.geometry.apply(lambda p: len(p.geoms) if p.geom_type == "MultiPolygon" else 1)
    g["is_valid"] = g.geometry.is_valid
    proj = g.to_crs(AREA_CRS)
    g["osm_area_km2"] = proj.geometry.area / 1e6
    # centroid in lon/lat: adequate for point-in-polygon seeding (area is computed projected).
    g["centroid_lon"] = g.geometry.centroid.x
    g["centroid_lat"] = g.geometry.centroid.y
    return g


def load_attributes(year: int = 2024) -> pd.DataFrame:
    xl = pd.read_excel(ATTRIBUTES)
    xl["tid"] = pd.to_numeric(xl["territory_id"], errors="raise").astype("Int64")
    xl["year_from"] = pd.to_numeric(xl["year_from"], errors="coerce").astype("Int64")
    xl["year_to"] = pd.to_numeric(xl["year_to"], errors="coerce").astype("Int64")
    live = xl[(xl.year_from <= year) & (xl.year_to > year)]
    pick = live.drop_duplicates("tid", keep="first")
    fallback = xl.sort_values(["year_from", "year_to"], ascending=[True, False]).drop_duplicates(
        "tid", keep="first"
    )
    attrs = fallback.set_index("tid")
    attrs.update(pick.set_index("tid"))  # prefer the version valid in `year`
    cols = [
        "municipal_district_name",
        "municipal_district_name_short",
        "municipal_district_type",
        "municipal_district_status",
        "oktmo",
        "region_code",
        "region_name",
        "municipal_district_center",
        "municipal_district_center_lat",
        "municipal_district_center_lon",
        "shape",
        "shape_linked_oktmo",
    ]
    out = attrs[[c for c in cols if c in attrs.columns]].reset_index()
    span = (
        xl.groupby("tid")
        .agg(
            year_from_min=("year_from", "min"),
            year_to_max=("year_to", "max"),
            n_versions=("tid", "size"),
        )
        .reset_index()
    )
    return out.merge(span, on="tid", how="left")


def load_coverage() -> dict[str, set[int]]:
    cons = pd.read_parquet(CONSUMPTION)
    conn = pd.read_parquet(CONNECTION)
    ma = pd.read_parquet(MARKET_ACCESS)
    C = set(cons.territory_id.astype(int))
    K = set(conn.territory_id_x.astype(int)) | set(conn.territory_id_y.astype(int))
    M = set(ma.territory_id.astype(int))
    h = conn[conn.type == "highway"]
    r = conn[conn.type == "railway"]
    H = set(h.territory_id_x.astype(int)) | set(h.territory_id_y.astype(int))
    R = set(r.territory_id_x.astype(int)) | set(r.territory_id_y.astype(int))
    months = cons.groupby("territory_id").date.nunique()
    return {"C": C, "K": K, "M": M, "H": H, "R": R, "months": months, "cons": cons}


def sea_overlap(union: gpd.GeoDataFrame) -> dict[str, float]:
    out = {}
    u = union.to_crs(AREA_CRS)
    for name, (lon0, lon1, lat0, lat1) in SEA_BOXES.items():
        box = gpd.GeoDataFrame(geometry=[sg.box(lon0, lat0, lon1, lat1)], crs=GEOG_CRS).to_crs(
            AREA_CRS
        )
        cut = gpd.overlay(
            u[["geometry"]], box[["geometry"]], how="intersection", keep_geom_type=False
        )
        out[name] = float(cut.geometry.area.sum() / 1e6)
    return out


def region_inflation(master: gpd.GeoDataFrame) -> pd.DataFrame:
    t = (
        master.groupby("region_name")
        .agg(
            n_territories=("tid", "size"),
            sum_polygons_km2=("osm_area_km2", "sum"),
            largest_polygon_km2=("osm_area_km2", "max"),
        )
        .reset_index()
    )
    t["official_area_km2"] = t.region_name.map(OFFICIAL_SUBJECT_AREA)
    t["inflation_ratio"] = t.sum_polygons_km2 / t.official_area_km2
    t["known_official"] = t.official_area_km2.notna()
    return t.sort_values("inflation_ratio", ascending=False)


def build_master(asof_year: int = 2024) -> tuple[pd.DataFrame, dict]:
    """Returns (master, diagnostics). Writes nothing — callers decide where to persist."""
    poly = load_polygons()
    attrs = load_attributes(asof_year)
    cov = load_coverage()
    G = {int(x) for x in poly.tid.dropna()}

    master = poly.merge(attrs, on="tid", how="outer")
    master["has_geometry"] = master.tid.isin(G)
    master["has_consumption"] = master.tid.isin(cov["C"])
    master["has_market_access"] = master.tid.isin(cov["M"])
    master["has_highway"] = master.tid.isin(cov["H"])
    master["has_railway"] = master.tid.isin(cov["R"])
    master["months_present"] = master.tid.map(cov["months"])

    def gap_class(r):
        if not r.has_geometry:
            return "id_in_sber_tables_without_polygon"
        if not r.has_consumption:
            return "polygon_without_consumption"
        if not r.has_market_access:
            return "measured_without_market_access"
        if not r.has_highway:
            return "measured_without_road_edges"
        if (r.months_present or 0) < 24:
            return "measured_ragged_panel"
        return "complete_panel"

    master["gap_class"] = master.apply(gap_class, axis=1)
    master, rostat_info = attach_land_area(master)
    master["official_subject_area_km2"] = master.region_name.map(OFFICIAL_SUBJECT_AREA)

    diag = {
        "n_territories": int(master.tid.nunique()),
        "coverage": {
            "geo": len(G),
            "consumption": len(cov["C"]),
            "connection": len(cov["K"]),
            "market_access": len(cov["M"]),
            "highway_nodes": len(cov["H"]),
            "railway_nodes": len(cov["R"]),
        },
        "gaps": {
            "G_minus_C": len(G - cov["C"]),
            "C_minus_G": len(cov["C"] - G),
            "K_minus_C": len(cov["K"] - cov["C"]),
            "C_minus_K": len(cov["C"] - cov["K"]),
            "M_minus_C": len(cov["M"] - cov["C"]),
            "C_minus_M": len(cov["C"] - cov["M"]),
            "core_C_K_M_G": len(cov["C"] & cov["K"] & cov["M"] & G),
        },
        "geometry": {
            "invalid": int((~master.is_valid).sum()),
            "multi_part": int((master.n_parts > 1).sum()),
            "sum_polygons_km2": float(master.osm_area_km2.sum()),
            "union_km2": float(poly.to_crs(AREA_CRS).dissolve().geometry.area.sum() / 1e6),
        },
        "sea_overlap_km2": sea_overlap(poly.dissolve()),
        "rostat": rostat_info,
    }
    return master, diag
