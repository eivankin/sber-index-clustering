#!/usr/bin/env python3
"""Bake a vector choropleth of a clustering answer, with no basemap under it.

The teaser and the decision maps are images the reader cannot zoom into, so the geometry may be
generalised and rounded here without anyone being able to find the rounding. The interactive map is
the opposite case and gets the lossless geometry instead.

One territory becomes one path, projected into Web Mercator so the shapes match the interactive map
exactly. The colours come from the same palette the map page uses, so a colour means the same type in
both places. Nothing is invented: the partition comes from a committed assignment file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "processed" / "territory_master.parquet"

COLOURS = {
    0: "#3f7d5c",
    1: "#c8502f",
    2: "#2f6f8f",
    3: "#b08d3f",
    4: "#6a5a7e",
}
GREY = "#c2c7ce"

# A pale-to-deep scale for "how much a district spends". Five steps, chosen so the eye reads order
# rather than trying to compare two similar shades.
MONEY_RAMP = ["#dfe6ee", "#b9c9dd", "#8fa8c8", "#5f7fa8", "#2f5476"]

HALF_C = 20037508.342789244


def _mercator_xy(lon: float, lat: float):
    x = lon * HALF_C / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * HALF_C / math.pi
    return x, y


def _project(geom):
    def point(lon, lat):
        return _mercator_xy(lon, lat)

    def ring(coords):
        # A ring may carry a third ordinate from wherever it was built; only the two that place it
        # on the map are used.
        return [point(c[0], c[1]) for c in coords]

    if geom.geom_type == "Polygon":
        return [[ring(geom.exterior.coords)] + [ring(r.coords) for r in geom.interiors]]
    return [poly for part in geom.geoms for poly in _project(part)]


def _path(polygons, scale, dx, dy):
    """One SVG path for one territory, in projected pixel units."""
    out = []
    for rings in polygons:
        for ring in rings:
            parts = []
            for pt in ring:
                parts.append(f"{(pt[0] - dx) * scale:.2f},{(dy - pt[1]) * scale:.2f}")
            out.append("M" + "L".join(parts) + "Z")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path,
                        help="parquet with tid and cluster; the shipped partition by default")
    parser.add_argument("--demo-state", type=int,
                        help="index into artifacts/landing/demo_states.json instead of a partition")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ramp-column", default=None,
                        help="colour by this continuous column instead of by cluster label")
    parser.add_argument("--tol-deg", type=float, default=0.02,
                        help="one global generalisation tolerance, for a picture nobody zooms into")
    parser.add_argument("--width", type=int, default=1200)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from sberindex.dateline import shift_east
    from sberindex.regions import district_of

    if args.demo_state is not None:
        demo = json.loads((ROOT / "artifacts" / "landing" / "demo_states.json").read_text())
        state = demo["states"][args.demo_state]
        assignments = pd.read_parquet(ROOT / state["assignments"])
        caption = f"{state['space_title']} · k={state['k']}"
    else:
        assignments = pd.read_parquet(
            args.assignments or ROOT / "artifacts" / "icvi" / "cluster_assignments.parquet"
        )
        caption = ""

    master = pd.read_parquet(MASTER)
    geometries = [
        shapely.from_wkb(raw) if isinstance(raw, (bytes, bytearray)) else raw
        for raw in master.geometry
    ]
    usable = [g is not None and not g.is_empty and g.is_valid for g in geometries]
    keep = master[[bool(u) for u in usable]]
    geoms = [g for g, u in zip(geometries, usable) if u]
    districts = keep["region_name"].map(district_of).tolist()
    shifted = [shift_east(g, d)[0] for g, d in zip(geoms, districts)]
    gdf = gpd.GeoDataFrame(
        keep.reset_index(drop=True),
        geometry=gpd.GeoSeries(shifted, crs="EPSG:4326"),
        crs="EPSG:4326",
    ).merge(assignments[["tid", "cluster"]], on="tid", how="left")
    gdf["geometry"] = gdf.geometry.simplify(args.tol_deg, preserve_topology=True)

    ramp_cuts = None
    if args.ramp_column and args.ramp_column not in gdf.columns:
        # The value to ramp by is not carried by the master table, so it is joined from the same
        # feature space the clustering read. No number is invented for the picture.
        space = pd.read_parquet(ROOT / "data" / "processed" / "feature_space.parquet")
        gdf = gdf.merge(space[["tid", args.ramp_column]], on="tid", how="left")
    if args.ramp_column:
        if args.ramp_column not in gdf.columns:
            raise SystemExit(f"no column {args.ramp_column!r} to ramp by")
        series = pd.to_numeric(gdf[args.ramp_column], errors="coerce").dropna()
        ramp_cuts = [float(series.quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]

    projected = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            projected.append(None)
            continue
        projected.append(_project(geom))

    xs = [pt[0] for polys in projected if polys for poly in polys for ring in poly for pt in ring]
    ys = [pt[1] for polys in projected if polys for poly in polys for ring in poly for pt in ring]
    west, east = min(xs), max(xs)
    top, bottom = max(ys), min(ys)
    # In Web Mercator y grows northward, so the height of the picture is the top minus the bottom,
    # not the other way round; writing it the other way gave a negative canvas height.
    span = max(east - west, top - bottom)
    scale = args.width / span
    width = round(args.width)
    height = round((top - bottom) * scale)
    dx, dy = west, top
    pad = 6

    paths = []
    rows_in_order = list(gdf.itertuples())
    for polys, row in zip(projected, rows_in_order):
        cluster = row.cluster
        if polys is None:
            continue
        if args.ramp_column:
            value = getattr(row, args.ramp_column, None)
            if value is None or pd.isna(value):
                fill = GREY
            else:
                step = 0
                while step < len(ramp_cuts) and value > ramp_cuts[step]:
                    step += 1
                fill = MONEY_RAMP[min(step, len(MONEY_RAMP) - 1)]
        else:
            fill = COLOURS.get(int(cluster), GREY) if cluster is not None and not pd.isna(cluster) else GREY
        paths.append(f'<path fill="{fill}" d="{_path(polys, scale, dx, dy)}"/>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width + 2 * pad} '
        f'{height + 2 * pad}" width="{width + 2 * pad}" height="{height + 2 * pad}">\n'
        f'<!-- {caption or "shipped partition"} · geometry generalised at {args.tol_deg} deg and '
        'projected into Web Mercator, the same projection the interactive map uses; no basemap '
        'under it, so nothing third party is drawn here. -->\n'
        f'<!-- No background rectangle: the picture sits on whatever the page paints, so it does\n'
        '       not arrive with its own white field that fights the page background. -->\n'
        + "\n".join(paths)
        + "\n</svg>\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(
        f"baked choropleth: {args.out} {args.out.stat().st_size / 1e6:.2f} MB · paths {len(paths):,} "
        f"· {width + 2 * pad}x{height + 2 * pad} · {caption or 'shipped partition'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
