"""Move territories that cross the antimeridian to the east edge of the map, for display only.

The analytical master stores every longitude inside [-180, 180]. For Chukotka that is correct
arithmetic and terrible cartography: the parts of a district that lie east of the antimeridian end
up at -177, which puts them off the coast of Alaska on a map centred at 71.5 E - so the Far East
appears cut off, or, in a figure, shoved to the far left. This module moves only the rings that
are clearly far-east territory drawn in the western hemisphere: a ring whose largest longitude is
below -100, in a territory that also reaches past 150 E, gets 360 added to every longitude. Then
the two halves of the same district sit next to each other at the right edge of the map.

It is a display transform. Nothing here touches data/processed/territory_master.parquet, the
feature panel or any clustering artifact, and it returns the number of rings it moved so the
figure and map builders can report the change instead of hiding it.

Moving the rings is only half the job. Once the two halves of a district sit side by side they
still get drawn as two polygons, which paints a line straight through the middle of the district at
the antimeridian and makes it look like two separate districts. So the parts of a qualifying
territory are also merged where they already touch. Measured on the master: городской округ
Эгвекинот has four parts, two of which touch exactly at the antimeridian (distance 0.0 m) and
merge into two; Анадырский муниципальный район has two parts that touch and merge into one.
The other Far-East territories whose parts do not touch keep their gaps - those are real geography,
islands and separated blocks, and this module never bridges a real gap.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

WEST_LIMIT = -100.0
FAR_EAST_ANCHOR = 150.0
FAR_EAST_DISTRICT = "Far Eastern"

# A degenerate ring is one that encloses no land at all. Measured by scripts/check_map_full.py on the
# merged outline of городской округ Эгвекинот, before the module drops it: a hole of 5 vertices, area
# 0.000 km2, bounding box 1.2 m wide and 69 307.2 m tall, running along longitude 180 to within 2.2 m -
# a line, not a lake. Drawn, it paints the thin ring the reader complained about. Anything below this
# area is such a line.
HOLE_EPS_DEG2 = 1e-9


def _lon_span(ring):
    lons = [c[0] for c in ring.coords]
    return min(lons), max(lons)


def reaches_far_east(geom) -> bool:
    """True when any ring of this territory extends past 150 E, i.e. it is Far-East territory."""
    if geom is None or geom.is_empty:
        return False
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in parts:
        for ring in [p.exterior, *p.interiors]:
            if _lon_span(ring)[1] > FAR_EAST_ANCHOR:
                return True
    return False


def _drop_line_holes(geom):
    """Drop interior rings that enclose no land. They carry no area and only paint a line.

    Returns (geometry, holes_dropped). Real lakes and enclaves keep their rings: the test is area.
    """
    if geom is None or geom.is_empty:
        return geom, 0
    if geom.geom_type == "Polygon":
        keep = [r for r in geom.interiors if r.area > HOLE_EPS_DEG2]
        return Polygon(geom.exterior, keep), len(list(geom.interiors)) - len(keep)
    if geom.geom_type == "MultiPolygon":
        rebuilt, dropped = [], 0
        for p in geom.geoms:
            keep = [r for r in p.interiors if r.area > HOLE_EPS_DEG2]
            dropped += len(list(p.interiors)) - len(keep)
            rebuilt.append(Polygon(p.exterior, keep))
        out = MultiPolygon(rebuilt) if len(rebuilt) > 1 else rebuilt[0]
        return out, dropped
    return geom, 0


def stitch(geom):
    """Merge the parts of a territory that already touch, so a shared boundary stops being drawn.

    Only parts that touch are merged - a union does that by construction, so a genuine gap between
    two blocks is preserved. Returns (geometry, parts_before, parts_after).
    """
    if geom is None or geom.is_empty or geom.geom_type != "MultiPolygon":
        return geom, 1 if geom is not None and not geom.is_empty else 0, 1
    parts = list(geom.geoms)
    try:
        merged = unary_union(parts)
    except Exception:  # a union that fails must not take the whole map down
        return geom, len(parts), len(parts)
    if merged.is_empty:
        return geom, len(parts), len(parts)
    if not merged.is_valid:
        return geom, len(parts), len(parts)
    after = len(list(merged.geoms)) if merged.geom_type == "MultiPolygon" else 1
    if after > len(parts):
        return geom, len(parts), len(parts)
    return merged, len(parts), after


def shift_east(geom, district=None):
    """Return (geometry, rings_moved, sliver_rings_dropped): the same territory with far-east rings in the western
    hemisphere translated by +360 degrees. The input is never modified.

    Two ways a territory qualifies as far east: its federal district is the Far Eastern one - which
    covers districts that lie entirely west of the antimeridian, such as Провиденский at -177 - or
    some ring of it reaches past 150 E. A district drawn in the western hemisphere with no far-east
    anchor would otherwise stay off the coast of Alaska.
    """
    if geom is None or geom.is_empty:
        return geom, 0, 0
    if district != FAR_EAST_DISTRICT and not reaches_far_east(geom):
        return geom, 0, 0
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    moved = 0
    polys = []
    for p in parts:

        def fix(ring):
            nonlocal moved
            _, hi = _lon_span(ring)
            if hi < WEST_LIMIT:
                moved += 1
                return [(x + 360.0, y) for x, y in ring.coords]
            return [(x, y) for x, y in ring.coords]

        rings = [fix(p.exterior)] + [fix(r) for r in p.interiors]
        polys.append(Polygon(rings[0], rings[1:]))
    out = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    # Stitch the halves that meet at the antimeridian. Without this the district is drawn as two
    # polygons and the shared edge reads as a cut through the middle of it.
    out, _before, _after = stitch(out)
    slivers = 0
    if moved or _after < _before:
        # Only where this module actually changed the shape. A territory left exactly as the master
        # drew it keeps every ring it had, including a tiny real one.
        out, slivers = _drop_line_holes(out)
    return out, moved, slivers
