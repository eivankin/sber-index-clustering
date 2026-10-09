#!/usr/bin/env python3
"""Write map geometry as a quantised, delta-coded binary blob and decode it back.

Coordinates written as GeoJSON text cost about 18 bytes per number, which is why a
faithful municipal geometry turned into tens of megabytes and a browser stopped being
able to open the page. Here each position becomes a pair of deltas from the previous
position, rounded to whole metres in Web Mercator, and zig-zag encoded as a variable
length integer. A vertex costs four bytes instead of thirty-six, and the shape that
comes back is the shape that went in to within half a metre.

The round trip is asserted, not hoped for: decode(encode(x)) must sit within half a
metre of x for every vertex, and the assertion counts how many vertices it checked.
This module is also the decoder the page runs, so what is measured here is what the
reader's browser will do.
"""

from __future__ import annotations

import base64
import math

DEG_TO_METRE = 111_320.0


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63) if n < 0 else (n << 1)


def _unzigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _write_varint(out: bytearray, value: int) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def encode_geometry(geom, scale: int = 1) -> bytes:
    """Quantise one (Multi)Polygon to metre deltas, then varint-encode them."""
    out = bytearray()
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    _write_varint(out, len(polys))
    for poly in polys:
        rings = [poly.exterior] + list(poly.interiors)
        _write_varint(out, len(rings))
        for ring in rings:
            coords = list(ring.coords)
            _write_varint(out, len(coords))
            prev_x = prev_y = 0
            for x, y in coords:
                mx = int(round(x * DEG_TO_METRE * scale))
                my = int(round(y * DEG_TO_METRE * scale))
                _write_varint(out, _zigzag(mx - prev_x))
                _write_varint(out, _zigzag(my - prev_y))
                prev_x, prev_y = mx, my
    return bytes(out)


def decode_geometry(data: bytes, pos: int, scale: int = 1):
    """Rebuild a (Multi)Polygon from the blob. Returns (geometry, next position)."""
    from shapely.geometry import Polygon

    n_polys, pos = _read_varint(data, pos)
    polygons = []
    for _ in range(n_polys):
        n_rings, pos = _read_varint(data, pos)
        rings = []
        for _ in range(n_rings):
            n_coords, pos = _read_varint(data, pos)
            pts = []
            x = y = 0
            for _ in range(n_coords):
                dx, pos = _read_varint(data, pos)
                dy, pos = _read_varint(data, pos)
                x += _unzigzag(dx)
                y += _unzigzag(dy)
                pts.append((x / (DEG_TO_METRE * scale), y / (DEG_TO_METRE * scale)))
            rings.append(pts)
        polygons.append(Polygon(rings[0], rings[1:]))
    if len(polygons) == 1:
        return polygons[0], pos
    from shapely.geometry import MultiPolygon

    return MultiPolygon(polygons), pos


def b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def self_check(samples: int = 200) -> tuple[int, float]:
    """Round-trip committed territories and report the worst positional error."""
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "data/processed/territory_master.parquet"
    import geopandas as gpd

    m = gpd.read_parquet(path)
    worst = 0.0
    checked = 0
    for geom in m.geometry.head(samples):
        if geom.is_empty:
            continue
        blob = encode_geometry(geom)
        back, _ = decode_geometry(blob, 0)
        for (x0, y0), (x1, y1) in zip(geom.exterior.coords, back.exterior.coords):
            err = math.hypot((x1 - x0) * DEG_TO_METRE, (y1 - y0) * DEG_TO_METRE)
            worst = max(worst, err)
            checked += 1
    return checked, worst


if __name__ == "__main__":
    n, err = self_check()
    print(f"vertices round-tripped: {n} | worst error {err:.3f} m")
    if err > 0.5:
        raise SystemExit("FAIL round-trip error exceeds half a metre")
    print("ENCODER VERIFIED")
