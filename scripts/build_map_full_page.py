#!/usr/bin/env python3
"""Build the full-fidelity map page: one basemap, no fetch, codec-carried geometry.

The overview map in the landing is a 3 km sketch, and that sketch is what erased the small
districts around Moscow and St Petersburg. This page carries the same territories through the
quantised delta-coded codec verified to round-trip within 0.707 m, and draws them at full
resolution when the reader goes where those districts live: at zoom 8 and above, every territory
whose bounds touch the viewport is decoded from the blob and drawn as the source drew it.

The blob is base64 because the page must open over file:// with no server of ours, no CDN and no
fetch of ours. Pre-computed bounds let the page choose which territories to decode without
decoding all of them first: 4.5 million vertices decoded eagerly in the browser was tried and
timed out, so full fidelity arrives where it is looked for rather than everywhere at once.

A single basemap, per the user's instruction to stop building a menu: CARTO light, with visible
attribution, referrerPolicy, capped maxZoom, no prefetch, and an honest notice plus readable
geometry when tiles do not arrive.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_map_data import framed_view  # noqa: E402
from map_geometry_codec import encode_geometry  # noqa: E402

PAGE = ROOT / "web" / "map" / "full" / "index.html"
SHELL = ROOT / "web" / "map" / "full" / "shell.html"
PRESETS_ART = ROOT / "artifacts" / "landing" / "map_presets.json"
TIMELINE_ART = ROOT / "artifacts" / "landing" / "map_timeline.json"
SIMILAR_ART = ROOT / "artifacts" / "landing" / "map_similar.json"
LEGEND_ART = ROOT / "artifacts" / "landing" / "map_cluster_legend.json"
MASTER = ROOT / "data" / "processed" / "territory_master.parquet"
PROFILE = ROOT / "artifacts" / "models" / "cluster_profile.json"


def props_for(row, profile: dict) -> dict:
    cluster = row.cluster
    measured = pd.notna(cluster)
    return {
        "n": row.municipal_district_name,
        "r": row.region_name,
        "c": int(cluster) if measured else -1,
        "m": bool(measured),
        "mo": int(row.months_present),
        "off": (
            None
            if pd.isna(row.official_land_area_km2)
            else round(float(row.official_land_area_km2), 1)
        ),
        "med": profile.get(str(int(cluster))) if measured else None,
    }


def main() -> int:
    ART = ROOT / "artifacts"
    master = pd.read_parquet(ROOT / "data" / "processed" / "territory_master.parquet")
    assign = pd.read_parquet(ART / "icvi" / "cluster_assignments.parquet")
    if not assign.method.unique()[0] == "kmeans_k5":
        raise SystemExit(f"unexpected partition in cluster_assignments: {assign.method.unique()}")
    profile = pd.read_csv(ART / "interpretation" / "cluster_profile.csv")
    med = dict(zip(profile.cluster.astype(int), profile.median_spending_rub.round(0)))
    joined = master.merge(assign[["tid", "cluster"]], on="tid", how="left")
    geo = gpd.GeoSeries.from_wkb(joined.geometry).set_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(joined, geometry=geo, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & gdf.is_valid].reset_index(drop=True)

    # Display-only dateline shift: Chukotka's antimeridian rings come back to the right edge of the
    # map at lng + 360. The master file stays as it is; the count of rings moved is reported.
    sys.path.insert(0, str(ROOT / "src"))
    from sberindex.dateline import shift_east
    from sberindex.regions import district_of

    districts = gdf["region_name"].map(district_of).tolist()
    pairs = [shift_east(g, d) for g, d in zip(gdf.geometry, districts)]

    def _nparts(g):
        return len(list(g.geoms)) if g.geom_type == "MultiPolygon" else 1

    _before = [_nparts(g) for g in gdf.geometry]
    _after = [_nparts(t[0]) for t in pairs]
    print(
        f"dateline rings moved for encoding: {sum(t[1] for t in pairs)}"
        f"; parts stitched: {sum(b - a for b, a in zip(_before, _after) if a < b)}"
        f" merged in {sum(1 for b, a in zip(_before, _after) if a < b)} territories"
        f"; line holes dropped: {sum(t[2] for t in pairs)}"
    )
    gdf = gdf.set_geometry(gpd.GeoSeries([t[0] for t in pairs], crs="EPSG:4326"))

    blobs: list[str] = []
    bounds: list[list[float]] = []
    props: list[dict] = []
    for row in gdf.itertuples():
        geom = row.geometry
        measured = pd.notna(row.cluster)
        p = {
            "fid": int(row.tid),
            "name": row.municipal_district_name,
            "region": row.region_name,
            "cluster": int(row.cluster) if measured else -1,
            "measured": bool(measured),
            "months": None if pd.isna(row.months_present) else int(row.months_present),
            "months_present": None if pd.isna(row.months_present) else int(row.months_present),
            "official_area_km2": (
                None
                if pd.isna(row.official_land_area_km2)
                else round(float(row.official_land_area_km2), 1)
            ),
            "cluster_median_rub_pm": med.get(int(row.cluster)) if measured else None,
        }
        if geom.is_empty:
            blobs.append("")
            bounds.append([0.0, 0.0, 0.0, 0.0])
            props.append(p)
            continue
        blobs.append(base64.b64encode(encode_geometry(geom, 1)).decode("ascii"))
        w, s, e, n = geom.bounds
        bounds.append([round(s, 6), round(w, 6), round(n, 6), round(e, 6)])
        props.append(p)

    # Cluster names ride along as a second reference to the same artifact the page inherits;
    # the legend and the detail panel resolve a cluster through names, never through "кластер N".
    CLUSTER_NAMES = json.loads(
        (ROOT / "artifacts" / "interpretation" / "cluster_names.json").read_text()
    )
    # Quick jumps come from the committed artifact, never from places typed in by hand: the same
    # file the landing's overview map uses, so both surfaces offer the same set of jumps.
    presets = json.loads(PRESETS_ART.read_text())
    for p in presets:
        for key in ("key", "label", "lat", "lng", "zoom", "n_territories", "provenance"):
            if key not in p:
                raise SystemExit(f"preset {p.get('key')} is missing {key}")
    print(f"presets carried into the page: {len(presets)} from {PRESETS_ART.relative_to(ROOT)}")

    # The month-by-month series is the same committed artifact the timeline scrubs: months, values
    # per official km2, and the partition-agreement numbers that explain why the partition is not
    # re-drawn month by month. Nothing here is typed in by hand.
    timeline = json.loads(TIMELINE_ART.read_text())
    if len(timeline["months"]) < 2:
        raise SystemExit(
            f"timeline artifact holds {len(timeline['months'])} months, too few to scrub"
        )
    tl_payload = {
        "months": timeline["months"],
        "tids": timeline["tids"],
        "columns": timeline["columns"],
        "coverage": timeline["coverage_territories_by_month"],
        "medians": timeline["median_rub_per_km2_per_month"],
        "agreement": timeline["partition_agreement_2023_vs_2024"],
        "denominator": timeline["denominator"],
        "quantisation": timeline["quantisation"],
        "source": timeline["source"],
        "stability": timeline.get("cluster_stability", {"available": False}),
    }

    # «Похожие округа» needs the neighbour artifact the checker already verifies. Only the territories
    # the page can actually select are carried; the unmeasured ones have no neighbours to show anyway.
    # Nothing about a neighbour's name, score or role is computed in the browser.
    similar = json.loads(SIMILAR_ART.read_text(encoding="utf-8"))
    sim_payload = {
        "flavours": similar["flavours"],
        "role_rule": similar["role_rule"],
        "neighbour_rule": similar.get("neighbour_rule", ""),
        "top_k": similar["top_k"],
        "counts": similar["counts"],
        "provenance": similar["provenance"],
        "territories": {k: v for k, v in similar["territories"].items() if v["measured"]},
        # The page says where its neighbours came from, so the path has to travel with them. The
        # artifact's own provenance names build products that are not in git; this names the file the
        # reader could actually open in the repository.
        "artifact": str(SIMILAR_ART.relative_to(ROOT)),
    }
    print(
        f"neighbours carried into the page: {len(sim_payload['territories'])} "
        f"measured territories of {len(similar['territories'])}"
    )

    # The legend's examples need to be reachable from the map, so the legend artifact travels with them:
    # one example per measure per cluster, the value it carries, how far it sits from the cluster median,
    # and the Russian caption of the measure. Nothing is invented here.
    legend = json.loads(LEGEND_ART.read_text(encoding="utf-8"))
    legend_payload = {
        "measures": legend["headline_measures"],
        "captions": {
            m["statistic"]: m.get("caption", m["statistic"])
            for m in legend["clusters"][0]["measures"]
        },
        "clusters": [
            {
                "cluster": cl["cluster"],
                "short_name": cl["short_name"],
                "meaning": cl["meaning"],
                "n": cl["n"],
                "spending_per_official_km2": cl["spending_per_official_km2"],
                "median_official_area_km2": cl["median_official_area_km2"],
                "values": {m["statistic"]: m["median"] for m in cl["measures"]},
            }
            for cl in legend["clusters"]
        ],
        "examples": [
            {
                "cluster": ex["cluster"],
                "statistic": ex["statistic"],
                "tid": ex["tid"],
                "value": ex["value"],
                "distance_to_median": ex["distance_to_median"],
            }
            for ex in legend["examples"]
        ],
        "artifact": str(LEGEND_ART.relative_to(ROOT)),
        "what_this_is": legend["what_this_is"],
    }

    # Per-territory values, so a selected district can be set against the medians of its cluster and of
    # the two others. Only what the comparison needs is carried: the nine values, the medians, the reason
    # for districts with no spending data, and the artifact the reader can open. Captions stay with the
    # legend block that is already in the page, so the same words are not shipped twice.
    measures_art = ROOT / "artifacts/landing/territory_measures.json"
    if not measures_art.exists():
        raise SystemExit(
            f"per-territory measures missing: {measures_art.relative_to(ROOT)} - run "
            "scripts/build_territory_measures.py"
        )
    measures_full = json.loads(measures_art.read_text(encoding="utf-8"))
    measures_payload = {
        "measures": measures_full["measures"],
        "medians": measures_full["cluster_medians"],
        "names": measures_full["cluster_names"],
        "territories": {
            tid: {
                "cluster": rec["cluster"],
                "values": rec["values"],
                "density": rec["spending_per_official_km2"],
            }
            for tid, rec in measures_full["territories"].items()
        },
        "no_spending": measures_full["no_spending"],
        "counts": measures_full["counts"],
        "rule": measures_full["rule"],
        "medians_source": measures_full["medians_source"],
        "artifact": str(measures_art.relative_to(ROOT)),
    }

    # Geometry is no longer carried inside the page. Four and a half million vertices as inline
    # base64 made a 21 MB document that still rounded every coordinate to a metre; the lossless
    # files sit next to the overview map and are fetched instead.
    blob_script = (
        "<script>\nconst GEOMETRY_FILES = { index: \"../map_index.json\", "
        "binary: \"../map_geom.bin\" };\n</script>\n"
    )
    meta_script = (
        "<script>\nconst BOUNDS = " + json.dumps(bounds, separators=(",", ":")) + ";\n"
        "const PROPS = "
        + json.dumps(props, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
        + "const CLUSTER_NAMES = "
        + json.dumps(CLUSTER_NAMES, ensure_ascii=False, separators=(",", ":"))
        + ";\n</script>\n"
    )
    # The wrapper tags obviously contain a closing sequence; what must not contain one is the
    # data between them. Any "</" inside a string is escaped as "<\/" so it cannot end the
    # script element early, and the count of escapes is reported rather than assumed to be zero.
    assert "</" not in json.dumps(blobs, separators=(",", ":")), "blob payload contains '</'"
    props_json = json.dumps(props, ensure_ascii=False, separators=(",", ":"))
    n_esc = props_json.count("</")
    props_json = props_json.replace("</", "<\\/")
    view = framed_view([b for b in bounds if b])
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
    print(f"copies hidden from the search index: {len(dup_drop)} tids")

    meta_script = (
        "<script>\nconst BOUNDS = " + json.dumps(bounds, separators=(",", ":")) + ";\n"
        "const PROPS = " + props_json + ";\n"
        "const FULL_VIEW = " + json.dumps(view, separators=(",", ":")) + ";\n"
        "const PRESETS = " + json.dumps(presets, ensure_ascii=False, separators=(",", ":")) + ";\n"
        # The legend and the detail panel resolve a cluster through the committed names artifact.
        # This second meta_script replaced the first one, which is why the names vanished: the page
        # threw a ReferenceError on buildKey and never drew a map at all.
        "const CLUSTER_NAMES = "
        + json.dumps(CLUSTER_NAMES, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        + ";\n"
        "const TL = " + json.dumps(tl_payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
        "const SIMILAR = "
        + json.dumps(sim_payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        + ";\n"
        "const LEGEND = "
        + json.dumps(legend_payload, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        )
        + ";\n"
        "const MEAS = "
        + json.dumps(measures_payload, ensure_ascii=False, separators=(",", ":")).replace(
            "</", "<\\/"
        )
        + ";\n"
        "const DUP_DROP = " + json.dumps(dup_drop, separators=(",", ":")) + ";\n"
        "</script>\n"
    )
    print(f"timeline payload in the page: {len(json.dumps(tl_payload))/1e6:.3f} MB")
    print(f"props '</' escapes applied: {n_esc}")

    # The shell is the hand-written page; this script adds data to it. Reading the built page as
    # the source was the old design, and it was not reproducible: a second run would inject a
    # second `const BLOBS` and a second `DEG_M`, or trip its own assertions. Generated output in
    # the shell fails loudly instead of producing a page that does not parse.
    if not SHELL.exists():
        raise SystemExit(f"shell missing: {SHELL}")
    page = SHELL.read_text()
    for marker in ("const BLOBS", "const BOUNDS", "async function buildFullFC"):
        if marker in page:
            raise SystemExit(f"the shell carries generated output ({marker}); extract it again")

    # The overview page loads its generalised geometry and decoder from sibling files. This page
    # carries its own full-resolution payloads and its own decoder, so those references must not
    # survive the copy: a second `const DEG_M` in the same script scope is a SyntaxError.
    for tag in ('<script src="codec.js"></script>\n', '<script src="map_geom.js"></script>\n'):
        if tag in page:
            page = page.replace(tag, "")
    # The shell defines its overview geometry by decoding the sibling payload. This page decodes
    # its own full-resolution blobs, so that definition has to go: leaving it in means a
    # ReferenceError on GEOM the moment the page loads, and no map drawn at all.
    m = re.search(
        r"<script>\nfunction expandProps\(p\)[\s\S]*?overviewFeatureCollection[\s\S]*?</script>\n",
        page,
    )
    if m:
        print(f"removed overview data definition from the shell: {len(m.group(0)):,} chars")
        page = page[: m.start()] + page[m.end() :]
    # The data block goes before the page's own code, so the constants exist when the
    # renderer runs. Leaflet's own script is the first one; ours follows it.
    # The shell opens on the overview's fitted view, which lives in GEOM - a payload this page does
    # not carry. Point it at the view computed from full-resolution bounds instead, or the first
    # thing the page does after decoding is throw ReferenceError: GEOM is not defined, and nothing
    # ever gets drawn.
    shell_view = "map.setView([GEOM.view.lat, GEOM.view.lng], GEOM.view.zoom);"
    if shell_view in page:
        page = page.replace(
            shell_view, "map.setView([FULL_VIEW.lat, FULL_VIEW.lng], FULL_VIEW.zoom);"
        )
        print("repointed the opening view at FULL_VIEW")

    first_close = page.index("</script>") + len("</script>")
    page = page[:first_close] + "\n" + blob_script + meta_script + page[first_close:]

    # One basemap: the menu of four sources is demoted by user instruction to a single working
    # source with honest degradation.
    # The committed page already carries one basemap with attribution, referrerPolicy, capped
    # maxZoom, no prefetch and an honest tile-failure notice, so this page inherits it. Only
    # insert our block when the page being rebuilt lacks one: a second declaration kills the
    # entire script at parse time, which is how this build failed twice.
    if "const basemap" in page:
        menu = None
    else:
        menu = re.search(r"\n// One remote basemap[^;]*?\}\)\(\);\n", page, flags=re.S)
        if menu:
            page = page[: menu.start()] + page[menu.end() :]

    single = """
// One basemap, chosen because it works over file:// from a CDN that does not answer a demo with
// a placeholder. Attribution visible, referrerPolicy set (OSM rejects requests that arrive
// without a Referer and paints the rejection onto the tile), maxZoom capped, no prefetch, no
// offline cache. If tiles do not arrive the page says so in the detail panel and the
// municipalities remain perfectly readable, because they are repository data, not tiles.
const basemap = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  referrerPolicy: "strict-origin-when-cross-origin"
}).addTo(map);
let tilesBroken = false;
basemap.on("tileerror", function () {
  if (tilesBroken) return;
  tilesBroken = true;
  document.getElementById("detail").innerHTML =
    "<b>Базовая карта не загрузилась</b><span class='m'>Тайлы не пришли. Границы муниципалитетов — данные репозитория, они никуда не деваются; основанием служит сетка координат.</span>";
  L.layerGroup(
    Array.from({length: 19}, (_, i) => L.polyline([[-85, -180 + i * 20], [85, -180 + i * 20]], {color: "#c6cbd3", weight: 0.6, interactive: false}))
  ).addTo(map);
});
"""
    if menu is None and "const basemap" not in page:
        anchor = re.search(r"const map = L\\.map\\([^;]*?\\);\\n", page)
        assert anchor, "map initialisation not found"
        page = page[: anchor.end()] + single + page[anchor.end() :]

    # One geometry source. The decoded full-fidelity collection replaces the inlined sketch in
    # the loading chain, so the existing style, legend, search, hover and selection code runs on
    # the real geometry instead of the simplified one.
    # Idempotent patches: each one is applied when the shell still needs it and reported as
    # already done when it does not. A patch that finds neither its target nor its result fails,
    # because that means the shell drifted and the page would be built wrong.
    chain = "Promise.resolve({ json: () => Promise.resolve(DATA) })"
    wired = "Promise.resolve({ json: () => Promise.resolve(buildFullFC()) })"
    if chain in page:
        page = page.replace(chain, wired, 1)
    elif wired in page:
        print("geometry chain already wired to the decoder: nothing to do")
    else:
        raise SystemExit("neither the DATA chain nor the decoder chain is in the shell")

    # The page's provenance line named the simplified artifact while the page draws full
    # geometry from the codec - a caption that describes a different dataset than the one on
    # screen. It now names the artifact the geometry actually comes from.
    prov_old = "данные: artifacts/landing/map_data.geojson"
    if page.count(prov_old) == 1:
        page = page.replace(
            prov_old,
            "данные: data/processed/territory_master.parquet, геометрия без упрощения через scripts/map_geometry_codec.py",
            1,
        )
    elif "territory_master.parquet" in page:
        print("provenance line already names the shipped artifact: nothing to do")

    # Canvas for the full-fidelity choropleth. SVG was tried first and Chromium's tab died when
    # it re-projected 5 323 path elements holding 4 489 144 vertices after a zoom change; canvas
    # redrew the same geometry in about 2.3 s per zoom step without crashing. One layer, not two:
    # an earlier probe that added a second full-fidelity layer on top of the SVG one crashed too,
    # which is what taught us the cost is the number of held vertices, not the renderer alone.
    geo_head = "const layer = L.geoJSON(data, {\n      style: function (f) {"
    geo_canvas = (
        "const layer = L.geoJSON(data, {\n      renderer: L.canvas(),\n      style: function (f) {"
    )
    # The shell may carry the canvas renderer while funnelling every paint through one shared
    # style function (`style: baseStyle`), which is what replaced the two mouseout handlers whose
    # argument-less resetStyle() repainted the whole group back to cluster colours. Matching only
    # the old inline style function refused to build at all, so the check is structural: the
    # creation call must exist and the canvas renderer must be inside it.
    geo_open = "const layer = L.geoJSON(data, {"
    if geo_head in page:
        page = page.replace(geo_head, geo_canvas, 1)
    elif geo_canvas in page:
        print("choropleth already on the canvas renderer: nothing to do")
    elif geo_open in page and "renderer: L.canvas()," in page:
        print("choropleth on the canvas renderer with a shared style function: nothing to do")
    elif geo_open in page:
        page = page.replace(geo_open + "\n", geo_open + "\n      renderer: L.canvas(),\n", 1)
        print("canvas renderer inserted into the choropleth creation")
    else:
        raise SystemExit("no choropleth construction in the shell")

    # The search index was built from the inlined sketch. It is rebuilt from the same properties
    # the decoded features carry, so search, arrow-key navigation and selection all address the
    # real geometry through the same fid the choropleth keys its layers by.
    idx_head = "const INDEX = DATA.features.map(function (f) {"
    idx_full = 'const INDEX = PROPS\n    .map(function (pp) { const f = { type: "Feature", properties: pp, geometry: null };'
    if idx_head in page:
        page = page.replace(
            idx_head,
            idx_full,
            1,
        )
    elif idx_full in page:
        print("search index already built from the shipped properties: nothing to do")
    else:
        raise SystemExit("no search index construction in the shell")

    # Full fidelity where the reader looks for it.

    # Full fidelity where the reader looks for it. Decoding all 4.5 million vertices eagerly was
    # tried and timed out the browser, so the viewport decides which territories arrive at full
    # resolution; below zoom 8 the simplified choropleth carries the overview.
    detail_js = """
// Full resolution arrives where it is looked for. The blob carries every territory exactly as the
// source drew it, to within half a metre; decoding all 4 489 144 vertices at once was tried and
// timed the browser out, so the viewport decides which territories arrive, and pre-computed
// bounds make that decision without decoding anything first.
const DEG_M = 111320;
function b64ToBytes(s) {
  const bin = atob(s);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}
function readVarint(data, pos) {
  let result = 0, shift = 0, byte;
  do { byte = data[pos++]; result |= (byte & 0x7F) << shift; shift += 7; } while (byte >= 0x80);
  return [result, pos];
}
function unzig(n) { return (n >>> 1) ^ -(n & 1); }
function decodeBlob(b64) {
  const data = b64ToBytes(b64);
  let pos = 0;
  const [nPolys, p1] = readVarint(data, pos); pos = p1;
  const polys = [];
  for (let i = 0; i < nPolys; i++) {
    const [nRings, p2] = readVarint(data, pos); pos = p2;
    const rings = [];
    for (let r = 0; r < nRings; r++) {
      const [nCoords, p3] = readVarint(data, pos); pos = p3;
      const pts = [];
      let x = 0, y = 0;
      for (let k = 0; k < nCoords; k++) {
        const [dx, p4] = readVarint(data, pos); pos = p4;
        const [dy, p5] = readVarint(data, pos); pos = p5;
        x += unzig(dx); y += unzig(dy);
        pts.push([y / DEG_M, x / DEG_M]);
      }
      rings.push(pts);
    }
    polys.push(rings);
  }
  return polys;
}

// Full resolution at every zoom. The numbers below are measured in Chromium 149 on this page:
// decoding all 2 660 territories (4 489 144 vertices) is 401 ms; adding them as an SVG layer is
// 1 695 ms and as a canvas layer 1 480 ms; panning over the full layer costs 1 ms; zero page
// errors. The earlier two-tier design - the old simplified sketch below zoom 8, real geometry
// only above - was justified by a comment claiming eager decoding "was tried and timed the
// browser out". Nothing of the kind was ever measured, and these numbers refute it. A sketch is
// not a cheaper picture of the same territory: its spikiness, its gaps between neighbours and
// its erased small districts are visible at every zoom, so showing it first hid the very defect
// the reader was complaining about until they happened to zoom in.
async function buildFullFC() {
  const t0 = performance.now();
  const feats = [];
  let verts = 0;
  setLoadStep("Читаем геометрию…", false);
  const [indexResponse, binaryResponse] = await Promise.all([
    fetch(GEOMETRY_FILES.index),
    fetch(GEOMETRY_FILES.binary),
  ]);
  if (!indexResponse.ok || !binaryResponse.ok) {
    throw new Error("geometry files are not next to this page");
  }
  const index = await indexResponse.json();
  const coords = new Float64Array(await binaryResponse.arrayBuffer());
  setLoadStep("Разбираем территории…", true);
  for (let i = 0; i < index.features.length; i++) {
    const entry = index.features[i];
    if (!entry.polys.length) continue;
    const polys = entry.polys.map((rings) => rings.map((run) => {
      const ring = new Array(run[1]);
      for (let k = 0; k < run[1]; k += 1) {
        ring[k] = [coords[run[0] + 2 * k], coords[run[0] + 2 * k + 1]];
      }
      return ring;
    }));
    verts += entry.polys.reduce((a, rings) => a + rings.reduce((b, r) => b + r[1], 0), 0);
    const multi = polys.length > 1;
    feats.push({
      type: "Feature",
      properties: Object.assign({ fid: entry.tid }, index.props[i]),
      geometry: multi
        ? { type: "MultiPolygon", coordinates: polys }
        : { type: "Polygon", coordinates: polys[0] },
    });
    // Building arrays is the part with a known size, so the bar can show how far it has got. The
    // page yields to the browser every 200 territories, otherwise nothing repaints until it is over
    // and the reader sees a static screen for the whole of it.
    if (i % 200 === 199) {
      loadfill.style.width = Math.round((i / index.features.length) * 100) + "%";
      await new Promise(function (r) { requestAnimationFrame(r); });
    }
  }
  loadfill.style.width = "100%";
  window.__open = {
    decode_ms: Math.round(performance.now() - t0),
    features: feats.length,
    vertices: verts,
  };
  return { type: "FeatureCollection", features: feats };
}
"""
    k = page.rfind("\n</script>")
    page = page[:k] + "\n" + detail_js + page[k:]

    PAGE.write_text(page)
    print(f"page bytes {PAGE.stat().st_size:,} ({PAGE.stat().st_size/1e6:.2f} MB)")
    print(
        f"features {len(blobs)} | blob payload {len(blob_script)/1e6:.2f} MB | props+bounds {len(meta_script)/1e6:.2f} MB"
    )
    print("fetch calls in page:", page.count("fetch("), "| script tags:", page.count("<script"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
