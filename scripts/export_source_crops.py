#!/usr/bin/env python3
"""Exports per-district source crops and draws our municipal boundary over each one.

The point of this surface is that a viewer can see what each pixel means. So every source is exported over the
same window, at the same 500 m grid, at the centroid of the same three named districts, and every crop is drawn with
the municipal outline taken from data/processed/territory_master.parquet - the committed geometry, not a shape
traced by hand. Outside the boundary the pixel is dimmed, because "the district ends here" is the thing a crop
without a border never says.

Dynamic World crops are re-rendered from the GeoTIFFs already in the cache outside the repository. Rebuilding from
cache is what reproducibility means here; re-downloading would change bytes without changing the measurement. The
five remaining sources are small single-window exports, seconds each, and each one is verified to carry numbers
before it is called a crop: a raster that is entirely masked is not an observation, however confidently it was
downloaded.

Writes data crops to the cache directory named by SBER_GEE_CACHE (outside the repository), committed PNGs to artifacts/landing/provenance/,
and the records to artifacts/landing/source_crops.json and artifacts/landing/source_legends.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from PIL import Image
from rasterio import features
from google.auth.transport.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberindex import gee, geecache  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "landing" / "provenance"
CROPS_ARTIFACT = ROOT / "artifacts" / "landing" / "source_crops.json"
LEGEND_ARTIFACT = ROOT / "artifacts" / "landing" / "source_legends.json"
DW_CLASSES_ARTIFACT = ROOT / "artifacts" / "landing" / "provenance_crops.json"

SCALE_M = 500
UPSCALE = 4
# One grid cell is ten source pixels: at 500 m per pixel that is a 5 km square, which a viewer can use as a ruler.
GRID_CELL_PX = 10
GRID_KM = round(SCALE_M * GRID_CELL_PX / 1000.0, 1)

# The window is wider than it is tall so the picture fills a landscape page without letterboxing.
# Pixels stay square - 500 m on the ground in both directions - so nothing is stretched to look bigger.
WIDE_DEG = 0.8
TALL_DEG = 0.5

# A shared ruler is what makes two examples comparable. Drawn against its own brightest and darkest pixel,
# every district looks like the most contrasty place on earth. The endpoints here come from what the catalog
# page documents for the band, widened to what is actually in view, and are written into the artifact rather
# than typed into the page. Night lights are held back until the entrant chooses between a linear and a
# logarithmic ruler - see the comparison renders.
FIXED_SCALE = {
    "ndvi": {"low": 0.0, "high": 8000.0, "scale": "linear"},
    # Elevation is drawn as relief above the lowest measured point of each window. The entrant first chose a
    # shared 0-600 m ruler so districts stayed comparable, then on 2026-10-09 put detail first: a flat district
    # occupies 10 % of a national ruler and reads as a colour swatch. Each window now uses its own relief, and the
    # legend states the zero and the size of the relief so the reader knows what the top of the ramp means here.
    "terrain": {"low": 0.0, "high": 600.0, "scale": "linear"},
    "viirs": {"low": 0.1, "high": 3000.0, "scale": "log"},
}
# Night lights use the logarithmic ruler: the entrant chose it on 2026-10-09 after seeing both variants, because
# Moscow intra-city peaks at 2 329 units of radiance while Nizhny Tagil tops out at 5.8, and a linear ruler made
# every rural window black. The legend says the ruler is logarithmic and shows the real marks.
VIIRS_SCALE_MODE = "fixed"

# The binary line the clustering itself uses. The registry says the indicator is "average >= 0.5 nW/cm2/sr",
# so drawing the picture with the same line means the viewer sees exactly what the feature sees: a pixel is
# either lit or it is not. Drawing radiance on a log ruler made the rural windows almost invisible.
VIIRS_LIT_THRESHOLD = 0.5

# Steps for the picture, chosen from what the three windows actually measure rather than from a round number: in
# the rural windows almost every pixel sits below 1 unit of radiance, while Moscow intra-city has a median of 39.
# So the line between "nothing" and "a settlement" is around 1-3 units and the line between a settlement and a
# city is around 10. These are a reading of the picture, not the threshold the clustering uses - the drawer and the
# card both say which is which.
VIIRS_STEP_EDGES = [1.0, 3.0, 10.0]
VIIRS_STEP_COLORS = [(16, 18, 26), (74, 58, 32), (176, 120, 48), (246, 224, 150)]
VIIRS_STEP_SHORTS = ["тёмное", "слабое", "поселение", "город"]
VIIRS_STEP_LABELS = [
    "света почти нет",
    "слабое свечение: отдельные объекты",
    "видно поселение",
    "город: яркое плотное свечение",
]

# Greenness is drawn as a continuous ramp because that reads well; the steps are only a way of counting how much
# of the window is bare, weakly green or actively green. The band holds MODIS integers, so the edges are 0,1 and
# 0,3 multiplied by 10 000.
NDVI_STEP_EDGES = [1000.0, 3000.0]
NDVI_STEP_SHORTS = ["без растительности", "слабая", "активная"]
NDVI_STEP_LABELS = ["почти без растительности", "растительность слабая", "растительность активная"]


def _step_shapes(values: np.ndarray, edges: list) -> list:
    """Share of measured pixels in each band, as percents, with nothing computed later in the browser."""
    bounds = [float("-inf")] + list(edges) + [float("inf")]
    shares = []
    total = float(values.size)
    for low, high in zip(bounds, bounds[1:]):
        hits = float(np.count_nonzero((values >= low) & (values < high)))
        shares.append(round(hits / total * 100, 1) if total else None)
    return shares

# Which observation year the dated sources are pulled for. A second run with SBER_CROP_YEAR=2024 puts the same
# windows and the same districts under the next year, so the year control moves more than the land map. Static
# layers ignore it: elevation and forest loss are not observations of a year.
CROP_YEAR = int(os.environ.get("SBER_CROP_YEAR", "2023"))


def _period_window(source: dict) -> tuple[str, str]:
    """Shift the dated window to the crop year being produced."""
    start, end = source["window"]
    if source["kind"] not in ("collection-mean", "collection-rgb"):
        return start, end
    shift = CROP_YEAR - 2023

    def _bump(text: str) -> str:
        year, rest = text.split("-", 1)
        return f"{int(year) + shift}-{rest}"

    return _bump(start), _bump(end)


def _window(lon: float, lat: float) -> list:
    return [
        lon - WIDE_DEG / 2.0,
        lat - TALL_DEG / 2.0,
        lon + WIDE_DEG / 2.0,
        lat + TALL_DEG / 2.0,
    ]


def _fit_window(shape) -> list:
    """A window that holds the municipality rather than a fixed box around its centroid.

    A fixed box is why two of the three examples showed less than half of the district and why the smallest one
    looked like a speck in a frame meant for a region. The window is the district's own bounding box grown by a
    margin: a wide margin around a small district, because the surroundings are the story there, and a narrow one
    around a large district, because a national-scale frame would swallow the detail. The box stays rectangular so
    a landscape page carries no letterboxing. The floors keep a small district from being the whole picture: at the
    floor the district fills about two fifths of the frame, which is between filling nothing and filling it.
    """
    min_x, min_y, max_x, max_y = shape.bounds
    wide = max_x - min_x
    tall = max_y - min_y
    margin = 0.30 if wide < 0.15 else 0.15
    wide = max(wide * (1.0 + margin), 0.25)
    tall = max(tall * (1.0 + margin), 0.16)
    centre_x = (min_x + max_x) / 2.0
    centre_y = (min_y + max_y) / 2.0
    return [
        centre_x - wide / 2.0,
        centre_y - tall / 2.0,
        centre_x + wide / 2.0,
        centre_y + tall / 2.0,
    ]


def _grid_step(bbox: list) -> int:
    """The grid step asked of Earth Engine for this window.

    A district of three kilometres drawn on the regional 500 m grid is a dozen pixels across, which the browser
    then smears. A finer grid is asked for when the window is small, and it buys no new information from a coarse
    sensor: it only places the same measurement on a finer lattice. The native pixel size of each source is stated
    beside the picture, so the reader can tell a finer grid from finer data.
    """
    return 100 if (bbox[2] - bbox[0]) < 0.35 else SCALE_M


def _window_tag(bbox: list) -> str:
    """A cache-file tag that records which window a raster covers.

    The tag is part of the cache name, so a raster fetched for one window can never be mistaken for a picture of
    another: the name carries the extent in thousandths of a degree and the grid step in metres.
    """
    return "{}x{}_{}m".format(
        int(round((bbox[2] - bbox[0]) * 1000)),
        int(round((bbox[3] - bbox[1]) * 1000)),
        _grid_step(bbox),
    )


DW_PALETTE = {
    "trees": (46, 93, 58),
    "grass": (127, 168, 96),
    "crops": (201, 162, 39),
    "built": (176, 65, 62),
    "bare": (184, 160, 122),
}

# Color stops are chosen so the reading direction is obvious: low to high in the same direction every time,
# and a legend that says what the colors mean sits next to the image that uses them.
RAMPS = {
    "viirs": [(0.00, (11, 16, 32)), (0.55, (58, 63, 92)), (1.00, (242, 193, 74))],
    "ndvi": [(0.00, (138, 122, 94)), (0.45, (169, 181, 127)), (1.00, (46, 93, 58))],
    "terrain": [(0.00, (95, 122, 85)), (0.55, (168, 148, 104)), (1.00, (232, 227, 216))],
    "soil": [(0.00, (217, 205, 180)), (0.50, (138, 107, 69)), (1.00, (59, 42, 28))],
}
LOSS_COLORS = {0: (120, 124, 118)}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token() -> str:
    credentials = ee.data._state.get_state().credentials
    credentials.refresh(Request())
    return str(credentials.token)


def _valid(band: np.ndarray, nodata) -> np.ndarray:
    ok = np.isfinite(band)
    if nodata is not None:
        ok &= band != nodata
    return ok


def _ramp(
    values: np.ndarray,
    valid: np.ndarray,
    stops: list[tuple[float, tuple[int, int, int]]],
    fixed: dict | None = None,
) -> np.ndarray:
    # Scaling to the whole band would let nodata decide the colors: a -32768 sea floor or a -999 fill
    # would flatten every real value into the bottom of the ramp. The scale comes from valid pixels only.
    if fixed is None:
        sample = values[valid]
        low, high = float(sample.min()), float(sample.max())
    else:
        # A shared ruler: every district is drawn against the same endpoints, so the eye can compare them.
        low, high = float(fixed["low"]), float(fixed["high"])
    if fixed is not None and fixed.get("scale") == "log":
        floor = math.log10(max(low, 1e-6))
        ceiling = math.log10(max(high, floor + 1e-6))
        logged = np.log10(np.clip(values, 10.0**floor, 10.0**ceiling))
        norm = np.clip((logged - floor) / ((ceiling - floor) or 1.0), 0.0, 1.0)
    else:
        span = (high - low) or 1.0
        norm = np.clip((values - low) / span, 0.0, 1.0)
    rgb = np.zeros(values.shape + (3,), dtype="uint8")
    rgb[~valid] = (58, 60, 66)
    for index in range(len(stops) - 1):
        start_at, start_color = stops[index]
        end_at, end_color = stops[index + 1]
        band_mask = (norm >= start_at) & (norm <= end_at)
        if not band_mask.any():
            continue
        t = (norm[band_mask] - start_at) / ((end_at - start_at) or 1.0)
        for channel, color_slot in enumerate((start_color, end_color)):
            mix = np.array(start_color[channel]) + t * (
                np.array(end_color[channel]) - np.array(start_color[channel])
            )
            rgb[band_mask, channel] = np.clip(mix, 0, 255).astype("uint8")
    return rgb


def _boundary_edge(shape, out_shape, transform) -> np.ndarray:
    interior = features.rasterize(
        [shape], out_shape=out_shape, transform=transform, fill=0, all_touched=False, dtype="uint8"
    )
    edge = np.zeros(out_shape, dtype=bool)
    edge[1:, :] |= interior[1:, :] != interior[:-1, :]
    edge[:-1, :] |= interior[1:, :] != interior[:-1, :]
    edge[:, 1:] |= interior[:, 1:] != interior[:, :-1]
    edge[:, :-1] |= interior[:, 1:] != interior[:, :-1]
    return edge


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        out[1:, :] |= out[:-1, :]
        out[:-1, :] |= out[1:, :]
        out[:, 1:] |= out[:, :-1]
        out[:, :-1] |= out[:, 1:]
    return out


def _draw_boundary(rgb: np.ndarray, interior: np.ndarray, dim: bool = True) -> np.ndarray:
    """Dims the pixels that fall outside the municipal boundary.

    Only the fill is burned into the raster. The boundary line itself is carried as vector geometry into the
    page: a line rasterized at 500 m and then nearest-upscaled reads as blocks, and at this source resolution
    that is exactly the garbage the eye rejects. A dimmed surround is a fill, so it survives upscaling.
    """
    painted = rgb.copy()
    if not dim:
        return painted
    outside = interior == 0
    painted[outside] = (painted[outside] * 0.55).astype("uint8")
    return painted


def _pixel_rings(shape, bounds, out_shape: tuple, upscale: int) -> list:
    """The municipal boundary as SVG path data in the picture's own pixel coordinates.

    Degrees become pixel rows and columns through the raster's geotransform where that transform is usable; where
    it is degenerate - a zero pixel height, which some exports carry - the recorded window extent is used instead,
    and which basis was used is reported. Coordinates are then multiplied by the upscaling the published picture
    uses, so the browser places the line over the image without deriving any geography of its own.
    """
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    window = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    clipped = shape.intersection(window)
    if clipped.is_empty:
        return []
    shape = clipped.simplify(0.0015, preserve_topology=True)
    parts = shape.geoms if shape.geom_type == "MultiPolygon" else [shape]
    paths = []
    for part in parts:
        coords = list(part.exterior.coords)
        if len(coords) < 3:
            continue
        points = []
        for lon, lat in coords:
            col = (lon - bounds.left) / width * out_shape[1]
            row = (bounds.top - lat) / height * out_shape[0]
            points.append(f"{round(col * upscale)},{round(row * upscale)}")
        paths.append("M " + " L ".join(points) + " Z")
    return paths


def _interior(shape, out_shape, transform) -> np.ndarray:
    """Pixels covered by the municipal boundary.

    This used to rasterize with fill=1, which made the mask come back all ones: the district never differed from
    its surroundings, the boundary ring had nothing to follow, and every crop was drawn as if the whole window
    were the district. A mask of the inside is what a boundary overlay needs, so the fill is empty.
    """
    return features.rasterize(
        [shape], out_shape=out_shape, transform=transform, fill=0, all_touched=False, dtype="uint8"
    )


def _stretch_photo(band: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Sends each colour channel through the range this scene actually uses.

    Surface reflectance in visible light sits between 0.03 and 0.30, so mapping 0-1 straight to 0-255 makes
    almost every pixel sit below 70: the photo reads as a black square. Stretching over the 2nd-98th percentile
    of what is really in the frame keeps the colours comparable without inventing brightness that is not there.
    """
    height, width = ok.shape
    stretched = np.zeros((height, width, 3), dtype="float64")
    for channel in range(3):
        plane = band[channel].astype("float64")
        inside = plane[ok]
        if inside.size < 32:
            stretched[..., channel] = np.clip(plane, 0.0, 1.0) * 255.0
            continue
        low = float(np.percentile(inside, 2.0))
        high = float(np.percentile(inside, 98.0))
        if not high > low:
            stretched[..., channel] = np.clip(plane, 0.0, 1.0) * 255.0
            continue
        scaled = (plane - low) / (high - low)
        stretched[..., channel] = np.clip(scaled, 0.0, 1.0) ** 0.9 * 255.0
    return np.clip(stretched, 0, 255).astype("uint8")


def _grid_overlay(rgb: np.ndarray, cell_px: int) -> np.ndarray:
    """Marks the pixel structure of the source: a cell is cell_px source pixels wide.

    The grid is baked into a second committed image rather than drawn by the browser, so switching it on cannot
    invent geometry that the export does not have, and both versions line up pixel for pixel.
    """
    lined = rgb.astype("float64").copy()
    rows = np.zeros(lined.shape[0], dtype=bool)
    cols = np.zeros(lined.shape[1], dtype=bool)
    rows[::cell_px] = True
    cols[::cell_px] = True
    lined[rows, :, :] = lined[rows, :, :] * 0.45 + 255.0 * 0.55
    lined[:, cols, :] = lined[:, cols, :] * 0.45 + 255.0 * 0.55
    return np.clip(lined, 0, 255).astype("uint8")


def _publish(painted: np.ndarray, name: str, plain: np.ndarray) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for suffix, array in (("", painted), ("_plain", plain)):
        image = Image.fromarray(array, "RGB")
        resized = image.resize(
            (array.shape[1] * UPSCALE, array.shape[0] * UPSCALE), Image.NEAREST
        )
        target = OUT_DIR / f"{name}{suffix}.png"
        resized.save(target, "PNG", optimize=True)
        results[f"committed_file{suffix}"] = str(target.relative_to(ROOT))
        results[f"committed_bytes{suffix}"] = target.stat().st_size
        results[f"committed_pixels{suffix}"] = [resized.size[0], resized.size[1]]
    return results


def _district_shapes(wanted=None) -> dict[int, object]:
    master = pd.read_parquet(ROOT / "data" / "processed" / "territory_master.parquet")
    master["tid"] = master["tid"].astype("int64")
    wanted = wanted or {2518, 2208, 2000}
    rows = master[master["tid"].isin(wanted)]
    shapes = {}
    for tid, wkb in zip(rows["tid"].astype("int64"), rows["geometry"]):
        geometry = gpd.GeoSeries.from_wkb([wkb]).set_crs("EPSG:4326").geometry.iloc[0]
        shapes[int(tid)] = geometry
    return shapes


def _stats_from_tif(tif: Path) -> dict:
    with rasterio.open(tif) as src:
        band = src.read(1).astype("float64")
        masked = src.read(1, masked=True)
        # When nothing is masked rasterio hands back a scalar False, and "valid & False" silently
        # erases every pixel. getmaskarray always returns a full-shaped array.
        valid = np.isfinite(band) & ~np.ma.getmaskarray(masked)
        values = band[valid]
        total = band.size
        return {
            "valid_pixels": int(values.size),
            "total_pixels": int(total),
            "valid_share": round(float(values.size / total), 4) if total else None,
            "min": round(float(values.min()), 4) if values.size else None,
            "max": round(float(values.max()), 4) if values.size else None,
            "mean": round(float(values.mean()), 4) if values.size else None,
            "carries_values": bool(values.size > 0 and np.isfinite(values).all()),
            "transform_scale_m": round(float(src.transform.a), 2),
        }


def main() -> int:
    dw_reference = json.loads(DW_CLASSES_ARTIFACT.read_text(encoding="utf-8"))
    districts = [
        record
        for record in dw_reference["records"]
        if record["kind"] == "real observation" and record["asset_year"] == "2023"
    ]
    by_tid = {int(record["tid"]): record for record in districts}
    shapes = _district_shapes()
    missing = [tid for tid in by_tid if tid not in shapes]
    if missing:
        print(f"no committed geometry for {missing}, so no boundary can be drawn")
        return 1

    asset_root = os.environ.get("SBER_GEE_ASSET", "")
    asset_root = asset_root.rsplit("/", 1)[0] if asset_root else ""
    if not asset_root:
        print("SBER_GEE_ASSET is not set, so no asset root can be derived")
        return 1

    gee.init()
    token = _token()
    cache = geecache.cache_dir()

    sources = [
        {
            "key": "viirs",
            "label": "ночные огни",
            "meaning": "сколько света видно сверху ночью: чем ярче, тем больше освещённых объектов",
            "asset": "NOAA/VIIRS/DNB/ANNUAL_V22",
            "band": "average",
            "kind": "collection-mean",
            "window": ("2023-01-01", "2024-01-01"),
            "ramp": "viirs",
        },
        {
            "key": "ndvi",
            "label": "зелёность",
            "meaning": "индекс зелёности MODIS: чем выше, тем активнее фотосинтез на этом пикселе",
            "asset": "MODIS/061/MOD13Q1",
            "band": "NDVI",
            "kind": "collection-mean",
            "window": ("2023-01-01", "2024-01-01"),
            "ramp": "ndvi",
        },
        {
            "key": "terrain",
            "label": "высота местности",
            "meaning": "высота над уровнем моря, метры",
            "asset": "USGS/SRTMGL1_003",
            "band": "elevation",
            "kind": "image",
            "ramp": "terrain",
        },
        {
            "key": "hansen",
            "label": "потеря леса",
            "meaning": "год, когда лес на этом пикселе был потерян; серый - потери не было",
            "asset": "UMD/hansen/global_forest_change_2025_v1_13",
            "band": "lossyear",
            "kind": "image",
            "ramp": "lossyear",
        },
    ]
    soil = {
        "key": "soil",
        "label": "влага в почве",
        "meaning": "содержание органического углерода в слое 15-30 см, тёмнее - больше",
        "asset": "ISRIC/SoilGrids250m/v2_0",
        "band": "val_15_30cm_mean",
        "kind": "collection-mosaic-scaled",
        "scale_factor": 1000.0,
        "ramp": "soil",
    }
    sources.append(soil)
    sources.append(
        {
            "key": "sentinel2",
            "label": "настоящая фотография",
            "meaning": "истинные цвета со Sentinel-2: что на этом месте действительно видно сверху",
            "asset": "COPERNICUS/S2_SR_HARMONIZED",
            "band": "B4,B3,B2",
            "kind": "collection-rgb",
            "window": ("2023-05-01", "2023-09-30"),
            "ramp": "rgb",
        }
    )

    records = []
    source_scale: dict[str, dict] = {}
    for source in sources:
        if source["key"] == "sentinel2":
            records.append(
                {
                    "tid": None,
                    "source": "sentinel2",
                    "kind": "not exported",
                    "error": "sent by the entrant on 2026-10-09: a 500 m photographic mosaic reads as noise next to classified layers, so it is not offered as a source and not re-cropped",
                }
            )
            continue
        values_by_district = []
        for tid, reference in sorted(by_tid.items()):
            name = f"crop_{tid}_{source['key']}"
            year_tag = (
                f"_{CROP_YEAR}" if source["kind"] in ("collection-mean", "collection-rgb") else ""
            )
            name = f"crop_{tid}_{source['key']}{year_tag}"
            bbox = _fit_window(shapes[tid])
            scale_m = _grid_step(bbox)
            tif = cache / f"{name}_{_window_tag(bbox)}.tif"
            from_cache = False
            try:
                if tif.exists():
                    with rasterio.open(tif) as probe:
                        if probe.count >= 1 and probe.height > 0 and probe.width > 0:
                            from_cache = True
                if not from_cache:
                    region = ee.Geometry.Rectangle(bbox)
                    if source["kind"] == "collection-mean":
                        raw = (
                            ee.ImageCollection(source["asset"])
                            .filterDate(*_period_window(source))
                            .select(source["band"])
                            .mean()
                        )
                    elif source["kind"] == "collection-rgb":
                        raw = (
                            ee.ImageCollection(source["asset"])
                            .filterDate(*_period_window(source))
                            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                            .median()
                            .select(["B4", "B3", "B2"])
                            .multiply(0.0001)
                        )
                    elif source["kind"] == "collection-mosaic-scaled":
                        raw = (
                            ee.ImageCollection(source["asset"])
                            .mosaic()
                            .select(source["band"])
                            .multiply(source["scale_factor"])
                        )
                    elif source["kind"] == "image-scaled":
                        raw = (
                            ee.Image(source["asset"])
                            .select(source["band"])
                            .multiply(source["scale_factor"])
                        )
                    else:
                        raw = ee.Image(source["asset"]).select(source["band"])
                    url = raw.getDownloadUrl(
                        {
                            "region": region,
                            "scale": scale_m,
                            "crs": "EPSG:4326",
                            "format": "GEO_TIFF",
                            "maxPixels": 8_000_000,
                            "filename": name,
                        }
                    )
                    request = urllib.request.Request(  # nosec B310
                        url, headers={"Authorization": f"Bearer {token}"}
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310
                        tif.write_bytes(response.read())
            except Exception as failure:  # noqa: BLE001 - the failure is the finding, not something to hide
                records.append(
                    {
                        "tid": tid,
                        "district": reference["district"],
                        "cluster": int(reference["cluster"]),
                        "source": source["key"],
                        "asset": source["asset"],
                        "band": source["band"],
                        "region_bbox": bbox,
                        "kind": "unavailable",
                        "error": f"{type(failure).__name__}: {str(failure)[:180]}",
                    }
                )
                continue

            stats = _stats_from_tif(tif)
            if not stats["carries_values"]:
                records.append(
                    {
                        "tid": tid,
                        "district": reference["district"],
                        "cluster": int(reference["cluster"]),
                        "source": source["key"],
                        "asset": source["asset"],
                        "band": source["band"],
                        "region_bbox": bbox,
                        "kind": "empty raster",
                        "error": "the exported band carries no finite pixel values",
                        "stats": stats,
                        "cache_path": str(tif),
                        "payload_sha256": _sha(tif),
                    }
                )
                continue

            with rasterio.open(tif) as src:
                if source["kind"] == "collection-rgb":
                    band = np.stack(
                        [src.read(index + 1).astype("float64") for index in range(3)]
                    )
                    transform = src.transform
                    bounds = src.bounds
                    nodata = src.nodata
                    out_shape = (src.height, src.width)
                else:
                    band = src.read(1).astype("float64")
                    transform = src.transform
                    bounds = src.bounds
                    nodata = src.nodata
                    out_shape = (band.shape[0], band.shape[1])
            if source["kind"] == "collection-rgb":
                ok = np.isfinite(band).all(axis=0)
                rgb = _stretch_photo(band, ok)
                rgb[~ok] = (58, 60, 66)
                basis, fixed = "настоящие цвета, а не цветовая шкала", None
            else:
                ok = _valid(band, nodata)
                shape = shapes[tid]
                interior = _interior(shape, out_shape, transform)
                ok_inside = ok & (interior > 0)
                if source["ramp"] == "lossyear":
                    rgb = np.zeros(band.shape + (3,), dtype="uint8")
                    rgb[~ok] = (58, 60, 66)
                    for year in range(0, 25):
                        color = LOSS_COLORS.get(year)
                        if color is None:
                            t = (year - 1) / 23.0
                            color = (
                                int(198 - 60 * t),
                                int(178 - 110 * t),
                                int(120 - 60 * t),
                            )
                        rgb[ok & (band == year)] = color
                    basis, fixed = "год потери леса - категория, а не уровень", None
                else:
                    use_fixed = (
                        source["key"] in FIXED_SCALE
                        and source["key"] != "terrain"
                        and (source["key"] != "viirs" or VIIRS_SCALE_MODE == "fixed")
                    )
                    fixed = FIXED_SCALE[source["key"]] if use_fixed else None
                    zero_m = None
                    relief_max_m = None
                    display_band = band
                    if source["key"] == "terrain" and ok_inside.any():
                        zero_m = round(float(band[ok_inside].min()), 1)
                        relief_max_m = round(float(band[ok_inside].max()) - zero_m, 1)
                        display_band = band - zero_m
                    basis = (
                        "высота над уровнем моря: от самой низкой точки этого окна до самой высокой точки района"
                        if source["key"] == "terrain"
                        else (
                            "одна шкала на все примеры: " + fixed["scale"]
                            if use_fixed
                            else "шкала этого окна"
                        )
                    )
                    lit_share_pct = None
                    step_shares = None
                    if source["key"] == "viirs":
                        basis = "четыре ступени яркости - это способ читать картинку; кластеризация считает по порогу 0,5"
                        lit = ok_inside & (band >= VIIRS_LIT_THRESHOLD)
                        lit_share_pct = (
                            round(float(lit.sum()) / float(ok_inside.sum()) * 100, 2)
                            if ok_inside.any()
                            else None
                        )
                        step_shares = (
                            _step_shapes(band[ok_inside], VIIRS_STEP_EDGES)
                            if ok_inside.any()
                            else None
                        )
                        rgb = np.zeros(ok.shape + (3,), dtype="uint8")
                        rgb[~ok] = (58, 60, 66)
                        bins = np.digitize(band, VIIRS_STEP_EDGES)
                        for index, color in enumerate(VIIRS_STEP_COLORS):
                            rgb[ok & (bins == index)] = color
                    elif source["key"] == "ndvi" and ok_inside.any():
                        step_shares = _step_shapes(band[ok_inside], NDVI_STEP_EDGES)
                        rgb = _ramp(display_band, ok, RAMPS[source["ramp"]], fixed)
                    else:
                        rgb = _ramp(display_band, ok, RAMPS[source["ramp"]], fixed)
            painted = _draw_boundary(rgb, interior)
            published = _publish(painted, name, rgb)
            values_by_district.append(stats)
            records.append(
                {
                    "tid": tid,
                    "district": reference["district"],
                    "cluster": int(reference["cluster"]),
                    "source": source["key"],
                    "source_label": source["label"],
                    "meaning": source["meaning"],
                    "asset": source["asset"],
                    "band": source["band"],
                    "region_bbox": bbox,
                    "scale_m": scale_m,
                    "native_pixels": list(out_shape),
                    "kind": "real observation",
                    "interior_share_pct": round(float((interior > 0).mean()) * 100, 2),
                    "district_pixels": int((interior > 0).sum()),
                    "inside_pixels": int(ok_inside.sum()),
                    "boundary_paths": _pixel_rings(shape, bounds, out_shape, UPSCALE),
                    "boundary_basis": "the extent the exported raster actually covers",
                    "boundary_pixels": [int(out_shape[1] * UPSCALE), int(out_shape[0] * UPSCALE)],
                    "stats": stats,
                    "scale_basis": basis,
                    "scale_low": (
                        VIIRS_LIT_THRESHOLD
                        if source["key"] == "viirs"
                        else zero_m
                        if source["key"] == "terrain" and relief_max_m is not None
                        else None if fixed is None else fixed["low"]
                    ),
                    "scale_high": (
                        None
                        if source["key"] == "viirs"
                        else zero_m + relief_max_m
                        if source["key"] == "terrain" and relief_max_m is not None
                        else None if fixed is None else fixed["high"]
                    ),
                    "scale_mode": (
                        None
                        if source["key"] == "viirs"
                        else "linear"
                        if source["key"] == "terrain"
                        else None if fixed is None else fixed["scale"]
                    ),
                    "asset_year": (
                        CROP_YEAR if source["kind"] in ("collection-mean", "collection-rgb") else None
                    ),
                    "lit_share_pct": lit_share_pct,
                    "step_shares_pct": step_shares,
                    "zero_m_above_sea": zero_m,
                    "relief_max_m": relief_max_m,
                    "cache_path": str(tif),
                    "taken_from_cache": from_cache,
                    "payload_sha256": _sha(tif),
                    **published,
                }
            )

        if values_by_district:
            scale_of_this_source = {
                "min": min(value["min"] for value in values_by_district),
                "max": max(value["max"] for value in values_by_district),
                "valid_share_mean": round(
                    float(np.mean([value["valid_share"] for value in values_by_district])), 4
                ),
            }
            source_scale[source["key"]] = scale_of_this_source

    # Dynamic World: redrawn from the cached GeoTIFFs, so the boundary appears on the class map too.
    for tid, reference in sorted(by_tid.items()):
        for year in ("2023", "2024"):
            bbox = _fit_window(shapes[tid])
            scale_m = _grid_step(bbox)
            tif = cache / f"crop_{tid}_{year}_{_window_tag(bbox)}.tif"
            if not tif.exists():
                records.append(
                    {
                        "tid": tid,
                        "district": reference["district"],
                        "cluster": int(reference["cluster"]),
                        "source": "dynamic_world",
                        "asset_year": year,
                        "kind": "cache miss",
                        "error": f"no cached GeoTIFF at {tif}",
                        "owed": "run scripts/export_provenance_crops.py for this window",
                    }
                )
                continue
            with rasterio.open(tif) as src:
                names = [d for d in (src.descriptions or ()) if d in DW_PALETTE]
                if len(names) != src.count:
                    # GEE writes the bands in the order they were renamed on the asset; rasterio does not
                    # always carry a description, so the order is taken from the export and checked by count.
                    names = list(DW_PALETTE)
                bands = {
                    name: src.read(index + 1).astype("float64")
                    for index, name in enumerate(names)
                }
                transform = src.transform
                bounds = src.bounds
                out_shape = (src.height, src.width)
            names = list(bands)
            stacked = np.stack([bands[name] for name in names])
            winner = stacked.argmax(axis=0)
            # A pixel counts as observed when some class fraction came back at all; without this the drawer
            # has nothing to report about coverage and says "нет данных" about a picture that is full of data.
            dw_observed = stacked.sum(axis=0) > 0
            shape = shapes[tid]
            interior = _interior(shape, out_shape, transform)
            inside_now = interior > 0
            dw_share = (
                round(float(dw_observed[inside_now].mean()), 4) if inside_now.any() else None
            )
            rgb = np.zeros(out_shape + (3,), dtype="uint8")
            rgb[~stacked.any(axis=0)] = (58, 60, 66)
            for index, name in enumerate(names):
                rgb[winner == index] = DW_PALETTE[name]
            painted = _draw_boundary(rgb, interior)
            published = _publish(painted, f"crop_{tid}_dw_{year}", rgb)
            records.append(
                {
                    "tid": tid,
                    "district": reference["district"],
                    "cluster": int(reference["cluster"]),
                    "source": "dynamic_world",
                    "source_label": "земля по классам",
                    "interior_share_pct": round(float(inside_now.mean()) * 100, 2),
                    "district_pixels": int(inside_now.sum()),
                    "boundary_paths": _pixel_rings(shape, bounds, out_shape, UPSCALE),
                    "boundary_pixels": [int(out_shape[1] * UPSCALE), int(out_shape[0] * UPSCALE)],
                    "boundary_basis": "the extent the exported raster actually covers",
                    "meaning": "каждый пиксель - класс земли с наибольшей уверенностью: деревья, травы, пашня, застройка, открытая земля",
                    "asset_year": year,
                    "asset_leaf": f"dw_mean_{year}_500m",
                    "region_bbox": reference["region_bbox"],
                    "scale_m": scale_m,
                    "native_pixels": list(out_shape),
                    "stats": {"valid_share": dw_share, "min": None, "max": None},
                    "scale_basis": "класс земли - категория, а не уровень",
                    "kind": "rebuilt from cache",
                    "cache_path": str(tif),
                    "payload_sha256": _sha(tif),
                    **published,
                }
            )

    # Pooling has to follow the merged records too: the legend numbers describe every crop the artifact holds,
    # not just the ones this particular run happened to produce.
    source_scale = {}
    for key in sorted({record["source"] for record in records}):
        stats = [
            record["stats"]
            for record in records
            if record["source"] == key and isinstance(record.get("stats"), dict)
        ]
        usable = [item for item in stats if item.get("min") is not None and item.get("max") is not None]
        if usable:
            source_scale[key] = {
                "min": min(item["min"] for item in usable),
                "max": max(item["max"] for item in usable),
                "valid_share_mean": round(
                    float(np.mean([item["valid_share"] for item in usable])), 4
                ),
            }

    # A run for another year must not erase the year already recorded: two runs of this script wrote the same
    # artifact and the later one won, which left the page without the 2023 crops it had already published.
    # Records are merged by source, district and period so each run adds to the artifact instead of replacing it.
    stored_records: dict[tuple, dict] = {}
    if CROPS_ARTIFACT.exists():
        try:
            stored = json.loads(CROPS_ARTIFACT.read_text(encoding="utf-8"))
            for record in stored.get("records", []):
                stored_records[(record["source"], str(record["tid"]), str(record.get("asset_year")))] = record
        except (ValueError, OSError):
            stored_records = {}
    for record in records:
        stored_records[(record["source"], str(record["tid"]), str(record.get("asset_year")))] = record
    records = sorted(
        stored_records.values(),
        key=lambda item: (item["source"], str(item["tid"]), str(item.get("asset_year"))),
    )

    CROPS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    CROPS_ARTIFACT.write_text(
        json.dumps(
            {
                "generated_by": "scripts/export_source_crops.py",
                "what_this_is": "source crops for three named districts, each drawn with the municipal boundary from the committed master geometry",
                "scale_m": SCALE_M,
                "windows_shared_by_all_sources": True,
                "boundary_source": "data/processed/territory_master.parquet, column geometry (well-known binary, EPSG:4326)",
                "outside_the_boundary_is_dimmed": True,
                "grid_cell_pixels": GRID_CELL_PX,
                "grid_cell_km": GRID_KM,
                "grid_is_a_second_committed_image": "the browser switches files; it draws nothing",
                "never_names_the_granting_project": dw_reference["never_names_the_granting_project"],
                "cache_location": str(cache),
                "cache_inside_the_repository": False,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    legends = {}
    for source in sources:
        scale = source_scale.get(source["key"])
        if scale is None:
            continue
        fixed = (
            FIXED_SCALE.get(source["key"])
            if source["key"] != "viirs" or VIIRS_SCALE_MODE == "fixed"
            else None
        )
        legends[source["key"]] = {
            "label": source["label"],
            "meaning": source["meaning"],
            "asset": source["asset"],
            "band": source["band"],
            "scale_min": round(scale["min"], 4),
            "scale_max": round(scale["max"], 4),
            "shared_scale": (
                {"low": fixed["low"], "high": fixed["high"], "mode": fixed["scale"]} if fixed else None
            ),
            "coverage_mean": scale["valid_share_mean"],
            "ramp_colors": (
                [[120, 124, 118], [198, 178, 120], [138, 68, 60]]
                if source["ramp"] == "lossyear"
                else ([list(color) for color in RAMPS[source["ramp"]]] if source["ramp"] in RAMPS else None)
            ),
            "reading_direction": (
                "серый - пикселя данных нет"
                if source["ramp"] == "lossyear"
                else ("настоящие цвета, а не цветовая шкала" if source["ramp"] == "rgb" else "от низкого значения к высокому")
            ),
        }
    # Night lights are drawn with the binary line the analysis uses, so the legend has to say the same thing:
    # a pixel is lit or it is not, and the share of lit pixels is the number the clustering actually reads.
    lit_shares = [r["lit_share_pct"] for r in records if r.get("lit_share_pct") is not None]
    if "viirs" in legends and lit_shares:
        legends["viirs"]["classes"] = [
            {"label": label, "short": VIIRS_STEP_SHORTS[index], "color": list(VIIRS_STEP_COLORS[index]), "edge_below": VIIRS_STEP_EDGES[index] if index < len(VIIRS_STEP_EDGES) else None}
            for index, label in enumerate(VIIRS_STEP_LABELS)
        ]
        legends["viirs"]["step_edges"] = VIIRS_STEP_EDGES
        legends["viirs"]["threshold"] = VIIRS_LIT_THRESHOLD
        legends["viirs"]["lit_share_mean_pct"] = round(sum(lit_shares) / len(lit_shares), 2)
        legends["viirs"]["reading_direction"] = "от почти тёмного к яркому: ступени 1, 3 и 10 единиц яркости"
        legends["viirs"]["ramp_colors"] = None

    ndvi_steps = [r["step_shares_pct"] for r in records if r.get("step_shares_pct")]
    if "ndvi" in legends and ndvi_steps:
        legends["ndvi"]["steps"] = [
            {
                "label": NDVI_STEP_LABELS[index],
                "short": NDVI_STEP_SHORTS[index],
                "edge_below": NDVI_STEP_EDGES[index] if index < len(NDVI_STEP_EDGES) else None,
            }
            for index in range(len(NDVI_STEP_LABELS))
        ]

    if "terrain" in legends:
        legends["terrain"]["reading_direction"] = "от самой низкой точки этого окна к возвышенности"

    legends["dynamic_world"] = {
        "label": "земля по классам",
        "meaning": "каждый пиксель - класс земли с наибольшей уверенностью",
        "classes": [
            {"label": label, "color": list(color)}
            for label, color in (
                ("деревья", DW_PALETTE["trees"]),
                ("травы", DW_PALETTE["grass"]),
                ("пашня", DW_PALETTE["crops"]),
                ("застройка", DW_PALETTE["built"]),
                ("открытая земля", DW_PALETTE["bare"]),
            )
        ],
    }
    def _loss_color(year: int) -> list:
        if year == 0:
            return list(LOSS_COLORS[0])
        t = min((year - 1) / 23.0, 1.0)
        return [int(198 - 60 * t), int(178 - 110 * t), int(120 - 60 * t)]

    # Forest loss is a year, not a continuous quantity, so it gets a discrete legend: five stops of the same
    # ramp the picture is painted with, plus the "no loss" colour that covers most pixels here.
    legends["hansen"]["classes"] = [{"label": "нет потерь леса", "color": _loss_color(0)}] + [
        {"label": f"{2000 + year} год", "color": _loss_color(year)}
        for year in (1, 6, 12, 18, 24)
    ]

    LEGEND_ARTIFACT.write_text(
        json.dumps(
            {
                "generated_by": "scripts/export_source_crops.py",
                "rule": "no crop ships without a legend that says what a pixel means in that source",
                "sources": legends,
                "scale_per_source": source_scale,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    usable = [r for r in records if r["kind"] in ("real observation", "rebuilt from cache")]
    broken = [r for r in records if r["kind"] not in ("real observation", "rebuilt from cache")]
    print(f"crops usable {len(usable)}, failures recorded {len(broken)}")
    for failure in broken[:6]:
        print(f"  {failure['source']} tid={failure['tid']}: {failure['kind']} - {failure.get('error', '')[:90]}")
    print(f"records written to {CROPS_ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
