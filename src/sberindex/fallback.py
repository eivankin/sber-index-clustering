"""No-auth covariate stack (insurance against Earth Engine being unavailable).

Sources verified reachable without credentials from this machine:
* Copernicus Global Land Service LC100 v2019 - Zenodo record 3939050 (CC BY 4.0).
* Copernicus DEM GLO-30 Cloud-Optimized GeoTIFFs - public S3 bucket `copernicus-dem-30m`.
Everything is aggregated with *windowed* reads: Russia at 100 m is ~8.6e9 pixels, so nothing is
ever loaded whole into memory. Large territories are read at a coarser sampling step, which keeps
class proportions intact (nearest-neighbour downsampling of a categorical raster).
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import rasterize

ZENODO = "https://zenodo.org/api/records/3939050/files/"
RASTER_DIR = Path("data/raw/gls")

# only the small files are worth downloading; the float cover-fraction layers are 3-7 GB each
LAYERS = {
    "gls_lc": "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif",
    "gls_urban": "PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif",
    "gls_water": "PROBAV_LC100_global_v3.0.1_2019-nrt_PermanentWater-CoverFraction-layer_EPSG-4326.tif",
    "gls_forest_type": "PROBAV_LC100_global_v3.0.1_2019-nrt_Forest-Type-layer_EPSG-4326.tif",
    # the change-confidence band is what lets a spending-type transition be tested against the land:
    # without a change layer there is no way to say whether the physical fingerprint moved with it.
    "gls_change": "PROBAV_LC100_global_v3.0.1_2019-nrt_Change-Confidence-layer_EPSG-4326.tif",
}

DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _total_size(url: str) -> int:
    """Zenodo and the DEM bucket accept Range requests, so a 1-byte probe reveals the whole size."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec B310
        cr = r.headers.get("content-range") or ""
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return int(r.headers.get("content-length") or 0)


def _fetch(url: str, dest: Path, attempts: int = 8) -> Path:
    """Download with resume. A Wi-Fi restart killed a 1.7 GB GeoTIFF at 916 MB with
    ContentTooShortError, so a plain urlretrieve is not good enough for layers this large."""
    tmp = dest.with_suffix(".part")
    total = _total_size(url)
    for _ in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total and have >= total:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        mode = "ab" if have else "wb"
        print(f"  {dest.name}: resuming at {have:,} of {total:,} bytes", flush=True)
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, mode) as fh:  # nosec B310
            shutil.copyfileobj(r, fh, 1024 * 1024)
    size = tmp.stat().st_size if tmp.exists() else 0
    if total and size < total:
        raise RuntimeError(
            f"{dest.name}: stopped at {size:,} of {total:,} bytes after {attempts} tries"
        )
    tmp.replace(dest)
    return dest


def download(name: str, force: bool = False) -> Path:
    dest = RASTER_DIR / name
    # a file smaller than 1 MB is never a real GeoTIFF: Zenodo answered with record JSON before
    if dest.exists() and dest.stat().st_size < 1_000_000:
        dest.unlink()
    if dest.exists() and not force:
        # A truncated download leaves a real-looking but unreadable GeoTIFF behind (the discrete
        # classification map stopped at 874 MB of 1.71 GB and still opened as a file, but GDAL refused
        # its TIFF directory). Existence is not completeness, so verify by opening it, and treat a
        # broken file as resumable rather than done: the byte prefix on disk is a valid resume point.
        try:
            import rasterio

            with rasterio.open(dest) as ds:
                if ds.count >= 1:
                    return dest
        except Exception:
            part = dest.with_suffix(".part")
            if not part.exists():
                dest.rename(part)
            else:
                dest.unlink()
    if dest.exists() and not force:
        return dest
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    url = ZENODO + name + "/content"  # /files/<name>?download=1 returns JSON, not bytes
    print(f"downloading {name}", flush=True)
    _fetch(url, dest)
    size = dest.stat().st_size
    if size < 1_000_000:
        dest.unlink()
        raise RuntimeError(f"{name}: got {size} bytes, not a GeoTIFF")
    return dest


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bbox_in_grid(geom, ds):
    """Intersect a geometry's bbox with the grid it needs, so windows never run off the raster."""
    tol = abs(ds.transform.a) * 0.5 + abs(ds.transform.e) * 0.5
    left = max(geom.bounds[0], ds.bounds[0] - tol)
    bottom = max(geom.bounds[1], ds.bounds[1] - tol)
    right = min(geom.bounds[2], ds.bounds[2] + tol)
    top = min(geom.bounds[3], ds.bounds[3] + tol)
    if right <= left or top <= bottom:
        return None
    return (left, bottom, right, top)


def _read_window(ds, window, max_px: int):
    """Read a window, downsampling if it would blow memory. Returns (array, transform)."""
    w, h = int(window.width), int(window.height)
    step = max(1, int(np.ceil(np.sqrt(w * h / max_px))))
    out_shape = (max(1, h // step), max(1, w // step))
    if step > 1:
        arr = ds.read(
            1, window=window, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest
        )
        tr = ds.window_transform(window).scale(step, step)
    else:
        arr = ds.read(1, window=window)
        tr = ds.window_transform(window)
    return arr, tr


def zonal_class_counts(ds, geom, max_px: int = 40_000_000):
    """Class histogram of pixels inside `geom`; nodata/0 counted separately."""
    bbox = _bbox_in_grid(geom, ds)
    if bbox is None:
        return {}, 0, 0
    window = windows.from_bounds(*bbox, transform=ds.transform)
    arr, tr = _read_window(ds, window, max_px)
    mask = rasterize([geom], out_shape=arr.shape, transform=tr, all_touched=False, fill=0)
    inside = arr[mask.astype("bool")]
    if inside.size == 0:
        return {}, 0, 0
    nodata = ds.nodata
    valid = inside != nodata if nodata is not None else inside > 0
    valid = valid & (inside > 0)
    vals, counts = np.unique(inside[valid], return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    return hist, int(valid.sum()), int((~valid).sum())


def zonal_mean(ds, geom, max_px: int = 40_000_000):
    bbox = _bbox_in_grid(geom, ds)
    if bbox is None:
        return None, 0
    window = windows.from_bounds(*bbox, transform=ds.transform)
    arr, tr = _read_window(ds, window, max_px)
    mask = rasterize([geom], out_shape=arr.shape, transform=tr, all_touched=False, fill=0).astype(
        "bool"
    )
    inside = arr[mask]
    if inside.size == 0:
        return None, 0
    nodata = ds.nodata
    valid = inside != nodata if nodata is not None else np.ones(inside.shape, "bool")
    if not valid.any():
        return None, 0
    return float(np.mean(inside[valid])), int(valid.sum())


def dem_tiles(minx: float, miny: float, maxx: float, maxy: float):
    """Copernicus GLO-30 tile URLs covering a bounding box in (minLon, minLat, maxLon, maxLat)
    order - exactly what GeoDataFrame.bounds returns (5 degree grid)."""
    lats = range(int(np.floor(miny / 5) * 5), int(np.floor(maxy / 5) * 5) + 5, 5)
    lons = range(int(np.floor(minx / 5) * 5), int(np.floor(maxx / 5) * 5) + 5, 5)
    for la in lats:
        for lo in lons:
            if la < -90 or la > 80 or lo < -180 or lo > 175:
                continue
            name = f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM"
            yield f"{DEM_BASE}/{name}/{name}.tif"


DEM_DIR = Path("data/raw/copernicus_dem")


def ensure_dem_tile(url: str) -> Path:
    """Fetch one Copernicus GLO-30 COG locally. GDAL has its own DNS stack, so /vsicurl cannot use
    the TCP-resolver patch - the only reliable route is to download the tile and open it locally."""
    name = url.split("/")[-1]
    dest = DEM_DIR / name
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://copernicus-dem-30m.s3.amazonaws.com/" + name + "/" + name + ".tif"
    return _fetch(url, dest)


def prefetch_dem_tiles(geoms, log=None) -> int:
    urls = sorted({u for geom in geoms for u in dem_tiles(*geom.bounds)})
    n = 0
    for u in urls:
        try:
            ensure_dem_tile(u)
            n += 1
        except Exception as exc:
            if log:
                log(f"dem tile failed {u.split('/')[-1]}: {type(exc).__name__}")
    return n


def dem_stats(geom, max_px: int = 25_000_000):
    """Elevation statistics from remote COGs, read windowed, nothing downloaded to disk.

    Tiles are aggregated tile-by-tile and combined pixel-count-weighted: the mean is exact, the
    median is a count-weighted average of tile medians (documented approximation), min/max are
    exact over the union of tiles.
    """
    urls = list(dem_tiles(*geom.bounds))
    if not urls:
        return {}
    parts = []
    for url in urls:
        try:
            with rasterio.open(str(ensure_dem_tile(url))) as ds:
                bbox = _bbox_in_grid(geom, ds)
                if bbox is None:
                    continue
                win = windows.from_bounds(*bbox, transform=ds.transform)
                arr, tr = _read_window(ds, win, max_px)
                mask = rasterize(
                    [geom], out_shape=arr.shape, transform=tr, all_touched=False, fill=0
                ).astype("bool")
                vals = arr[mask]
                vals = vals[np.isfinite(vals) & (vals > -999)]
                if vals.size:
                    parts.append(
                        (
                            int(vals.size),
                            float(vals.mean()),
                            float(np.median(vals)),
                            float(vals.min()),
                            float(vals.max()),
                        )
                    )
        except Exception as exc:
            print(
                f"   dem tile failed {url.split('/')[-2]}: {type(exc).__name__}: {str(exc)[:90]}",
                flush=True,
            )
    if not parts:
        return {}
    n = sum(p[0] for p in parts)
    mean = sum(p[0] * p[1] for p in parts) / n
    median = sum(p[0] * p[2] for p in parts) / n
    return {
        "dem_mean_m": round(mean, 2),
        "dem_median_m": round(median, 2),
        "dem_min_m": min(p[3] for p in parts),
        "dem_max_m": max(p[4] for p in parts),
        "dem_px": n,
        "dem_tiles": len(parts),
        "dem_tiles_needed": len(urls),
    }
