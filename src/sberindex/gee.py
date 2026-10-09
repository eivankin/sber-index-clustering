"""GEE aggregation for per-territory land covariates.

Design constraints that shaped this module:
* Restricted mode is real: jobs run but the *request* payload is capped at 10 MB, so polygons are
  simplified and territories are chunked (<=100 per request).
* Histograms are returned inline and converted client-side; denominators are classified pixels only
  (`unclassified_pct` is reported per layer so nodata can never masquerade as a share).
* Every row must carry provenance (asset id, year, scale, reducer) - see `SOURCES`.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

import ee
import geopandas as gpd

SA_EMAIL = os.environ.get("SBER_GEE_SA_EMAIL")
SA_KEY = os.environ.get("SBER_GEE_SA_KEY")

# registry consumed by configs/gee_sources.yaml (license + citation live there)
SOURCES: dict[str, dict] = {
    "worldcover": {
        "asset": "ESA/WorldCover/v200/2021",
        "kind": "image",
        "band": "Map",
        "year": 2021,
        "scale": 10,
        "reducer": "frequencyHistogram",
    },
    "gls_lcz": {
        "asset": "COPERNICUS/Landcover/100m/Proba-V-C3/Global",
        "kind": "collection",
        "band": "LSM",
        "year": 2019,
        "scale": 100,
        "reducer": "frequencyHistogram",
    },
    "gls_fraction": {
        "asset": "COPERNICUS/Landcover/100m/Proba-V-C3/Global",
        "kind": "collection",
        "band": "tree_fraction,builtup_fraction,bare_fraction,shrub_fraction",
        "year": 2019,
        "scale": 100,
        "reducer": "mean",
    },
    "viirs": {
        "asset": "NOAA/VIIRS/DNB/ANNUAL_V22",
        "kind": "collection",
        "band": "average,average_masked,cf_cvg",
        "year": 2023,
        "scale": 464,
        "reducer": "mean",
    },
    "viirs_lit": {
        "asset": "NOAA/VIIRS/DNB/ANNUAL_V22",
        "kind": "collection",
        "band": "average>=0.5",
        "year": 2023,
        "scale": 464,
        "reducer": "mean(binary)",
    },
    "modis_lcc": {
        "asset": "MODIS/061/MCD12Q1",
        "kind": "collection",
        "band": "LC_Prop1",
        "year": 2023,
        "scale": 500,
        "reducer": "frequencyHistogram",
    },
    "modis_burn": {
        "asset": "MODIS/061/MCD64A1",
        "kind": "collection",
        "band": "burned_area",
        "year": 2023,
        "scale": 500,
        "reducer": "sum",
    },
    "modis_ndvi": {
        "asset": "MODIS/061/MOD13Q1",
        "kind": "collection",
        "band": "NDVI(QA-filtered)",
        "year": 2023,
        "scale": 250,
        "reducer": "mean",
    },
    "worldclim": {
        "asset": "WORLDCLIM/V1/BIO",
        "kind": "image",
        "band": "bio01,bio04,bio12,bio15",
        "year": None,
        "scale": 1000,
        "reducer": "mean",
    },
    "elevation": {
        "asset": "TERRAIN/SRTM/SRTM_ELEVATION_30M",
        "kind": "image",
        "band": "SRTM高程",
        "year": None,
        "scale": 30,
        "reducer": "mean",
    },
}


def init() -> None:
    if not SA_EMAIL or not SA_KEY:
        raise RuntimeError("SBER_GEE_SA_EMAIL / SBER_GEE_SA_KEY not set in the environment")
    # Earth Engine's getInfo() rides a request with no read timeout, so a stream the server stops feeding
    # blocks that worker until the process is killed. Measured, not assumed: two sockets sat in CLOSE-WAIT
    # with a byte still queued, 333 s had passed since anything was last received, and the run's CPU time
    # stayed flat while the cache gained nothing - a hang rather than slowness. A ceiling converts that
    # into a recorded failure the run can continue past, which is what the per-chunk error list is for.
    socket.setdefaulttimeout(float(os.environ.get("SBER_GEE_SOCKET_TIMEOUT", "300")))
    # ee.data.setDeadline() was tried and does not work with this authentication path: it calls
    # _install_cloud_api_resource(), which asserts state.requests_session is not None, and that session is
    # never created when Initialize is given ServiceAccountCredentials directly. Calling it therefore
    # raised AssertionError inside init(), which would have broken every run that imports this module. The
    # socket ceiling above stays as the mechanism, because it is the one that was measured to apply.
    # One variable, one meaning: it either names a key file that exists on disk, or it carries the
    # key text itself. Both are accepted, and the choice is made by looking, not by guessing. The
    # dynamic-world probe used to call .splitlines() on the same value and hand the library a list,
    # which failed far away from here with "'list' object has no attribute 'endswith'".
    if Path(SA_KEY).exists():
        credentials = ee.ServiceAccountCredentials(SA_EMAIL, key_file=SA_KEY)
        try:
            project = os.environ.get("SBER_GEE_PROJECT", "")
            if not project:
                project = str(
                    json.loads(Path(SA_KEY).read_text(encoding="utf-8")).get("project_id", "")
                )
        except Exception:  # noqa: BLE001
            project = ""
    else:
        # The library's parameters are email, key_file and key_data. This branch once passed key_content, a name
        # that does not exist, so any run handing over the key text instead of a path would have died here with a
        # TypeError - which is exactly what mypy reported, and what no run had exercised because the key on disk
        # took the branch above. The correct name for "the key text itself" is key_data.
        credentials = ee.ServiceAccountCredentials(SA_EMAIL, key_data=SA_KEY)
        project = os.environ.get("SBER_GEE_PROJECT", "")
    # Initialize needs to know which Earth Engine project to bill work to. Without it the library
    # asks for the default project list and then treats that list as a name.
    if not project:
        raise RuntimeError(
            "no Earth Engine project to initialize against: set SBER_GEE_PROJECT or use a key "
            "file that carries project_id"
        )
    ee.Initialize(credentials, project=project)


def simplify(gdf: gpd.GeoDataFrame, tol_deg: float = 0.0005) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(tol_deg, preserve_topology=True)
    return out


def feature_collection(geoms, tids) -> ee.FeatureCollection:
    import shapely.geometry as sg

    feats = []
    for geom, tid in zip(geoms, tids):
        feats.append(ee.Feature(ee.Geometry(sg.mapping(geom), geodesic=False), {"tid": str(tid)}))
    return ee.FeatureCollection(feats)


@dataclass
class Job:
    name: str
    image: ee.Image
    scale: int
    reducer: ee.Reducer
    asset: str = ""
    year: object = None
    note: str = ""
    reducer_label: str = "reducer"


def first_available(candidates: list[str]) -> str | None:
    """Return the first asset id this project can actually load, else None."""
    for aid in candidates:
        try:
            ee.Image(aid).bandNames().getInfo()
            return aid
        except Exception:
            continue
    return None


def _collection(aid: str, year: int) -> ee.ImageCollection:
    return ee.ImageCollection(aid).filterDate(f"{year}-01-01", f"{year + 1}-01-01")


def build_jobs(year: int = 2023) -> list[Job]:
    jobs: list[Job] = []
    s = SOURCES
    jobs.append(
        Job(
            "worldcover",
            ee.Image(s["worldcover"]["asset"]),
            10,
            ee.Reducer.frequencyHistogram(),
            s["worldcover"]["asset"],
            2021,
            "13-class shares over classified pixels",
            "frequencyHistogram",
        )
    )
    gls = _collection(s["gls_lcz"]["asset"], 2019).first()
    jobs.append(
        Job(
            "gls_lcz",
            gls.select("discrete_classification"),
            100,
            ee.Reducer.frequencyHistogram(),
            s["gls_lcz"]["asset"],
            2019,
            "LCZ class shares",
            "frequencyHistogram",
        )
    )
    frac_bands = [
        "tree-coverfraction",
        "urban-coverfraction",
        "bare-coverfraction",
        "shrub-coverfraction",
        "crops-coverfraction",
        "grass-coverfraction",
        "snow-coverfraction",
        "water-permanent-coverfraction",
        "water-seasonal-coverfraction",
        "moss-coverfraction",
    ]
    jobs.append(
        Job(
            "gls_fraction",
            gls.select(frac_bands).rename([b.replace("-coverfraction", "") for b in frac_bands]),
            100,
            ee.Reducer.mean(),
            s["gls_fraction"]["asset"],
            2019,
            "cover fractions (0-100)",
            "mean",
        )
    )
    viirs = _collection(s["viirs"]["asset"], year).first()
    jobs.append(
        Job(
            "viirs",
            viirs.select(["average", "average_masked", "cf_cvg", "maximum", "median", "minimum"]),
            464,
            ee.Reducer.mean(),
            s["viirs"]["asset"],
            year,
            "annual DNB statistics",
            "mean",
        )
    )
    jobs.append(
        Job(
            "viirs_lit",
            viirs.select("average").gte(0.5),
            464,
            ee.Reducer.mean(),
            s["viirs_lit"]["asset"],
            year,
            "fraction of lit pixels (>=0.5 nW/cm2/sr)",
            "mean(binary)",
        )
    )
    jobs.append(
        Job(
            "modis_lcc",
            _collection(s["modis_lcc"]["asset"], year).first().select("LC_Prop1"),
            500,
            ee.Reducer.frequencyHistogram(),
            s["modis_lcc"]["asset"],
            year,
            "IGBP class shares",
            "frequencyHistogram",
        )
    )
    burnmax = _collection(s["modis_burn"]["asset"], year).select("BurnDate").max()
    # BurnDate is masked outside burned pixels, so an unmasked constant is required: without it the
    # histogram comes back empty and a territory looks "burned 0%" when in fact nothing was counted.
    jobs.append(
        Job(
            "modis_burn",
            ee.Image.constant(0).where(burnmax.gt(0), 1).toInt(),
            500,
            ee.Reducer.frequencyHistogram(),
            s["modis_burn"]["asset"],
            year,
            "binary burned-at-least-once flag over an unmasked constant",
            "frequencyHistogram(binary)",
        )
    )
    ndvi = _collection(s["modis_ndvi"]["asset"], year).map(
        lambda img: img.updateMask(img.select("SummaryQA").eq(0))
    )
    jobs.append(
        Job(
            "modis_ndvi",
            ndvi.select("NDVI").mean(),
            250,
            ee.Reducer.mean(),
            s["modis_ndvi"]["asset"],
            year,
            "QA-filtered mean NDVI",
            "mean",
        )
    )
    jobs.append(
        Job(
            "worldclim",
            ee.Image(s["worldclim"]["asset"]).select(
                ["bio01", "bio04", "bio12", "bio15", "bio18", "bio19"]
            ),
            1000,
            ee.Reducer.mean(),
            s["worldclim"]["asset"],
            None,
            "bioclimatic means",
            "mean",
        )
    )
    elev = first_available(
        [
            "TERRAIN/SRTM/SRTM_ELEVATION_30M",
            "USGS/SRTMGL1_003",
            "COPERNICUS/DGANoteS202014001/GLO-30",
            "CGIAR/SRTM90_V4",
            "TERRAIN/SRTM90_V1",
        ]
    )
    if elev is not None:
        SOURCES["elevation"]["asset"] = elev
        jobs.append(
            Job(
                "elevation",
                ee.Image(elev),
                30,
                ee.Reducer.mean(),
                elev,
                None,
                "mean elevation",
                "mean",
            )
        )
    return jobs


def run_job(job: Job, fc: ee.FeatureCollection) -> list[dict]:
    """Returns [{tid, <props>}] for one chunk; raises on failure so the caller can retry smaller."""
    res = job.image.reduceRegions(fc, job.reducer, scale=job.scale).getInfo()
    rows = []
    for f in res["features"]:
        p = dict(f["properties"])
        rows.append(p)
    return rows


def provenance(job: Job) -> str:
    return (
        f"{job.asset} | year={job.year} | scale={job.scale} m | "
        f"reducer={job.reducer_label} | {job.note}"
    )
