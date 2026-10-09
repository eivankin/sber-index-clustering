# Covariates: what each number means, and what it does not mean

Land data enters this entry in exactly one place — the **supply** side of the complementarity edge — and
in a second, deliberately separate place: as **out-of-sample validation** of spending types. It is never
inside a demand weight, because night-light-derived demand built from land cover would be circular.

## Where covariates come from

GEE-first, aggregated per territory with `reduceRegions`, chunked by federal district. The layers actually
produced, with the band names corrected against the catalogue (several assumed names were wrong):

| job | asset | what the column means |
|---|---|---|
`ESA/WorldCover/v200/2021` | class shares over classified pixels (`wc_c*_pct`) |
`COPERNICUS/Landcover/100m/Proba-V-C3/Global`, `artificial`, `cropland`, `herbaceous`, `mangrove`, `peat_bog`, `snow_ice`, `bare_sparse` | same denominator (`gls_c*_pct`) |
`COPERNICUS/Landcover/100m/Proba-V-C3/Global` | cover *fractions* (`gls_fraction_*`) — 0 is a measurement, not a gap |
| `modis_lcc` | `MODIS/061/MCD12Q1` | class shares over classified pixels (`mcd_c*_pct`) |
| `modis_burn` | `MODIS/061/MCD64A1` | burn dates aggregated to per-territory area/dates (`burn_*`) |
| `modis_ndvi` | `MODIS/061/MOD13Q1` | mean NDVI over the window, `SummaryQA`-masked |
`NOAA/VIIRS/DNB/ANNUAL_V22` | annual night-light mass and mean |
| `worldclim` | `WORLDCLIM/V1/BIO` | 19 bioclimatic variables (an Image, not a Collection) |
| `elevation` | `USGS/SRTMGL1_003` | mean elevation in metres |

Assets that could not be used are recorded with the exact error, not silently dropped. That list was
re-measured on 2026-10-08 under a different, registered Earth Engine project, and part of it turned out to
be names this repository invented rather than took from the catalog. `ISRIC/SoilGrids250m/v2_0` answers, so
the soil source recorded as absent is available; `UMD/hansen/global_forest_change_2025_v1_13` answers with
`treecover2000`, `loss`, `gain` and `lossyear`, so the forest-change source recorded as no access is
available. The names that do not answer are `ISRIC/soilgrids`, `Hansen/GFSID2017`,
`UMD/hansen-global-tree-cover`, `UMD/GLAD_/TREECOVER/CIRCUITRY/2020v1`,
`UMD/gladsat-foundation-data-v1`, `GLAD/IFL/2021`, `UMD/ifl_2021` and
`WORLDCLIM/V1/MONTHLY/PRECIP` - measured in `artifacts/geo/asset_id_probe.json`. The last of those is not
needed: `WORLDCLIM/V1/BIO` supplies `worldclim_bio12`, annual precipitation, already in the panel. Legacy
identifier forms such as `MODIS/0.061/MCD12Q1` no longer load where `MODIS/061/MCD12Q1` does, which explains
some recorded failures without any change in access. Two families are dropped by the entrant's decision
rather than by failure of the source - GLAD tree cover and Intact Forest Landscapes - and that is recorded as
a decision, with the reason, in `configs/brief_requirements.yaml`.

## Denominators

Class shares are computed **over classified pixels only**. Pixels with no class assigned are excluded
from the denominator and reported separately as `*_unclassified_pct` for every territory. This matters in
the far north: a territory that is 60 % unclassified on a satellite grid is not a territory where 60 % of
the land "is not forest" — it is a territory where the sensor did not classify the surface. Reporting the
share of unclassified pixels keeps that difference visible instead of letting it be absorbed into the
class percentages.

Per-km² densities use **official Rosstat land area** (BDPMO Y48006001) where available, otherwise the
official subject area as a documented proxy, otherwise nothing. They never use the raw OSM municipal
polygon area, because those polygons are inflated ~2.3× by marine and extremity extensions (see
`docs/method/master-table-and-geometry.md`).

## Scale overrides are explicit, never silent

Restricted mode throttles heavy aggregations, so some layers are requested at a coarser resolution than
their native grid — `SBER_SCALE="worldcover=100,gls_lcz=300,gls_fraction=300,elevation=250"`. Every such
row carries the applied scale in its provenance, and the run prints each override. A coarser denominator
is therefore visible in the data rather than hidden in a config file.

## Provenance

Every row carries the asset id, the year/epoch, the scale actually requested, and the reducer used
(`Image.reduceRegions` with a per-class frequency histogram over an EE GeometryCollection). Provenance is
part of the schema, not a side note: without it, a share is just a number with no stated basis.

## How the geometry reaches Earth Engine, and what that costs

The territory outlines live in an Earth Engine table asset, `municipal_bounds_shp`, uploaded from a shapefile
built outside the repository: 2 660 features, read back carrying `tid`, `osm_ref`, `osm_vers`, `yr_from` and
`yr_to`, so the version window an outline came from travels with the geometry instead of living only in our
master table. The upload simplified geometry to about 55 m because zonal statistics at 250-500 m do not need
the vertex density of the 74 MB source, and a 15.9 MB zip uploads where that source cannot. When
`SBER_GEE_ASSET` names that asset, a chunk selects its districts with `ee.Filter.inList` on `tid` and sends
no geometry, which lifts the 10 MB request ceiling that had forced Hansen into 25-district chunks; the
default chunk becomes 300 districts and remains overridable, because a refused chunk is a measurement.

Two things were measured rather than assumed. Streaming results is not available for this kind of expression:
`ee.data.computeFeatures` was tried in three forms - the object, serialized JSON, and the result wrapped in a
FeatureCollection - and every one refused, so an asset buys payload size and composite reuse, not one request
per period. And whether this service account may *write* assets is untested: the asset root appeared only after
it was created through the console, so nothing here claims that per-period composites can be materialized as
assets of their own.

Every Earth Engine response is cached outside the repository under `SBER_GEE_CACHE`, and `cache_dir()` refuses
a cache pointed at `/tmp`, because that is a tmpfs which does not survive a reboot. The proof is timed and
needs no assumptions about Earth Engine: an identical second run took 0.1 s against 8.9 s for the first and
reported `cache_replays: 1`. Reproducibility therefore means rebuilding from what was already paid for rather
than downloading again, so artifacts may name the cache location and the payload hash but never credentials or
the project that granted access.

## The coverage gate

`cluster.land_columns` admits a covariate column into the model only when at least **50 %** of territories
have a non-null value. Before the gate, missing values were median-filled, which made the land block look
95 % complete while it was mostly imputed. Now a partially finished run cannot masquerade as a land
effect: with a 100-territory smoke table the gate dropped 6 of 7 covariate columns and kept only density,
which is exactly the correct behaviour.

## What is not claimed

Soil is absent. Terrain is absent in the current fallback run (the GLO-30 tile row at 80° N does not
exist; our polygons reach it only through the inflated marine extensions). Neither is load-bearing for any
contract, and both are stated rather than worked around with invented numbers.


## Official Rosstat BD PMO indicators (the socio-economic block)

`scripts/build_socio_covariates.py` builds `data/processed/covariates_socio.parquet` from five official
indicators of the BD PMO collection 118 (version v20250918): population as of 1 January (`Y48112027`),
headcount by economic activity (`Y48423005`), accrued monthly wage by economic activity (`Y48423007`),
the share of tax and non-tax revenue in local budgets (`Y48013015`) and the share of local public roads
in good condition (`Y48106005`). The files come from the `tochno-st-catalog` mirror on
`storage.yandexcloud.net`, because `ws.rosstat.gov.ru` has no address in the authoritative zone of
`rosstat.gov.ru` (proved over DNS-over-HTTPS against the zone nameserver, see `BLOCKERS.md` on the meta
branch). Licensing is CC BY 4.0 and the citation is recorded per column in a `prov_` column.

Two defects were caught during this build and are worth repeating, because both produce a number that
looks measured while being an artifact:

* population is reported under the period label `На 1 января` and has no full-year row, so filtering on
  `Январь-декабрь` returned an all-NaN column with 0% coverage;
* wages are reported per OKVED2 activity and must be weighted by per-activity headcount. Deduplicating
  on OKTMO alone collapsed the activities, and aggregating with national totals repeated on every row
  handed all 2,590 territories the identical value 97,461 rubles. The fix computes a per-territory
  weighted ratio, which yields min 27,107 / median 58,763 / max 408,518 rubles.

The feature builder puts every column through the same 50% coverage gate used for the land block; all
five passed (0.98, 0.99, 0.99, 0.91 and 0.85 respectively).

## What the official block does to the land thesis

`scripts/validate_land_hypothesis.py` now reports R^2 for access-only, socio+access, land+access and
land+socio+access over 3 region-grouped folds. The number that matters is the gain of land **over the
socio+access baseline**, since that is the combination official statistics make possible and the one any
well-resourced entry will build: mean gain +0.1107. Per target it is +0.197 for health, +0.141 for
marketplaces, +0.102 for transport, +0.070 for public catering and +0.043 for food retail. Wages and
population add little over market access on their own (+0.006 to +0.040), and on health they make the
access-only model worse, so the geospatial block is not a decoration on top of the socio-economic one -
it carries information the official indicators do not have.

## Dynamic World: reached, not yet used - and the earlier failure was our own typo

`NASA/DYNAMICWORLD/V1` was the identifier this repository first asked with. It was invented here rather than
taken from the catalog, and the "asset not found" it produced was reported as a fact about our access and as
evidence that Earth Engine could not be reached from here at all. Both conclusions are withdrawn: they
described a name we had made up. The catalog name is `GOOGLE/DYNAMICWORLD/V1`, and asked that way it answers.

Measured on 2026-10-08 by `scripts/probe_dynamic_world.py`, recorded in
`artifacts/geo/dynamic_world_probe.json`: **78 images** fall inside 2023-01-01 to 2024-12-31 overlapping a
point in Moscow; ten bands, nine class confidences and a label; a real class-confidence read at that point
returns built 0.0931, bare 0.0638, crops 0.0408, the sample image dated 2023-01-06. A point in Kaliningrad
returns no data, which is recorded rather than smoothed over. The total size of the collection was not asked:
`size()` over the whole catalog collection ran past a 150 second timeout, so the probe asks only for images
near a point inside the window, which the index can answer.

It is still not built into a covariate layer, so the requirement remains unmet - reaching a collection is not
the same as having used it. The role it was named for, land cover changing inside the observation window
rather than a single snapshot, is currently carried by the GLC-LC100 monthly 2019 layer, MODIS `MCD12Q1`
class shares and `MCD64A1` burn dates. `COPERNICUS/Landcover/100m/Proba-V/Global`, the other collection named
for that role, holds one composite dated 2015-01-01 and nothing inside the window, so it cannot carry a
within-window signal at all.
