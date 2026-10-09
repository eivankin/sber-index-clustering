# The territory master, and the geometry defect it exposes

The master table is 2,660 territories: one polygon per `territory_id`, EPSG:4326, zero geometrically
invalid geometries, and **191 multi-part** polygons (exclaves are real in Russian municipal geography, not
an artefact). It is built from the OSM municipal polygon set that SberIndex ships with the hackathon
package, joined to Sber's `connection` graph (2,574 territories) and `market_access` (2,571), and to the
municipality dictionary (`t_dict_municipal_districts.xlsx`, 3,101 rows, ids 1…3101, 85 distinct region
names, OKTMO codes in the form `AA-BBB-CCC-DDD`).

## Join coverage, recomputed

```
geo 2660  ⊇  connection 2574  ⊇  market_access 2571
consumption 2190 (0 of them missing geometry)
G-C = 470 polygons with geometry but no consumption
K-C = 401 graph ids with no consumption
C-K = 17   C-M = 20
core (all three sources) = 2170 ; 2016 territories have all 24 months observed
```

These are the numbers every later artifact is checked against, and `scripts/check_master.py` reproduces
them from the raw inputs rather than trusting the stored table.

## The defect

Summing the municipal polygon areas gives **40,001,827.79 km²**. Their union is **39,432,101.33 km²**.
Russia's land area is about **17.1 million km²**. The polygons therefore claim roughly 2.3× the country's
land, because municipal boundaries in this dataset extend to the edge of the territorial sea and, for the
Arctic municipalities, far past it.

Slices of the excess:

| slice | km² |
|---|---|
| Kara Sea + Laptev Sea extension | 4,789,810.83 |
| Black Sea + Caspian extension | 663,649 |
| north of 78° N | 940,550 |

Single territories make the absurdity concrete: Krasnoyarsk Krai's municipal polygons total **6,967,718
km²** against an official subject area of **2,360,000 km²** — an inflation of **2.95×** — and the
Taymyr municipal district alone covers **3,738,134 km²**, more than all but a handful of countries.

Consequences that were applied rather than noted:

1. **No density may use raw OSM municipal area.** Per-km² figures use official Rosstat land area (BDPMO
   **Y48006001**, obtained from the `tochno-st` mirror) and, where a municipal value is missing, the
   official subject area as a documented proxy. 2,426 territories matched the official register
   (match rate 0.912), 234 did not, and **2,085** carry an `area_suspect` flag where the municipal figure
   exceeds or contradicts the subject total.
2. **Marine area is not land.** Land-cover shares are taken over classified pixels only, with a
   per-territory `*_unclassified_pct`, so a sea-covered polygon cannot dilute a land statistic silently.
3. **Some "unmeasured" territories are not missing data at all.** Typing all 470 polygons that have
   geometry but no consumption gives: `geometry_artifact_sea_or_extremity` **231**,
   `unmeasured_large_low_density` 106, `unmeasured_rural` 75, `unclassifiable_no_official_area` 57,
   `geometry_artifact_fragmented` 1. Counted across all 470 the largest regional totals are Rostov Oblast
   55, Krasnodar Krai 46 and the Republic of Dagestan 39 - but most of those are sea polygons. Among the
   **181** genuinely measurable ones the concentration is Krasnodar Krai 38, Rostov Oblast 36, Republic of
   Dagestan 33: a real coverage gap in the North Caucasus and the South, not a nationwide hole.

## Implementation note that cost an hour

A GeoDataFrame must not have a column named `area`: geopandas shadows it and the value becomes the
geometry's own measurement rather than the official one. The master therefore carries `osm_area_km2`,
`official_land_area_km2` and `official_subject_area_km2` explicitly, and the BDPMO CSV turned out to be
UTF-8 with `;` separators rather than the cp1251/comma assumption that Rosstat's own documentation
suggests.
