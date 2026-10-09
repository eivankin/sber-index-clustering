# Why this attribute space

*Status: deliverable for the clustering track, criterion 1 ("обоснование выбора пространства атрибутов").*

## What is being clustered
Municipalities of the Russian Federation, described by **how residents spend** and - in the second
stage - by **what the land under them produces**. 2,016 territories have a complete 24-month series
(2023-01..2024-12) in the SberIndex consumption dataset and form the modelling panel; 2,131 with at
least 12 months form the reporting panel. Nothing is silently dropped: the shorter series are listed
in `artifacts/tables/panel_by_region.csv` with their `months_available`.

## The feature groups, and the three attribute spaces built from them

| group | columns | what it answers |
|---|---|---|
| levels | `total_log_mean` + `*_log_mean` per category | how much is spent, per resident per month |
| shares | `*_share` per category (category mean / `Все категории` mean) | how the spending is distributed across categories |
| dynamics | `total_cv`, `h2_over_h1`, `dec24_over_mean`, `total_slope_rub_per_month` | how stable or how volatile the pattern is |

Levels alone cannot separate a rich agricultural district from a rich industrial one; shares alone
cannot separate a small but fast-growing district from a large stagnant one. Levels give scale,
shares give composition, dynamics give trajectory - that is the reason all three groups exist, and
`density` is a fourth block rather than a relabelling of `levels`: `<category>_per_official_km2`, raw
mean roubles per municipality divided by the official Rosstat land area, never by the OSM polygon area
that this repository measures as inflated. It exists because the rubric's "production" is read per
capita while our claim is about production per square kilometre, so territory size has to enter the
attribute space on its own terms instead of hiding inside a logged total. Clustering therefore runs
three times - `levels+shares`, `levels+shares+land`, `density+shares+land` - and the third space was
previously an artifact with no recorded producing command, which is what adding this block fixes.

the reason we compare `levels`, `levels+shares` and `levels+shares+dynamics` rather than assuming
one of them is right.

## Why these transformations
* `log1p` on levels: the distribution of spending per resident is strongly right-skewed (Moscow and
  city okrugs sit far out on the tail). A raw level would let a handful of territories dominate the
  variance and pull the cluster centroids.
* rank/quantile normalization to a normal distribution inside `cluster.prepare`, then z-scoring: this
  removes the influence of extreme values without discarding them, and makes Euclidean distance
  meaningful for `kmeans`, `ward` and silhouette-based criteria.
* shares instead of raw category levels: a category level is mostly a copy of the total level. The
  share is the part that carries information about consumption structure, which is what the land
  hypothesis is about.
* no imputation, no zero-filling: a missing month stays NaN in the long panel and is excluded from
  the modelling panel by the documented threshold. Zero-filling would invent "zero spending", which
  is factually wrong.

## What the land side is for (the falsifiable part)
The claim under test: **the composition of spending is explained by what the land produces, out of
sample.** For each territory we compute land-cover class shares over *classified pixels only*
(ESA WorldCover 10 m, Copernicus GLS-LC100 100 m), night-light intensity and lit-pixel fraction
(VIIRS DNB), burned area and greenness (MODIS), climate (WorldClim) and elevation (Copernicus DEM).
Then we check whether agricultural/tree/built-up shares predict the food/health/marketplace shares
of territories that were **held out** from the fit, and we compare that against a baseline that uses
population and market access only.

## Data provenance and licence
SberIndex consumption, connection and market-access data are used under **CC BY-SA 4.0**
(licence text: `data/raw/sber_hackathon/Данные_СберИндекс_лицензия.pdf`). The municipality
attribute dictionary and its polygon layer come from the same package. Land area denominators use
Rosstat's БДПМО indicator Y48006001 where it matched, with a documented fallback for the 234
territories it does not cover. External layers are cited in `configs/gee_sources.yaml`.

## How the land block is admitted into the model
Three rules keep the land side from becoming an artefact of missing data:

1. **Denominator.** Land-cover shares are computed over classified pixels only, and every territory
   carries an explicit `*_unclassified_pct`. Unclassified area is never counted as some class.
2. **Density denominator.** Per-km2 spending uses **official** Rosstat land area (БДПМО Y48006001,
   2,015 of 2,190 model territories), the subject's official area as an explicitly labelled proxy for
   42 more, and nothing at all for 133. The OSM municipal polygon area is never used: the master table
   shows the polygons sum to 40.0M km2 against roughly 17.1M km2 of RF land, so a per-km2 figure built
   on them would be meaningless.
3. **Coverage gate.** A covariate column enters the feature space only when it has non-null values for
   at least 50% of model territories (`cluster.land_columns`, gate reported in
   `artifacts/features/feature_space_report.json`). A partially finished covariate run therefore cannot
   masquerade as a land effect: with the 100-territory Earth Engine smoke table the gate dropped 6 of 7
   covariate columns and kept only the density. Before the gate, missing values were filled with the
   column median, which would have made 95% of the "land" block imputed.

The same asymmetry is what makes the network defensible: **land enters only the supply side** of the
edge rule (`supply_i` = cropland + 0.5 grazing + 0.25 forest, divided by classified coverage, rescaled
so its median is 1.0), while **demand stays Sber's own** market-access index. Night lights and built-up
fraction are used as validation variables, never inside the weight, so the network is not built from
the quantity it is later asked to explain.

## Deliberate exclusions
* Raw monthly series as 24 separate features: would make the dimensionality equal to the number of
  months and let noise drive the partition.
* Prices, tariffs or infrastructure indicators from other sources: not part of the mandatory Sber
  dataset, and the rules require SberIndex to stay the analytical core.
* Territory names/region codes as features: they are categorical labels, not measurements; region is
  kept as an *interpretation* dimension only.
* Rosstat wages, employment and population: not obtainable from this environment (see BLOCKERS W7 -
  `rosstat.gov.ru` delivers an unverifiable GOST chain). Scale and demand are carried by Sber's
  `market_access`, the spending level and the land-based density instead, and this substitution is
  stated in the methodology report rather than hidden.
