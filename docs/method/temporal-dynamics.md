# Temporal dynamics: stability, transitions, and what counts as evidence

Everything here is reproducible from `configs/dynamics.yaml` plus `scripts/run_dynamics.py`; the
independent recomputation lives in `scripts/check_dynamics.py`, which rebuilds monthly shares from the
raw consumption table and re-derives the deltas rather than trusting the stored ones.

## Definitions

A territory's **spending share** for a category is its spend in that category divided by its total spend
in the same month. All shares are in `[0, 1]`.

We compare two windows:

* **baseline** `b` — the mean of the monthly shares over 2023 (12 months),
* **current** `a` — the mean of the monthly shares over 2024 (12 months),
* **delta** `D = a - b`.

`b` and `a` are **means of ratios**, one ratio per month, then averaged. They are not the ratio of two
means. The difference matters: a ratio-of-means recomputation initially disagreed with the stored delta
on 59 of 10,000 rows (`max|diff| = 0.0059`), which is what forced `check_dynamics.py` to adopt the same
mean-of-ratios definition.

A territory must have at least `min_months = 20` observed months to be considered. A territory with
fewer months cannot generate a transition at all — it is excluded rather than imputed, because an
imputed share would fabricate a trend.

## The persistence rule

A monthly change that appears once is not a shift; it may be one unusual month, a promotion, or a data
outage. So a delta is only declared a transition when the same direction persists at the end of 2024:

1. For each month `m` of 2024 compute the deviation `d_m = share_m - b`.
2. Walk backwards from December 2024 while the sign of `d_m` matches the sign of `D`.
3. The run length `L` is the number of consecutive months with the same sign.
4. The transition is **strong** when `L >= persist_months = 3` and `|D| >= delta_min = 0.05`; otherwise
   it is recorded but not counted as a shift.

The run is computed per territory and frozen there: it does not borrow persistence from a neighbouring
territory or from the category average.

## Stability

Partitioning the same territories by 2023 features and by 2024 features gives an **ARI of 0.0147** and a
label-stability of **26.5 %**. Two consecutive years of spending therefore rearrange the clusters almost
completely. The consequence for the entry: the static pooled partition is the primary result, and the
year split is reported as a sensitivity check rather than as a second finding.

## What the transitions say

546 transitions were declared:

| direction | count | mean delta |
|---|---|---|
| Маркетплейсы up | 523 | +0.0603 |
| Продовольствие down | 21 | -0.0576 |
| Продовольствие up | 2 | +0.0584 |

All 546 are `strong` under the rule above. The dominant story is marketplace share displacing grocery
share, and it is broad rather than local: it appears across most federal districts.

## Tracking the partition itself across time

The brief asks for a mathematical approach to following how clusters change over time. The approach we
use is a partition sequence with three measured quantities, and it is implemented rather than
described: `scripts/run_monthly_clustering.py` writes `artifacts/dynamics/monthly_clusters.json`, and
the per-month assignments ship as base64 so every figure in it is recomputable from the committed file
alone - the feature and spending tables it was built from are build products and are not tracked.

Let `P_m` be the clustering of the `levels+shares` block over the model-panel territories, built only
from data available up to month `m` and never from later months, with the method and `k` the published
partition used (`kmeans`, `k=3`). Because `kmeans` numbers its clusters in an order that carries no
meaning, `P_m` is aligned to `P_{m-1}` by greedy maximal overlap - the pair of groups sharing the most
territories is matched first - so that a colour change means a real change and not a renumbering.

Three quantities follow:

* **Agreement.** `ARI(P_m, P_published)` and `NMI(P_m, P_published)`, plus agreement with the previous
  month. Across 24 months the mean ARI against the published partition is **0.9263**, rising from
  **0.7713** in January 2023 to **1.0** in December 2024: a partition built on a partial window already
  resembles the published one, and by the end of the window it is the published one.
* **Churn.** `c_m = |{t : label_m(t) != label_{m-1}(t)}|`, reported as a share of the 2 016 territories.
  Mean **0.99%** per month, minimum 2 territories, maximum 44.
* **Label stability.** Per territory: how many distinct labels it is ever given, and how many times it
  changes. **1 831** of 2 016 never change their number; 185 change at least once; the median number of
  distinct labels seen is 1, the maximum 3.

This separates two phenomena that a single monthly re-colouring would blur together. *Label churn* is a
territory moving between clusters - rare, and the reason the map's colour does not move with the
slider. *Profile drift* is a territory staying in the same cluster while its spending per official
square kilometre changes - which is exactly what the slider animates, and what the month-by-month
presence bitmap makes distinguishable from zero spending.

A separate result belongs here rather than being folded into the monthly series: partitions built on
the 2023 window and the 2024 window agree barely at all - ARI 0.0147 and NMI 0.0097 for `kmeans`, 0.0136
and 0.0122 for `ward` (`artifacts/dynamics/dynamics_summary.json`, under `kmeans` and `ward`, quoted in
`artifacts/landing/map_timeline.json` as `partition_agreement_2023_vs_2024`). The year chosen as the
observation window is itself a modelling decision that visibly changes the partition; that is worth
saying out loud even though the land-side claim does not depend on it.



A spending shift becomes credible only if a physical land signal changed inside the same window
(2023-01 to 2024-12). The available layers were separated by whether they can possibly say something
about that window:

* **Within-window**: MODIS `MCD64A1` burn dates for 2023, Hansen tree-cover loss/gain year if the tiles
  are ever downloaded. These are the only signals that can corroborate.
* **Prior only**: the GLS-LC100 change-confidence band. It describes **2000-2015**, which ends eight
  years before the spending window. It can describe the background a territory changed within, but it
  can never corroborate a 2023-2024 shift, and `scripts/corroborate_transitions.py` refuses to use it
  that way.

A within-window layer **did** arrive: MODIS `MCD64A1` burn dates for 2023, aggregated per territory as
`burn_km2` and `burn_pct_of_territory`. Against it the 546 transitions split into **303 corroborated**
(at least 0.5 % of the territory's official area burned in 2023) and **243 land_stable** (no burn signal
at all). The `verdict` column in `transitions.csv` still reads `pending_physical_evidence` on every row,
because the rule below reserves `credible` and `noise` for a verdict that Hansen loss/gain years can
also support. Both layers are now on disk and reach **97.4 %** of the transition districts, so coverage
was never the lasting obstacle: `scripts/corroborate_transitions_hansen.py` ran over the full panel and
returns **8 corroborated, 1 refuted and 537 unresolvable**, because **523 of the 546 hypotheses named built-up
land**, and only three of those districts show a built-share change of even one percentage point between 2023 and
2024. The other classes did move, and that has to be said here rather than left for a reader to discover: crops
changed by a point in **238** of the 526 transition districts with coverage, tree cover in **384**. Nationally the
movement is larger still - tree cover by a point in **1458** of 2016 model districts, crops in 667, built-up in 326
(`artifacts/validation/dw_temporal_signal.json`). So the finding is not that the land did not move; it is that the
land did not move in the way the named hypothesis expected. The honest reading is that the spending
composition shifted while built-up land beneath it did not measurably move - a finding about the hypothesis, not
a missing-data excuse.

Two bugs in that script are named rather than buried, because they produced a headline that was wrong.
The first: when the class change was too small, the script fell back to "forest loss above 0.5 % therefore
corroborated" - 492 of 546 rows were called corroborated on that basis, with a median built-up change of
**−0.0002**, meaning the share had actually fallen. Forest loss measures clearing, not class composition, so
accepting it answered a different question and presented it as answering this one. The second: the threshold
was written as `1.0` and described as percentage points, while the confidences in this table are fractions, so
it demanded a change of one hundred points - unreachable by construction, which is what made the fallback the
branch that decided almost every row. Both are fixed: the threshold is now `0.01` in the units the data uses,
and a change below it is recorded as unresolvable with the measured value written beside it, while forest loss
stays in its own column as an observation about clearing rather than evidence about share.

That label is deliberate: `credible` and `noise` are both claims about evidence we do not have, and
writing either would be fabrication.

## Known limitation

`MCD64A1` carries only `BurnDate`, `Uncertainty`, `QA`, `FirstDay` and `LastDay` — there is no burned-area
fraction band on that asset, so burn evidence is expressed as dates and area derived from the per-pixel
class counts, not as a percentage-cover column. `ISRIC/soilgrids` does not resolve, but that sentence was
written before the collection was asked by its catalog name: `ISRIC/SoilGrids250m/v2_0` answers under the
registered project, with depth-stratified property bands scaled by 0.001, and is being brought into the panel
by `scripts/build_added_layers.py`. The property behind those bands is not named by the collection's own
metadata, so it is identified by value range - 87-209 in Voronezh, 115-276 in Krasnodar, which is organic
carbon concentration rather than texture, pH or bulk density - and the reasoning is recorded rather than
presented as a catalog statement.
