# Edge rules: a directed complementarity edge against a similarity baseline

Every other public entry in this contest builds municipal networks from *similarity*: two territories are
linked when their indicator vectors are close. That produces cliques — groups of territories that resemble
each other — and nothing about how places depend on one another.

A verified scan of 34 public SberIndex 2026 repositories narrows that sentence honestly: distance
decay is **not** unoccupied. At least one public entry already runs a road-gravity rule of the form
`road_gravity(D_road, lam_km=200)`, another uses `exp(-d/scale)` over a mean-DTW distance, a third
writes `w = exp(-D^2/(2sigma^2))` and `w = 1/(1+D)` over a weighted-L1 economic distance, and several
use knn/top-k sparsification and the word `directed`. Named attribution for each of these is held in
internal working material that is deliberately not part of the deliverable branch. Within the reach
of that scan, none of them make the weight a **product of a production-supply term and a demand
term**, and none derive supply from land cover; where the scan could not read a repository, the claim
is not made.
land cover at all. Our novelty is that composition, not the decay, not the sparsification, not having
several rules.

## The time-alignment family, what it cost to build, and the story it did not find

The brief names dynamic time alignment, so a family called `dynamic_alignment` was added: monthly totals are
summed across spending categories first, log1p-transformed and z-scored, compared over a bounded lag of −3 to
+3 months, and the warping distance is the minimum over those lags. The weight is
`exp(-D/h) · exp(-d/600)` with `h = 0.2623`, the scale of the log-distance distribution, so a pair that moves
together is worth more than a pair that merely sits near each other. Candidates are the 50 nearest districts by
Sber's own stored road distance, symmetrized and deduplicated: **160,054** candidate pairs, of which **115,352**
were scored and **44,702** were refused for having fewer than twelve months in common. Missing months are never
filled - a district that stopped reporting does not get an invented series to align against.

Structurally it behaves like the other families. Dense: **57,676** edges, density **0.016309**, **607** weak
components. Sparsified to ten neighbours: **15,136** edges, reciprocity **0.5225**, degree gini **0.3547**,
label-propagation modularity **0.7213**. The bounded lag is an approximation, and the approximation was checked
rather than trusted: against full warping on 400 complete series pairs, Spearman correlation is **0.9702**.

**The null result is the finding.** The offset that wins is **0 for 99.0 % of scored pairs**, and the mean
winning offset is **0.0 months**. In other words, districts whose spending moves together do so at the same time,
not one after another; the leader-and-follower structure this family was built to look for did not appear in the
spending window. That is reported rather than smoothed over, because presenting it as a discovered lead-lag
structure would be a claim about a quantity that measurement says is absent.

One defect in building it is worth keeping in the record: the node set has to be the master table's 2,660 sorted
territory ids. Constructing the family on the spending table's 2,190 rows instead made degree gini read **0.2162**
where the correct figure is **0.3547** - a structural statistic describing whoever happened to have spending data
rather than the country, which is the population every other family is measured over.

This family is wired into `scripts/check_edges.py`, which is why the edge contract now reads **99/99** rather than
the 85/85 it read before. Separately, `scripts/measure_edge_influence.py` measured what happens if edge-derived
features enter the clustering space, and the answer was that they should not: adding 21 network features moved
ARI to **0.1342**, churn to **81.69 %**, and separability from **5 of 15** contrasts to **0 of 15**, while each
family's own label-propagation communities agreed with the attribute clustering only at ARI **0.002-0.064**.

This entry's headline edge is **directed** and states a flow:

```
w_ij = f(d_ij) · supply_i · demand_j
```

`i` supplies what the land can produce, `j` demands what it can absorb, and `f` decays with the road
distance SberIndex publishes in its `connection` graph (5,942,364 stored pairs; the full matrix implied by
the municipality dictionary is n = 3,102). Direction is not decoration: `w_ij != w_ji`, so the graph
carries an asymmetry that a similarity graph cannot express.

## The six families actually run

| family | rule |
|---|---|
| `complementarity_exp` | `f(d) = exp(-d / tau)` |
| `complementarity_power` | `f(d) = (1 + d)^-p` |
| `complementarity_trunc` | `f(d) = 1` for `d <= d_max`, else 0 |
| `similarity` | RBF kernel over spending features (the baseline every competitor uses) |
| `growth_correlation` | correlation of standardized month-over-month log growth of total spending |
| `lagged_leader` | cosine between the 2023 category shares of `i` and the 2024 shares of `j` |

each sparsified three ways — `dense`, `knn10`, `knn25` — which is 6 rules x 3 sparsifications = 18
measured variants, flattened into `artifacts/edges/edge_rules_summary.csv` and reproduced by 85
structural checks in `scripts/check_edges.py`. A run below six families is refused.

## A stated deviation on the demand side

The brief describes the headline edge as supply times a **nightlight or built-up** demand term times road
decay. This entry uses SberIndex's own `market_access` index as the demand term instead, for a reason
worth stating rather than burying: night lights and built-up fraction are used elsewhere in this work as
**validation** variables, never as edge weights. If they were also the demand term, the edge and the
evidence used to check it would be built from the same signal, and a agreement between them would prove
nothing. Using the official access index keeps the edge and the validation independent, and it keeps the
demand term inside the mandatory SberIndex data. The deviation is therefore deliberate, recorded here and
in `configs/edge_rules.yaml`, and it is the kind of thing a reviewer should be able to find without
digging.

## Supply and demand, and why they are kept apart

`supply_i = crops + 0.5 · grazing + 0.25 · forest`, taken over **classified pixels only** and divided by
the classified coverage of the territory, then median-rescaled so the median is 1.0. Rescaling to a median
of 1 changes only the ordering scale, not the global level: it prevents one country-wide unit decision from
dominating every weight.

`demand_j = market_access / 1000`, Sber's own index. Land never enters demand. If night-light or built-up
cover were used to build demand, and land cover also built supply, the edge would be measuring land against
itself — a circular definition that could not be falsified. Night lights and built-up fraction are
therefore kept outside the weight and used only as **validation** variables.

## Hub/catchment versus clique: measured, not asserted

On the knn-10 sparsification:

| family | reciprocity | clustering coeff. | degree gini | top-1 % degree share | LP communities | modularity Q |
|---|---|---|---|---|---|---|
| complementarity_exp | 0.3594 | 0.0046 | 0.3565 | 0.1720 | 147 | 0.7602 |
| complementarity_power | 0.7038 | 0.0203 | 0.1685 | 0.0468 | 231 | 0.7371 |
| complementarity_trunc | 0.1601 | 0.0020 | 0.4480 | 0.2571 | 118 | 0.7099 |
| similarity | **1.0000** | 0.0247 | **0.1144** | **0.0171** | 103 | **0.5996** |

The signature is exactly what the two constructions predict. The similarity baseline is perfectly
reciprocal (by definition, it is symmetric), the least concentrated and the least modular: a clique
structure has no reason to be strongly partitioned and no hubs. The directed complementarity families carry
partial reciprocity — mutual supply/demand pairs exist but are not universal — high degree concentration
(the trunc variant puts 25.7 % of all degree in the top 1 % of territories) and higher modularity, which is
the hub-and-catchment pattern: a few productive places serve many districts, and the community structure
follows that service map rather than visual resemblance.

## Sparsification effect

knn-10 removes **99.4 %** of the dense complementarity edges while the network keeps **90 components** and
a largest-component share of **0.9665**. That is the justification for sparsifying: it is measured on the
same diagnostics before and after, not claimed as a modelling preference.

## Node sets differ, and that is recorded

The complementarity families are built on all **2,660** polygon territories (89 of them end up isolated);
the similarity baseline runs on its own **2,190**-territory endpoint set, because it needs spending
features on both ends. Consequently **430 territories that have spending features never appear in the
similarity edge file**. `scripts/check_edges.py` recomputes density and mean degree on each family's own
node set, so the reported numbers are self-consistent rather than averaged over a set the family does not
use.


## The two families that read time

A static feature space cannot express time: it averages twenty-four months into one vector and loses
the ordering. Two further rules were added for that reason, and both reach all 2,660 territories rather
than the 2,190 the similarity baseline can cover.

**Growth correlation** takes month-over-month log change of each territory's monthly spending total,
standardizes each territory's growth series, and correlates them pairwise over 23 differences. Negative
correlations are clipped to zero: a negative weight would invert the meaning of an edge, turning
"these places move together" into a subtraction, and the network is defined on non-negative weights.
The result is undirected by construction, since correlation is symmetric. At kNN-10 it keeps 13,153
edges at density 0.00372, mean degree 9.89, with 654 territories left isolated.

**Lagged leader** compares the category-share vector of territory *i* in 2023 with that of territory *j*
in 2024, as a cosine over the five spending categories, and keeps the reverse orientation as its own
weight so that a leader edge and a follower edge stay distinguishable. It is directed in both senses,
clipped to non-negative, and damped by the same road-distance factor `exp(-d/600)` as the growth rule.
At kNN-10: 14,076 edges, density 0.00398, mean degree 10.58, the same 654 isolated territories.

Isolation here is not a defect. A territory whose spending never moves has no growth series to correlate
with anything, and a territory whose composition no neighbour resembles has no top-k partner. Both are
recorded as measured facts about the network rather than smoothed away.

## Two traps that produced wrong numbers while looking correct

* `pandas` 3 removed `groupby(axis=1)`. The category aggregation now transposes and groups, and tolerates
  the month level either surviving or disappearing after month selection.
* Sber territory ids are not matrix coordinates. The largest id in the municipality dictionary exceeds
  3,000 while the square matrix is 2,660 wide, so ids must be translated through the id to row-position
  map the complementarity families already use; using them directly raised `axis 0 index 3013 exceeds
  matrix dimension 2660`.
* The checker verifies each family on a declared node set. The complementarity kernels and both time-based
  families use all 2,660 territories; the similarity baseline uses only the territories it actually links.
  Reporting over one node set while the checker recomputes over another makes density, mean degree,
  largest component, isolated count and degree gini disagree without the graph being wrong.
