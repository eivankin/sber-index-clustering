"""Cluster change over time - the formal part required by the contest.

Definitions used throughout:

* Attribute space per period. For period p in {2023, 2024} the vector of a territory is built the
  same way as in the static analysis (log1p levels, category shares), normalized with the *static*
  quantile transform so that the two periods live in the same coordinates. Comparing clusters across
  differently-normalized spaces would be meaningless.

* Membership flow matrix. F_ij = |{t : c_2023(t)=i and c_2024(t)=j}| / N. Row sums are the 2023
  cluster sizes, column sums the 2024 sizes. F is the primary object: every statement about "how
  clusters changed" is read off F, not invented.

* Cluster correspondence. Clusters are unlabeled, so "cluster 2 became cluster 1" is only meaningful
  after a matching. We match by maximum Jaccard overlap J_ij = |A_i ∩ B_j| / |A_i ∪ B_j| (greedy,
  one-to-one), and report the matched pairs plus the residual mass that could not be matched.

* Centroid drift. For a matched pair (i, j), delta_ij = mu_i(2023) - mu_j(2024) in the normalized
  space; its norm is the drift magnitude. Drift without membership flow means the cluster's economic
  profile moved while its members stayed; flow without drift means members moved between profiles
  that stayed put. Both are reported, because they are different phenomena.

* Stability. stability = sum_i F_ii (fraction of territories that keep their cluster). Adjusted Rand
  index and Jaccard between the two partitions are reported as chance-corrected alternatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import PROCESSED
from .panel import CATEGORIES, TOTAL

PERIODS = {
    "2023": [f"2023-{m:02d}" for m in range(1, 13)],
    "2024": [f"2024-{m:02d}" for m in range(1, 13)],
}


def period_features(min_months: int = 12):
    """Per-period level/shares features from the long panel, for territories observed long enough."""
    long = pd.read_parquet(PROCESSED / "spending_long.parquet")
    rows = []
    for period, months in PERIODS.items():
        sub = long[long.month.isin(months)]
        observed_months = sub.groupby("territory_id")["month"].nunique()
        keep_ids = observed_months[observed_months >= min_months].index
        piv = sub.pivot_table(
            index="territory_id", columns="category", values="value", aggfunc="mean"
        ).reindex(columns=CATEGORIES + [TOTAL])
        piv = piv.loc[piv.index.isin(keep_ids)]
        total = piv[TOTAL]
        base = total
        row = pd.DataFrame({"tid": piv.index, "period": period})
        row["total_log_mean"] = np.log1p(base)
        row["total_mean_rub"] = base
        row["months_observed"] = observed_months.reindex(piv.index)
        row["composition_cv"] = piv[CATEGORIES].std(axis=1) / base
        for cat in CATEGORIES:
            row[f"{cat}_log_mean"] = np.log1p(piv[cat])
            row[f"{cat}_share"] = piv[cat] / base
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def category_transitions(
    delta: float = 0.05,
    persist: int = 3,
    min_months: int = 20,
    covariates: pd.DataFrame | None = None,
):
    """Spending-type transitions with a persistence rule, 2023 -> 2024.

    For territory i and category c let s(i,c,m) be the category's share of total spending in month m.
    baseline b(i,c) = mean share over the 2023 months, current a(i,c) = mean over the 2024 months, and
    D = a - b. A transition is recorded only when |D| >= delta AND the deviation keeps the same sign as
    D for at least `persist` consecutive months taken from the end of 2024 - a single spiky month is not
    a transition. Territories observed in fewer than `min_months` of the 24 months are excluded so that
    ragged series cannot generate transitions.

    Every row carries the spending-side change (D, direction, persistent months) and, when a covariate
    table exists, the physical evidence used to decide credible versus noise. With only single-year
    covariates the physical column is reported but the verdict stays `pending_physical_evidence`: a
    spending change cannot be called geospatially corroborated until a change layer (GLS
    change-confidence, MODIS burn, Hansen loss year) says the land actually moved.
    """
    long = pd.read_parquet(PROCESSED / "spending_long.parquet")
    total = long[long.category == TOTAL].set_index(["territory_id", "month"])["value"]
    piv = long.pivot_table(
        index="territory_id", columns=["category", "month"], values="value", aggfunc="mean"
    )
    months = sorted(long.month.unique())
    ids = np.sort(pd.to_numeric(piv.index, errors="raise").to_numpy())

    tot = np.full((len(ids), len(months)), np.nan)
    idx = pd.MultiIndex.from_product([ids, months], names=["territory_id", "month"])
    tot[:] = total.reindex(idx).to_numpy().reshape(len(ids), len(months))

    m2023 = [m for m in months if m in PERIODS["2023"]]
    m2024 = [m for m in months if m in PERIODS["2024"]]
    i2023 = [months.index(m) for m in m2023]
    i2024 = [months.index(m) for m in m2024]

    rows = []
    for cat in CATEGORIES:
        block = piv.xs(cat, level="category", axis=1).reindex(columns=months).reindex(ids)
        S = block.to_numpy(dtype="float64") / tot
        base = np.nanmean(S[:, i2023], axis=1)
        cur = np.nanmean(S[:, i2024], axis=1)
        D = cur - base
        dev = S[:, i2024] - base[:, None]
        sign_ok = np.sign(dev) == np.sign(D)[:, None]
        # tail run: consecutive months from the end of 2024 that agree with the direction of D, per
        # territory - once a territory's month disagrees, its run is frozen at the length reached.
        tail = np.zeros(len(ids), dtype=int)
        stopped = np.zeros(len(ids), dtype=bool)
        for j in range(len(i2024) - 1, -1, -1):
            ok = np.isfinite(S[:, i2024[j]]) & sign_ok[:, j]
            tail = np.where(stopped, tail, np.where(ok, tail + 1, tail))
            stopped = stopped | (~ok)
        months_seen = np.isfinite(S[:, i2024]).sum(axis=1) + np.isfinite(S[:, i2023]).sum(axis=1)
        flag = (np.abs(D) >= delta) & (tail >= persist) & (months_seen >= min_months)
        sub = pd.DataFrame(
            {
                "tid": ids[flag],
                "category": cat,
                "baseline_share": np.round(base[flag], 4),
                "current_share": np.round(cur[flag], 4),
                "delta": np.round(D[flag], 4),
                "direction": np.where(D[flag] > 0, "up", "down"),
                "persistent_months": tail[flag],
                "months_observed": months_seen[flag],
            }
        )
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "tid",
                "category",
                "baseline_share",
                "current_share",
                "delta",
                "direction",
                "persistent_months",
                "months_observed",
                "strength",
                "physical_evidence",
                "verdict",
            ]
        )
    out["strength"] = np.where(
        out.persistent_months >= 6,
        "strong",
        np.where(out.persistent_months >= 4, "moderate", "thin"),
    )
    if covariates is not None:
        covariates = covariates.copy()
        # Earth Engine returns the territory id as a string; transitions carry it as an integer
        covariates["tid"] = pd.to_numeric(covariates["tid"], errors="coerce").astype("Int64")
        covariates = covariates.dropna(subset=["tid"]).drop_duplicates("tid")
        phys = [
            c
            for c in covariates.columns
            if c.startswith(("gls_", "wc_", "burn_", "mcd_")) and not c.startswith("prov_")
        ]
        cols = phys[:4]
        if cols:
            merged = out.merge(covariates[["tid"] + cols], on="tid", how="left")
            out["physical_evidence"] = merged[cols].apply(
                lambda r: "; ".join(f"{k}={v:.4g}" for k, v in r.items() if pd.notna(v)), axis=1
            )
        else:
            out["physical_evidence"] = "covariate table has no land-cover columns"
    else:
        out["physical_evidence"] = "no covariate table present"
    out["verdict"] = np.where(
        out.physical_evidence.str.startswith(("wc_", "gls_", "burn_", "mcd_")),
        "pending_physical_evidence",
        "unverifiable_no_physical_layer",
    )
    return out.sort_values(["category", "delta"], ascending=[True, False]).reset_index(drop=True)


def build_matrices(periods: pd.DataFrame):
    """Return {period: (tids, Z, cols)} using one shared quantile normalization."""
    from sklearn.preprocessing import QuantileTransformer

    cols = (
        ["total_log_mean"]
        + [f"{c}_log_mean" for c in CATEGORIES]
        + [f"{c}_share" for c in CATEGORIES]
    )
    out = {}
    X_all = np.vstack([periods[periods.period == p][cols].to_numpy("float64") for p in PERIODS])
    X_all = np.nan_to_num(X_all, nan=np.nanmedian(X_all, axis=0))
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X_all)))
    qt.fit(X_all)
    for p in PERIODS:
        g = periods[periods.period == p]
        X = g[cols].to_numpy("float64")
        X = np.nan_to_num(X, nan=np.nanmedian(X_all, axis=0))
        out[p] = (g.tid.to_numpy("int64"), qt.transform(X), cols)
    return out


def flow_matrix(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    F = np.zeros((ka, kb))
    for i in range(ka):
        for j in range(kb):
            F[i, j] = np.sum((a == i) & (b == j))
    F = F / len(a)
    return pd.DataFrame(
        F, index=[f"c2023_{i}" for i in range(ka)], columns=[f"c2024_{j}" for j in range(kb)]
    )


def match_clusters(F: np.ndarray):
    """Greedy one-to-one matching by maximum Jaccard overlap. Returns pairs and residual mass."""
    _ka, _kb = F.shape
    A = F.sum(axis=1, keepdims=True)
    B = F.sum(axis=0, keepdims=True)
    J = F / np.maximum(A + B - F, 1e-12)
    used_a, used_b, pairs = set(), set(), []
    order = np.dstack(np.unravel_index(np.argsort(-J, axis=None), J.shape))[0]
    for i, j in order:
        if i in used_a or j in used_b:
            continue
        if J[i, j] <= 0:
            break
        pairs.append((int(i), int(j), float(J[i, j])))
        used_a.add(int(i))
        used_b.add(int(j))
    matched_mass = sum(F[i, j] for i, j, _ in pairs)
    return pairs, float(matched_mass), J


def centroid_drift(Za: np.ndarray, Zb: np.ndarray, a: np.ndarray, b: np.ndarray, pairs):
    rows = []
    for i, j, jacc in pairs:
        ma = Za[a == i].mean(axis=0)
        mb = Zb[b == j].mean(axis=0)
        d = mb - ma
        rows.append(
            {
                "c2023": i,
                "c2024": j,
                "jaccard": round(jacc, 4),
                "drift_norm": round(float(np.linalg.norm(d)), 4),
                "n_2023": int((a == i).sum()),
                "n_2024": int((b == j).sum()),
                **{
                    f"delta_{c}": round(float(v), 4)
                    for c, v in zip(["dim" + str(k) for k in range(len(d))], d)
                },
            }
        )
    return pd.DataFrame(rows)


def summarize(a: np.ndarray, b: np.ndarray) -> dict:
    F = flow_matrix(a, b)
    pairs, matched, J = match_clusters(F.values)
    return {
        "n": len(a),
        "k_2023": int(a.max()) + 1,
        "k_2024": int(b.max()) + 1,
        "stability_F_ii": round(float(np.trace(F.values)), 4),
        "adjusted_rand": round(float(adjusted_rand_score(a, b)), 4),
        # NMI alongside ARI: the objective asks for both, and they answer different questions.
        # ARI counts pairs that stay together or apart after relabelling, so it is strict about
        # whether the partition itself is preserved; NMI only asks how much the two labellings
        # share information, so a genuine re-cut with stable membership can score high on NMI
        # while ARI stays modest. Reporting one without the other hides which kind of stability
        # the clustering actually has.
        "normalized_mutual_info": round(float(normalized_mutual_info_score(a, b)), 4),
        "matched_pairs": len(pairs),
        "matched_mass": round(matched, 4),
        "max_jaccard": round(float(J.max()), 4),
        "flow": F.round(4).to_dict(),
    }
