"""Spending panel: per-territory monthly consumption by category, plus derived model features.

NaN policy (documented, enforced here):
* A territory enters the modelling panel only if it has >= MIN_MONTHS_REPORT of the 24 months in
  2023-01..2024-12 for the "Все категории" series. Shorter series stay in the reporting panel with
  `months_available` so nothing is silently dropped.
* Missing category-months are NaN, never zero-filled: a zero would look like "no spending".
* Categories never sum to "Все категории"; shares are computed against the "Все категории" mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONSUMPTION

TOTAL = "Все категории"
CATEGORIES = ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
MIN_MONTHS_REPORT = 12


def load_consumption() -> pd.DataFrame:
    c = pd.read_parquet(CONSUMPTION)
    c["month"] = c["date"].astype("string")
    c["territory_id"] = c["territory_id"].astype("int64")
    c["value"] = c["value"].astype("float64")
    return c


def build_long(c: pd.DataFrame | None = None) -> pd.DataFrame:
    c = load_consumption() if c is None else c
    months = sorted(c.month.unique())
    grid = pd.MultiIndex.from_product(
        [sorted(c.territory_id.unique()), months, CATEGORIES + [TOTAL]],
        names=["territory_id", "month", "category"],
    )
    obs = (
        c[["territory_id", "month", "category", "value"]]
        .drop_duplicates(["territory_id", "month", "category"])
        .set_index(["territory_id", "month", "category"])["value"]
        .reindex(grid)
    )
    out = obs.reset_index()
    out = out.dropna(subset=["value"])
    out["value"] = out.value.astype("float64")
    return out


def build_features(c: pd.DataFrame | None = None) -> pd.DataFrame:
    c = load_consumption() if c is None else c
    months = sorted(c.month.unique())
    rows = []
    for tid, g in c.groupby("territory_id"):
        piv = g.pivot_table(index="month", columns="category", values="value", aggfunc="mean")
        piv = piv.reindex(index=months, columns=CATEGORIES + [TOTAL])
        total = piv[TOTAL]
        present = total.notna()
        row: dict[str, float | None] = {
            "tid": int(tid),
            "months_available": int(present.sum()),
            "months_expected": len(months),
        }
        if present.sum() == 0:
            rows.append(row)
            continue
        months_idx = np.arange(len(months))
        good = present.values
        # slope of the total series (per month, in RUB), on observed points only
        if good.sum() >= 3:
            x = months_idx[good]
            y = total.values[good]
            row["total_slope_rub_per_month"] = float(np.polyfit(x, y, 1)[0])
        base = np.nanmean(total.values[good])
        row["total_mean_rub"] = float(base)
        row["total_log_mean"] = float(np.log1p(max(base, 0.0)))
        row["total_cv"] = float(np.nanstd(total.values[good]) / base) if base else None
        row["total_max_rub"] = float(np.nanmax(total.values[good]))
        row["total_min_rub"] = float(np.nanmin(total.values[good]))
        h1 = total.loc[[m for m in months if m.startswith("2023")]].mean()
        h2 = total.loc[[m for m in months if m.startswith("2024")]].mean()
        row["h2_over_h1"] = float(h2 / h1) if h1 and not np.isnan(h1) else None
        dec = total.loc["2024-12"]
        row["dec24_over_mean"] = float(dec / base) if base and not np.isnan(dec) else None
        for cat in CATEGORIES:
            s = piv[cat]
            ok = s.notna()
            m = float(np.nanmean(s.values[ok])) if ok.any() else None
            row[f"{cat}_mean_rub"] = m
            row[f"{cat}_log_mean"] = float(np.log1p(m)) if m is not None else None
            row[f"{cat}_share"] = float(m / base) if m is not None and base else None
            row[f"{cat}_months"] = int(ok.sum())
            row[f"{cat}_cv"] = float(np.nanstd(s.values[ok]) / m) if m else None
        rows.append(row)
    return pd.DataFrame(rows)


def attach_master(features: pd.DataFrame) -> pd.DataFrame:
    from .config import PROCESSED

    m = pd.read_parquet(PROCESSED / "territory_master.parquet")
    m["tid"] = m.tid.astype("int64")
    out = features.merge(
        m[
            [
                "tid",
                "region_name",
                "gap_class",
                "osm_area_km2",
                "official_land_area_km2",
                "land_inflation_ratio",
                "n_parts",
            ]
        ],
        on="tid",
        how="left",
    )
    out["in_report_panel"] = out.months_available >= MIN_MONTHS_REPORT
    out["in_model_panel"] = out.months_available >= 24
    return out
