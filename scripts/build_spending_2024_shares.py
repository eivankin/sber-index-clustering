#!/usr/bin/env python3
"""Category shares for 2024, computed the way the shipped panel computes them.

The 2024 partition we shipped kept the 2023 shape of spending, because at the time it was written
nobody asked whether a 2024 shape existed. It does: the SberIndex series runs through 2024-12, and
12 months of 2024 are in the long table. So the shape of spending can move too, and the claim that
only land and greenness changed was incomplete.

The rule is the one in src/sberindex/panel.py: a category share is that category's mean over the
months it was observed, divided by the mean of "Все категории" over the same months. Nothing is
averaged over months that are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPENDING = ROOT / "data" / "processed" / "spending_long.parquet"
OUT = ROOT / "artifacts" / "validation" / "spending_2024_shares.csv"

CATEGORIES = ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
TOTAL = "Все категории"
YEAR = "2024"


def main() -> int:
    spending = pd.read_parquet(SPENDING)
    spending["month"] = spending["month"].astype(str)
    spending = spending[spending.month.str.startswith(YEAR)].dropna(subset=["value"])
    spending["value"] = spending["value"].astype("float64")

    months = sorted(spending.month.unique())
    if len(months) != 12:
        print(f"note: {len(months)} months of {YEAR} carry values, not 12")

    rows = []
    for tid, group in spending.groupby("territory_id"):
        pivot = group.pivot_table(index="month", columns="category", values="value", aggfunc="mean")
        pivot = pivot.reindex(index=months, columns=CATEGORIES + [TOTAL])
        total = pivot[TOTAL]
        present = total.notna()
        if present.sum() == 0:
            continue
        base = float(np.nanmean(total.values[present.values]))
        row = {
            "tid": int(tid),
            "months_2024": int(present.sum()),
            "total_mean_rub_2024": base,
        }
        for category in CATEGORIES:
            series = pivot[category]
            observed = series.notna().values
            mean = float(np.nanmean(series.values[observed])) if observed.any() else None
            row[f"{category}_mean_rub_2024"] = mean
            row[f"{category}_share_2024"] = (
                float(mean / base) if mean is not None and base else None
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    coverage = float(table[f"{CATEGORIES[0]}_share_2024"].notna().mean())
    print(
        f"2024 spending shape written: {OUT.relative_to(ROOT)} · rows {len(table):,} · "
        f"territories {table.tid.nunique():,} · months {len(months)} · "
        f"coverage of the first share {coverage:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
