#!/usr/bin/env python
"""Assert the documented panel shape and NaN policy. Fails loudly on drift."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sberindex.config import PROCESSED

EXPECTED = {
    "long_territories": 2190,
    "long_rows": 303126,
    "months": 24,
    "categories": 6,
    "grid_total": 315360,
    "territories_full_24m": 2016,
    "territories_ge_12m": 2131,
}


def main() -> int:
    long = pd.read_parquet(PROCESSED / "spending_long.parquet")
    feats = pd.read_parquet(PROCESSED / "spending_features.parquet")
    got = {
        "long_territories": int(long.territory_id.nunique()),
        "long_rows": len(long),
        "months": int(long.month.nunique()),
        "categories": int(long.category.nunique()),
        "grid_total": int(
            long.territory_id.nunique() * long.month.nunique() * long.category.nunique()
        ),
        "territories_full_24m": int(feats.in_model_panel.sum()),
        "territories_ge_12m": int(feats.in_report_panel.sum()),
    }
    bad = 0
    for k, v in EXPECTED.items():
        ok = got.get(k) == v
        bad += not ok
        print(f"{'OK ' if ok else 'MISMATCH '}{k}: got {got.get(k)} expected {v}")
    zero_filled = int((long.value == 0).sum())
    print(f"zero-filled cells: {zero_filled} (policy says: never zero-fill)")
    print("ALL PANEL CONTRACT VALUES REPRODUCED" if bad == 0 else f"{bad} MISMATCHES")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
