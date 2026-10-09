#!/usr/bin/env python
"""Build the spending panel (long) and per-territory model features (wide).

Writes data/processed/spending_long.parquet, data/processed/spending_features.parquet and
artifacts/tables/panel_summary.csv. No network needed - everything comes from data/raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sberindex import panel
from sberindex.config import ARTIFACTS, PROCESSED, ensure_dirs


def main() -> int:
    ensure_dirs()
    c = panel.load_consumption()
    print(
        f"consumption rows={len(c)} months={c.month.nunique()} territories={c.territory_id.nunique()}"
    )
    long = panel.build_long(c)
    feats = panel.build_features(c)
    feats = panel.attach_master(feats)
    (PROCESSED).mkdir(parents=True, exist_ok=True)
    long.to_parquet(PROCESSED / "spending_long.parquet", index=False)
    feats.to_parquet(PROCESSED / "spending_features.parquet", index=False)

    summary = pd.DataFrame(
        {
            "long_rows": [len(long)],
            "long_territories": [long.territory_id.nunique()],
            "feature_rows": [len(feats)],
            "report_panel": [int(feats.in_report_panel.sum())],
            "model_panel_24m": [int(feats.in_model_panel.sum())],
            "model_panel_ge18m": [int((feats.months_available >= 18).sum())],
            "expected_2016": [2016],
        }
    )
    summary.to_csv(ARTIFACTS / "tables" / "panel_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("feature columns:", len(feats.columns))
    # a coverage table by region so gaps are visible to the reader
    by_region = (
        feats.groupby("region_name")
        .agg(
            n=("tid", "size"),
            mean_months=("months_available", "mean"),
            model_panel=("in_model_panel", "sum"),
        )
        .sort_values("n", ascending=False)
    )
    by_region.to_csv(ARTIFACTS / "tables" / "panel_by_region.csv")
    print(by_region.head(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
