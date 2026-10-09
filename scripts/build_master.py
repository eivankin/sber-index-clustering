#!/usr/bin/env python
"""Day-1 build: territory master + geometry-defect diagnostics + measurement-gap tables.

Usage:  python scripts/build_master.py            # writes artifacts and prints a summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from sberindex.config import ARTIFACTS, PROCESSED, ensure_dirs
from sberindex.master import build_master, region_inflation

EXPECTED = {
    "geo": 2660,
    "consumption": 2190,
    "connection": 2574,
    "market_access": 2571,
    "highway_nodes": 2573,
    "railway_nodes": 1627,
    "G_minus_C": 470,
    "C_minus_G": 0,
    "K_minus_C": 401,
    "C_minus_K": 17,
    "M_minus_C": 401,
    "C_minus_M": 20,
    "core_C_K_M_G": 2170,
    "multi_part": 191,
    "invalid": 0,
}


def main() -> int:
    ensure_dirs()
    master, diag = build_master(asof_year=2024)

    master.to_parquet(PROCESSED / "territory_master.parquet", index=False)
    (ARTIFACTS / "geo" / "geometry_diagnostics.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    reg = region_inflation(master)
    reg.to_csv(ARTIFACTS / "geo" / "region_inflation.csv", index=False)
    top = master.nlargest(15, "osm_area_km2")[
        [
            "tid",
            "municipal_district_name",
            "region_name",
            "osm_area_km2",
            "official_land_area_km2",
            "land_inflation_ratio",
            "n_parts",
        ]
    ]
    top.to_csv(ARTIFACTS / "geo" / "top_polygons_by_area.csv", index=False)

    for cls, frame in master.groupby("gap_class"):
        frame[
            [
                "tid",
                "municipal_district_name",
                "region_name",
                "municipal_district_type",
                "osm_area_km2",
                "months_present",
            ]
        ].to_csv(ARTIFACTS / "gaps" / f"gap_{cls}.csv", index=False)

    # ---- verification against independently established values ----
    problems = []
    for k, v in EXPECTED.items():
        got = diag["coverage"].get(k, diag["gaps"].get(k, diag["geometry"].get(k)))
        if got != v:
            problems.append(f"{k}: expected {v}, got {got}")
    print(json.dumps(diag, indent=2, ensure_ascii=False))
    print("\nrow count:", len(master), "| unique tid:", master.tid.nunique())
    print("gap_class counts:\n", master.gap_class.value_counts().to_string())
    print("\ntop inflation:\n", reg.head(6).to_string())
    if problems:
        print("\nMISMATCHES:", problems)
        return 1
    print("\nALL EXPECTED VALUES REPRODUCED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
