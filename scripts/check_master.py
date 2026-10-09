#!/usr/bin/env python
"""Re-assert the Day-1 contract numbers from the committed artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sberindex.config import ARTIFACTS, PROCESSED

EXPECTED = {
    "rows": 2660,
    "G_minus_C": 470,
    "K_minus_C": 401,
    "C_minus_K": 17,
    "C_minus_M": 20,
    "core_C_K_M_G": 2170,
    "invalid": 0,
    "multi_part": 191,
}
AREA = {
    "sum_polygons_km2": 40.0e6,
    "union_km2": 39.4e6,
    "kara_laptev": 4.79e6,
    "black_caspian": 0.66e6,
    "north_of_78N": 0.94e6,
}


def main() -> int:
    diag = json.loads((ARTIFACTS / "geo" / "geometry_diagnostics.json").read_text())
    m = pd.read_parquet(PROCESSED / "territory_master.parquet")
    diag.get("coverage", {})
    gaps = diag.get("gaps", {})
    geom = diag.get("geometry", {})
    sea = diag.get("sea_overlap_km2", {})
    got = {
        "rows": len(m),
        **{k: gaps.get(k) for k in EXPECTED if k != "rows"},
        "invalid": geom.get("invalid"),
        "multi_part": geom.get("multi_part"),
    }
    bad = 0
    for k, v in EXPECTED.items():
        ok = got.get(k) == v
        bad += not ok
        print(f"{'OK ' if ok else 'MISMATCH '}{k}: got {got.get(k)} expected {v}")
    for k, v in AREA.items():
        g = geom.get(k, sea.get(k))
        if g is None:
            print(f"MISSING {k} in diagnostics")
            bad += 1
            continue
        ok = abs(g - v) / v < 0.02
        bad += not ok
        print(f"{'OK ' if ok else 'DRIFT '}{k}: {g/1e6:.3f}M km2 vs ~{v/1e6:.2f}M")
    ros = diag.get("rostat", {})
    print(
        f"rostat: status={ros.get('rostat_status')} matched={ros.get('matched_territories')} "
        f"rate={ros.get('match_rate')} unmatched={ros.get('unmatched_territories')}"
    )
    print(
        f"area_suspect rows: {int(m.area_suspect.sum())}; multipart rows: {int((m.n_parts > 1).sum())}"
    )
    print("DAY-1 CONTRACT REPRODUCED" if bad == 0 else f"{bad} PROBLEMS")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
