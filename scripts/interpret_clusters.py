#!/usr/bin/env python
"""Economic interpretation of the chosen partition: per-cluster profile + regional composition.

Nothing here is decorative: every column is a measured quantity that a reader can use to decide
whether a cluster means something economically, and the geometry-suspect count shows how much of a
cluster's apparent "large area" is an artefact of the OSM polygons rather than the land.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from sberindex.config import (
    ARTIFACTS,
    MARKET_ACCESS,
    PROCESSED,
    ROOT,
    ensure_dirs,
)

SHARES = [
    f"{c}_share"
    for c in ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
]


def main() -> int:
    ensure_dirs()
    assign = pd.read_parquet(ARTIFACTS / "icvi" / "cluster_assignments.parquet")
    # One named partition, not whichever method row was written first: interpretation, profiles and the map
    # have to describe the same clustering or they contradict each other on the page.
    from sberindex import cluster  # noqa: E402

    shipped = yaml.safe_load((ROOT / "configs" / "clustering.yaml").read_text(encoding="utf-8")).get(
        "shipped_method"
    )
    assign = cluster.select_partition(assign, shipped)
    feats = pd.read_parquet(PROCESSED / "spending_features.parquet")
    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    ma = pd.read_parquet(MARKET_ACCESS)
    df = (
        assign.merge(
            feats[
                [
                    "tid",
                    "region_name",
                    "total_mean_rub",
                    "total_cv",
                    "h2_over_h1",
                    "dec24_over_mean",
                    "months_available",
                ]
                + SHARES
            ],
            on="tid",
            how="left",
        )
        .merge(
            master[
                [
                    "tid",
                    "osm_area_km2",
                    "official_land_area_km2",
                    "land_inflation_ratio",
                    "area_suspect",
                    "n_parts",
                ]
            ],
            on="tid",
            how="left",
        )
        .merge(
            ma.rename(columns={"market_access": "market_access_index", "territory_id": "tid"}),
            on="tid",
            how="left",
        )
    )

    prof = df.groupby("cluster").agg(
        n=("tid", "size"),
        median_spending_rub=("total_mean_rub", "median"),
        mean_cv=("total_cv", "mean"),
        mean_h2_h1=("h2_over_h1", "mean"),
        mean_dec24=("dec24_over_mean", "mean"),
        mean_market_access=("market_access_index", "mean"),
        median_osm_area_km2=("osm_area_km2", "median"),
        median_official_area_km2=("official_land_area_km2", "median"),
        mean_inflation=("land_inflation_ratio", "mean"),
        suspect_share=("area_suspect", "mean"),
        multipart_share=("n_parts", lambda s: float((s > 1).mean())),
        mean_months=("months_available", "mean"),
        **{f"mean_{c}": (c, "mean") for c in SHARES},
    )
    prof = prof.round(4).reset_index()
    ARTIFACTS.joinpath("interpretation").mkdir(parents=True, exist_ok=True)
    prof.to_csv(ARTIFACTS / "interpretation" / "cluster_profile.csv", index=False)

    regions = (
        df.groupby(["cluster", "region_name"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["cluster", "n"], ascending=[True, False])
    )
    regions.to_csv(ARTIFACTS / "interpretation" / "cluster_regions.csv", index=False)
    top = regions.groupby("cluster").head(5)
    print(prof.to_string(index=False))
    print("\ntop regions per cluster:")
    print(top.to_string(index=False))
    print(
        f"\nmethod={assign.method.iloc[0]} k={int(assign.cluster.nunique())} "
        f"assigned={len(assign)} with_profile={int(df.region_name.notna().sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
