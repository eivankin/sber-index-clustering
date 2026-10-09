#!/usr/bin/env python3
"""Build the modelling attribute space: spending behaviour + land covariates + land-based density.

Writes data/processed/feature_space.parquet and artifacts/features/feature_space_report.json.
It runs with or without covariates: without them the report records exactly which blocks are missing
and the density is still computed from official land area, so the pipeline is never blocked.

Density rule (measured, not assumed): spending per km2 uses **official** Rosstat land area; where a
municipality has no official figure the subject's official area is used as an explicitly documented
proxy. The OSM polygon area is never used - the master table shows it is inflated by marine and
extremity areas (sum 40.0M km2 against ~17.1M km2 of RF land).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sberindex import cluster
from sberindex.config import (
    ARTIFACTS,
    MARKET_ACCESS,
    PROCESSED,
    ensure_dirs,
)

DIAGNOSTIC_SUFFIX = "_unclassified_pct"


def find_covariates():
    """Pick the covariate table with the most territories, not the first one that exists.

    The 100-territory smoke GEE table stays on disk after a full no-auth run. Taking it just because it
    sorts first would build the attribute space from 100 territories while the coverage gate still
    reported the land block as usable - a silent downgrade, so completeness decides the choice."""
    best_name, best = None, None
    for name in ("covariates_gee.parquet", "covariates_fallback.parquet"):
        p = PROCESSED / name
        if p.exists():
            df = pd.read_parquet(p)
            if best is None or len(df) > len(best):
                best_name, best = name, df
    return best_name, best


def main() -> int:
    ensure_dirs()
    (ARTIFACTS / "features").mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(PROCESSED / "territory_master.parquet")
    feats = pd.read_parquet(PROCESSED / "spending_features.parquet")
    keep_cols = [
        "tid",
        "region_name",
        "district",
        "official_land_area_km2",
        "area_year",
        "official_subject_area_km2",
        "area_suspect",
        "land_inflation_ratio",
        "n_parts",
        # the BD PMO crosswalk keys, needed to join official Rosstat indicators to territories
        "oktmo_norm",
        "oktmo_stable",
    ]
    master = master[[c for c in keep_cols if c in master.columns]]
    # spending_features already carries some master columns; keep the master's version and avoid
    # pandas creating *_x / *_y suffixes
    dup = [c for c in feats.columns if c != "tid" and c in master.columns]
    if dup:
        print(f"dropping from spending_features (master owns these): {dup}")
        feats = feats.drop(columns=dup)
    out = feats.merge(master, on="tid", how="left")

    # Sber's own market-access index: the demand side of the edge rule and the baseline that the land
    # thesis must beat. Missing for 89 territories, which the report records as a coverage count.
    ma = pd.read_parquet(MARKET_ACCESS)
    ma["tid"] = pd.to_numeric(ma["territory_id"], errors="raise").astype("int64")
    ma = ma.rename(columns={"market_access": "market_access_index"})[
        ["tid", "market_access_index"]
    ].drop_duplicates("tid")
    out = out.merge(ma, on="tid", how="left")

    # land-based density: official municipality area, else the subject's official area as a proxy
    area = pd.to_numeric(out["official_land_area_km2"], errors="coerce")
    proxy = pd.to_numeric(out.get("official_subject_area_km2"), errors="coerce")
    basis = np.where(
        area.notna(),
        "official_municipal",
        np.where(proxy.notna(), "official_subject_proxy", "none"),
    )
    area_used = area.fillna(proxy)
    spend = pd.to_numeric(out["total_mean_rub"], errors="coerce")
    density = np.log1p(spend / area_used.where(area_used > 0))
    out["spending_per_km2_log"] = density
    out["area_basis"] = basis
    # Per-category spending densities on the official-area denominator. The entry's claim is about
    # production per square kilometre, and "production" in the rubric is a per-capita reading, so a
    # density block has to exist as its own attribute space rather than being one logged total. The
    # denominator is the Rosstat area alone - not the proxy-filled `area_used` - because an OSM or
    # subject-area denominator is exactly the inflated number this repository measures as a defect.
    # Territories without an official area stay NaN and are imputed at clustering time, which is
    # recorded rather than hidden: their density is then the cohort median, not a measurement.
    official_area = area.where(area > 0)
    for col in [c for c in out.columns if c.endswith("_mean_rub")]:
        out[col.replace("_mean_rub", "_per_official_km2")] = (
            pd.to_numeric(out[col], errors="coerce") / official_area
        )

    cov_name, cov = find_covariates()
    cov_cols: list[str] = []
    diag_cols: list[str] = []
    prov_cols: list[str] = []
    if cov is not None:
        cov["tid"] = pd.to_numeric(cov["tid"], errors="coerce").astype("Int64")
        cov = cov.drop_duplicates("tid")
        cov_cols = [
            c
            for c in cov.columns
            if c != "tid" and not c.startswith("prov_") and not c.endswith(DIAGNOSTIC_SUFFIX)
        ]
        diag_cols = [c for c in cov.columns if c.endswith(DIAGNOSTIC_SUFFIX)]
        prov_cols = [c for c in cov.columns if c.startswith("prov_")]
        out = out.merge(cov[["tid"] + cov_cols + diag_cols + prov_cols], on="tid", how="left")

    # The GLS change-confidence band is aggregated separately (it arrived later than the main layers and
    # is a different kind of evidence: prior, not within-window), so it is merged here if present.
    chg_path = PROCESSED / "covariates_change.parquet"
    if chg_path.exists():
        chg = pd.read_parquet(chg_path)
        chg["tid"] = pd.to_numeric(chg["tid"], errors="coerce").astype("Int64")
        chg = chg.drop_duplicates("tid")
        chg_cols = [c for c in chg.columns if c != "tid"]
        out = out.merge(chg[["tid"] + chg_cols], on="tid", how="left")
        cov_name = f"{cov_name} + covariates_change" if cov is not None else "covariates_change"

    # Official Rosstat BD PMO indicators: the socio-economic block every leading competitor pairs with
    # SberIndex spending. Joined on the same BD PMO OKTMO crosswalk that produced official land area,
    # and subjected to the same coverage gate, so a column the official file barely covers cannot enter
    # the attribute space and later be presented as a measured characteristic of the territory.
    socio_path = PROCESSED / "covariates_socio.parquet"
    socio_cols: list[str] = []
    socio_prov: list[str] = []
    socio_gate = {"gate": 0.5, "kept": [], "dropped_low_coverage": []}
    if socio_path.exists():
        soc = pd.read_parquet(socio_path)
        soc["oktmo_stable"] = soc["oktmo_stable"].astype("string").str.strip().str.zfill(8)
        soc = soc.drop_duplicates("oktmo_stable")
        cand = [c for c in soc.columns if c != "oktmo_stable" and not c.startswith("prov_")]
        socio_cols = [c for c in cand if float(soc[c].notna().mean()) >= 0.5]
        socio_gate["kept"] = socio_cols
        socio_gate["dropped_low_coverage"] = [c for c in cand if c not in socio_cols]
        socio_prov = [c for c in soc.columns if c.startswith("prov_")]
        if socio_cols:
            # The feature table may carry the BD PMO stable code directly (it came in with the land-area
            # join) or only the normalized Sber code, so the join uses whichever is present.
            key_col = "oktmo_stable" if "oktmo_stable" in out.columns else "oktmo_norm"
            soc = soc.rename(columns={"oktmo_stable": key_col})
            out = out.merge(soc[[key_col] + socio_cols + socio_prov], on=key_col, how="left")

    added_gate = {
        "gate": 0.5,
        "status": "file absent",
        "kept": [],
        "dropped_low_coverage": [],
        "resolution_used": None,
    }
    added_path = PROCESSED / "covariates_added.parquet"
    if added_path.exists():
        added = pd.read_parquet(added_path)
        if "tid" not in added.columns:
            added = added.reset_index(names="tid")
        added["tid"] = pd.to_numeric(added["tid"], errors="coerce")
        added = added.dropna(subset=["tid"]).drop_duplicates("tid")
        added["tid"] = added["tid"].astype("int64")
        derived = pd.DataFrame(index=added["tid"])
        loss_columns = [c for c in added.columns if c in ("hly_c23_pct", "hly_c24_pct")]
        if loss_columns:
            derived["hansen_window_forest_loss_pct"] = (
                added[loss_columns].fillna(0.0).sum(axis=1).to_numpy()
            )
        if "hansen_canopy_treecover2000" in added.columns:
            derived["hansen_tree_cover_2000_pct"] = added["hansen_canopy_treecover2000"].to_numpy()
        if "hansen_canopy_gain" in added.columns:
            derived["hansen_forest_gain_pct"] = added["hansen_canopy_gain"].to_numpy()
        if "soil_val_0_5cm_mean" in added.columns:
            derived["soil_organic_carbon_surface"] = added["soil_val_0_5cm_mean"].to_numpy()

        # Dynamic World arrives at three resolutions and all three cannot enter the space at once: the
        # choice is the one the resolution study made by measurement. Until it exists, annual is used and
        # the artifact says the choice was still open when this run happened.
        resolution = None
        resolution_choice = ARTIFACTS / "validation" / "dynamic_world_resolution.json"
        if resolution_choice.exists():
            resolution = json.loads(resolution_choice.read_text(encoding="utf-8")).get("choice")
        resolution = resolution or "annual"
        added_gate["resolution_used"] = resolution
        pattern = re.compile(r"^dw_(trees|grass|crops|built|bare)@")
        for column in added.columns:
            match = pattern.match(column)
            if not match:
                continue
            period = column.split("@", 1)[1]
            is_annual = not period[4:].strip("-q")
            is_quarterly = "q" in period
            is_monthly = bool(re.search(r"-\d{2}$", period))
            wanted = (
                is_annual
                if resolution == "annual"
                else is_quarterly if resolution == "quarterly" else is_monthly
            )
            if not wanted:
                continue
            klass = match.group(1)
            name = f"dynamic_world_{klass}"
            if name in derived:
                # derived is indexed by territory id; a bare ndarray has no index, so pandas refused to
                # align the two and raised before any panel could be written. The incoming column is put
                # on the same index as the thing it is filling, which is the only comparison that makes
                # sense here - values for the same district, not values in the same row position.
                values = pd.Series(added[column].to_numpy(), index=derived.index)
                derived[name] = derived[name].fillna(values).combine_first(values)
            else:
                derived[name] = added[column].to_numpy()

        cand = list(derived.columns)
        # Coverage is measured over the panel's rows, not over the rows that were fetched. A gate computed
        # on the fetched subset says 100% about a hundred districts while 2,560 others get nothing, which
        # is how a sliver of data gets reported as a usable layer.
        out = out.merge(derived, on="tid", how="left")
        kept = [c for c in cand if float(out[c].notna().mean()) >= 0.5]
        dropped = [c for c in cand if c not in kept]
        if dropped:
            out = out.drop(columns=dropped)
        added_gate["kept"] = kept
        added_gate["dropped_low_coverage"] = dropped
        added_gate["panel_rows_measured"] = int(len(out))
        added_gate["status"] = (
            "present" if kept else "present but every measure is below the 50% coverage gate"
        )

    blocks = {
        "levels+shares": cluster.LEVELS + cluster.SHARES,
        "dynamics": [c for c in cluster.DYNAMICS if c in out.columns],
        "land": cluster.land_columns(out),
        "unclassified_diagnostics": diag_cols,
        "provenance": prov_cols,
        "area": ["official_land_area_km2", "area_year", "official_subject_area_km2", "area_basis"],
        "access": ["market_access_index"],
        "socio": socio_cols,
        "socio_provenance": socio_prov,
        "dynamic_world": [c for c in out.columns if c.startswith("dynamic_world_")],
        "hansen": [c for c in out.columns if c.startswith("hansen_")],
        "soil": [c for c in out.columns if c.startswith("soil_")],
    }
    report = {
        "rows": len(out),
        "columns": int(out.shape[1]),
        "covariate_source": cov_name or "none yet (spending + official area only)",
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "socio_coverage_gate": socio_gate,
        "added_layers_gate": added_gate,
        "land_coverage_gate": {
            "gate": 0.5,
            "kept": cluster.land_columns(out),
            "dropped_low_coverage": [
                c
                for c in cluster.land_columns(out, min_coverage=0.0)
                if c not in cluster.land_columns(out)
            ],
        },
        "non_null_share": {
            k: round(float(out[v].notna().mean().mean()), 4)
            for k, v in blocks.items()
            if v and all(c in out.columns for c in v)
        },
        "area_basis_counts": pd.Series(basis).value_counts().to_dict(),
        "density_computed_rows": int(np.isfinite(density).sum()),
        "model_panel_rows": (
            int(out.loc[out.get("in_model_panel", False).astype(bool), "tid"].nunique())
            if "in_model_panel" in out.columns
            else None
        ),
        "note": (
            "spending_per_km2_log = log1p(total_mean_rub / official land area km2); OSM polygon "
            "area never used because the master table proves it is inflated by marine and "
            "extremity areas"
        ),
    }
    out.to_parquet(PROCESSED / "feature_space.parquet", index=False)
    with open(ARTIFACTS / "features" / "feature_space_report.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(
        f"feature space: {out.shape[0]} rows x {out.shape[1]} cols | covariates: {report['covariate_source']}"
    )
    print("blocks:", report["block_sizes"])
    print(
        "area basis:",
        report["area_basis_counts"],
        "| density rows:",
        report["density_computed_rows"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
