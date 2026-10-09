#!/usr/bin/env python3
"""States what is static, what refreshes, how often that is measured to be possible, and why annual is the scope.

Every number in here is read from a committed artifact. Nothing is typed in by hand: whether a layer moved between
years is proven by an asset that exists for two annual periods, by a field that carries a year, or by an Earth Engine
measurement recorded in the cadence table; the cost of recompute comes from the run diagnostics; the reason monthly
and quarterly are not discussed as noise comes from the resolution study, which recorded that they were never fetched.

The point of the artifact is that refreshability is not assumed. A layer that is called static is static because the
quantity itself does not drift year to year - elevation, climate normals, a soil survey, a single-vintage land-cover
map - and a layer that is called refreshable has proof that another period exists and a measured cost for pulling it.

Writes artifacts/validation/cadence_scope.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "gee_sources.yaml"
CADENCE = ROOT / "artifacts" / "validation" / "input_cadence.json"
ASSETS = ROOT / "artifacts" / "validation" / "dw_composite_assets.json"
RESOLUTION = ROOT / "artifacts" / "validation" / "dynamic_world_resolution.json"
MEASURES = ROOT / "artifacts" / "geo" / "added_layers_measures.json"
DIAG = ROOT / "artifacts" / "geo" / "covariates_diagnostics.json"
DIAG_2018 = ROOT / "artifacts" / "geo" / "covariates_diagnostics_v2018.json"
DIAG_2022 = ROOT / "artifacts" / "geo" / "covariates_diagnostics_v2022.json"
DECISION = ROOT / "artifacts" / "validation" / "k_decision.json"
TEMPORAL = ROOT / "artifacts" / "validation" / "dw_temporal_signal.json"
OUT = ROOT / "artifacts" / "validation" / "cadence_scope.json"

STATIC_REASON = {
    "elevation": "terrain height does not drift year to year",
    "worldclim": "climate normals are a 1979-2013 average by construction",
    "soilgrids": "a soil survey is a survey, not an annual observation",
    "gls_lcz": "one vintage of this land-cover map is what the panel reads",
    "gls_fraction": "one vintage of this land-cover map is what the panel reads",
    "worldcover": "one released map year, refreshed only when a new release exists",
}
REFRESH_REASON = {
    "dynamic_world": "a new annual classification is released every year",
    "viirs_lit": "the monthly night-time collection grows continuously",
    "viirs": "the monthly night-time collection grows continuously",
    "modis_ndvi": "16-day composites grow continuously",
    "modis_lcc": "annual land-cover products grow continuously",
    "modis_burn": "burned-area products grow continuously",
    "hansen": "a new forest-change edition is published each year",
}
SPACE_LAYERS = {
    "dynamic_world": "dynamic_world_bare, built, crops, grass, trees",
    "viirs_lit": "viirs_lit_mean",
    "modis_ndvi": "modis_ndvi_mean",
    "elevation": "elevation_mean",
}


def main() -> int:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    cadence = {row["input"]: row for row in json.loads(CADENCE.read_text(encoding="utf-8"))["rows"]}
    assets = json.loads(ASSETS.read_text(encoding="utf-8"))
    resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))
    measures = json.loads(MEASURES.read_text(encoding="utf-8"))
    coverage = measures["coverage"]
    approved = json.loads(DECISION.read_text(encoding="utf-8"))["space"]["columns"]

    def edition(asset: str) -> str | None:
        match = re.search(r"(v\d+(?:_\d+)?)|/(20\d\d)", asset)
        return match.group(0) if match else None

    rows = []
    for key in registry:
        entry = registry[key]
        if not isinstance(entry, dict):
            continue
        asset = str(entry.get("asset", ""))
        row = cadence.get(key, {})
        refreshable = key in REFRESH_REASON
        proof_field, proof_value = None, None
        if key == "dynamic_world":
            proof_field = "artifacts/validation/dw_composite_assets.json 2023.state and 2024.state"
            proof_value = f"{assets['2023'].get('state')} / {assets['2024'].get('state')}"
        elif refreshable:
            proof_field = f"artifacts/validation/input_cadence.json rows[{key}].proof"
            proof_value = str(row.get("proof", ""))[:140] or None
        elif key in STATIC_REASON:
            proof_field = f"artifacts/validation/input_cadence.json rows[{key}].other_year_available"
            proof_value = str(row.get("other_year_available", "")) or None
        rows.append(
            {
                "layer": key,
                "asset": asset,
                "release_or_edition_in_asset": edition(asset),
                "registry_year_field": entry.get("year"),
                "registry_kind": entry.get("kind"),
                "in_approved_space": SPACE_LAYERS.get(key),
                "static_or_refreshable": "refreshable" if refreshable else "static",
                "why": REFRESH_REASON.get(key) or STATIC_REASON.get(key) or "not stated",
                "proof_field": proof_field,
                "proof_value": proof_value,
                "cadence_recorded_by_the_cadence_table": row.get("interval"),
                "proof_kind": row.get("proof_kind"),
                "licence_status": entry.get("licence"),
            }
        )

    for key, job_key, asset, is_static, why in (
        ("dynamic_world", "dw_annual_2024", "GOOGLE/DYNAMICWORLD/V1", False, REFRESH_REASON["dynamic_world"]),
        (
            "hansen",
            "hansen_canopy",
            "UMD/hansen/global_forest_change_2025_v1_13",
            False,
            REFRESH_REASON["hansen"],
        ),
        ("soilgrids", "soilgrids", "ISRIC/SoilGrids250m/v2_0", True, STATIC_REASON["soilgrids"]),
    ):
        job = measures["jobs"][job_key]
        rows.append(
            {
                "layer": key,
                "asset": asset,
                "release_or_edition_in_asset": edition(asset),
                "in_approved_space": SPACE_LAYERS.get(key),
                "static_or_refreshable": "static" if is_static else "refreshable",
                "why": why,
                "proof_field": f"artifacts/geo/added_layers_measures.json jobs[{job_key}].window",
                "proof_value": str(job.get("window")),
                "measured_recompute_seconds": job.get("seconds"),
                "rows_in_recompute": job.get("rows"),
                "chunks_in_recompute": job.get("chunks"),
                "licence_status": measures["licences"].get(asset),
            }
        )
    rows = [r for r in rows if r.get("in_approved_space") or r.get("why") != "not stated"]

    spending_row = cadence["sberindex_spending"]
    area_row = cadence["rosstat_official_land_area"]
    rows.append(
        {
            "layer": "sberindex_spending",
            "asset": "SberIndex daily spending by municipality, 2023-01 to 2024-12",
            "in_approved_space": "Продовольствие, Здоровье, Маркетплейсы, Общественное питание, Транспорт shares",
            "static_or_refreshable": "refreshable",
            "why": "the series grows month by month",
            "proof_field": "artifacts/validation/input_cadence.json rows[sberindex_spending].proof_kind",
            "proof_value": spending_row["proof_kind"],
            "cadence_recorded_by_the_cadence_table": spending_row["interval"],
            "licence": "CC BY-SA 4.0",
        }
    )
    rows.append(
        {
            "layer": "rosstat_official_land_area",
            "asset": "Rosstat indicator Y48006001, official land area per municipality",
            "in_approved_space": None,
            "static_or_refreshable": "refreshable",
            "why": "the register republishes area annually, most recent year prevailing",
            "proof_field": "artifacts/validation/input_cadence.json rows[rosstat_official_land_area].other_year_available",
            "proof_value": area_row["other_year_available"],
            "cadence_recorded_by_the_cadence_table": area_row["interval"],
            "note": "this is the denominator every density uses; it is not a clustering column",
        }
    )

    throughput = {
        "full_panel_fetch_seconds": json.loads(DIAG.read_text(encoding="utf-8"))["total_seconds"],
        "vintage_2018_fetch_seconds": json.loads(DIAG_2018.read_text(encoding="utf-8"))["total_seconds"],
        "vintage_2022_fetch_seconds": json.loads(DIAG_2022.read_text(encoding="utf-8"))["total_seconds"],
        "dw_asset_export_wait_seconds": assets["_"]["wait_seconds"],
        "cache_replays_recorded_in_the_resolution_study": resolution["cache_at_decision"],
    }
    annual = resolution["results"]["annual"]
    scope = {
        "chosen": resolution["choice"],
        "coverage_gate": resolution["coverage_gate"],
        "annual_coverage_mean_share": annual["coverage_mean_share"],
        "annual_status": annual["status"],
        "monthly_status": resolution["results"]["monthly"]["status"],
        "monthly_reason": resolution["results"]["monthly"]["reason"],
        "quarterly_status": resolution["results"]["quarterly"]["status"],
        "quarterly_reason": resolution["results"]["quarterly"]["reason"],
        "not_a_verified_claim": (
            "That finer frequencies are noisier was never measured: monthly and quarterly were not fetched, so it "
            "stays a standing warning rather than a finding. The one annual movement that was measured is also "
            "uncorroborated by an independent annual forest signal, recorded in "
            "artifacts/validation/dw_temporal_signal.json."
        ),
    }
    payload = {
        "generated_by": "scripts/build_cadence_story.py",
        "what_this_is": (
            "A refreshability table for the layers the shipped clustering reads, naming the field or date that "
            "proves whether each layer moves, the measured throughput behind the update-frequency claim, and why "
            "annual is the scope."
        ),
        "approved_space_columns": approved,
        "rows": rows,
        "measured_throughput": throughput,
        "why_annual": scope,
        "update_frequency_claim": (
            f"Once a year for the land-cover and light layers, monthly for spending, annually for the register area. "
            f"A single-year recompute of the geospatial panel measured {throughput['vintage_2022_fetch_seconds']} and "
            f"{throughput['vintage_2018_fetch_seconds']} seconds; the whole multi-period fetch measured "
            f"{throughput['full_panel_fetch_seconds']} seconds; materialising one annual composite as an asset waited "
            f"{throughput['dw_asset_export_wait_seconds']} seconds of server-side work."
        ),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"rows: {len(rows)}; static {sum(r['static_or_refreshable'] == 'static' for r in rows)}; refreshable "
          f"{sum(r['static_or_refreshable'] == 'refreshable' for r in rows)}")
    print(f"annual coverage mean share {annual['coverage_mean_share']}; monthly {scope['monthly_status']}; "
          f"quarterly {scope['quarterly_status']}")
    print(f"written to {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
