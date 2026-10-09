#!/usr/bin/env python3
"""Write the per-source preprocessing story, from the registry and the decision, not from prose.

The entrant asked the right-hand card to show how a raw source becomes a number in the clustering table,
instead of generic dataset-level figures. So every step here is read from a committed artifact or the registry:
the native step of the sensor, how the window is reduced, which year, and whether the resulting columns entered
the clustering space of the approved decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/gee_sources.yaml"
DECISION = ROOT / "artifacts/validation/k_decision.json"
ADDED = ROOT / "artifacts/geo/added_layers_measures.json"
CROPS = ROOT / "artifacts/landing/source_crops.json"
LEGENDS = ROOT / "artifacts/landing/source_legends.json"
OUT = ROOT / "artifacts/landing/source_pipeline.json"

# Which registry entry and which panel columns belong to which source on the rail.
RAIL = {
    "dynamic_world": {"registry": None, "prefix": "dynamic_world_"},
    "viirs": {"registry": "viirs", "prefix": "viirs_"},
    "ndvi": {"registry": "modis_ndvi", "prefix": "modis_ndvi_"},
    "terrain": {"registry": "elevation", "prefix": "elevation_"},
    "hansen": {"registry": None, "prefix": "hansen_"},
    "soil": {"registry": None, "prefix": "soil_"},
}

REDUCER_RU = (
    ("frequencyhistogram", "считает доли классов по пикселям района"),
    ("mean of the monthly mean", "среднее по месяцам, затем по году"),
    ("mean of binary", "доля пикселей, где свет ярче 0,5"),
    ("mean", "среднее по пикселям внутри района"),
    ("sum", "сумма по пикселям внутри района"),
)


def ru_reduce(reducer: str) -> str:
    lowered = (reducer or "").lower()
    for needle, phrase in REDUCER_RU:
        if needle in lowered:
            return phrase
    return "усреднение по пикселям внутри района"


def main() -> int:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    added = json.loads(ADDED.read_text(encoding="utf-8"))
    crops = json.loads(CROPS.read_text(encoding="utf-8"))
    legends = json.loads(LEGENDS.read_text(encoding="utf-8"))
    columns = [str(name) for name in decision["space"]["columns"]]

    coverage_by_layer = {}
    for name, block in (added.get("coverage") or {}).items():
        if isinstance(block, dict) and block.get("share") is not None:
            coverage_by_layer[name] = float(block["share"])

    sources = {}
    for key, facts in RAIL.items():
        entry = registry.get(facts["registry"]) if facts["registry"] else None
        crop = next(
            (record for record in crops["records"] if record["source"] == key),
            None,
        )
        matched = [name for name in columns if name.startswith(facts["prefix"])]
        steps = []
        scale_m = (entry or {}).get("scale_m") or (crop or {}).get("scale_m")
        if scale_m:
            steps.append(f"шаг съёмки {int(scale_m)} м на пиксель")
        if entry:
            steps.append(ru_reduce(entry.get("reducer")))
            if entry.get("year"):
                steps.append(f"год наблюдения: {entry['year']}")
        elif key == "dynamic_world":
            years = sorted(
                {
                    str(record.get("asset_year"))
                    for record in crops["records"]
                    if record["source"] == key and record.get("asset_year")
                }
            )
            steps.append("окно усредняется до 500 м и получает один класс на пиксель")
            steps.append("год наблюдения: " + " и ".join(years))
        if entry and entry.get("denominator"):
            steps.append("доли считаются только по классифицированным пикселям")
        layer = f"{key}_layer"
        share = coverage_by_layer.get(key) or coverage_by_layer.get(facts["prefix"].rstrip("_"))
        if share is not None:
            steps.append(f"пикселей с данными по стране: {round(share * 100, 1)} %")
        if matched:
            steps.append(f"входит в пространство кластеризации: {len(matched)} колонк" + ("и" if len(matched) > 1 else "а"))
        else:
            steps.append("не входит в пространство кластеризации")
        sources[key] = {
            "steps": steps,
            "in_space": bool(matched),
            "matched_columns": matched,
            "registry_entry": facts["registry"],
            "label": (legends.get("sources", {}).get(key, {}) or {}).get("label", key),
        }

    OUT.write_text(
        json.dumps(
            {
                "generated_by": "scripts/build_source_pipeline.py",
                "rule": "steps are read from configs/gee_sources.yaml, the approved k decision and the added-layer measures; nothing is typed by hand",
                "sources": sources,
                "space_column_count": len(columns),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"source pipeline written to {OUT.relative_to(ROOT)}; sources: {len(sources)}")
    for key, block in sources.items():
        print(f"  {key}: in_space={block['in_space']} steps={len(block['steps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
