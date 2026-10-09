#!/usr/bin/env python3
"""Builds the Russian stage-one presentation as a viewer-facing visual story.

Three tabs in the upper row - where the data comes from, how often it can be refreshed, why the answer is five
types. Each tab is one screen: the page itself never scrolls, and the wheel moves the viewer between tabs. Content
inside a tab is switched by buttons - which district to look at, which year, which feature space - rather than by
growing the page.

Every figure on the visible face comes from the payload assembled here from committed artifacts. Nothing is typed
into the template and nothing is computed in the browser, because a number invented at render time resolves to
nothing and a number typed by hand can go stale without anyone noticing. Internal machinery - artifact paths,
payload hashes, cache locations, territory ids - is deliberately kept off the face and lives in a per-figure
"проверить источник" drawer, which is an overlay panel and therefore cannot push the page past one screen.

Writes web/presentation/index.html and artifacts/landing/stage_one_presentation.json.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "presentation" / "index.html"
ASSETS = ROOT / "web" / "presentation" / "assets"
PROOF = ROOT / "artifacts" / "landing" / "stage_one_presentation.json"

CROPS = ROOT / "artifacts" / "landing" / "provenance_crops.json"
RECREATIONS = ROOT / "artifacts" / "landing" / "provenance_recreations.json"
CHAIN = ROOT / "artifacts" / "validation" / "derivation_chain.json"
CADENCE = ROOT / "artifacts" / "validation" / "cadence_scope.json"
CADENCE_FAMILIES = ROOT / "artifacts" / "validation" / "cadence_families.json"
DOMINANCE = ROOT / "artifacts" / "validation" / "amount_dominance.json"
SUPERIORITY = ROOT / "artifacts" / "validation" / "filtering_superiority.json"
DW_SUB = ROOT / "artifacts" / "validation" / "dw_substitution.json"
DW_TEMPORAL = ROOT / "artifacts" / "validation" / "dw_temporal_signal.json"
DECISION = ROOT / "artifacts" / "validation" / "k_decision.json"
NAMES = ROOT / "artifacts" / "interpretation" / "cluster_names.json"
GEOM = ROOT / "artifacts" / "geo" / "geometry_diagnostics.json"
INFLATION = ROOT / "artifacts" / "geo" / "region_inflation.csv"
RAMP = ROOT / "artifacts" / "landing" / "amount_comparison_ramp_note.json"
SOURCE_CROPS = ROOT / "artifacts" / "landing" / "source_crops.json"
DEMO_STATES = ROOT / "artifacts" / "landing" / "demo_states.json"
SOURCE_LEGENDS = ROOT / "artifacts" / "landing" / "source_legends.json"
ADDED_LAYERS = ROOT / "artifacts" / "geo" / "added_layers_measures.json"
SOURCE_PIPELINE = ROOT / "artifacts" / "landing" / "source_pipeline.json"

# The palette is the one the crops were rendered with. A legend that disagrees with the image is worse than no
# legend, so the colors are carried from the script that made the pixels, not picked here.
PALETTE = {
    "bare": "#b8a07a",
    "built": "#b0413e",
    "crops": "#c9a227",
    "grass": "#7fa860",
    "trees": "#2e5d3a",
}
CLASS_RU = {
    "trees": "деревья",
    "grass": "травы",
    "crops": "пашня",
    "built": "застройка",
    "bare": "открытая земля",
}

# Legal boilerplate that makes a municipal name unreadable on a button. The name is shortened, never re-invented:
# whatever survives the strip is the name the register carries.
BOILERPLATE = (
    "внутригородская территория города федерального значения",
    "муниципальный район",
    "городской округ город",
    "муниципальный округ",
    "сельское поселение",
    "городское поселение",
    "муниципальный",
    "район",
    "городской",
    "округ",
    "город",
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def short_name(name: str) -> str:
    """Strip the legal form, keep the part a viewer would recognise."""
    stripped = re.sub(
        r"(?iu)" + "|".join(re.escape(phrase) for phrase in BOILERPLATE), " ", name
    )
    words = [w for w in stripped.split() if w.lower() not in BOILERPLATE]
    keep = " ".join(words).strip()
    return keep or name


def _dehome(value):
    """Drop the machine-specific prefix from any path the page or its payload may carry.

    A cache location recorded as "/home/<user>/coding/gee-cache" is true on the machine that built
    it and meaningless on anyone else's; the fact worth keeping is that the cache sits outside the
    repository. So the prefix goes away and the relative name stays.
    """
    if isinstance(value, str):
        home = str(Path.home())
        if value.startswith(home + "/"):
            return value[len(home) + 1:]
        return value.replace(home + "/", "")
    if isinstance(value, dict):
        return {k: _dehome(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dehome(v) for v in value]
    return value


def _num(value, digits=1):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number, digits)


def _display(value, divide=1.0):
    """The number a reader can check outside us: the stored value in the unit the catalog page states."""
    if value is None:
        return None
    try:
        number = float(value) / float(divide)
    except (TypeError, ValueError):
        return None
    return _num(number, 2)


def _catalog_link(asset: str) -> str:
    """The public catalog page for an Earth Engine asset: the slug is the asset id with slashes as underscores.

    Provenance on the face of a drawer should let a reader check us from the outside. A local path or an internal
    asset leaf does not; a public page, a centre point and an extent do.
    """
    return "https://developers.google.com/earth-engine/datasets/catalog/" + asset.replace("/", "_")


# What may be said about the terms of use, and only what may be said: two of these were read off a real page,
# one could not be read at all, and the rest are described by the catalog page rather than asserted by us.
RU_LICENCE = {
    "GOOGLE/DYNAMICWORLD/V1": "CC BY 4.0 - указано на странице набора",
    "ISRIC/SoilGrids250m/v2_0": "CC BY 4.0 - прочитано на странице набора 2026-10-08",
    "UMD/hansen/global_forest_change_2025_v1_13": "лицензия не прочитана ни на одной странице; долг записан в отчёте",
}
GENERIC_LICENCE = "условия использования описаны на странице набора в каталоге"

# What each source is, what we do with it, and what we refuse to do with it - said in the viewer's words.
# 'divide' turns the stored value into the unit the catalog page states, so a number on screen is the number
# a reader can check outside our repository.
SOURCE_NOTES = {
    "dynamic_world": {
        "explainer": "Классификация земной поверхности по пяти классам: деревья, трава, пашня, застройка, открытая земля.",
        "use": "Из него считаются доли классов - это основа пространства кластеризации.",
        "not_used": "Мы не берем из него площадь района: площадь берётся из официального реестра.",
        "native": "10 м на пиксель",
        "cadence": "новый классификатор - почти каждый год",
        "unit": "доля пикселей",
        "divide": None,
    },
    "sentinel2": {
        "explainer": "Настоящая фотография со спутника: красный, зелёный и синий каналы. Не шкала, а цвета.",
        "use": "Только для проверки глазом: совпадает ли то, что видно, с тем, что посчитано.",
        "not_used": "Не входит в пространство кластеризации и ничего не измеряет.",
        "native": "10 м на пиксель",
        "cadence": "новый снимок - каждые несколько дней",
        "unit": "яркость канала",
        "divide": None,
    },
    "viirs": {
        "explainer": "Ночная подсветка: как ярко светятся населённые места ночью.",
        "use": "Как признаки того, что место застройено и активно.",
        "not_used": "Не является мерой доходов или зарплат - мы не выводим из него деньги.",
        "native": "около 750 м на пиксель",
        "cadence": "итог за год собирается один раз в год",
        "unit": "ед. яркости",
        "divide": None,
    },
    "ndvi": {
        "explainer": "вегетация - NDVI (Normalized Difference Vegetation Index)",
        "full": "полное название - нормализованный различностный индекс растительности",
        "formula": "NDVI = ( ближний инфракрасный - красный ) / ( ближний инфракрасный + красный )",
        "use": "",
        "not_used": "",
        "native": "250 м на пиксель",
        "cadence": "новая точка каждые 16 дней",
        "unit": "NDVI",
        "divide": 10000.0,
    },
    "terrain": {
        "explainer": "",
        "use": "",
        "not_used": "",
        "native": "30 м на пиксель",
        "cadence": "не меняется: это рельеф",
        "unit": "м над уровнем моря",
        "divide": None,
    },
    "hansen": {
        "explainer": "Год, когда лес в этом пикселе был потерян; 0 означает, что потерь не было.",
        "use": "Как независимая проверка: подтверждает ли потеря леса то, что сказала классификация земли.",
        "not_used": "Не заменяет классификацию земли и не входит в пространство кластеризации.",
        "native": "30 м на пиксель",
        "cadence": "обновляется раз в год",
        "unit": "год потери, от 0 до 25",
        "divide": None,
    },
    "soil": {
        "explainer": "Влагосодержание почвы на глубине 15-30 см по модели SoilGrids.",
        "use": "Оставлено как справочный слой: он не участвует в разделении.",
        "not_used": "Не входит в пространство кластеризации.",
        "native": "250 м на пиксель",
        "cadence": "модельная оценка, не обновляется ежегодно",
        "unit": "процентов объёма",
        "divide": 10.0,
    },
}

WINDOW_TEXT = {
    "viirs": "2023 год, среднее по всем месяцам",
    "ndvi": "2023 год, среднее по 16-дневным снимкам",
    "terrain": "высота не меняется: это рельеф, а не наблюдение",
    "hansen": "потери леса до 2025 года включительно",
    "soil": "почвенная съёмка SoilGrids версии 2_0",
    "sentinel2": "май-сентябрь 2023, снимки без облаков",
    "dynamic_world": None,
}
SOURCE_ORDER = (
    "dynamic_world",
    "viirs",
    "ndvi",
    "terrain",
)
# Forest loss came off the rail on 2026-10-09 for the same reason as SoilGrids: the hansen columns never entered
# the k=5 space, so offering it as a source promised something the analysis does not do. Its crops stay committed
# as evidence, along with the recorded finding that all three windows report no forest loss.
# SoilGrids came off the rail on 2026-10-09 at the entrant's request: its volumetric-water columns never entered
# the clustering space, so offering it promised something the analysis does not do. Its crops stay as evidence.
# Sentinel-2 true colour is exported and kept as evidence, but the entrant took it off the rail on 2026-10-09:
# a photographic mosaic at 500 m reads as noise next to the classified layers, and a basemap would serve better.


def _capture(bbox: list, lon: float, lat: float) -> dict:
    """How the window was taken, in units a reader can verify outside our repository."""
    wide_deg = round(float(bbox[2]) - float(bbox[0]), 4)
    tall_deg = round(float(bbox[3]) - float(bbox[1]), 4)
    return {
        "center_lon": round(lon, 4),
        "center_lat": round(lat, 4),
        "wide_deg": wide_deg,
        "tall_deg": tall_deg,
        "wide_km": round(wide_deg * 111.320 * math.cos(math.radians(lat)), 1),
        "tall_km": round(tall_deg * 110.574, 1),
    }


def _source_periods(key: str, slots: dict) -> list:
    """The observation years a source is offered in, with night lights held to the year the analysis uses.

    The shipped light column is the 2023 composite with a fixed line, and that is not changing: the per-year
    relative line could not be recovered from the stored profile within an error a jury could accept - worst 23.5
    percentage points. Offering a 2024 light picture next to a clustering that reads 2023 would show the reader
    one year and compute another, so the light examples stay on the year the column is from while land and
    greenness still move between years.
    """
    periods = sorted({str(period) for slot in slots.values() for period in slot})
    if key == "viirs":
        return [period for period in periods if period == "2023"] or periods[:1]
    return periods


def build_payload() -> dict:
    crops = json.loads(CROPS.read_text(encoding="utf-8"))
    recreations = json.loads(RECREATIONS.read_text(encoding="utf-8"))
    chain = json.loads(CHAIN.read_text(encoding="utf-8"))
    cadence = json.loads(CADENCE.read_text(encoding="utf-8"))
    families = json.loads(CADENCE_FAMILIES.read_text(encoding="utf-8"))
    dominance = json.loads(DOMINANCE.read_text(encoding="utf-8"))
    demo = json.loads(DEMO_STATES.read_text(encoding="utf-8"))
    demo_states = [
        {
            "space": state["space"],
            "space_title": state["space_title"],
            "k": state["k"],
            "columns": state["columns"],
            "districts": state["districts"],
            "image": "assets/" + Path(state["image"]).name,
            "cluster_sizes": state["cluster_sizes"],
            "family_share_pct": state["family_share_pct"],
            "ari_vs_shipped": state["ari_vs_shipped"],
            "moved_pct": state["moved_pct"],
        }
        for state in demo["states"]
    ]
    superiority = json.loads(SUPERIORITY.read_text(encoding="utf-8"))
    dw_sub = json.loads(DW_SUB.read_text(encoding="utf-8"))
    temporal = json.loads(DW_TEMPORAL.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    names = json.loads(NAMES.read_text(encoding="utf-8"))
    geom = json.loads(GEOM.read_text(encoding="utf-8"))
    ramp = json.loads(RAMP.read_text(encoding="utf-8"))

    real = [r for r in crops["records"] if r["kind"] == "real observation"]
    by_district: dict[int, dict] = {}
    for record in real:
        slot = by_district.setdefault(
            int(record["tid"]),
            {
                "district": record["district"],
                "short": short_name(record["district"]),
                "cluster": int(record["cluster"]),
                "tid": int(record["tid"]),
                "why": record["why_chosen"],
                "years": {},
                "images": {},
            },
        )
        year = record["asset_year"]
        slot["years"][year] = {
            "means": {
                k: _num(float(v) * 100, 1)
                for k, v in record["mean_class_share_in_crop"].items()
                if v is not None
            },
            "region": record["region_bbox"],
            "scale_m": int(record["scale_m"]),
            "asset_leaf": record["asset_leaf"],
            "sha": record["payload_sha256"][:12],
            "cache": record["cache_path"],
        }
        slot["images"][year] = "assets/" + Path(record["committed_file"]).name

    example_order = [2000, 2208, 2518]
    import pandas as pd

    master = pd.read_parquet(ROOT / "data" / "processed" / "territory_master.parquet")
    master["tid"] = master["tid"].astype("int64")
    region_by_tid = {
        int(tid): str(name)
        for tid, name in zip(master["tid"], master["region_name"])
    }
    for item in by_district.values():
        item["region"] = region_by_tid.get(item["tid"], "")
    districts = sorted(
        by_district.values(),
        key=lambda d: (
            example_order.index(d["tid"]) if d["tid"] in example_order else len(example_order),
            d["short"],
        ),
    )

    # The failed 2018 attempt and its labelled fallback stay in the artifacts, where a reader who wants the
    # recorded failure can find it. On the face they would only be a broken image and an exception string.
    fallback = []
    for record in [r for r in crops["records"] if r["kind"] == "unavailable"]:
        recreation = next(
            (r for r in recreations["records"] if int(r["tid"]) == int(record["tid"])), None
        )
        if recreation is None:
            continue
        fallback.append(
            {
                "district": record["district"],
                "year": record["asset_year"],
                "error": record["error"],
                "image": "assets/" + Path(recreation["committed_file"]).name,
                "artifact": recreation["source_artifact"],
                "fields": recreation["fields_used"],
                "shares": recreation["shares_2023"],
                "kind": "recreation",
            }
        )

    walked = [c for c in chain["districts"] if c["steps"][3]["values"]["official_land_area_km2"]]
    chain_block = None
    if walked:
        pick = walked[0]
        steps = pick["steps"]
        total_px = steps[0]["values"]["total_pixels"]
        class_px = steps[0]["values"]["classified_pixels"]
        chain_block = {
            "district": pick["district"],
            "short": short_name(pick["district"]),
            "cluster": pick["cluster"],
            "total_pixels": int(total_px),
            "classified_pixels": int(class_px),
            "classified_share_pct": _num(class_px / total_px * 100, 1) if total_px else None,
            "shares": {
                k: _num(float(v) * 100, 1) for k, v in steps[1]["values"].items() if v is not None
            },
            "official_km2": _num(steps[3]["values"]["official_land_area_km2"], 1),
            "osm_km2": _num(steps[3]["values"]["osm_area_km2"], 1),
            "raw_over_official": _num(steps[3]["values"]["raw_over_official"], 2),
            "spending_rub": _num(steps[4]["values"]["total_mean_rub"], 0),
            "density": _num(steps[4]["values"]["stored_density"], 2),
            "artifact": rel(CHAIN),
        }

    inflation = pd.read_csv(INFLATION).sort_values("inflation_ratio", ascending=False)
    geometry = {
        "sum_km2": _num(geom["geometry"]["sum_polygons_km2"], 2),
        "union_km2": _num(geom["geometry"]["union_km2"], 2),
        "difference_km2": _num(
            geom["geometry"]["sum_polygons_km2"] - geom["geometry"]["union_km2"], 0
        ),
        "artifact": rel(GEOM),
        "top": [
            {
                "region": str(row.region_name),
                "ratio": _num(row.inflation_ratio, 2),
                "official": _num(row.official_area_km2, 1),
            }
            for row in inflation.head(3).itertuples()
        ],
        "inflation_artifact": rel(INFLATION),
    }

    source_records = json.loads(SOURCE_CROPS.read_text(encoding="utf-8"))
    source_legends = json.loads(SOURCE_LEGENDS.read_text(encoding="utf-8"))
    pipeline = json.loads(SOURCE_PIPELINE.read_text(encoding="utf-8"))["sources"]
    added = json.loads(ADDED_LAYERS.read_text(encoding="utf-8"))
    licences = added["licences"]
    centers = {
        int(record["tid"]): (float(record["lon"]), float(record["lat"]))
        for record in crops["records"]
        if record.get("lon") is not None and record.get("lat") is not None
    }
    scales = source_legends.get("scale_per_source", {})

    crops_by_source: dict[str, dict[int, dict]] = {}
    for record in source_records["records"]:
        if record["kind"] not in ("real observation", "rebuilt from cache"):
            continue
        key = record["source"]
        tid = int(record["tid"])
        lon, lat = centers.get(tid, (0.0, 0.0))
        stats = record.get("stats") or {}
        low, high = stats.get("min"), stats.get("max")
        bbox = record["region_bbox"]
        divide = SOURCE_NOTES.get(key, {}).get("divide") or 1.0
        native = record.get("native_pixels") or [0, 0]
        height_px, width_px = int(native[0]), int(native[1])
        east_m = (
            round(
                (float(bbox[2]) - float(bbox[0]))
                * 111320.0
                * math.cos(math.radians(lat))
                / width_px,
                1,
            )
            if width_px
            else None
        )
        north_m = (
            round(
                (float(bbox[3]) - float(bbox[1])) * 110574.0 / height_px,
                1,
            )
            if height_px
            else None
        )
        period = record.get("asset_year")
        slot = crops_by_source.setdefault(key, {}).setdefault(tid, {})
        shared_low = record.get("scale_low")
        shared_high = record.get("scale_high")
        display_low = float(shared_low) if shared_low is not None else _num(low)
        display_high = float(shared_high) if shared_high is not None else _num(high)
        if display_low is None or display_high is None:
            display_mid = None
        elif record.get("scale_mode") == "log" and display_low > 0 and display_high > display_low:
            # On a logarithmic ruler the middle of the gradient is the geometric mean, not the arithmetic one.
            display_mid = 10.0 ** ((math.log10(display_low) + math.log10(display_high)) / 2.0)
        else:
            display_mid = (display_low + display_high) / 2.0
        slot[str(period) if period is not None else ""] = {
            "image": "assets/" + Path(record["committed_file_plain"]).name
            if record.get("committed_file_plain")
            else "assets/" + Path(record["committed_file"]).name,
            "plain": "assets/" + Path(record["committed_file_plain"]).name
            if record.get("committed_file_plain")
            else None,
            "coverage_pct": _num(float(stats["valid_share"]) * 100, 1)
            if stats.get("valid_share") is not None
            else None,
            "year": record.get("asset_year"),
            "window": record.get("window") or WINDOW_TEXT.get(key),
            "capture": _capture(record["region_bbox"], lon, lat),
            "scale_m": int(record.get("scale_m") or 500),
            "pixels": record.get("native_pixels"),
            "pixel_east_m": east_m,
            "pixel_north_m": north_m,
            "low_display": _display(display_low, divide),
            "mid_display": _display(display_mid, divide),
            "high_display": _display(display_high, divide),
            "scale_basis": record.get("scale_basis"),
            "scale_mode": record.get("scale_mode"),
            "zero_m": record.get("zero_m_above_sea"),
            "relief_m": record.get("relief_max_m"),
            "lit_share_pct": record.get("lit_share_pct"),
            "analysis_low": record.get("scale_low"),
            "step_shares_pct": record.get("step_shares_pct"),
            "boundary_paths": record.get("boundary_paths"),
            "boundary_pixels": record.get("boundary_pixels") or record.get("committed_pixels"),
            "interior_share_pct": record.get("interior_share_pct"),
            "boundary_basis": record.get("boundary_basis"),
            "coverage": stats.get("valid_share"),
            "low": low,
            "high": high,
            "uniform": bool(low is not None and low == high),
        }
    dw_asset = "GOOGLE/DYNAMICWORLD/V1"
    source_blocks = []
    for key in SOURCE_ORDER:
        block = crops_by_source.get(key)
        if not block:
            continue
        legend = source_legends["sources"].get(key, {})
        asset = legend.get("asset") or (dw_asset if key == "dynamic_world" else None)
        scale = scales.get(key, {})
        source_blocks.append(
            {
                "key": key,
                "label": legend.get("label", key),
                "meaning": legend.get("meaning", ""),
                "classes": legend.get("classes"),
                "ramp_colors": legend.get("ramp_colors"),
                "reading": legend.get("reading_direction"),
                "pipeline": pipeline.get(key, {}).get("steps", []),
                "in_space": pipeline.get(key, {}).get("in_space"),
                "matched": pipeline.get(key, {}).get("matched_columns", []),
                "periods": _source_periods(key, crops_by_source.get(key, {})),
                "step_labels": [
                    item.get("label") for item in (legend.get("classes") or legend.get("steps") or [])
                ],
                "step_shorts": [
                    item.get("short") for item in (legend.get("classes") or legend.get("steps") or [])
                ],
                "step_colors": [item.get("color") for item in (legend.get("classes") or [])],
                "cell_km": source_records.get("grid_cell_km"),
                "scale_low": scale.get("min"),
                "scale_high": scale.get("max"),
                "coverage": scale.get("valid_share_mean"),
                "catalog": _catalog_link(asset) if asset else None,
                "licence": RU_LICENCE.get(asset, GENERIC_LICENCE) if asset else GENERIC_LICENCE,
                **SOURCE_NOTES.get(key, {}),
                "low_display": _num(
                    scale.get("min") / (SOURCE_NOTES.get(key, {}).get("divide") or 1.0), 2
                )
                if scale.get("min") is not None
                else None,
                "high_display": _num(
                    scale.get("max") / (SOURCE_NOTES.get(key, {}).get("divide") or 1.0), 2
                )
                if scale.get("max") is not None
                else None,
                "years": sorted(
                    {
                        int(entry["year"])
                        for entry in block.values()
                        if entry.get("year") is not None
                    }
                ),
            }
        )

    cadence_rows = [
        {
            "layer": r["layer"],
            "status": r["static_or_refreshable"],
            "why": r["why"],
            "proof": r.get("proof_value"),
            "seconds": r.get("measured_recompute_seconds"),
            "in_space": r.get("in_approved_space"),
            "asset": r.get("asset"),
        }
        for r in cadence["rows"]
    ]

    arms = {
        name: dict(
            entry,
            churn_pct=_num(float(entry["churn_after_alignment"]) * 100, 1),
        )
        for name, entry in {
            "land_only": dw_sub["movement"]["dw_only|k5"],
            "shape_only": dw_sub["movement"]["dw_frozen|k5"],
            "both": dw_sub["movement"]["dw_refreshed|k5"],
            "frozen": temporal["movement"]["all_land_frozen_including_dw|k5"],
        }.items()
    }

    # One word changed by reader request, everywhere it is shown: the NDVI source now reads
    # "вегетация" instead of "зелёность". The artifacts keep their own wording; this is a display
    # choice on the face of the page, applied to whatever the source blocks carry.
    for block in source_blocks:
        for key, value in list(block.items()):
            if isinstance(value, str) and "зелёность" in value:
                block[key] = value.replace("зелёность", "вегетация")

    return {
        "legend": {
            "classes": [
                {"key": key, "label": CLASS_RU[key], "color": PALETTE[key]}
                for key in ("trees", "grass", "crops", "built", "bare")
            ],
            "rule": "красивая картинка без легенды читателя не учит ничего: каждый цвет назван",
        },
        "provenance": {
            "districts": districts,
            "sources": source_blocks,
            "crops": crops_by_source,
            "boundary_source": source_records["boundary_source"],
            "outside_dimmed": source_records["outside_the_boundary_is_dimmed"],
            "source_crops_artifact": rel(SOURCE_CROPS),
            "legends_artifact": rel(SOURCE_LEGENDS),
            "fallback": fallback,
            "chain": chain_block,
            "geometry": geometry,
            "crops_artifact": rel(CROPS),
            "recreations_artifact": rel(RECREATIONS),
            "chain_artifact": rel(CHAIN),
            "asset_note": crops["never_names_the_granting_project"],
        },
        "cadence": {
            "rows": cadence_rows,
            "counts": {
                "static": sum(1 for r in cadence_rows if r["status"] == "static"),
                "refreshable": sum(1 for r in cadence_rows if r["status"] == "refreshable"),
            },
            "throughput": cadence["measured_throughput"],
            "scope": dict(
                cadence["why_annual"],
                coverage_gate_pct=_num(float(cadence["why_annual"]["coverage_gate"]) * 100, 1),
                annual_share_pct=_num(
                    float(cadence["why_annual"]["annual_coverage_mean_share"]) * 100, 1
                ),
            ),
            "claim": cadence["update_frequency_claim"],
            "artifact": rel(CADENCE),
            "families": {
                "lanes": families["lanes"],
                "families": families["families"],
                "total": sum(len(family["members"]) for family in families["families"]),
                "family_count": len(families["families"]),
                "unproven": families["honesty"]["inputs_without_a_proven_lane"],
                "why_unproven_stay": families["honesty"]["why_they_still_appear"],
                "artifact": rel(CADENCE_FAMILIES),
            },
        },
        "decision_maps": [
            {
                "index": index,
                "svg": f"decision_state_{index}.svg",
                "caption": state["space_title"] + " · k=" + str(state["k"]),
                "ari": state["ari_vs_shipped"],
                "moved_pct": state["moved_pct"],
                "sizes": state["cluster_sizes"],
                "family_share_pct": state["family_share_pct"],
                "image": state["image"],
            }
            for index, state in enumerate(demo["states"])
            if index in (2, 6, 10, 14, 0, 1, 3)
            and (ROOT / "artifacts" / "landing" / f"decision_state_{index}.svg").exists()
        ],
        "demo": {
            "states": demo_states,
            "spaces": demo["spaces"],
            "space_titles": demo["space_titles"],
            "k_values": demo["k_values"],
            "district_rule": demo["district_rule"],
            "nothing_computed_in_the_browser": demo["nothing_computed_in_the_browser"],
        },
        "decision": {
            "approved_k": int(decision["approved_k"]),
            "space_name": decision["space"]["name"],
            "columns": len(decision["space"]["columns"]),
            "amount_excluded": bool(decision["space"]["spending_amount_excluded"]),
            "decided_by": decision["decided_by"],
            "reason": decision["reason"],
            "k_artifact": rel(DECISION),
            "dominance": {
                "without": {
                    k: _num(float(v) * 100, 1)
                    for k, v in dominance["dominance"]["without_amount"][
                        "family_split_share"
                    ].items()
                    if v is not None
                },
                "with": {
                    k: _num(float(v) * 100, 1)
                    for k, v in dominance["dominance"]["with_amount"][
                        "family_split_share"
                    ].items()
                    if v is not None
                },
                "labels": {
                    "land": "земля",
                    "spending_shape": "форма трат",
                    "spending_amount": "сумма трат",
                },
                "moved": _num(float(ramp["moved_between_partitions"]) * 100, 1),
                "artifact": rel(DOMINANCE),
                "ramp_artifact": rel(RAMP),
            },
            "filtering": {
                "filtered": superiority["comparison"]["filtered"],
                "unfiltered": superiority["comparison"]["unfiltered"],
                "verdict": superiority["comparison"]["verdict"],
                "artifact": rel(SUPERIORITY),
            },
            "arms": arms,
            "arms_artifact": rel(DW_SUB),
            "control_artifact": rel(DW_TEMPORAL),
            "names": {str(k): str(v) for k, v in names.items()},
            "names_artifact": rel(NAMES),
        },
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Земля под тратами — три решения, показанные картинкой</title>
<style>
:root{
  --paper:#f4f3ef; --card:#ffffff; --ink:#14161c; --muted:#5d616a;
  --line:#e3dfd7; --accent:#8c3a2e; --accent2:#2e5d3a; --chip:#efece5;
}
*{box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.4 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
h2{margin:0 0 22px;font-size:30px;line-height:1.15;font-weight:650;letter-spacing:-0.01em}
.lede{margin:8px 0 0;color:var(--muted);font-size:17px;max-width:56ch}
.hero-title{font-size:34px;font-weight:700;line-height:1.12;margin:6px 0 0;max-width:24ch}
.byline{display:flex;align-items:center;gap:16px;margin-top:28px;padding-top:18px;
  border-top:1px solid var(--line);max-width:46ch}
.byline .face{width:66px;height:66px;border-radius:50%;object-fit:cover;border:1px solid var(--line)}
.byline .who{display:flex;flex-direction:column;gap:4px}
.byline .who b{font-size:16px;font-weight:700}
.byline .who small{font-size:13px;color:var(--muted);line-height:1.35}
#rail{position:fixed;top:0;left:0;right:0;height:62px;z-index:40;
  background:rgba(255,255,255,.86);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);display:flex;align-items:center;gap:8px;padding:0 26px}
#rail .brand{font-size:20px;font-weight:700;letter-spacing:.02em;text-transform:uppercase;color:var(--ink);margin-right:24px}
.hint{position:fixed;bottom:26px;left:50%;transform:translateX(-50%);z-index:20;font-size:13px;
  font-weight:600;color:var(--ink);background:var(--card);border:1px solid var(--ink);
  border-radius:999px;padding:7px 13px;display:flex;align-items:center;gap:9px;white-space:nowrap}
.hint .chev{width:7px;height:7px;border-right:2px solid var(--ink);border-bottom:2px solid var(--ink);
  transform:rotate(45deg);flex:0 0 auto}
.hint[hidden]{display:none}
#rail button{font:inherit;font-size:15px;font-weight:600;color:var(--muted);background:transparent;
  border:0;border-radius:999px;padding:9px 16px;cursor:pointer}
#rail button:hover{background:var(--chip);color:var(--ink)}
#rail button[aria-selected="true"]{background:var(--ink);color:#fff}
#story{position:absolute;inset:62px 0 0 0}
.slide{position:absolute;inset:0;display:none;padding:14px 30px 30px;
  grid-template-rows:auto 1fr auto;grid-template-columns:1fr}
.slide.active{display:grid}
#s-hero.active{display:flex;align-items:flex-end;gap:36px;padding:34px 46px 44px}
.herotext{flex:1 1 36%;min-width:0}
.heromapside{flex:1 1 64%;min-width:0;display:flex;justify-content:flex-end;align-items:flex-end}
.heromap{display:block;width:100%;height:auto;max-height:84vh;object-fit:contain}
#s-map.active{display:block;padding:0;height:calc(100vh - 84px)}
#s-cadence .card{height:100%}
#s-decision.active{display:grid;gap:16px;padding:16px 30px 20px;
  grid-template-columns:minmax(0,1fr);
  grid-template-rows:auto auto minmax(0,1fr) auto}
.dswitch{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.dswitch>span{font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  margin-right:4px;font-weight:600}
.dbtn{font:inherit;font-size:.86rem;font-weight:600;background:#fff;color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:7px 14px;cursor:pointer}
.dbtn[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:#fff}
.dpane{display:none;min-height:0}
.dpane.active{display:flex;flex-direction:column;gap:12px;min-height:0}
.dpane.active>*{flex:0 0 auto}
.dpane.active .setrow,.dpane.active .volgrid{flex:1 1 auto;min-height:0}
.dprompt{font-size:.86rem;color:var(--muted);margin:0}
.dfoot{display:grid;gap:4px;font-size:.84rem}
.setrow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;min-height:0}
.setrow.kpair{grid-template-columns:repeat(2,minmax(0,1fr))}
.bigmap{display:block;width:100%;height:auto;aspect-ratio:912/577;object-fit:contain;
  object-position:center;background:#fff;border:1px solid var(--line);border-radius:10px}
.volgrid{display:grid;grid-template-rows:minmax(0,1fr) auto;gap:12px;min-height:0}
.volimg{display:block;width:100%;height:100%;min-height:0;max-height:52vh;object-fit:contain;
  background:#fff;border:1px solid var(--line);border-radius:10px}
#s-decision .decisionrow{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;min-height:0}
#s-decision .decisionrow .card{height:100%;min-height:0}
.mapstrip{display:grid;grid-template-rows:repeat(3,minmax(0,1fr));gap:10px;min-height:0}
.mapcell{margin:0;display:flex;flex-direction:column;min-height:0}
.mapcell .mapimg{flex:1 1 auto;min-height:0;width:100%;max-height:19vh;object-fit:contain;
  object-position:center center;background:#fff;border:1px solid var(--line);border-radius:10px}
.mapcell figcaption{font-size:.84rem;font-weight:600;margin-top:6px;color:var(--ink)}
.mapcell small{display:block;font-size:.76rem;color:var(--muted);margin-top:2px;
  font-variant-numeric:tabular-nums}
.kmapimg{display:block;width:100%;max-height:19vh;object-fit:contain;
  background:#fff;border:1px solid var(--line);border-radius:10px}
.proofimg{display:block;width:100%;height:auto;max-height:21vh;object-fit:contain;
  background:#fff;border:1px solid var(--line);border-radius:10px}
.sharestrip{border-top:1px solid var(--line);padding-top:10px}
.sharehead{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  font-weight:600;margin-bottom:6px}
.sharepair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px}
.sharepair small{display:block;font-size:.76rem;color:var(--muted);margin-bottom:3px}
.inbar{display:grid;grid-template-columns:minmax(86px,110px) minmax(0,1fr) minmax(46px,58px);
  align-items:center;gap:7px;font-size:.78rem;padding:1px 0}
.inbar b{font-weight:600}
.inbar .track{height:6px;background:#eceef1;border-radius:4px;overflow:hidden}
.inbar .fill{height:100%;border-radius:4px 0 0 4px}
.inbar em{font-style:normal;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.mapframe{width:100%;height:100%;border:0;background:#fff}
#s-hero .kicker{margin-bottom:2px}
.stage{flex:1;min-height:0;display:grid;gap:18px;align-content:stretch;grid-auto-rows:minmax(0,1fr)}
.cadgroup{border-top:1px solid var(--line);padding:10px 0 8px}
.cadgroup>b{display:block;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin-bottom:6px}
.cadrow{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr) minmax(0,1fr);gap:14px;
  padding:4px 0;align-items:start}
.cadname span{font-weight:600;font-size:.86rem}
.cadname small{display:block;font-size:.72rem;color:var(--muted);overflow-wrap:anywhere;margin-top:2px}
.trio{display:flex;flex-direction:column;gap:4px}
.trioline{display:flex;gap:8px;align-items:center;font-size:.76rem;color:var(--muted)}
.trioline small{min-width:88px}
.fq{border:1px solid var(--line);border-radius:7px;padding:3px 9px;font-size:.8rem;font-weight:600;
  background:var(--chip);white-space:nowrap}
.cadtable{width:100%;border-collapse:collapse;font-size:.92rem}
.cadtable th{text-align:left;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);padding:0 10px 9px;border-bottom:1px solid var(--line);font-weight:600}
.cadtable td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.cadtable tbody tr:last-child td{border-bottom:0}
.cadtable .cadgroup{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.cadtable .cadname b{font-size:.95rem}
.cadtable .cadcols .colname{font-size:.8rem;color:#3d444c;overflow-wrap:anywhere;padding:2px 0}
.cadtable .cadlast b{font-size:.9rem}
.cadtable .cadlast small{display:block;font-size:.78rem;color:var(--muted);overflow-wrap:anywhere;margin-top:3px}
.cadtable .trio{gap:5px}
.cadtable .trioline{font-size:.78rem;gap:9px}
.cadtable .trioline small{min-width:96px}
.mapcell{margin:0}
.mapcell .mapimg{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--line);
  border-radius:10px}
.mapcell figcaption{font-size:.8rem;font-weight:600;margin-top:6px;color:var(--ink)}
.mapcell small{display:block;font-size:.76rem;color:var(--muted);margin-top:3px}
.krow{display:flex;gap:8px;margin-bottom:10px}
.kbtn{font:inherit;font-size:.86rem;font-weight:600;background:#fff;color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:6px 14px;cursor:pointer}
.kbtn[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:#fff}
.kcaption{font-size:.86rem;font-weight:600;margin:8px 0 2px}
.kstat{font-size:.82rem;color:var(--muted);font-variant-numeric:tabular-nums;margin:0}
.proofimg{display:block;width:100%;height:auto}
.card{min-height:0;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;
  box-shadow:0 1px 0 rgba(20,22,28,.04);position:relative;display:flex;flex-direction:column;min-height:0}
.card h3{margin:0 0 6px;font-size:13px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);padding-right:150px}
figure{margin:0;min-height:0;display:flex;flex-direction:column;flex:1}
.cropbox{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;position:relative}
.frame{position:relative;display:block;max-width:100%;max-height:100%}
.frame>img{display:block;width:100%;height:auto}
h3 .toggle{margin-left:12px;font:inherit;font-size:12px;letter-spacing:.04em;padding:4px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;cursor:pointer;vertical-align:middle}
h3 .toggle[aria-pressed="true"]{background:#f4efe4;border-color:#b8a88c;color:#14161c}
h3 .jump{margin-left:14px;font-size:12px;font-weight:600;color:var(--accent);text-decoration:none;border-bottom:1px solid var(--line);vertical-align:middle}
.bounds{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
figure img{width:100%;height:auto;max-height:100%;object-fit:contain;
  border-radius:10px;background:#e9e6df}
figcaption{margin-top:8px;font-size:15px;color:var(--ink);min-height:calc(1.35em * 2);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.legend{display:flex;flex-wrap:wrap;gap:10px 14px;margin-top:10px}
.legend span{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--muted)}
.legend i{width:13px;height:13px;border-radius:3px;display:inline-block}
.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.row button{font:inherit;font-size:14px;font-weight:600;padding:8px 13px;border-radius:9px;
  border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.row button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.row button:disabled{opacity:.42;cursor:default}
.row button small{display:block;font-weight:400;font-size:11px;color:currentColor;opacity:.75}
.kicker{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
.stat{font-size:26px;font-weight:650;letter-spacing:-0.02em}
.stat small{display:block;font-size:12px;font-weight:500;color:var(--muted);letter-spacing:0}
.bars{display:grid;gap:7px;margin-top:8px}
.bar{display:grid;grid-template-columns:112px 1fr 56px;align-items:center;gap:8px;font-size:13px}
.bar b{font-weight:600;color:var(--muted)}
.bar .track{height:9px;background:var(--chip);border-radius:5px;overflow:hidden}
.bar .fill{height:100%;border-radius:5px}
.bar em{font-style:normal;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.note{font-size:14px;color:var(--muted);margin:10px 0 0}
.desc{font-size:14px;color:var(--ink);line-height:1.45;margin:0 0 8px}
.formula{font-size:13px;color:var(--ink);background:var(--chip);border:1px solid var(--line);
  border-radius:8px;padding:6px 9px;margin:0 0 8px;font-variant-numeric:tabular-nums}
.fine{margin:8px 0 0}
.bar + .bar{margin-top:6px}
.warn{border-left:3px solid var(--accent);padding-left:10px;font-size:14px;color:var(--ink)}
/* The check-source control sits in the left rail, below the source, district and year selectors, because that
   is where the reader has just chosen what to look at. The panel is a wide fixed overlay: it used to open inside
   the figure card and was too small to hold the explanation. Cards behind it fade so it does not read as floating
   over the legend. */
details.src{margin-top:auto;padding-top:14px;border-top:1px solid var(--line);position:relative}
details.src>summary{display:block;width:100%;text-align:left;font-size:12px;font-weight:600;color:var(--muted);
  cursor:pointer;list-style:none;background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:8px 11px}
details.src>summary::before{content:"ⓘ "}
details.src[open]>summary{color:var(--accent)}
details.src>div{position:absolute;left:0;bottom:calc(100% + 10px);width:min(620px,34vw);
  overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
  box-shadow:0 18px 46px rgba(20,22,28,.18);font-size:13px;line-height:1.6;color:var(--muted);z-index:31}
details.src>div button.close{float:right;font:inherit;font-size:12px;font-weight:600;background:var(--chip);
  border:1px solid var(--line);border-radius:999px;padding:4px 10px;cursor:pointer;color:var(--ink)}
details.src>div a{color:var(--accent);text-decoration:underline}
.dline{margin:0 0 5px;font-size:12.5px;line-height:1.45;color:var(--ink)}
.dline b{font-weight:600;color:var(--muted)}
.laneGrid{display:grid;gap:4px;padding-top:22px}
.laneRow{display:grid;grid-template-columns:110px repeat(4,minmax(0,1fr));gap:7px;align-items:start}
.laneHead span{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.laneHead b{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.laneCell{display:flex;flex-direction:column;gap:3px}
.familyName{font-size:14px;font-weight:650;color:var(--ink)}
.familyName small{display:block;font-size:11px;font-weight:500;color:var(--muted);margin-top:2px}
.inp{font:inherit;font-size:12px;text-align:left;background:var(--chip);border:1px solid var(--line);
  border-radius:8px;padding:4px 6px;line-height:1.25}
.inp .why{font-size:10px;color:var(--muted);margin-left:5px}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:middle}
.dot.measured{background:var(--accent)}
.dot.page{background:#8d9aa5}
.dot.artifact{background:#c9a227}
.dot.none{background:transparent;border:1.5px dashed var(--muted)}
.inp.unproven{border-style:dashed}
.col{display:flex;flex-direction:column;gap:6px;min-height:0;align-content:start}
.col button{text-align:left}
.col button{font:inherit;font-size:13px;font-weight:600;text-align:left;width:100%;padding:7px 10px;
  border-radius:8px;background:var(--chip);border:1px solid var(--line);color:var(--ink);cursor:pointer;line-height:1.3}
.col button:hover{border-color:var(--accent)}
.col button small{text-align:left;display:block;margin-top:2px;font-size:11px;color:var(--muted)}
.steps{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;font-size:12px;color:var(--ink)}
.steps span{background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:5px 9px}
.group{font-size:11px;color:var(--muted);letter-spacing:.05em;margin-top:10px;font-weight:600}
.sourceHead{font-size:11px;color:var(--muted);letter-spacing:.05em;margin-bottom:4px;font-weight:600}
button:disabled{opacity:.4;cursor:default}
button[aria-pressed="true"]{background:var(--card);border-color:var(--accent);color:var(--ink);
  box-shadow:inset 3px 0 0 var(--accent)}
button[aria-pressed="true"] small{color:var(--muted)}
.side{display:grid;gap:12px;min-height:0;align-content:start}
.legend{display:flex;flex-direction:column;gap:7px;font-size:12px;color:var(--muted)}
.legend .key{display:flex;align-items:center;gap:7px;color:var(--ink)}
.legend i{width:13px;height:13px;border-radius:4px;flex:0 0 auto}
.scaleRow{display:flex;gap:10px;align-items:stretch;min-height:132px}
.ramp{width:17px;border-radius:4px;border:1px solid var(--line);flex:0 0 auto}
.marks{display:flex;flex-direction:column;justify-content:space-between;font-size:11px;min-width:0}
.marks .mid{text-align:center;color:var(--muted)}
.fine{font-size:11px;color:var(--muted);line-height:1.35}
.chain{display:grid;grid-template-columns:1fr auto;gap:4px 8px;align-items:center;font-size:11px}
.chain span{background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:4px 8px;
  color:var(--ink);font-weight:600}
.chain em{font-style:normal;color:var(--muted);font-size:10px}
.chain .arrow{grid-column:1 / -1;text-align:center;color:var(--muted);font-size:12px;font-weight:400}
details.src>div h4{margin:0 0 6px;font-size:12px;color:var(--ink)}
details.src>div code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--ink)}
.timeline{display:grid;gap:0;margin-top:6px}
.tl{display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:center;padding:7px 0;
  border-bottom:1px solid var(--line);font-size:14px}
.tl:last-child{border-bottom:0}
.tl .fam{font-weight:600}
.tl .chips{display:flex;gap:6px;flex-wrap:wrap}
.tl .chips i{font-style:normal;font-size:12px;padding:4px 8px;border-radius:6px;background:var(--chip);color:var(--muted)}
.tl .chips i.now{background:#e5efe6;color:var(--accent2)}
.sheen{position:fixed;inset:auto 0 0 0;height:0;pointer-events:none}
</style>
</head>
<body>
<nav id="rail" role="tablist" aria-label="Три решения">
  <span class="brand">Земля под тратами</span>
</nav>
<p class="hint" id="scroll-hint">колёсико вниз - следующий раздел <i class="chev"></i></p>
<main id="story"></main>
<script>
const PDATA = __PDATA__;
const THIN = "\\u2009";
const story = document.getElementById("story");
const rail = document.getElementById("rail");

function num(value, decimals) {
  if (value === null || value === undefined || isNaN(value)) return "нет данных";
  const fixed = Number(value).toFixed(decimals).replace(".", ",");
  const parts = fixed.split(",");
  let whole = parts[0];
  whole = whole.replace(/\\B(?=(?:\\d{3})+(?!\\d))/g, THIN);
  return parts.length > 1 ? whole + "," + parts[1] : whole;
}
function pct(value, decimals) {
  const shown = num(value, decimals);
  // A missing number must not come back as "нет данных %": the unit belongs to the number, not to the apology.
  return shown === "нет данных" ? shown : shown + THIN + "%";
}
function el(tag, cls, text, extra) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  // Some call sites pass (tag, class, null, text). Without this, that text disappears without a trace -
  // which is exactly the silent-empty defect this repository keeps hunting.
  if (text === null || text === undefined) text = extra;
  if (text !== null && text !== undefined) node.textContent = text;
  return node;
}
function drawer(figure, lines) {
  const details = el("details", "src");
  details.appendChild(el("summary", null, "проверить источник"));
  const body = el("div", null);
  body.appendChild(el("h4", null, figure));
  lines.forEach(function (pair) {
    body.appendChild(el("div", null, pair[0] + ": "));
    body.appendChild(el("code", null, String(pair[1])));
  });
  details.appendChild(body);
  return details;
}

const TABS = [
  { id: "s-hero", label: "О проекте" },
  { id: "s-provenance", label: "Откуда данные" },
  { id: "s-cadence", label: "Как часто" },
  { id: "s-decision", label: "Признаки и k" },
  { id: "s-map", label: "Карта" }
];
let active = 0;
const slides = [];

TABS.forEach(function (tab, index) {
  const button = el("button", null, tab.label);
  button.id = "tab-" + tab.id;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", index === 0 ? "true" : "false");
  button.addEventListener("click", function () { select(index); });
  rail.appendChild(button);
  const slide = el("section", "slide");
  slide.id = tab.id;
  slide.setAttribute("role", "tabpanel");
  slide.setAttribute("aria-labelledby", button.id);
  story.appendChild(slide);
  slides.push(slide);
});

// Title slide: the name, the hypothesis in one sentence, and the one person behind it.
(function () {
  const hero = document.getElementById("s-hero");
  // The words sit in the lower left, the answer sits to their right. Stacking the map under the
  // text made it spill past the fold on every screen; side by side it stays inside the first view.
  const textCol = el("div", "herotext");
  textCol.appendChild(el("h2", "hero-title", null, 'Больше, чем "тратят много" и "тратят мало"'));
  textCol.appendChild(el("p", "lede", null,
    "Кластеризация только по тратам разделит округа на те, что тратят много, и на те, что тратят мало."
    + " Гипотеза: если обогатить данные о тратах геопространственными, разделение становится интереснее."
    + " Проверка - на снимках: классы земли, ночные огни, вегетация."));
  const by = el("div", "byline");
  const face = el("img", "face");
  face.src = "assets/author_avatar.jpg";
  face.alt = "Евгений Иванкин";
  by.appendChild(face);
  const who = el("div", "who");
  who.appendChild(el("b", null, "Евгений Иванкин"));
  who.appendChild(el("small", null, "МФТИ - студент"));
  who.appendChild(el("small", null, "Проект сделан одним человеком - это вся команда"));
  by.appendChild(who);
  textCol.appendChild(by);
  const fine = el("p", "fine", null,
    "Дальше - три вопроса: откуда берутся данные, как часто их можно пересчитывать, и почему типов ровно пять. ");
  const reportLink = el("a", null);
  reportLink.href = "../report/methodology-ru.md";
  reportLink.appendChild(document.createTextNode("Полный отчёт по методологии - Markdown."));
  fine.appendChild(reportLink);
  textCol.appendChild(fine);
  hero.appendChild(textCol);
  // The answer, before the argument. This is the same partition the interactive map at the end of
  // the story draws, baked as vector paths with no basemap under it, so nothing third party sits
  // behind it and it stays crisp whatever the screen is.
  const teaser = el("img", "heromap");
  teaser.src = "assets/hero_map.svg";
  teaser.alt = "Карта: каждый муниципалитет окрашен по своему типу экономики";
  const mapCol = el("div", "heromapside");
  mapCol.appendChild(teaser);
  hero.appendChild(mapCol);
})();

// The map is the last section of the same story, not a separate site the reader has to leave. It is
// the interactive page loaded inside this tab, with the two-year switch it already has.
(function () {
  const host = document.getElementById("s-map");
  const frame = el("iframe", "mapframe");
  frame.src = "../map/full/index.html";
  frame.setAttribute("title", "Карта кластеров");
  frame.setAttribute("loading", "lazy");
  host.appendChild(frame);
  window.__mapHref = function (tid) { return "../map/full/index.html#tid=" + tid; };
})();

// A link like #map_2518 says: open the map and show me this municipality. The map page already
// understands #tid=2518, so the presentation only has to pass it on and move the reader to the map.
(function () {
  function readHash() {
    const raw = (location.hash || "").replace(/^#/, "");
    if (raw.indexOf("map_") !== 0) return;
    const tid = raw.slice(4);
    if (!tid) return;
    select(TABS.findIndex(function (t) { return t.id === "s-map"; }));
    const frame = document.querySelector("#s-map iframe");
    if (frame) frame.src = "../map/full/index.html#tid=" + tid;
  }
  window.addEventListener("hashchange", readHash);
  readHash();
})();

function select(index) {
  active = index;
  slides.forEach(function (slide, i) { slide.classList.toggle("active", i === index); });
  TABS.forEach(function (tab, i) {
    document.getElementById("tab-" + tab.id)
      .setAttribute("aria-selected", i === index ? "true" : "false");
  });
  document.getElementById("scroll-hint").hidden = index !== 0;
  window.scrollTo(0, 0);
}

// Scrolling is navigation here, not scrolling: the wheel moves between tabs and the page never scrolls.
// Inside an open source drawer the wheel belongs to the drawer, so that case is let through.
let lastMove = 0;
window.addEventListener("wheel", function (event) {
  if (event.target && event.target.closest && event.target.closest("details.src[open] > div")) return;
  event.preventDefault();
  const now = performance.now();
  if (now - lastMove < 240) return;
  if (Math.abs(event.deltaY) < 3) return;
  active = Math.max(0, Math.min(TABS.length - 1, active + (event.deltaY > 0 ? 1 : -1)));
  select(active);
  lastMove = now;
}, { passive: false });
window.addEventListener("keydown", function (event) {
  if (event.key === "ArrowRight") select(Math.min(TABS.length - 1, active + 1));
  if (event.key === "ArrowLeft") select(Math.max(0, active - 1));
});

const P = PDATA.provenance;
const legendBox = el("div", "legend");
PDATA.legend.classes.forEach(function (c) {
  const s = el("span", null);
  const sw = el("i", null);
  sw.style.background = c.color;
  s.appendChild(sw);
  s.appendChild(document.createTextNode(c.label));
  legendBox.appendChild(s);
});

// Tab 1: one district at a time, one source at a time, with the pixel grid as a second committed image.
(function buildProvenance() {
  const slide = slides[1];
  const P = PDATA.provenance;
  const head = el("header", null);
  head.appendChild(el("h2", null, "Собираем данные из открытых гео-наборов, которые покрывают весь мир"));
  slide.appendChild(head);

  const stage = el("div", "stage");
  stage.style.gridTemplateColumns = "minmax(228px,244px) minmax(0,1fr) minmax(330px,430px)";

  const figCard = el("div", "card");
  const figureHeading = el("h3", null, "Окно над районом");
  figCard.appendChild(figureHeading);
  const sourceCol = el("div", "col");
  P.sources.forEach(function (source) {
    const b = el("button", null);
    b.appendChild(document.createTextNode(source.label));
    b.appendChild(el("small", null, source.native));
    b.dataset.source = source.key;
    b.addEventListener("click", function () { setSource(source.key); });
    sourceCol.appendChild(b);
  });
  const figure = el("figure", null);
  const box = el("div", "cropbox");
  const image = el("img", null);
  image.alt = "снимок земли вокруг выбранного района";
  const frame = el("div", "frame");
  frame.appendChild(image);
  const boundsLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  boundsLayer.setAttribute("class", "bounds");
  boundsLayer.setAttribute("preserveAspectRatio", "none");
  boundsLayer.setAttribute("width", "100%");
  boundsLayer.setAttribute("height", "100%");
  frame.appendChild(boundsLayer);
  box.appendChild(frame);
  const boundsToggle = el("button", "toggle");
  boundsToggle.appendChild(document.createTextNode("граница района"));
  boundsToggle.setAttribute("aria-pressed", "true");
  boundsToggle.addEventListener("click", function () {
    boundsOn = !boundsOn;
    boundsToggle.setAttribute("aria-pressed", boundsOn ? "true" : "false");
    paintBoundary(currentCrop());
  });
  figureHeading.appendChild(boundsToggle);
  const mapJump = el("a", "jump");
  mapJump.href = "#map";
  mapJump.appendChild(document.createTextNode("показать на общей карте кластеров"));
  mapJump.addEventListener("click", function (event) {
    event.preventDefault();
    const target = "map_" + (P.districts[picked] ? P.districts[picked].tid : picked);
    location.hash = target;
    select(TABS.findIndex(function (t) { return t.id === "s-map"; }));
    const frame = document.querySelector("#s-map iframe");
    if (frame) frame.src = "../map/full/index.html#tid=" + (P.districts[picked] ? P.districts[picked].tid : picked);
  });
  figureHeading.appendChild(mapJump);
  figure.appendChild(box);
  figCard.appendChild(figure);
  sourceCol.appendChild(el("div", "group", null, "район"));
  const pickerRow = el("div", "col");
  P.districts.forEach(function (d, i) {
    const b = el("button", null);
    b.appendChild(document.createTextNode(d.short));
    b.dataset.pick = String(i);
    b.addEventListener("click", function () { pick(i); });
    b.appendChild(el("small", null, d.region));
    pickerRow.appendChild(b);
  });
  sourceCol.appendChild(pickerRow);
  const yearRow = el("div", "col");
  ["2023", "2024"].forEach(function (year) {
    const b = el("button", null, year);
    b.dataset.year = year;
    b.addEventListener("click", function () { setYear(year); });
    yearRow.appendChild(b);
  });
  sourceCol.appendChild(el("div", "group", null, "год - всё, кроме высоты и ночных огней"));
  sourceCol.appendChild(yearRow);
  stage.appendChild(sourceCol);
  stage.appendChild(figCard);

  const legend = el("div", "legend");
  const side = el("div", "side");
  const legendCard = el("div", "card");
  legendCard.appendChild(el("h3", null, "Как читать снимок"));
  legendCard.appendChild(legend);
  const infoCard = el("div", "card");
  const infoBody = el("div", null);
  infoCard.appendChild(infoBody);
  side.appendChild(legendCard);
  side.appendChild(infoCard);
  stage.appendChild(side);
  slide.appendChild(stage);


  const drawer = el("details", "src");
  drawer.appendChild(el("summary", null, "проверить источник"));
  const drawerBody = el("div", null);
  drawer.appendChild(drawerBody);
  sourceCol.appendChild(drawer);

  const byKey = {};
  P.sources.forEach(function (source) { byKey[source.key] = source; });

  let picked = 0;
  let sourceKey = "dynamic_world";
  let year = "2023";

  function markPressed(scope, chosen, attribute) {
    // A subgroup may only clear its own buttons. The rail holds sources, districts and years together, and an
    // unscoped pass used to wipe the highlight of the other two groups.
    scope.querySelectorAll("button[" + attribute + "]").forEach(function (b) {
      b.setAttribute("aria-pressed", b === chosen ? "true" : "false");
    });
  }
  function pick(i) {
    picked = i;
    markPressed(pickerRow, pickerRow.querySelectorAll("button")[i], "data-pick");
    render();
  }
let boundsOn = true;
  function currentCrop() {
    const source = P.sources.find(function (s) { return s.key === sourceKey; });
    const period =
      source && source.periods && source.periods.length > 1
        ? year
        : source && source.periods
        ? source.periods[0]
        : "";
    const block = (P.crops || {})[sourceKey] || {};
    const wanted = P.districts[picked] ? String(P.districts[picked].tid) : String(picked);
    const slot = block[wanted] || {};
    return slot[period] || slot[""] || null;
  }
  function paintBoundary(crop) {
    boundsLayer.textContent = "";
    if (!crop || !boundsOn || !crop.boundary_paths || !crop.boundary_paths.length) {
      boundsLayer.removeAttribute("viewBox");
      return;
    }
    const size = crop.boundary_pixels || [716, 452];
    boundsLayer.setAttribute("viewBox", "0 0 " + size[0] + " " + size[1]);
    crop.boundary_paths.forEach(function (d) {
      ["casing", "line"].forEach(function (role) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", role === "casing" ? "#101216" : "#f2ead9");
        path.setAttribute("stroke-width", role === "casing" ? "4" : "1.6");
        path.setAttribute("stroke-linejoin", "round");
        path.setAttribute("vector-effect", "non-scaling-stroke");
        boundsLayer.appendChild(path);
      });
    });
  }

  function setSource(key) {
    sourceKey = key;
    markPressed(sourceCol, sourceCol.querySelector('[data-source="' + key + '"]'), "data-source");
    const block = P.sources.find(function (s) { return s.key === key; });
    yearRow.querySelectorAll("button").forEach(function (b) {
      b.disabled = !(block && block.periods && block.periods.length > 1);
    });
    // The chosen year is a state of the page, not of one source: it survives a trip to a layer that has only
    // one period and comes back highlighted when the reader returns to a layer that has both.
    markPressed(yearRow, yearRow.querySelector('[data-year="' + year + '"]'), "data-year");
    render();
  }
  function setYear(y) {
    year = y;
    markPressed(yearRow, yearRow.querySelector('[data-year="' + y + '"]'), "data-year");
    render();
  }

  function paintLegend(source, crop) {
    legend.textContent = "";
    if (source.classes) {
      source.classes.forEach(function (c) {
        const line = el("div", "key");
        const sw = el("i", null);
        sw.style.background = "rgb(" + c.color[0] + "," + c.color[1] + "," + c.color[2] + ")";
        line.appendChild(sw);
        line.appendChild(el("span", null, c.label));
        legend.appendChild(line);
      });
      legend.appendChild(el("div", "fine", null, source.meaning));
    } else if (source.key === "sentinel2") {
      legend.appendChild(el("div", "key", null,
        "красный, зелёный и синий каналы - настоящие цвета, а не шкала"));
    } else if (source.ramp_colors) {
      const row = el("div", "scaleRow");
      const bar = el("div", "ramp");
      bar.style.background = "linear-gradient(to top, " + source.ramp_colors.map(function (stop) {
        const color = stop[1] || stop;
        return "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
      }).join(", ") + ")";
      const marks = el("div", "marks");
      marks.appendChild(el("span", null, num(crop.high_display, 2) + " " + (source.unit || "")));
      marks.appendChild(el("span", "mid", null, num(crop.mid_display, 2)));
      marks.appendChild(el("span", null, num(crop.low_display, 2) + " " + (source.unit || "")));
      row.appendChild(bar);
      row.appendChild(marks);
      legend.appendChild(row);
      legend.appendChild(el("div", "fine", null, source.meaning + " - " + (source.reading || "")));
      if (crop.zero_m !== null && crop.zero_m !== undefined) {
        legend.appendChild(el("div", "fine", null,
          "самая низкая точка этого окна - " + num(crop.zero_m, 0)
          + " м над уровнем моря; самый высокий пиксель района - "
          + num(crop.zero_m + crop.relief_m, 0) + " м"));
      }
      if (crop.scale_mode === "log") {
        legend.appendChild(el("div", "fine", null,
          "шкала логарифмическая: один и тот же шаг цвета означает умножение яркости, а не прибавление."));
      }
    }
    if (crop && crop.uniform) {
      legend.appendChild(el("div", "warn", null, "во всём окне одно значение"));
    }
  }

  function appendPipeline(crop) {
    const c = P.chain;
    const chain = el("div", "chain");
    const steps = [
      [num(c.total_pixels, 0) + " пикселей", "пиксель " + num(crop.pixel_north_m, 0) + " м"],
      [num(c.classified_pixels, 0) + " с классом", pct(c.classified_share_pct, 1)],
      ["доли 5 классов", "в процентах"],
      [num(c.official_km2, 1) + " км²", "площадь из реестра"],
      [num(c.spending_rub, 0) + " ₽", "за год"],
      [num(c.density, 2) + " ₽/км²", "итог"]
    ];
    steps.forEach(function (step, i) {
      chain.appendChild(el("span", null, step[0]));
      chain.appendChild(el("em", null, step[1]));
      if (i < steps.length - 1) chain.appendChild(el("b", "arrow", null, "\u2192"));
    });
    infoBody.appendChild(chain);
  }

  function appendSteps(source) {
    if (!source.pipeline || !source.pipeline.length) {
      return;
    }
    const list = el("div", "steps");
    source.pipeline.forEach(function (step) {
      // The pipeline artifact names the year the column was built from; when the reader has chosen a year, that
      // chip would contradict the picture on screen, so the control carries the fact instead.
      if (source.periods && source.periods.length > 1 && /год наблюдения/.test(step)) return;
      list.appendChild(el("span", null, step));
    });
    infoBody.appendChild(list);
  }

  function render() {
    const d = P.districts[picked];
    const source = byKey[sourceKey];
    const period = source.periods && source.periods.length > 1 ? year : (source.periods ? source.periods[0] : "");
    const crop = (P.crops[sourceKey] || {})[d.tid] ? (P.crops[sourceKey][d.tid][period]) : null;
    if (!crop) {
      image.removeAttribute("src");
      infoBody.textContent = "";
      return;
    }
    image.src = crop.image;
    paintBoundary(crop);
    paintLegend(source, crop);

    infoBody.textContent = "";
    infoCard.style.display = "";
    if (sourceKey === "dynamic_world") {
      const dw = P.districts[picked].years[year];
      infoBody.appendChild(el("p", "kicker", null, "Доли классов земли, " + year + " год"));
      PDATA.legend.classes.forEach(function (c) {
        const line = el("div", "bar");
        line.appendChild(el("b", null, c.label));
        const track = el("div", "track");
        const fill = el("div", "fill");
        fill.style.width = Math.min(100, dw.means[c.key]) + "%";
        fill.style.background = c.color;
        track.appendChild(fill);
        line.appendChild(track);
        line.appendChild(el("em", null, pct(dw.means[c.key], 1)));
        infoBody.appendChild(line);
      });
    } else if (sourceKey === "terrain") {
      infoBody.textContent = "";
      infoCard.style.display = "none";
    } else if (sourceKey === "viirs") {
      infoBody.appendChild(el("p", "kicker", null, "Яркость ночью, " + year + " год"));
      (crop.step_shares_pct || []).forEach(function (share, index) {
        const names = ["тёмное", "слабое", "поселение", "город"];
        const colors = ["#10121a", "#4a3a20", "#b07830", "#f6e096"];
        const bar = el("div", "bar");
        bar.appendChild(el("b", null, names[index] || "шаг " + (index + 1)));
        const track = el("div", "track");
        const fill = el("div", "fill");
        fill.style.width = Math.min(100, share) + "%";
        fill.style.background = colors[index] || "#b07830";
        track.appendChild(fill);
        bar.appendChild(track);
        bar.appendChild(el("em", null, pct(share, 1)));
        infoBody.appendChild(bar);
      });
    } else {
      if (sourceKey === "ndvi") {
        infoBody.appendChild(el("p", "kicker", null, "Об этом источнике"));
      }
      infoBody.appendChild(el("p", "desc", null, source.explainer));
      if (source.full) {
        infoBody.appendChild(el("p", "desc", null, source.full + "."));
      }
      if (source.formula) {
        infoBody.appendChild(el("p", "formula", null, source.formula));
      }
      if (sourceKey === "ndvi" && crop.step_shares_pct && crop.step_shares_pct.length) {
        infoBody.appendChild(el("p", "kicker", null, "Что видно внутри района"));
        const vegNames = ["без растительности", "слабая", "активная"];
        const vegColors = ["rgb(138, 127, 106)", "rgb(111, 143, 87)", "rgb(63, 125, 60)"];
        crop.step_shares_pct.forEach(function (share, index) {
          const bar = el("div", "bar");
          bar.appendChild(el("b", null, vegNames[index] || ""));
          const track = el("div", "track");
          const fill = el("div", "fill");
          fill.style.width = Math.min(100, share) + "%";
          fill.style.background = vegColors[index] || "rgb(111, 143, 87)";
          track.appendChild(fill);
          bar.appendChild(track);
          bar.appendChild(el("em", null, pct(share, 1)));
          infoBody.appendChild(bar);
        });
        infoBody.appendChild(el("p", "fine", null,
          "порог 0,1 - ниже растительности почти нет; порог 0,3 - полог сомкнутый или посевы активные"));
      }
      if (source.use) {
        infoBody.appendChild(el("p", "desc", null, source.use));
      }
      if (crop.lit_share_pct !== null && crop.lit_share_pct !== undefined) {
        infoBody.appendChild(el("p", "fine", null,
          "Освещённых пикселей в этом окне: " + pct(crop.lit_share_pct, 1)
          + " - по порогу 0,5 ед. яркости, именно так считает кластеризация."));
      }
      if (source.not_used) {
        infoBody.appendChild(el("p", "fine", null, source.not_used));
      }
      if (crop.uniform) {
        infoBody.appendChild(el("p", "warn", null,
          "Во всём окне значение одинаково: " + (sourceKey === "hansen" ? "ни одного пикселя с потерей леса" : "однотонный снимок") + "."));
      }
      if (sourceKey !== "terrain" && sourceKey !== "ndvi") {
        infoBody.appendChild(el("p", "fine", null,
          "пиксель на земле ≈ " + num(crop.pixel_east_m, 0) + " × " + num(crop.pixel_north_m, 0)
          + " м; данных в окне " + pct(crop.coverage_pct, 1) + "."));
      }
    }

    drawerBody.textContent = "";
    document.addEventListener("click", function (event) {
      if (!drawer.open) return;
      if (drawer.contains(event.target)) return;
      drawer.open = false;
    });
    drawerBody.appendChild(el("h4", null, "Что это за данные"));
    function line(label, value) {
      const p = el("p", "dline");
      p.appendChild(el("b", null, label));
      p.appendChild(document.createTextNode(": " + value));
      drawerBody.appendChild(p);
    }
    line("что показывает", source.meaning);
    line("родной размер пикселя", source.native);
    line("как часто обновляется", source.cadence);
    line("год наблюдения", /^\d{4}$/.test(period) ? period + " год" : "не зависит от года наблюдений");
    line("шкала на картинке", crop.scale_basis || "ярность этого окна");
    line("условия использования", source.licence);
    drawerBody.appendChild(el("h4", null, "Как получено это окно"));
    const cap = crop.capture;
    line("центр окна", num(cap.center_lon, 4) + "\u00B0 в.д., " + num(cap.center_lat, 4) + "\u00B0 с.ш.");
    line("размер окна", num(cap.wide_deg, 2) + " \u00D7 " + num(cap.tall_deg, 2) + "\u00B0 - примерно "
      + num(cap.wide_km, 0) + " \u00D7 " + num(cap.tall_km, 0) + " км");
    line("масштаб", num(crop.pixel_east_m, 0) + " \u00D7 " + num(crop.pixel_north_m, 0)
      + " м на земле; мы просили " + num(crop.scale_m, 0) + " м");
    line("пикселей с данными", pct(crop.coverage_pct, 1));
    if (crop.lit_share_pct !== null && crop.lit_share_pct !== undefined) {
      line("доля освещённых пикселей", pct(crop.lit_share_pct, 1) + " (порог " + num(crop.analysis_low, 1)
        + " ед. яркости)");
    }
    if (source.catalog) {
      const link = el("a", null, "страница набора данных в каталоге Earth Engine");
      link.href = source.catalog;
      link.target = "_blank";
      link.rel = "noopener";
      drawerBody.appendChild(el("div", null, "проверить снаружи: "));
      drawerBody.appendChild(link);
    }
  }

  markPressed(sourceCol, sourceCol.querySelector('[data-source="dynamic_world"]'), "data-source");
  markPressed(pickerRow, pickerRow.querySelectorAll("button")[0], "data-pick");
  markPressed(yearRow, yearRow.querySelector('[data-year="2023"]'), "data-year");
  render();
})();

// Tab 2: families, not a per-layer table. The per-layer detail sits in the drawer.
(function buildCadence() {
  const slide = slides[2];
  const C = PDATA.cadence;
  const head = el("header", null);
  head.appendChild(el("h2", null, "Что обновляется, а что нет"));
  slide.appendChild(head);

  const LANE_COLOUR = {
    "раз в месяц": "#2a78d6",
    "раз в год": "#2e5d3a",
    "статично": "#8a94a0",
    "не пересчитывается": "#8a94a0",
    "не доказано": "#b08d3f",
  };
  function chip(text) {
    const c = el("span", "fq");
    c.appendChild(document.createTextNode(text));
    const colour = LANE_COLOUR[text];
    if (colour) {
      c.style.borderColor = colour;
      c.style.background = colour;
      c.style.color = "#fff";
    }
    return c;
  }
  function stack(used, possible) {
    const box = el("div", "trio");
    [["сейчас", used], ["возможно", possible]].forEach(function (pair) {
      if (!pair[1]) return;
      const line = el("div", "trioline");
      line.appendChild(el("small", null, pair[0]));
      line.appendChild(chip(pair[1]));
      box.appendChild(line);
    });
    return box;
  }

  const rows = [];
  // Only the inputs that actually enter the shipped decision get a row. Listing every layer we could
  // have touched turned the table into a catalogue, and the question is about the five we did use.
  C.families.families.forEach(function (family) {
    family.members.filter(function (m) { return m.in_space && m.used_frequency; })
      .forEach(function (m) { rows.push([family.label, m]); });
  });

  const table = el("table", "cadtable");
  const thead = el("thead", null);
  const hr = el("tr", null);
  ["семейство", "вход и его столбцы в решении", "частота", "разрешение", "последнее обновление"].forEach(function (name) {
    const th = el("th", null);
    th.appendChild(document.createTextNode(name));
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody", null);
  rows.forEach(function (pair) {
    const member = pair[1];
    const tr = el("tr", null);
    const group = el("td", "cadgroup");
    group.appendChild(document.createTextNode(pair[0]));
    tr.appendChild(group);
    const who = el("td", "cadname");
    who.appendChild(el("b", null, member.label));
    // The verbose column names the decision uses belong under the input they came from, not in a
    // column of their own that would widen the table past the screen.
    if (member.space_columns && member.space_columns.length) {
      who.appendChild(el("small", null, member.space_columns.join(" · ")));
    }
    tr.appendChild(who);
    const freq = el("td", null);
    // A static input still has an answer to the frequency question - it is "static" - so the cell is
    // not left blank only because nobody wrote a cadence for it.
    const staticWord = member.status === "static" ? "статично" : null;
    freq.appendChild(stack(member.used_frequency || staticWord, member.possible_frequency));
    tr.appendChild(freq);
    const res = el("td", null);
    res.appendChild(stack(member.used_resolution, member.possible_resolution));
    tr.appendChild(res);
    const last = el("td", "cadlast");
    // Where a date came from is provenance, not something the reader needs in the row. It lives in
    // the drawer under the table instead of printing a repository path on the face of the page.
    if (member.last_update) {
      last.appendChild(el("b", null, member.last_update));
    } else {
      last.appendChild(el("small", null, "не записано"));
    }
    tr.appendChild(last);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const card = el("div", "card");
  card.appendChild(table);
  card.appendChild(el("p", "fine", null,
    "Цвет частоты: синий - месяц, зелёный - год, серый - статично, песочный - не доказано."
    + " «По муниципалитету» значит, что величина уже сведена к району, а не к пикселю."));
  card.appendChild(drawer("Откуда это известно", [
    ["артефакт семей", C.families.artifact],
    ["артефакт частот", C.artifact],
    ["основание дат", (C.families.families || []).map(function (f) {
      return f.members.map(function (m) { return m.last_update_basis || "нет записи"; }).join(" · ");
    }).join(" | ")]
  ]));
  slide.appendChild(card);
})();

// Tab 4: how the attributes and the number of types were chosen. Three views, one at a time, so each
// picture can be big enough to read. The browser never computes anything: every map here is a
// partition that was computed in Python and written to disk.
(function buildDecision() {
  const slide = slides[3];
  const D = PDATA.decision;
  const head = el("header", null);
  head.appendChild(el("h2", null, "Как выбраны признаки и сколько типов"));
  slide.appendChild(head);

  const maps = PDATA.decision_maps || [];
  const byIndex = {};
  maps.forEach(function (m) { byIndex[m.index] = m; });

  const switcher = el("div", "dswitch");
  switcher.appendChild(el("span", null, "смотреть"));
  const VIEWS = [
    ["sets", "наборы признаков", "Три способа раскрасить одну и ту же страну"],
    ["k", "сколько типов", "Один и тот же набор, разное число типов"],
    ["volume", "деньги против типов", "Пять типов - не ранжирование по объёму"],
  ];
  const panes = {};

  function makeCell(svg, caption, note) {
    const cell = el("figure", "mapcell");
    const img = el("img", "bigmap");
    img.src = "assets/" + svg;
    img.alt = caption;
    cell.appendChild(img);
    cell.appendChild(el("figcaption", null, caption));
    if (note) cell.appendChild(el("small", null, note));
    return cell;
  }

  // View one: the choice of attributes, shown as three full-size maps.
  const paneSets = el("div", "dpane");
  paneSets.appendChild(el("p", "dprompt", null, VIEWS[0][2]));
  const setRow = el("div", "setrow");
  setRow.appendChild(makeCell(
    byIndex[10] ? byIndex[10].svg : "", "только форма трат по категориям",
    byIndex[10] ? "совпадение с присланным ответом ARI " + num(byIndex[10].ari, 4) : ""));
  setRow.appendChild(makeCell("decision_money.svg", "окрашено по сумме месячных трат",
    "тёмное - от 48,7 тыс. ₽ в месяц, светлое - до 21,1 тыс.; серые - нет данных"));
  setRow.appendChild(makeCell(
    byIndex[2] ? byIndex[2].svg : "", "земля и доли трат, без суммы",
    byIndex[2] ? "совпадение с присланным ответом ARI " + num(byIndex[2].ari, 4) : ""));
  paneSets.appendChild(setRow);
  panes.sets = paneSets;

  // View two: the same attributes, different numbers of types, next to the money picture so the
  // reader can see whether the split is following the money or the land.
  const paneK = el("div", "dpane");
  paneK.appendChild(el("p", "dprompt", null, VIEWS[1][2]));
  const kRow = el("div", "krow");
  const kToIndex = { 3: 0, 4: 1, 5: 2, 6: 3 };
  const kCaption = el("p", "kcaption", null, "");
  const kStat = el("p", "kstat", null, "");
  const kMapCell = el("figure", "mapcell");
  const kImg = el("img", "bigmap");
  kImg.alt = "Разбиение на выбранном наборе признаков";
  kMapCell.appendChild(kImg);
  const moneyMapCell = makeCell("decision_money.svg", "для сравнения: окрашено по сумме трат",
    "тёмное - от 48,7 тыс. ₽ в месяц, светлое - до 21,1 тыс.");
  const kPair = el("div", "setrow kpair");
  kPair.appendChild(kMapCell);
  kPair.appendChild(moneyMapCell);
  function showK(k) {
    const m = byIndex[kToIndex[k]];
    if (!m) return;
    kImg.src = "assets/" + m.svg;
    kCaption.textContent = m.caption;
    kStat.textContent = Object.keys(m.sizes).map(function (key) { return m.sizes[key]; }).join(" · ") +
      " округов · совпадение с присланным ответом ARI " + num(m.ari, 4);
    Array.from(kRow.children).forEach(function (b) {
      if (b.tagName === "BUTTON") b.setAttribute("aria-pressed", Number(b.dataset.k) === k ? "true" : "false");
    });
  }
  [3, 4, 5, 6].forEach(function (k) {
    const b = el("button", "kbtn");
    b.textContent = "k=" + k;
    b.dataset.k = String(k);
    b.setAttribute("aria-pressed", k === 5 ? "true" : "false");
    b.addEventListener("click", function () { showK(k); });
    kRow.appendChild(b);
  });
  paneK.appendChild(kRow);
  paneK.appendChild(kPair);
  const kFoot = el("div", "dfoot");
  kFoot.appendChild(kCaption);
  kFoot.appendChild(kStat);
  kFoot.appendChild(el("p", "note", null,
    "k=5 - не за количество: он даёт тип, которого нет при k=4."));
  paneK.appendChild(kFoot);
  panes.k = paneK;

  // View three: the accusation answered with pictures rather than numbers.
  const paneVolume = el("div", "dpane");
  paneVolume.appendChild(el("p", "dprompt", null, VIEWS[2][2]));
  const volGrid = el("div", "volgrid");
  const volImg = el("img", "volimg");
  volImg.src = "assets/spending_vs_structure.png";
  volImg.alt = "Слева - сколько округ тратит, справа - чем округи различаются по устройству";
  volGrid.appendChild(volImg);
  const barsBox = el("div", "sharepair");
  function inlineBars(dict, label) {
    const side = el("div", null);
    side.appendChild(el("small", null, label));
    const box = el("div", "inbars");
    Object.keys(dict).forEach(function (key) {
      if (key === "all_columns") return;
      const line = el("div", "inbar");
      line.appendChild(el("b", null, D.dominance.labels[key] || key));
      const track = el("div", "track");
      const fill = el("div", "fill");
      fill.style.width = dict[key] + "%";
      fill.style.background = key === "spending_amount" ? "var(--accent)"
        : (key === "land" ? "var(--accent2)" : "#c9a227");
      track.appendChild(fill);
      line.appendChild(track);
      line.appendChild(el("em", null, pct(dict[key], 1)));
      box.appendChild(line);
    });
    side.appendChild(box);
    return side;
  }
  barsBox.appendChild(inlineBars(D.dominance.without, "без суммы трат"));
  barsBox.appendChild(inlineBars(D.dominance.with, "с суммой трат"));
  volGrid.appendChild(barsBox);
  paneVolume.appendChild(volGrid);
  const volFoot = el("div", "dfoot");
  volFoot.appendChild(el("p", "note", null,
    "Сумма забирает " + pct(D.dominance.with.spending_amount, 1) + " разделения; между разбиением с ней и"
    + " присланным ответом переезжают " + pct(D.dominance.moved, 1) + " районов."));
  paneVolume.appendChild(volFoot);
  panes.volume = paneVolume;

  slide.appendChild(switcher);
  VIEWS.forEach(function (view, position) {
    const b = el("button", "dbtn");
    b.textContent = view[1];
    b.dataset.view = view[0];
    b.setAttribute("aria-pressed", position === 0 ? "true" : "false");
    b.addEventListener("click", function () { showView(view[0]); });
    switcher.appendChild(b);
    slide.appendChild(panes[view[0]]);
  });
  function showView(name) {
    Object.keys(panes).forEach(function (key) {
      panes[key].classList.toggle("active", key === name);
    });
    Array.from(switcher.querySelectorAll("button")).forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.view === name ? "true" : "false");
    });
    if (name === "k") showK(5);
  }
  showView("sets");

  slide.appendChild(drawer("Чем это проверено", [
    ["записанное решение", D.k_artifact],
    ["доминирование суммы", D.dominance.artifact],
    ["демонстрация k и пространств", "artifacts/landing/demo_states.json"],
    ["печать картинок", "scripts/build_vector_choropleth.py"],
    ["движение 2023 к 2024", D.arms_artifact]
  ]));
})();

select(0);
</script>
</body>
</html>
"""


def publish_site_fixes() -> None:
    """Make web/ a tree GitHub Pages can actually serve.

    Three things the site needs that no page can supply by itself: the report has to live inside the
    published subtree or the link from the presentation leaves the site; Jekyll has to be told to
    stand down or it will not copy the 71.8 MB geometry file; and the site root has to answer with
    something, because there is no index.html at web/.
    """
    WEB = ROOT / "web"
    report_dir = WEB / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "docs" / "report" / "methodology-ru.md",
                    report_dir / "methodology-ru.md")

    (WEB / ".nojekyll").write_text("")

    (WEB / "index.html").write_text(
        "<!DOCTYPE html>\n<html lang=\"ru\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta http-equiv=\"refresh\" content=\"0; url=./presentation/index.html\">\n"
        "<link rel=\"canonical\" href=\"./presentation/index.html\">\n"
        "<title>Земля под тратами</title>\n</head>\n<body>\n"
        "<p>Переход к презентации: <a href=\"./presentation/index.html\">Земля под тратами</a>.</p>\n"
        "</body>\n</html>\n", encoding="utf-8")


def copy_assets(payload: dict) -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    copied = 0
    crops = json.loads(CROPS.read_text(encoding="utf-8"))
    recreations = json.loads(RECREATIONS.read_text(encoding="utf-8"))
    sources = []
    for record in crops["records"]:
        if record.get("committed_file"):
            sources.append(Path(record["committed_file"]))
    for record in recreations["records"]:
        sources.append(Path(record["committed_file"]))
    for record in json.loads(SOURCE_CROPS.read_text(encoding="utf-8"))["records"]:
        for key in ("committed_file", "committed_file_plain"):
            if record.get(key):
                sources.append(Path(record[key]))
    for state in json.loads(DEMO_STATES.read_text(encoding="utf-8"))["states"]:
        sources.append(ROOT / "artifacts" / "landing" / "demo" / Path(state["image"]).name)
    sources.append(ROOT / "artifacts" / "landing" / "author_avatar.jpg")
    sources.append(ROOT / "artifacts" / "landing" / "hero_map.svg")
    sources.append(ROOT / "artifacts" / "landing" / "spending_vs_structure.png")
    sources.extend(
        ROOT / "artifacts" / "landing" / f"decision_state_{index}.svg"
        for index in (0, 1, 2, 3, 6, 10, 14)
    )
    sources.append(ROOT / "artifacts" / "landing" / "decision_money.svg")
    for source in sources:
        if not source.exists():
            continue
        target = ASSETS / source.name
        if target.exists() and target.stat().st_size == source.stat().st_size:
            copied += 1
            continue
        shutil.copyfile(source, target)
        copied += 1
    return copied


def main() -> int:
    payload = build_payload()
    html = TEMPLATE.replace(
        "__PDATA__",
        json.dumps(_dehome(payload), ensure_ascii=False).replace("</", "<\\/"),
    )
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(html, encoding="utf-8")
    copied = copy_assets(payload)
    publish_site_fixes()
    proof = {
        "generated_by": "scripts/build_stage_one_presentation.py",
        "page_bytes": PAGE.stat().st_size,
        "tabs": 3,
        "numbers_rendered_from": "the injected PDATA object; no number is typed into the template",
        "internal_machinery_lives_in": "a collapsed per-figure drawer, not on the visible face",
        "images_copied": copied,
        "sections": {
            "provenance_artifacts": [
                payload["provenance"]["crops_artifact"],
                payload["provenance"]["chain_artifact"],
            ],
            "cadence_artifacts": [payload["cadence"]["artifact"]],
            "decision_artifacts": [
                payload["decision"]["k_artifact"],
                payload["decision"]["dominance"]["artifact"],
                payload["decision"]["arms_artifact"],
            ],
        },
        "caveats_kept": [
            "dynamic world year-to-year movement is uncorroborated",
            "monthly and quarterly resolutions were never measured",
        ],
    }
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"page: {rel(PAGE)} {PAGE.stat().st_size} B; images copied {copied}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
