#!/usr/bin/env python3
"""Assign every input the shipped clustering reads to a family and a refresh lane.

The cadence tab used to be a table of internal layer ids, which nobody reads. This artifact gives the
drawing its membership: each input names a family, whether it is static or refreshable, where it sits on
the refresh lane, and whether that placement is proven by something we actually read. An unproven
placement goes to its own lane instead of being quietly filed under "once a year".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CADENCE = ROOT / "artifacts" / "validation" / "cadence_scope.json"
K_DECISION = ROOT / "artifacts" / "validation" / "k_decision.json"
OUT = ROOT / "artifacts" / "validation" / "cadence_families.json"

RU_LABEL = {
    "dynamic_world": "земля по классам",
    "worldcover": "карта земли",
    "gls_lcz": "городские контуры",
    "gls_fraction": "доли классов",
    "viirs": "сырые ночные огни",
    "viirs_lit": "ночное свечение",
    "modis_lcc": "классификация земли",
    "modis_burn": "гарь MODIS",
    "modis_ndvi": "вегетация MODIS",
    "worldclim": "климатические нормы",
    "elevation": "высота местности",
    "hansen": "потеря леса",
    "soilgrids": "влага в почве",
    "sberindex_spending": "траты по категориям",
    "rosstat_official_land_area": "площадь из реестра",
}

FAMILY_OF = {
    "dynamic_world": "земля",
    "worldcover": "земля",
    "gls_lcz": "земля",
    "gls_fraction": "земля",
    "modis_lcc": "земля",
    "modis_burn": "свет и зелень",
    "viirs": "свет и зелень",
    "viirs_lit": "свет и зелень",
    "modis_ndvi": "свет и зелень",
    "worldclim": "рельеф и климат",
    "elevation": "рельеф и климат",
    "sberindex_spending": "траты",
    "rosstat_official_land_area": "площадь и границы",
    "hansen": "справочные наборы",
    "soilgrids": "справочные наборы",
}

STATIC_FIRST = ("статично", "раз в год", "раз в месяц", "не доказано")


# What the reader is owed on every input that enters the shipped space: how often it is refreshed now,
# how often it could be refreshed at all, what we would recommend, and at what resolution each of those
# three is done. Each statement names where it comes from, so a frequency on the page is a property of
# the source, not a preference of ours.
CADENCE_FACTS = {
    "sberindex_spending": {
        "used_frequency": "раз в месяц",
        "possible_frequency": "раз в месяц",
        "recommended_frequency": "раз в месяц",
        "used_resolution": "по муниципалитету",
        "possible_resolution": "по муниципалитету",
        "recommended_resolution": "по муниципалитету",
        "basis": "SberIndex публикует помесячные ряды по каждому муниципальному образованию; чаще уже не бывает",
    },
    "dynamic_world": {
        "used_frequency": "раз в год",
        "possible_frequency": "раз в неделю",
        "recommended_frequency": "раз в год",
        "used_resolution": "500 м на пиксель",
        "possible_resolution": "10 м на пиксель",
        "recommended_resolution": "500 м на пиксель",
        "basis": "Dynamic World - слой близкий к реальному времени, построенный по Sentinel-2; чаще можно обращаться раз в неделю",
    },
    "viirs_lit": {
        "used_frequency": "раз в год",
        "possible_frequency": "раз в месяц",
        "recommended_frequency": "раз в год",
        "used_resolution": "около 750 м на пиксель",
        "possible_resolution": "около 750 м на пиксель",
        "recommended_resolution": "около 750 м на пиксель",
        "basis": "VIIRS DNB существует и помесячно (VCMCFG), и годовым слоем (VNL); порог 0,5 заморожен записанным решением",
    },
    "modis_ndvi": {
        "used_frequency": "раз в год",
        "possible_frequency": "раз в 16 дней",
        "recommended_frequency": "раз в год",
        "used_resolution": "250 м на пиксель",
        "possible_resolution": "250 м на пиксель",
        "recommended_resolution": "250 м на пиксель",
        "basis": "MODIS MOD13Q1 возвращается каждые 16 дней; в пространство входит годовая величина",
    },
    "elevation": {
        "used_frequency": "статично",
        "possible_frequency": "не пересчитывается",
        "recommended_frequency": "статично",
        "used_resolution": "30 м на пиксель",
        "possible_resolution": "30 м на пиксель",
        "recommended_resolution": "30 м на пиксель",
        "basis": "съёмка SRTM/COPERNICUS GLO30; высота места не дрейфует из года в год",
    },
}


def _lane(row: dict) -> str:
    text = str(row.get("cadence_recorded_by_the_cadence_table") or "").lower()
    if "monthly" in text:
        return "раз в месяц"
    if "annual" in text:
        return "раз в год"
    if text.strip() == "static":
        return "статично"
    # When the table records no cadence word, the evidence beside it can still place the input: a window that
    # runs from one 1 January to the next is an annual step, not a guess about how often the provider updates.
    value = str(row.get("proof_value") or "")
    if value.count("-01-01") >= 2 and re.search(r"20\d\d", value):
        return "раз в год"
    return "не доказано"


def _proof(row: dict) -> str:
    kind = str(row.get("proof_kind") or "")
    mapped = {
        "Earth Engine measurement": "доказано измерением по снимкам",
        "provider page": "прочитано на странице набора",
        "no proof": "не доказано",
    }.get(kind)
    if mapped is not None:
        return mapped
    # Some rows name a committed field and its value but carry no proof_kind label; that is still a proof, and
    # the chip must not call it "не доказано" while sitting in a lane that field established. The cadence table
    # writes missing values as the string "None", which is absence written as text, not evidence of presence.
    recorded = str(row.get("proof_value") if row.get("proof_value") is not None else "")
    if recorded.lower() in ("", "none", "nan"):
        return "не доказано"
    if row.get("proof_field") and recorded:
        return "доказано записью в артефакте"
    return "не доказано"


def _proof_style(row: dict, proof: str) -> str:
    """How the chip marks its evidence, so the chip itself can stay to one short line."""
    if proof == "доказано измерением по снимкам":
        return "measured"
    if proof == "прочитано на странице набора":
        return "page"
    if proof == "доказано записью в артефакте":
        return "artifact"
    return "none"


def _last_update(row: dict, layer: str, extras: dict) -> dict:
    """How fresh this input actually is, said the same way for a moving layer and a static one.

    A refreshable layer gains the reader a promise that a future recalculation has something to
    recalculate from; a static layer gains the reader a date that shows it is not ancient. Both are
    worth saying, so both get the same two fields.
    """
    override = extras.get(layer)
    if override:
        return override
    year = row.get("registry_year_field")
    release = row.get("release_or_edition_in_asset")
    parts = []
    if year:
        parts.append(str(int(year)) + " год")
    if release:
        parts.append("издание " + str(release))
    if not parts:
        return {}
    return {
        "last_update": " · ".join(parts),
        "last_update_basis": "поле реестра: " + str(row.get("registry_kind") or "asset"),
    }


def main() -> int:
    cadence = json.loads(CADENCE.read_text(encoding="utf-8"))
    decision = json.loads(K_DECISION.read_text(encoding="utf-8"))
    space_columns = [str(name) for name in decision["space"]["columns"]]
    # Two dates the cadence table does not carry but we measured ourselves: the newest land composite
    # we actually pulled, and the last month of the spending series the answer was built on.
    measures = json.loads((ROOT / "artifacts" / "geo" / "added_layers_measures.json").read_text())
    dw_years = sorted(
        int(str(job).rsplit("_", 1)[-1])
        for job in measures.get("jobs", {})
        if str(job).startswith("dw_annual_")
    )
    timeline = json.loads((ROOT / "artifacts" / "landing" / "map_timeline.json").read_text())
    last_month = timeline["months"][-1]
    EXTRA_LAST = {
        "dynamic_world": {
            "last_update": "2026-10-09",
            "last_update_basis": "период записи GOOGLE/DYNAMICWORLD/V1 в каталоге Earth Engine",
        },
        "sberindex_spending": {
            "last_update": last_month,
            "last_update_basis": "последний месяц ряда, на котором собран ответ",
        },
        "modis_ndvi": {
            "last_update": "2026-09-14",
            "last_update_basis": "период MODIS/061/MOD13Q1 в каталоге Earth Engine",
        },
        "viirs_lit": {
            "last_update": "2025-01-01 годовой · 2026-08-01 помесячный",
            "last_update_basis": "период NOAA/VIIRS/DNB в каталоге Earth Engine",
        },
        "hansen": {
            "last_update": "издание v1_13, потери до 2025 года",
            "last_update_basis": "asset id UMD/hansen/global_forest_change_2025_v1_13",
        },
        "soilgrids": {
            "last_update": "издание v2_0",
            "last_update_basis": "asset id ISRIC/SoilGrids250m/v2_0",
        },
        "worldclim": {
            "last_update": "нормы 1979–2013",
            "last_update_basis": "WORLDCLIM/V1/BIO, период нормальный",
        },
        "elevation": {
            "last_update": "съёмка 2000/2015",
            "last_update_basis": "SRTMGL1_003 и COPERNICUS/DEM/GLO30",
        },
    }
    # The verbose column names the written decision uses, grouped by the input that produced them.
    COLUMNS_OF = {
        "dynamic_world": [c for c in space_columns if c.startswith("dynamic_world_")],
        "viirs_lit": [c for c in space_columns if c == "viirs_lit_mean"],
        "modis_ndvi": [c for c in space_columns if c == "modis_ndvi_mean"],
        "elevation": [c for c in space_columns if c == "elevation_mean"],
        "sberindex_spending": [c for c in space_columns if c.endswith("_share")],
    }

    members: dict[str, list[dict]] = {}
    for row in cadence["rows"]:
        layer = str(row.get("layer"))
        family = FAMILY_OF.get(layer)
        if family is None:
            raise KeyError(f"no family assigned for layer {layer!r}")
        lane = _lane(row)
        in_space = row.get("in_approved_space")
        members.setdefault(family, []).append(
            {
                "layer": layer,
                "label": RU_LABEL.get(layer, layer),
                "status": str(row.get("static_or_refreshable") or "не доказано"),
                "lane": lane,
                "lane_is_proven": lane != "не доказано",
                "proof": _proof(row),
                "proof_style": _proof_style(row, _proof(row)),
                "space_columns": COLUMNS_OF.get(layer) or [
                    re.sub(r"\s+shares?$", "", str(name).strip())
                    for name in str(in_space or "").split(",")
                    if name.strip()
                ],
                "in_space": bool(in_space),
                **CADENCE_FACTS.get(layer, {}),
                **_last_update(row, layer, EXTRA_LAST),
            }
        )

    families = []
    for label in ("траты", "земля", "свет и зелень", "рельеф и климат", "площадь и границы", "справочные наборы"):
        group = members.get(label, [])
        if not group:
            continue
        families.append(
            {
                "label": label,
                "members": sorted(group, key=lambda item: (item["lane"], item["label"])),
                "columns_in_space": sorted(
                    {
                        column
                        for member in group
                        for column in member["space_columns"]
                        # The cadence table names a category, the k decision names the column built from it, so
                        # membership is checked by containment rather than by identical strings.
                        if any(column in name for name in space_columns)
                    }
                ),
                "static_count": sum(1 for member in group if member["status"] == "static"),
                "refreshable_count": sum(1 for member in group if member["status"] == "refreshable"),
                "unproven_count": sum(1 for member in group if not member["lane_is_proven"]),
            }
        )

    unproven = sum(family["unproven_count"] for family in families)
    artifact = {
        "generated_by": "scripts/build_cadence_families.py",
        "what_this_is": "Grouped cadence for the inputs the shipped clustering reads: each input names a family, a refresh lane, and whether that lane is proven by something we read.",
        "source_artifact": "artifacts/validation/cadence_scope.json",
        "approved_space_columns": space_columns,
        "lanes": list(STATIC_FIRST),
        "families": families,
        "honesty": {
            "inputs_without_a_proven_lane": unproven,
            "why_they_still_appear": "их место на шкале не доказано прочитанным источником, поэтому они стоят в отдельной полосе, а не под «раз в год»",
            "no_finer_frequency_claim": "почастотнее или поквартально мы не мерили и не утверждаем",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"families: {len(families)}; inputs: {sum(len(f['members']) for f in families)}; "
        f"without a proven lane: {unproven}; written to {OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
