"""Rosstat БДПМО official municipal land area (indicator Y48006001, hectares).

Why this matters: the SberIndex polygon layer is OSM-derived and includes marine/extremity areas,
so `osm_area_km2` must never be used as a density denominator. This module supplies the official
land area of each municipality in km^2.

Key: BDPMO `oktmo_stable` (8-digit) matched against the Sber dictionary `oktmo` normalized to the
same 8-digit form ("79-701-000-000" -> "79701000").
"""

from __future__ import annotations

import pandas as pd

from .config import RAW

BDPMO_FILE = RAW / "rostat" / "data_Y48006001_112_v20250918.csv"
TOP_LEVEL = "Муниципальное образование верхнего уровня"


def normalize_oktmo(value: pd.Series) -> pd.Series:
    """'79-701-000-000' -> '79701000' (the 8-digit municipal OKTMO)."""
    digits = value.astype("string").str.replace("-", "", regex=False)
    return digits.str.slice(0, 8)


def load_land_area() -> pd.DataFrame:
    if not BDPMO_FILE.exists():
        raise FileNotFoundError(f"BDPMO indicator file not found: {BDPMO_FILE}")
    d = pd.read_csv(BDPMO_FILE, sep=";", encoding="utf-8", low_memory=False)
    d = d[d.mun_level == TOP_LEVEL].copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["indicator_value"] = pd.to_numeric(d["indicator_value"], errors="coerce")
    d = d.dropna(subset=["year", "indicator_value"])
    d["oktmo_stable"] = d["oktmo_stable"].astype("string").str.strip().str.zfill(8)
    # prefer the newest year per municipality, fall back to older ones
    d = d.sort_values("year").drop_duplicates("oktmo_stable", keep="last")
    out = d[["oktmo_stable", "year", "indicator_value"]].rename(
        columns={"year": "area_year", "indicator_value": "area_ha"}
    )
    out["official_land_area_km2"] = out.area_ha / 100.0
    return out.reset_index(drop=True)


def attach_land_area(master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adds official municipal land area + inflation ratio to the master frame."""
    try:
        ra = load_land_area()
    except FileNotFoundError as exc:
        master["official_land_area_km2"] = pd.NA
        master["area_year"] = pd.NA
        master["oktmo_norm"] = normalize_oktmo(master["oktmo"])
        return master, {
            "rostat_status": "blocked",
            "error": str(exc),
            "fallback": "subject official area (curated) + classified-pixel denominators only",
        }

    master["oktmo_norm"] = normalize_oktmo(master["oktmo"])
    before = len(master)
    master = master.merge(ra, left_on="oktmo_norm", right_on="oktmo_stable", how="left")
    assert len(master) == before, "land-area merge changed row count"
    matched = int(master.official_land_area_km2.notna().sum())
    info = {
        "rostat_status": "obtained",
        "rows_in_indicator": len(
            pd.read_csv(
                BDPMO_FILE, sep=";", encoding="utf-8", usecols=["oktmo_stable"], low_memory=False
            )
        ),
        "top_level_municipalities": int(ra.oktmo_stable.nunique()),
        "matched_territories": matched,
        "match_rate": round(matched / len(master), 4),
        "area_year_used": ra.area_year.value_counts().to_dict(),
        "unmatched_territories": int(len(master) - matched),
        "fallback_for_unmatched": "classified-pixel denominators for shares; subject official area as the density proxy",
    }
    master["land_inflation_ratio"] = master.osm_area_km2 / master.official_land_area_km2
    master["area_suspect"] = master.land_inflation_ratio.gt(1.5) | master.osm_area_km2.gt(50_000)
    return master, info
