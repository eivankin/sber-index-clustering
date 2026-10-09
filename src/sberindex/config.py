"""Shared paths, constants and small helpers for the SberIndex 2026 pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

HACK = RAW / "sber_hackathon"
DICT = RAW / "sber_dict"

CONSUMPTION = HACK / "consumption.parquet"
CONNECTION = HACK / "connection.parquet"
MARKET_ACCESS = HACK / "market_access.parquet"
ATTRIBUTES = DICT / "t_dict_municipal_districts.xlsx"
POLYGONS = DICT / "t_dict_municipal_districts_poly.gpkg"

# Equal-area cylindrical projection: areas are preserved at any latitude.
AREA_CRS = "EPSG:4087"
GEOG_CRS = "EPSG:4326"

SEA_BOXES = {
    "kara_laptev": (60.0, 100.0, 66.0, 82.0),
    "black_caspian": (20.0, 60.0, 40.0, 48.0),
    "north_of_78N": (-180.0, 180.0, 78.0, 90.0),
}

# Official subject land area, km^2 (Rosstat, rounded). Only the subjects needed to
# quantify the polygon inflation; missing entries are reported as "unknown".
OFFICIAL_SUBJECT_AREA = {
    "Красноярский край": 2_360_000,
    "Республика Саха (Якутия)": 3_083_523,
    "Архангельская область": 587_949,
    "Чукотский автономный округ": 731_500,
    "Ямало-Ненецкий автономный округ": 123_587,
    "Ненецкий автономный округ": 176_716,
    "Иркутская область": 767_534,
    "Республика Коми": 415_900,
    "Магаданская область": 461_400,
    "Камчатский край": 270_560,
    "Мурманская область": 141_584,
    "Ханты-Мансийский автономный округ - Югра": 52_725,
    "Томская область": 314_398,
    "Хабаровский край": 787_633,
    "Амурская область": 363_689,
    "Сахалинская область": 87_101,
    "Еврейская автономная область": 36_843,
    "Республика Бурятия": 351_251,
    "Забайкальский край": 431_505,
    "Краснодарский край": 75_485,
    "Ростовская область": 100_967,
    "Волгоградская область": 31_299,
    "Астраханская область": 44_149,
    "Республика Калмыкия": 75_900,
}


def ensure_dirs() -> None:
    for p in (
        PROCESSED,
        ARTIFACTS / "geo",
        ARTIFACTS / "gaps",
        ARTIFACTS / "tables",
        ARTIFACTS / "edges",
        ARTIFACTS / "icvi",
        ARTIFACTS / "dynamics",
        ARTIFACTS / "interpretation",
    ):
        p.mkdir(parents=True, exist_ok=True)
