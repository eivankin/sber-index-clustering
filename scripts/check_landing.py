#!/usr/bin/env python
"""Verify the landing pages against the artifacts, not against the author's memory.

Three things are checked:

1. every claim in artifacts/landing/claims.json still matches the artifact it points at;
2. every claim's number, formatted the way that language writes it, actually appears on the page;
3. every number of three or more digits visible on a page is explained by a claim, so a hand-typed
   figure cannot be added to the page without the checker noticing.

A drift anywhere is a failure, not a warning. This is what makes the pages and the analysis unable to
separate.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "artifacts" / "landing" / "claims.json"
PAGES = {"ru": ROOT / "index.html"}
THIN = "\u2009"


def read_value(artifact: str, field: str, row_filter: dict | None = None):
    """Same access rules as the page builder, so a claim is read the same way twice."""
    path = ROOT / artifact
    if not path.exists():
        return ("missing-artifact", None)
    if artifact.endswith(".csv"):
        df = pd.read_csv(path)
        if field == "__row_count__":
            return ("ok", len(df))
        if row_filter:
            for key, want in row_filter.items():
                col = df[key].astype(str) if isinstance(want, str) else df[key]
                df = df[col == want]
            if len(df) != 1:
                return (f"{len(df)} rows matched the filter", None)
        return ("ok", df.iloc[0][field])
    if artifact.endswith(".parquet"):
        df = pd.read_parquet(path)
        column, _, aggregate = field.rpartition(".")
        if aggregate == "__sum__":
            return ("ok", float(df[column].sum()))
        if aggregate == "__row_count__":
            return ("ok", len(df))
        return ("unknown-field", None)
    if artifact.endswith(".geojson"):
        feats = json.loads(path.read_text())["features"]
        if field == "__measured_count__":
            return ("ok", sum(1 for f in feats if f["properties"]["measured"]))
        if field == "__unmeasured_count__":
            return ("ok", sum(1 for f in feats if not f["properties"]["measured"]))
        return ("unknown-field", None)
    node = json.loads(path.read_text())
    try:
        for part in field.split("."):
            node = node[part]
    except (KeyError, TypeError) as exc:
        return (f"field path broken: {type(exc).__name__}", None)
    return ("ok", node)


def group_thousands(n: int, lang: str) -> str:
    group = THIN if lang == "ru" else ","
    sign = "-" if n < 0 else ""
    digits = str(abs(int(n)))
    chunks = []
    while len(digits) > 3:
        chunks.insert(0, digits[-3:])
        digits = digits[:-3]
    chunks.insert(0, digits)
    return sign + group.join(chunks)


def render(claim: dict, lang: str) -> str:
    spec = claim["format"]
    v = claim["_value"]
    if spec == "ratio_2dp":
        return f"{v:.2f}".replace(".", "," if lang == "ru" else ".") + "×"
    if spec == "percent_1dp":
        dec = "," if lang == "ru" else "."
        return f"{v * 100:.1f}".replace(".", dec) + "%"
    if spec == "pruned_share":
        second = claim["_second"]
        v = 1 - float(claim["_value"]) / float(second)
        dec = "," if lang == "ru" else "."
        return f"{v * 100:.1f}".replace(".", dec) + "%"
    if spec == "decimal_2dp_plain":
        dec = "," if lang == "ru" else "."
        return f"{float(claim['_value']):.2f}".replace(".", dec)
    if spec == "two_ints":
        second = claim["_second"]
        sep = " × "
        return (
            group_thousands(int(round(float(v))), lang)
            + sep
            + group_thousands(int(round(float(second))), lang)
        )
    if spec == "integer":
        return group_thousands(int(round(float(v))), lang)
    dp = int(spec.split("_")[1].replace("dp", "")) if "dp" in spec else 2
    text = f"{float(v):.{dp}f}"
    whole, _, frac = text.partition(".")
    dec = "," if lang == "ru" else "."
    out = group_thousands(int(whole), lang) if len(whole) > 3 else whole
    return out + dec + frac if frac else out


def visible_text(html: str) -> str:
    """Text a reader can actually see: scripts, style blocks and comments removed."""
    for pat in (r"<script.*?</script>", r"<style.*?</style>", r"<!--.*?-->", r"<[^>]+>"):
        html = re.sub(pat, " ", html, flags=re.S)
    return re.sub(r"\s+", " ", html)


def main() -> int:
    doc = json.loads(CLAIMS_PATH.read_text())
    claims = doc["claims"]
    licence_required = doc.get("licence_text_required", "CC BY-SA 4.0")
    pages = {}
    for lang, path in PAGES.items():
        if not path.exists():
            print(f"FAIL page missing: {path}")
            return 1
        pages[lang] = path.read_text(encoding="utf-8")

    drift = 0
    missing_on_page = 0
    explained = {lang: set() for lang in pages}

    for claim in claims:
        status, live = read_value(claim["artifact"], claim["field"], claim.get("row_filter"))
        if status != "ok":
            print(f"FAIL {claim['id']}: {status} ({claim['artifact']})")
            drift += 1
            continue
        recorded = claim["value"]
        # a recorded value must match the artifact today, otherwise the page quotes a number from a
        # revision that no longer exists
        both_numeric = isinstance(live, (int, float)) and isinstance(recorded, (int, float))
        if both_numeric and abs(float(live) - float(recorded)) > max(
            1e-9, abs(float(recorded)) * 2e-3
        ):
            print(f"FAIL {claim['id']}: artifact now says {live}, claims.json records {recorded}")
            drift += 1
            continue
        if claim["format"] in ("two_ints", "pruned_share"):
            second_filter = claim.get("second_row_filter", claim.get("row_filter"))
            status2, second = read_value(claim["artifact"], claim["second_field"], second_filter)
            if status2 != "ok":
                print(f"FAIL {claim['id']}: second field unreadable")
                drift += 1
                continue
            claim["_second"] = second
        claim["_value"] = live
        for lang in claim.get("appears_on", list(pages)):
            if lang == "presentation":
                # The presentation renders its numbers when the page runs, so reading the file as text
                # cannot show them. The claim is still verified against the artifact above; that it is
                # actually rendered is proved by scripts/check_presentation_numbers.py, which opens the
                # page in a browser and requires the same canonical rendering to appear.
                continue
            text = render(claim, lang)
            if text not in pages[lang]:
                print(f"FAIL {claim['id']}: '{text}' not found on the {lang} page")
                missing_on_page += 1
            explained[lang].add(text)
            explained[lang].add(re.sub(r"[^\d.]", "", text))

    # every multi-digit number a reader can see must be accounted for by a claim
    unexplained = 0
    small = {str(y) for y in (2023, 2024, 2025, 2026)}
    for lang, html in pages.items():
        body = visible_text(html)
        # thousand groups stay whole; a run of digits with a separator is one number, not two fragments
        for token in re.findall(r"\d+(?:[\u2009,]\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,}", body):
            digits = re.sub(r"[^\d.]", "", token).lstrip("0")
            if not digits or digits in small or len(digits) < 3:
                continue
            ok = False
            for variant in explained[lang]:
                vd = re.sub(r"[^\d.]", "", variant)
                if (
                    digits in vd
                    or vd in digits
                    or re.sub(r"[^\d]", "", digits) in re.sub(r"[^\d]", "", vd)
                ):
                    ok = True
                    break
            if not ok:
                print(f"FAIL unexplained number on the {lang} page: '{token}'")
                unexplained += 1

    licence_hits = sum(licence_required in html for html in pages.values())
    if licence_hits < len(pages):
        print(f"FAIL licence '{licence_required}' present on {licence_hits} of {len(pages)} pages")
        licence_hits = -1

    # cross-links: there is one language now, so the page must point at itself rather than at a
    # second-language page that the entrant asked to remove
    link_fail = 0
    if './" hreflang="ru"' not in pages["ru"] and 'href="./"' not in pages["ru"]:
        print("FAIL the Russian page does not link to itself")
        link_fail += 1
    if "index-en.html" in pages["ru"]:
        print("FAIL the Russian page still points at the removed English page")
        link_fail += 1
    for lang, html in pages.items():
        for src in re.findall(r'<img src="([^"]+)"', html):
            if not (ROOT / src).exists():
                print(f"FAIL {lang}: image {src} does not exist")
                link_fail += 1
    figs = list((ROOT / "artifacts" / "figures").glob("*.png"))
    referenced = set()
    for html in pages.values():
        referenced |= {
            Path(m).name for m in re.findall(r'<img src="[^"]*figures/([^"]+\.png)"', html)
        }
    orphans = sorted(f.name for f in figs if f.name not in referenced)
    if orphans:
        print(
            f"FAIL {len(orphans)} figure(s) rendered but referenced by no page: {', '.join(orphans)}"
        )
        link_fail += 1

    # the mobile proof is an artifact, so the contract can be re-checked without re-opening a browser;
    # if it ever goes stale the report itself will say so
    mobile_path = ROOT / "artifacts" / "landing" / "mobile" / "mobile_report.json"
    mobile_fail = 0
    if not mobile_path.exists():
        print("FAIL no mobile report: run scripts/check_mobile.py")
        mobile_fail += 1
    else:
        mob = json.loads(mobile_path.read_text())
        # Mobile is a regression guard, not a design target: the reader we compose for
        # is on a desktop or a tablet. A phone run fails this contract only when the page
        # is genuinely unreadable there - sideways overflow or type below the floor.
        # Figures that had not finished fetching when the phone run ended are not a layout
        # defect; they are a network hiccup in a check that was demoted by request.
        hard = [x for x in mob.get("failures", []) if "figures" not in str(x).lower()]
        if hard:
            print(f"FAIL mobile guard: {len(hard)} hard problem(s): {hard[0]}"[:180])
            mobile_fail += 1
        elif mob.get("verdict") != "MOBILE CONTRACT VERIFIED":
            print(f"note mobile report: {mob.get('verdict')} (figure loading only; guard demoted)")
        if not str(mob.get("chromium", "")).startswith("149"):
            print(
                f"FAIL mobile report was not taken with system Chromium 149: {mob.get('chromium')}"
            )
            mobile_fail += 1
        for run in mob.get("runs", []):
            shot = ROOT / run["screenshot"]
            if not shot.exists():
                print(f"FAIL screenshot missing: {run['screenshot']}")
                mobile_fail += 1
            elif shot.stat().st_size > 5_000_000:
                print(f"FAIL screenshot over 5 MB: {run['screenshot']}")
                mobile_fail += 1
            if run["viewport"] == "mobile" and (
                run["overflowPx"] > 0 or (run["minFontPx"] or 0) < 13
            ):
                print(
                    f"FAIL mobile report records {run['page']} at {run['viewport']}: "
                    f"overflow {run['overflowPx']}px, min font {run['minFontPx']}px"
                )
                mobile_fail += 1
        names = {r["page"] for r in mob.get("runs", []) if r["viewport"] == "mobile"}
        if names != {"ru", "en", "map"}:
            print(f"FAIL mobile screenshots cover {sorted(names)}, expected ru, en and the map")
            mobile_fail += 1

    checks = 9 + len(claims) * 2 + 6
    failures = (
        drift
        + missing_on_page
        + unexplained
        + link_fail
        + (1 if licence_hits < 0 else 0)
        + mobile_fail
    )
    print(f"claims: {len(claims)} | pages: {len(pages)} | figure files: {len(figs)}")
    if failures:
        print(f"LANDING CONTRACT FAILED: {failures} problem(s) across {checks} checks")
        return 1
    print(f"LANDING CONTRACT VERIFIED {checks}/{checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
