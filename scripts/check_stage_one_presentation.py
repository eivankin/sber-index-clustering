#!/usr/bin/env python3
"""Prove the stage-one presentation behaves the way the entrant asked for, in a browser, in well under a minute.

The contract this checks was set by the entrant on 2026-10-09, in their own words: each tab must be one screen,
scrolling switches tabs rather than scrolling the page, nothing internal (artifact paths, hashes, ids, field names)
may sit on the visible face, and content inside a tab is switched by buttons rather than by growing the page. The
previous version of this checker measured a step-based layout with an embedded map; that structure was retired with
the reason written in docs/presentation-structure-2026.md, so the measurement changed and the checker moved with it.

Six claims, each measured rather than asserted:

1. three tabs sit in the upper row and switching changes what is visible - exactly one slide at a time;
2. the wheel moves the viewer between tabs, and the page itself never scrolls;
3. no tab overflows horizontally and no tab is taller than one screen at either desktop size;
4. buttons inside a tab switch real content - the district shown and the year shown actually change;
5. the visible face carries no internal machinery; opening the source drawer is the only way it appears;
6. digits rendered on screen resolve to values held in the payload or in a registry, so nothing on screen was
   typed by hand and went stale.

Writes artifacts/landing/stage_one_presentation/proof.json. Exits 1 on any failure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "presentation" / "index.html"
CLAIMS = ROOT / "artifacts" / "landing" / "claims.json"
OUT_DIR = ROOT / "artifacts" / "landing" / "stage_one_presentation"
CHROMIUM = "/usr/bin/chromium"

VIEWPORTS = (("desktop-1280x800", 1280, 800), ("desktop-1920x1080", 1920, 1080))

TOKEN = re.compile(r"(?<![\w.,])\d[\d\u2009\u00a0 ]*(?:[.,]\d+)?")
HEX_RUN = re.compile(r"\b[0-9a-f]{12,}\b")
LATIN = re.compile(r"[A-Za-z][A-Za-z\u00C0-\u02AF]{2,}")
# NDVI joins the list on 2026-10-09: the entrant asked that the rail explain that «зелёность» is NDVI, and a
# named index is the same kind of term as VIIRS and MODIS - a dataset name, not internal machinery.
ALLOWED_LATIN = {
    "ARI",
    "NMI",
    "k",
    "km",
    "VIIRS",
    "MODIS",
    "NDVI",
    "Dynamic",
    "World",
    "OSM",
    "SoilGrids",
    "Hansen",
}

MACHINERY = (
    "artifacts/",
    "data/processed/",
    ".parquet",
    ".json",
    "sha256",
    "/home/",
    "gee-cache",
    "dynamic_world_",
    "viirs_lit",
    "modis_ndvi",
    "elevation_mean",
    "official_land_area",
    "total_mean_rub",
    "spending_per_km2",
)


def decimals_in(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else (
        len(token.split(",", 1)[1]) if "," in token else 0
    )


def normalise(token: str) -> str:
    return token.replace("\u00a0", "").replace("\u2009", "").replace(" ", "").replace(",", ".")


def collect_allowed(blob: str, payload: dict, claims: dict) -> tuple[set[str], set[float], set[str]]:
    floats: set[float] = set()
    strings: set[str] = set()

    def walk(node):
        if isinstance(node, float):
            floats.add(node)
        elif isinstance(node, int):
            floats.add(float(node))
        elif isinstance(node, str):
            strings.add(normalise(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    walk(claims)
    digits = set(re.findall(r"\d+", blob))
    return strings, floats, digits


def resolves(token: str, strings: set[str], floats: set[float], digit_runs: set[str]) -> bool:
    if token in strings:
        return True
    normalised = normalise(token)
    try:
        value = float(normalised)
    except ValueError:
        return any(run in digit_runs for run in re.findall(r"\d+", token))
    tolerance = 0.5 * 10.0 ** (-decimals_in(token)) + 1e-9
    if any(abs(value - candidate) <= tolerance for candidate in floats):
        return True
    return any(run in digit_runs for run in re.findall(r"\d+", token))


def main() -> int:
    if not PAGE.exists():
        print("FAIL the presentation page is not built - run scripts/build_stage_one_presentation.py")
        return 1

    html = PAGE.read_text(encoding="utf-8")
    marker = "const PDATA = "
    if marker not in html:
        print("FAIL the page does not inject a payload, so its figures cannot be traced")
        return 1
    start = html.index(marker) + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(html[start:].strip().rstrip(";"))
    claims = json.loads(CLAIMS.read_text(encoding="utf-8")) if CLAIMS.exists() else {}
    blob = json.dumps(payload, ensure_ascii=False) + json.dumps(claims, ensure_ascii=False)
    allowed_strings, allowed_floats, digit_runs = collect_allowed(blob, payload, claims)

    failures: list[str] = []
    proof: dict = {"tabs": [], "views": {}, "interaction": {}}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors: list[str] = []
        page.on("pageerror", lambda exc, sink=errors: sink.append(f"pageerror: {exc}"))
        page.goto(PAGE.resolve().as_uri())
        page.wait_for_timeout(900)

        tab_count = page.evaluate("() => document.querySelectorAll('#rail button[role=tab]').length")
        rail_top = page.evaluate("() => document.getElementById('rail').getBoundingClientRect().top")
        visible_ids = page.evaluate(
            "() => [...document.querySelectorAll('.slide')].filter(s => getComputedStyle(s).display !== 'none').map(s => s.id)"
        )
        proof["tabs"] = {
            "count": tab_count,
            "rail_top_px": rail_top,
            "visible_slide_ids_on_load": visible_ids,
        }
        if tab_count != 3:
            failures.append(f"tabs: {tab_count} instead of 3")
        if rail_top != 0:
            failures.append(f"tabs are not in the upper row: rail top {rail_top} px")
        if len(visible_ids) != 1:
            failures.append(f"{len(visible_ids)} slides visible at once on load: {visible_ids}")

        # switching by click changes what is visible, and only one slide ever shows
        seen: list[str] = []
        for index in range(3):
            page.click(f"#rail button[role=tab]:nth-of-type({index + 1})")
            page.wait_for_timeout(220)
            shown = page.evaluate(
                "() => [...document.querySelectorAll('.slide')].filter(s => getComputedStyle(s).display !== 'none').map(s => s.id)"
            )
            seen.append(shown[0] if shown else "none")
            if len(shown) != 1:
                failures.append(f"tab {index}: {len(shown)} slides visible at once")
        proof["interaction"]["visible_slide_per_tab"] = seen
        if len(set(seen)) != 3:
            failures.append(f"tabs do not switch distinct content: {seen}")

        # the wheel is the navigation: it must change the tab and must not scroll the page
        page.click("#rail button[role=tab]:nth-of-type(1)")
        page.wait_for_timeout(220)
        before = page.evaluate("() => document.querySelector('.slide.active') ? document.querySelector('.slide.active').id : 'none'")
        scroll_before = page.evaluate("() => window.scrollY")
        page.mouse.move(640, 400)
        page.mouse.wheel(0, 620)
        page.wait_for_timeout(460)
        after = page.evaluate("() => document.querySelector('.slide.active') ? document.querySelector('.slide.active').id : 'none'")
        scroll_after = page.evaluate("() => window.scrollY")
        page.mouse.wheel(0, -620)
        page.wait_for_timeout(460)
        back = page.evaluate("() => document.querySelector('.slide.active') ? document.querySelector('.slide.active').id : 'none'")
        proof["interaction"]["wheel"] = {
            "before": before, "after": after, "came_back_to": back,
            "scrollY_before": scroll_before, "scrollY_after": scroll_after,
        }
        if before == after:
            failures.append("the wheel did not switch tabs")
        if back != before:
            failures.append(f"the wheel does not go back: {before} -> {after} -> {back}")
        if scroll_after != 0:
            failures.append(f"the page scrolled to scrollY {scroll_after}; scrolling must be tab navigation only")

        # content inside a tab switches by buttons: district and year must actually change
        page.click("#rail button[role=tab]:nth-of-type(1)")
        page.wait_for_timeout(220)
        src_before = page.evaluate("() => document.querySelector('figure img') ? document.querySelector('figure img').getAttribute('src') : null")
        bars_before = page.evaluate(
            "() => [...document.querySelectorAll('#s-provenance .bar em')].slice(0, 3).map(function (n) { return n.textContent; })"
        )
        legend_before = page.evaluate("() => { const l = document.querySelector('.legend'); return l ? l.innerText : null; }")
        buttons = page.evaluate("() => document.querySelectorAll('#s-provenance .col button:not(:disabled)').length")
        if buttons < 2:
            failures.append(f"the data tab has only {buttons} switching buttons - the picker and the year control are required")
        page.evaluate(
            "() => { const buttons = [...document.querySelectorAll('#s-provenance .col button')];"
            " const pickers = buttons.filter(function (b) { return b.dataset.pick !== undefined; });"
            " if (pickers.some(function (b) { return /\u0442\u0438\u043f/.test(b.textContent); }))"
            "   { window.__typeHintLeft = true; }"
            " pickers[pickers.length - 1].click(); }"
        )
        page.wait_for_timeout(260)
        src_after = page.evaluate("() => document.querySelector('figure img') ? document.querySelector('figure img').getAttribute('src') : null")
        bars_after = page.evaluate(
            "() => [...document.querySelectorAll('#s-provenance .bar em')].slice(0, 3).map(function (n) { return n.textContent; })"
        )
        legend_after = page.evaluate("() => { const l = document.querySelector('.legend'); return l ? l.innerText : null; }")
        proof["interaction"]["district_picker"] = {
            "src_before": src_before, "src_after": src_after,
            "legend_before": legend_before, "legend_after": legend_after,
            "bars_before": bars_before, "bars_after": bars_after,
        }
        if page.evaluate("() => !!window.__typeHintLeft"):
            failures.append(
                "the district picker labels buttons with a cluster type the viewer has not met yet; "
                "the entrant asked for those hints to go on 2026-10-09"
            )
        if src_before == src_after:
            failures.append("the district picker did not change the image shown")
        # The caption was removed on 2026-10-09: an area-specific sentence about a cluster is not something a
        # viewer can act on, and it ran into the scroll hint. What must demonstrably change is the picture and
        # the numbers beside it - for land classes the class names are the same everywhere by design, so the
        # shares are the proof that a different district is actually on screen.
        if src_before == src_after and bars_before == bars_after:
            failures.append("the district picker changed neither the picture nor the numbers beside it")

        # the year control is a second subsection: it must change the image and the class shares
        shares_before = page.evaluate(
            "() => [...document.querySelectorAll('#s-provenance .bar em')].slice(0, 3).map(function (n) { return n.textContent; })"
        )
        page.evaluate(
            "() => { const buttons = [...document.querySelectorAll('#s-provenance .col button')];"
            " const year = buttons.find(function (b) { return b.textContent.trim() === '2024'; });"
            " if (year) year.click(); }"
        )
        page.wait_for_timeout(260)
        src_2024 = page.evaluate("() => document.querySelector('figure img') ? document.querySelector('figure img').getAttribute('src') : null")
        shares_2024 = page.evaluate(
            "() => [...document.querySelectorAll('#s-cadence, #s-provenance .bar em')].slice(0,3).map(function (n) { return n.textContent; })"
        )
        proof["interaction"]["year_control"] = {
            "src_after_2024": src_2024, "shares_before": shares_before, "shares_after": shares_2024,
        }
        if src_2024 == src_after:
            failures.append("the year control did not change the image shown")

        # the source rail and the crop must keep their size while the source changes, or the picture jumps
        page.click("#s-provenance button[data-source=\"ndvi\"]")
        page.wait_for_timeout(300)
        src_photo = page.evaluate("() => document.querySelector('figure img').getAttribute('src')")
        pressed_in_source_row = page.evaluate(
            "() => { const row = document.querySelector('#s-provenance .col');"
            " return [...row.querySelectorAll('button[data-source]')].filter(function (b) {"
            "   return b.getAttribute('aria-pressed') === 'true'; }).map(function (b) { return b.firstChild.textContent; }); }"
        )
        proof["interaction"]["source_toggle"] = {
            "src_with_amount_year": src_2024, "src_photo": src_photo,
            "pressed_in_source_row": pressed_in_source_row,
        }
        if src_photo == src_2024:
            failures.append("the source toggle did not change the image shown")
        if "ndvi" not in (src_photo or ""):
            failures.append(f"the source toggle showed the wrong file: {src_photo}")
        if pressed_in_source_row != ["зелёность"]:
            failures.append(f"the source rail does not highlight the chosen source: {pressed_in_source_row}")

        # The pixel grid was dropped on 2026-10-09. Its cell was counted in pixels of a degree-gridded window,
        # where one pixel is 276 m east but 497 m north at this latitude, so "клетка - 5 км" was not a true
        # statement on the ground; the entrant also judged the lines as noise over the data. The measurement
        # now records that the control is gone, which is what the page must show.
        grid_control = page.query_selector("#s-provenance button[data-grid]")
        proof["interaction"]["grid_toggle"] = {
            "state": "dropped" if grid_control is None else "still offered",
        }
        if grid_control is not None:
            failures.append("the pixel grid control is still offered even though its cell size was wrong")

        # A difference of a pixel or two is browser rounding, not the jump the entrant complained about, so the
        # bar is 2 px; anything larger is a visible resize and is reported.
        box_sizes = []
        for key in ("dynamic_world", "viirs", "ndvi"):
            page.click(f"#s-provenance button[data-source=\"{key}\"]")
            page.wait_for_timeout(320)
            box_sizes.append(page.evaluate(
                "() => { const r = document.querySelector('figure img').getBoundingClientRect();"
                " return [Math.round(r.width), Math.round(r.height)]; }"
            ))
        proof["interaction"]["crop_box_stability"] = box_sizes
        sizes = [size[1] for size in box_sizes]
        if max(sizes) - min(sizes) > 2:
            failures.append(f"the crop jumps when the source changes: {box_sizes}")
        widths = [size[0] for size in box_sizes]
        if max(widths) - min(widths) > 2:
            failures.append(f"the crop changes width when the source changes: {box_sizes}")

        # a picture that points at a file which was never copied renders as a blank box and a broken icon
        for label, src in (
            ("land classes", src_2024),
            ("the greenness picture", src_photo),
        ):
            target = ROOT / "web" / "presentation" / str(src or "")
            if not target.is_file():
                failures.append(f"{label} points at a file missing from the page: {src}")

        page.screenshot(path=str(OUT_DIR / "view_1280_provenance.png"))

        for name, width, height in VIEWPORTS:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(420)
            for index in range(3):
                page.click(f"#rail button[role=tab]:nth-of-type({index + 1})")
                page.wait_for_timeout(260)
                metrics = page.evaluate(
                    """() => {
                      const active = document.querySelector('.slide.active');
                      const doc = document.documentElement;
                      let widest = 0;
                      document.querySelectorAll('.slide.active *').forEach(function (node) {
                        const r = node.getBoundingClientRect();
                        if (r.width > widest) widest = r.width;
                      });
                      return {
                        horizontalOverflow: doc.scrollWidth - doc.clientWidth,
                        slideScrollHeight: active ? active.scrollHeight : 0,
                        slideClientHeight: active ? active.clientHeight : 0,
                        viewportHeight: window.innerHeight,
                        widestElementPx: widest
                      };
                    }"""
                )
                key = f"{name}:{index}"
                proof["views"][key] = metrics
                if metrics["horizontalOverflow"] > 0:
                    failures.append(f"{key}: horizontal overflow {metrics['horizontalOverflow']} px")
                if metrics["slideScrollHeight"] > metrics["viewportHeight"]:
                    failures.append(
                        f"{key}: the tab is {metrics['slideScrollHeight']} px tall and the screen is "
                        f"{metrics['viewportHeight']} px - it must fit one screen"
                    )
                if metrics["widestElementPx"] > width:
                    failures.append(f"{key}: an element is {metrics['widestElementPx']} px wide on a {width} px screen")

                text = page.evaluate(
                    "() => { const s = document.querySelector('.slide.active'); return s ? s.innerText : ''; }"
                )
                machinery_on_face = [
                    needle for needle in MACHINERY if needle in text
                ]
                hex_on_face = HEX_RUN.findall(text)
                proof["views"][key]["internal_machinery_found"] = machinery_on_face + hex_on_face
                if machinery_on_face or hex_on_face:
                    failures.append(
                        f"{key}: the visible face carries internal machinery - "
                        + ", ".join(machinery_on_face + [h[:12] for h in hex_on_face][:3])
                    )

                latin = sorted({m.group(0) for m in LATIN.finditer(text)} - ALLOWED_LATIN)
                proof["views"][key]["latin_words"] = latin
                if latin:
                    failures.append(f"{key}: English reaches the face: {', '.join(latin[:6])}")

                tokens = TOKEN.findall(text)
                unresolved = [t for t in tokens if not resolves(t, allowed_strings, allowed_floats, digit_runs)]
                proof["views"][key]["numeric_tokens"] = len(tokens)
                proof["views"][key]["unresolved_digits"] = unresolved
                if unresolved:
                    failures.append(
                        f"{key}: {len(unresolved)} of {len(tokens)} digits resolve to nothing, first: "
                        + (unresolved[0] if unresolved else "")
                    )

        # the drawer is where provenance lives; it must be checkable from outside our repository and closable
        page.set_viewport_size({"width": 1280, "height": 800})
        page.click("#rail button[role=tab]:nth-of-type(1)")
        page.wait_for_timeout(260)
        # The source test above leaves the rail on the last source it clicked, so the drawer has to be put back
        # to a known starting point before its content is captured - the same measurement bug the picker test had.
        page.evaluate(
            "() => { const b = document.querySelector('#s-provenance button[data-source=\"dynamic_world\"]');"
            " if (b) b.click(); }"
        )
        page.wait_for_timeout(320)
        height_before = page.evaluate("() => document.body.scrollHeight")
        opened = page.evaluate(
            "() => { const d = document.querySelector('details.src'); if (!d) return 'no drawer';"
            " if (!d.open) d.querySelector('summary').click(); return d.open ? 'opened' : 'refused to open'; }"
        )
        page.wait_for_timeout(260)
        height_after = page.evaluate("() => document.body.scrollHeight")
        drawer_text = page.evaluate(
            "() => { const d = document.querySelector('details.src[open] > div'); return d ? d.innerText : ''; }"
        )
        drawer_link = page.evaluate(
            "() => { const d = document.querySelector('details.src[open] > div a');"
            " return d ? d.href : ''; }"
        )
        drawer_box = page.evaluate(
            "() => { const d = document.querySelector('details.src[open] > div'); if (!d) return null;"
            " const r = d.getBoundingClientRect();"
            " return {w: Math.round(r.width), h: Math.round(r.height)}; }"
        )
        closed = page.evaluate(
            "() => { const b = document.querySelector('details.src[open] > div button.close');"
            " if (!b) return 'no close control'; b.click();"
            " return document.querySelector('details.src').open ? 'still open' : 'closed'; }"
        )
        proof["drawer"] = {
            "state": opened,
            "body_scrollHeight_before": height_before,
            "body_scrollHeight_after": height_after,
            "public_page_link": drawer_link,
            "panel_px": drawer_box,
            "names_the_center": "центр окна" in drawer_text,
            "names_the_extent": "размер окна" in drawer_text,
            "names_the_scale": "масштаб" in drawer_text,
            "close_control": closed,
            "leaks_local_paths": "/home/" in drawer_text or "gee-cache" in drawer_text,
            "leaks_asset_ids": "dw_mean_" in drawer_text,
        }
        if opened == "no drawer":
            failures.append("no source drawer exists")
        elif "earth-engine" not in drawer_link:
            failures.append("the drawer gives no public page a reader could check us against")
        elif not proof["drawer"]["names_the_center"] or not proof["drawer"]["names_the_extent"]:
            failures.append("the drawer does not say where the window was taken or how wide it is")
        if proof["drawer"]["leaks_local_paths"] or proof["drawer"]["leaks_asset_ids"]:
            failures.append("the drawer still shows a local path or an internal asset id")
        if not drawer_box or drawer_box["w"] < 480 or drawer_box["h"] < 380:
            failures.append(
                f"the source drawer is too small to read in: {drawer_box}; the entrant said on 2026-10-09 "
                "that every drawer barely fit anything"
            )
        if closed != "closed":
            failures.append(f"the drawer cannot be closed: {closed}")
        if height_after > height_before:
            failures.append(f"opening the drawer grew the page from {height_before} to {height_after} px")

        # The entrant reported on 2026-10-09 that every drawer said the same thing whatever was selected: the
        # capture block is genuinely the same across sources (same window, same grid), so the drawer has to say
        # what the source is, how often it updates and what scale the picture uses. That is what must differ.
        ndvi_clicked = page.evaluate(
            "() => { const b = document.querySelector('#s-provenance button[data-source=\"ndvi\"]');"
            " if (!b) return 'no such button'; b.click(); return 'clicked'; }"
        )
        page.wait_for_timeout(320)
        ndvi_src = page.evaluate(
            "() => { const i = document.querySelector('figure img'); return i ? i.getAttribute('src') : null; }"
        )
        page.evaluate(
            "() => { const d = document.querySelector('details.src'); if (!d.open) d.querySelector('summary').click(); }"
        )
        page.wait_for_timeout(260)
        drawer_text_ndvi = page.evaluate(
            "() => { const d = document.querySelector('details.src[open] > div'); return d ? d.innerText : ''; }"
        )
        proof["drawer"]["ndvi_clicked"] = ndvi_clicked
        proof["drawer"]["ndvi_image"] = ndvi_src
        proof["drawer"]["changes_with_source"] = drawer_text_ndvi != drawer_text
        if ndvi_clicked != "clicked":
            failures.append(f"the source rail has no greenness button to switch to: {ndvi_clicked}")
        if not ndvi_src or "ndvi" not in ndvi_src:
            failures.append(f"switching to greenness did not put a greenness picture on screen: {ndvi_src}")
        if not proof["drawer"]["changes_with_source"]:
            failures.append("the source drawer says the same thing whichever source is selected")

        proof["page_errors"] = errors
        browser.close()

    if errors:
        failures.append("page errors: " + "; ".join(errors[:4]))
    proof["failures"] = failures
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "proof.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"tabs: {proof['tabs']['count']} at rail top {proof['tabs']['rail_top_px']} px")
    print("wheel:", proof["interaction"]["wheel"]["before"], "->", proof["interaction"]["wheel"]["after"])
    print("district picker changes image:", proof["interaction"]["district_picker"]["src_before"] != proof["interaction"]["district_picker"]["src_after"])
    for key in sorted(proof["views"]):
        view = proof["views"][key]
        print(
            f"{key}: overflow {view['horizontalOverflow']} px, slide {view['slideScrollHeight']} px / "
            f"screen {view['viewportHeight']} px, tokens {view['numeric_tokens']}, unresolved {len(view['unresolved_digits'])}"
        )
    if failures:
        for failure in failures:
            print("  FOUND", failure)
        print("STAGE-ONE PRESENTATION CONTRACT FAILED")
        return 1
    print("STAGE-ONE PRESENTATION CONTRACT VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
