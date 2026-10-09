#!/usr/bin/env python3
"""Check the cluster legend against artifacts that are in git.

Every number in the legend has to resolve to a committed artifact: the medians, coverage, quartiles
and provenance strings come from artifacts/interpretation/cluster_fingerprints.csv; n, months and the
spending median come from artifacts/interpretation/cluster_profile.csv; the example territories have
to be territories that the committed assignment table actually placed in that cluster, and their
names have to be names the shipped map payload carries.

The chosen example for each measure is the territory whose value is closest to the cluster median.
That claim is checked from the artifact's own candidate shortlist, because the per-territory values
live in an untracked table and a check that leaned on it would pass here and fail anywhere else.

Rendering is another matter: until the legend is drawn under the selected-territory panel, the proof
that it fits without scrolling cannot be measured, so this says so out loud.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGEND = ROOT / "artifacts/landing/map_cluster_legend.json"
FINGERPRINTS = ROOT / "artifacts/interpretation/cluster_fingerprints.csv"
PROFILE = ROOT / "artifacts/interpretation/cluster_profile.csv"
ASSIGNMENTS = ROOT / "artifacts/icvi/cluster_assignments.parquet"
PAYLOAD = ROOT / "web/map/map_geom.js"


# The fingerprints artifact stores the cluster's median in a column called 'value', so the legend's
# field names are mapped onto the artifact's column names rather than assumed to match.
COLUMNS = {
    "median": "value",
    "coverage": "coverage",
    "n_members": "n_members",
    "p25": "p25",
    "p75": "p75",
}


def load_payload() -> dict:
    text = PAYLOAD.read_text(encoding="utf-8")
    start = text.find("const GEOM = ")
    body = text[start + len("const GEOM = ") :].rstrip().rstrip(";")
    return json.loads(body)


def main() -> int:
    problems: list[str] = []
    for path in (LEGEND, FINGERPRINTS, PROFILE, ASSIGNMENTS, PAYLOAD):
        if not path.exists():
            problems.append(f"{path.relative_to(ROOT)} missing")
    if problems:
        print("FAIL missing input: " + "; ".join(problems))
        return 1

    legend = json.loads(LEGEND.read_text(encoding="utf-8"))
    fp = pd.read_csv(FINGERPRINTS)
    profile = pd.read_csv(PROFILE)
    assign = pd.read_parquet(ASSIGNMENTS)
    payload = load_payload()

    fingerprint: dict[tuple[int, str], dict] = {
        (int(r.cluster), r.statistic): r._asdict() for r in fp.itertuples()
    }
    prof = {int(r.cluster): r._asdict() for r in profile.itertuples()}
    placed = dict(zip(assign["tid"].astype(int), assign["cluster"].astype(int)))
    territory_names = {int(p["i"]): (str(p["n"]), str(p["r"])) for p in payload["props"]}

    headline = legend.get("headline_measures") or []
    if len(headline) != 9:
        problems.append(f"headline measures: {len(headline)}, expected 9")

    checked_cells = 0
    for block in legend["clusters"]:
        cluster = int(block["cluster"])
        if cluster not in prof:
            problems.append(f"cluster {cluster} absent from cluster_profile.csv")
            continue
        p = prof[cluster]
        if int(block["n"]) != int(p["n"]):
            problems.append(f"cluster {cluster}: n {block['n']} vs profile {p['n']}")
        if float(block["months_of_observation_mean"]) != round(float(p["mean_months"]), 2):
            problems.append(f"cluster {cluster}: months do not match cluster_profile.csv")
        if float(block["median_spending_rub"]) != round(float(p["median_spending_rub"]), 2):
            problems.append(f"cluster {cluster}: spending median does not match the profile")
        if float(block["median_official_area_km2"]) != round(
            float(p["median_official_area_km2"]), 2
        ):
            problems.append(f"cluster {cluster}: official area does not match the profile")
        density = block.get("spending_per_official_km2")
        expected = round(float(p["median_spending_rub"]) / float(p["median_official_area_km2"]), 2)
        if density is None or abs(density - expected) > 0.02:
            problems.append(
                f"cluster {cluster}: spending per official km2 {density} is not "
                f"median_spending_rub / median_official_area_km2 ({expected})"
            )
        if "отношение медиан" not in str(block.get("spending_per_official_km2_note", "")):
            problems.append(f"cluster {cluster}: the density is not labelled as a ratio of medians")

        for cell in block["measures"]:
            stat = cell["statistic"]
            source = fingerprint.get((cluster, stat))
            if source is None:
                problems.append(f"cluster {cluster} / {stat}: no row in cluster_fingerprints.csv")
                continue
            checked_cells += 1
            for field, got in (
                ("median", float(cell["median"])),
                ("coverage", float(cell["coverage"])),
                ("n_members", int(cell["n_members"])),
                ("p25", float(cell["p25"])),
                ("p75", float(cell["p75"])),
            ):
                want = float(source[COLUMNS[field]])
                # The artifact rounds to four decimals, so equality would fail on the rounding, not
                # on the number. The tolerance is half of the last kept digit.
                ok = got == want if field == "n_members" else abs(got - round(want, 4)) <= 5e-5
                if not ok:
                    problems.append(f"cluster {cluster} / {stat}: {field} {got} vs artifact {want}")
            if not str(cell.get("provenance", "")).strip():
                problems.append(f"cluster {cluster} / {stat}: provenance string is empty")
            elif str(cell["provenance"]) != str(source["provenance"]):
                problems.append(
                    f"cluster {cluster} / {stat}: provenance does not match the artifact"
                )
            others = [c for c in prof if c != cluster]
            for contra in cell["contrast_with_others"]:
                other = int(contra["cluster"])
                want_other = round(fingerprint[(other, stat)]["value"], 4)
                if abs(float(contra["other_median"]) - want_other) > 5e-5:
                    problems.append(
                        f"cluster {cluster} / {stat}: contrast names median {contra['other_median']} "
                        f"for cluster {other}, the artifact says {want_other}"
                    )
                if (
                    abs(float(contra["difference"]) - round(float(cell["median"]) - want_other, 4))
                    > 1e-3
                ):
                    problems.append(
                        f"cluster {cluster} / {stat}: the stated difference is not median minus the other median"
                    )
            if len(cell["contrast_with_others"]) != len(others):
                problems.append(
                    f"cluster {cluster} / {stat}: contrasts with {len(cell['contrast_with_others'])} "
                    f"of the {len(others)} other clusters"
                )

    example_checks = 0
    for ex in legend["examples"]:
        tid = int(ex["tid"])
        cluster = int(ex["cluster"])
        candidates = ex.get("candidates") or []
        if not candidates:
            problems.append(f"example for cluster {cluster}/{ex['statistic']} ships no candidates")
            continue
        distances = [float(c["distance_to_median"]) for c in candidates]
        if distances[0] > min(distances) + 1e-9:
            problems.append(
                f"example for cluster {cluster}/{ex['statistic']} is not the closest of its own shortlist"
            )
        if int(candidates[0]["tid"]) != tid:
            problems.append(
                f"example for cluster {cluster}/{ex['statistic']} is not the first candidate it lists"
            )
        if placed.get(tid) != cluster:
            problems.append(
                f"example territory {tid} is not placed in cluster {cluster} by cluster_assignments.parquet"
            )
        if tid not in territory_names:
            problems.append(f"example territory {tid} is not in the shipped map payload")
        else:
            name, _region = territory_names[tid]
            ex["resolved_name"] = name
        example_checks += 1

    if "не признаки" not in str(legend.get("what_this_is", "")):
        problems.append("the legend does not say these are contrasts and not membership criteria")

    # A cluster number on the map is a claim about a district, so the district's own values have to be
    # available next to the medians of its cluster and of the two others, and a district with no spending
    # data must be told that plainly instead of being given a cluster number.
    measures_art = ROOT / "artifacts/landing/territory_measures.json"
    measures_report: dict = {"status": "not built"}
    if measures_art.exists():
        measures = json.loads(measures_art.read_text(encoding="utf-8"))
        clustered = measures["territories"]
        unclustered = measures["no_spending"]
        null_values = 0
        for tid, record in clustered.items():
            if "cluster" not in record:
                problems.append(f"territory {tid} has a record but no cluster")
                break
            if len(record["values"]) != len(measures["measures"]):
                problems.append(
                    f"territory {tid} carries {len(record['values'])} measures, the artifact lists "
                    f"{len(measures['measures'])}"
                )
                break
            null_values += sum(1 for v in record["values"].values() if v is None)
            if not str(record.get("name", "")).strip():
                problems.append(f"territory {tid} has no name")
                break
        for tid, reason in unclustered.items():
            if not str(reason).strip():
                problems.append(f"territory {tid} without a cluster carries no reason")
                break
            # A substring test on the word for "no" was both too weak and wrong: it accepted
            # "кластеризацию" as a mention of absence and rejected the precise wording now used, which
            # says either that no spending row exists or that spending exists and the coverage gate was
            # not passed. The reason must name which of the two kinds of absence it is.
            if not re.search(r"ни одной строки трат|траты в данных есть", str(reason)):
                problems.append(
                    f"territory {tid} says it is absent from the clustering without saying which kind "
                    "of absence it is"
                )
                break
        split = measures["counts"]
        kinds = {
            "rows": sum(1 for r in unclustered.values() if "ни одной строки трат" in str(r)),
            "gate": sum(1 for r in unclustered.values() if "траты в данных есть" in str(r)),
        }
        if kinds["rows"] != split.get("no_spending_rows") or kinds["gate"] != split.get(
            "spending_but_excluded"
        ):
            problems.append(
                f"the reasons split {kinds['rows']} without spending rows and {kinds['gate']} below the "
                f"panel gate, counts say {split.get('no_spending_rows')} and "
                f"{split.get('spending_but_excluded')}"
            )
        if len(unclustered) != split.get("without_spending"):
            problems.append(
                f"{len(unclustered)} districts carry an absence reason, counts say "
                f"{split.get('without_spending')}"
            )
        if int(measures["counts"]["with_cluster"]) != len(clustered):
            problems.append(
                f"the artifact counts {measures['counts']['with_cluster']} clustered districts but "
                f"carries {len(clustered)}"
            )
        if int(measures["counts"]["without_spending"]) != len(unclustered):
            problems.append(
                f"the artifact counts {measures['counts']['without_spending']} districts without "
                f"spending data but carries {len(unclustered)}"
            )
        # The medians the page will compare against must be the legend's medians, not a second opinion.
        for cluster_id, medians in measures["cluster_medians"].items():
            legend_cluster = next(
                (c for c in legend["clusters"] if str(c["cluster"]) == str(cluster_id)), None
            )
            if legend_cluster is None:
                problems.append(f"cluster {cluster_id} has medians but no legend entry")
                continue
            for measure in legend_cluster["measures"]:
                stat = measure["statistic"]
                if (
                    stat not in medians
                    or abs(float(medians[stat]) - float(measure["median"])) > 1e-9
                ):
                    problems.append(
                        f"cluster {cluster_id} / {stat}: the per-territory artifact's median "
                        f"{medians.get(stat)} is not the legend's {measure['median']}"
                    )
                    break
        measures_report = {
            "artifact": str(measures_art.relative_to(ROOT)),
            "artifact_bytes": measures_art.stat().st_size,
            "with_cluster": len(clustered),
            "without_spending": len(unclustered),
            "measures": len(measures["measures"]),
            "absent_values": null_values,
            "medians_match_legend": not any("is not the legend's" in p for p in problems),
        }
        print(
            f"territory measures: {len(clustered)} districts with a cluster, {len(unclustered)} "
            f"without spending data, {null_values} absent values kept as absent, "
            f"{measures_art.stat().st_size:,} B"
        )

    proof = {
        "artifact": str(LEGEND.relative_to(ROOT)),
        "artifact_bytes": LEGEND.stat().st_size,
        "clusters": len(legend["clusters"]),
        "headline_measures": len(headline),
        "cells_checked_against_fingerprints": checked_cells,
        "examples": len(legend["examples"]),
        "examples_resolved_in_assignments_and_payload": example_checks,
        "territory_measures": measures_report,
        "rendering": "not measured",
    }
    print(f"clusters: {proof['clusters']} | headline measures: {proof['headline_measures']}")
    print(f"cells checked against cluster_fingerprints.csv: {checked_cells}")
    print(f"examples: {proof['examples']} resolved against assignments and the map payload")
    # The legend's examples are drawn on the map, so the promise can become a measurement: the rows must
    # be the artifact's examples, opening them must not make the side column scroll, and a row must fly
    # to the district it names - which is the zoom-to-territory the map owes.
    full_page = ROOT / "web/map/full/index.html"
    drawn: dict = {"status": "not measured"}
    if full_page.exists():
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page_errors: list[str] = []
            page.on("pageerror", lambda e, sink=page_errors: sink.append(str(e)))
            page.goto(full_page.resolve().as_uri())
            page.wait_for_function(
                "() => document.getElementById('key').children.length > 0", timeout=40000
            )
            page.evaluate("() => document.querySelectorAll('#meaning button')[0].click()")
            page.wait_for_timeout(500)
            drawn = page.evaluate(
                "() => { const box = document.getElementById('examples');"
                " const rows = Array.from(box.querySelectorAll('.ex')).map(r => r.textContent);"
                " const s = box.getBoundingClientRect();"
                " const m = document.getElementById('map').getBoundingClientRect();"
                " const side = document.getElementById('side');"
                " return {hidden: box.hidden, header: (box.querySelector('h4')||{}).textContent || '',"
                " rows: rows, pastBottom: Math.round(s.bottom - m.bottom),"
                " pastRight: Math.round(s.right - m.right), inside: s.left >= m.left && s.top >= m.top,"
                " inner: box.scrollHeight + '/' + box.clientHeight,"
                " sideScroll: side.scrollHeight + '/' + side.clientHeight, scrollY: window.scrollY}; }"
            )
            drawn["flyTarget"] = page.evaluate(
                "() => { const btn = document.querySelector('#examples .ex button');"
                " if (!btn) return null; btn.click(); return true; }"
            )
            page.wait_for_timeout(900)
            drawn["detail"] = page.evaluate(
                "() => document.getElementById('detail').textContent.slice(0, 80)"
            )
            drawn["scrollYAfterFly"] = page.evaluate("() => window.scrollY")
            drawn["sideAfterFly"] = page.evaluate(
                "() => { const s = document.getElementById('side');"
                " return s.scrollHeight + '/' + s.clientHeight; }"
            )
            drawn["page_errors"] = page_errors[:3]
            browser.close()
        if drawn.get("hidden"):
            problems.append("clicking a legend row left the examples hidden")
        expected_rows = [
            e for e in legend["examples"] if e["cluster"] == legend["clusters"][0]["cluster"]
        ]
        if len(drawn.get("rows", [])) != len(expected_rows):
            problems.append(
                f"the examples box shows {len(drawn.get('rows', []))} districts, the artifact has "
                f"{len(expected_rows)} for that cluster"
            )
        for ex in expected_rows:
            want = f"значение {ex['value']}"
            if not any(want in r for r in drawn.get("rows", [])):
                problems.append(
                    f"example {ex['statistic']} of cluster {ex['cluster']} is not shown "
                    f"with the value the artifact carries ({ex['value']})"
                )
                break
        if drawn.get("pastBottom", 0) > 0 or drawn.get("pastRight", 0) > 0:
            problems.append(
                f"the examples box runs past the map frame (bottom {drawn.get('pastBottom')}, "
                f"right {drawn.get('pastRight')})"
            )
        if not drawn.get("inside"):
            problems.append("the examples box starts outside the map frame")
        for key in ("sideScroll", "sideAfterFly"):
            parts = [int(x) for x in str(drawn.get(key, "")).split("/") if x.isdigit()]
            if len(parts) == 2 and parts[0] > parts[1]:
                problems.append(f"the examples made the side column scroll ({drawn.get(key)})")
        if drawn.get("scrollY") != 0 or drawn.get("scrollYAfterFly") != 0:
            problems.append("opening or using the examples scrolled the page")
        if drawn.get("flyTarget") is None:
            problems.append("no example offers a way to zoom to its own territory")
        first_example = expected_rows[0]
        named = territory_names.get(first_example["tid"], (None, None))[0]
        if named and named not in drawn.get("detail", ""):
            problems.append(
                f"the fly went to a district other than the first example, {first_example['tid']} "
                f"({named})"
            )
        if drawn.get("page_errors"):
            problems.append(f"page errors while drawing the examples: {drawn['page_errors'][:2]}")
        proof["rendering"] = drawn
        print(
            f"examples drawn: {len(drawn.get('rows', []))} rows, box {drawn.get('inner')}, "
            f"side {drawn.get('sideScroll')} then {drawn.get('sideAfterFly')}, past frame "
            f"bottom {drawn.get('pastBottom')} right {drawn.get('pastRight')}, "
            f"scrollY {drawn.get('scrollY')}/{drawn.get('scrollYAfterFly')}"
        )

    print(f"findings: {len(problems)}")
    for p in problems[:8]:
        print(f"  FOUND {p}")
    if problems:
        print("CLUSTER LEGEND CONTRACT FAILED")
        return 1
    print("CLUSTER LEGEND CONTRACT VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
