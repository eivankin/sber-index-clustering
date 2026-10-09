#!/usr/bin/env python
"""Run every contract check and print one verdict line per contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ("day-1 master / geometry / gaps", "scripts/check_master.py"),
    ("federal districts mapped", "scripts/check_districts.py"),
    ("panel shape and NaN policy", "scripts/check_panel.py"),
    ("feature space + density denominator", "scripts/check_features.py"),
    ("edge construction + sparsification effect", "scripts/check_edges.py"),
    ("spending-type transition rule", "scripts/check_dynamics.py"),
    ("interpretation artifacts (fingerprints, profiles, gaps)", "scripts/check_interpretation.py"),
    ("source registries (license + citation per layer)", "scripts/check_sources_registry.py"),
    ("ICVI directions", "scripts/check_icvi.py"),
    ("artifact manifest", "scripts/make_manifest.py --verify"),
    ("cluster names resolve to the generated artifact", "scripts/check_cluster_names.py"),
    ("full-fidelity map page really draws full geometry", "scripts/check_map_full.py"),
    ("map zoom is exercised in a browser", "scripts/check_map_zoom.py"),
    (
        "credit line: flag gone, library link and basemap credits visible and not clipped",
        "scripts/check_attribution.py",
    ),
    ("map says what it is doing while it draws", "scripts/check_map_loading.py"),
    (
        "preset jumps land where the artifact says, and no menu exists",
        "scripts/check_map_presets.py",
    ),
    (
        "the month timeline moves the drawing and keeps no-data grey",
        "scripts/check_map_timeline.py",
    ),
    ("one-screen two-column layout at both desktop sizes", "scripts/check_map_layout.py"),
    (
        "neighbours that share an edge in the master still share it on the map",
        "scripts/check_neighbour_gaps.py",
    ),
    (
        "a duplicated place is offered once, and the duplication is counted",
        "scripts/check_duplicate_territories.py",
    ),
    ("our JavaScript parses and its names are defined", "scripts/check_js.py"),
    (
        "every brief requirement has an artifact behind it or a sentence naming its gap",
        "scripts/check_brief_audit.py",
    ),
    (
        "added Earth Engine layers carry provenance and licence, clear the coverage gate over the panel, "
        "and leave the files the published clustering reads untouched",
        "scripts/check_added_layers.py",
    ),
    ("no emoji glyphs on any shipped surface", "scripts/check_no_emoji.py"),
    (
        "region and place labels come from our own geometry and grow as you zoom",
        "scripts/check_map_labels.py",
    ),
    (
        "the cluster legend resolves to the fingerprint and profile artifacts",
        "scripts/check_cluster_legend.py",
    ),
    (
        "similar and complementary neighbours resolve to the shipped payload",
        "scripts/check_map_similar.py",
    ),
    (
        "every surface leads to the full map app and none of them embeds it",
        "scripts/check_map_centrepiece.py",
    ),
    (
        "every digit rendered on the presentation resolves to an artifact",
        "scripts/check_presentation_numbers.py",
    ),
    (
        "the stage-one presentation: tabs on top, switching, no overflow, lists within a screen, digits resolve",
        "scripts/check_stage_one_presentation.py",
    ),
    (
        "hover names one territory and only that territory, and clicking is what brings numbers",
        "scripts/check_map_hover.py",
    ),
    (
        "every monthly-cluster figure behind the map's note is recomputed from shipped assignments",
        "scripts/check_monthly_clusters.py",
    ),
    (
        "landing pages vs the claims map",
        "scripts/check_landing.py",
    ),
]


# Deferred contracts are listed here with the reason they are not being enforced in this goal, so a scope change
# stays visible instead of quietly reducing what the repository promises. The checker itself is untouched.
# The landing contract is no longer deferred: on 2026-10-09 scripts/build_claims.py re-read every claim from the
# artifact it names, which corrected the frozen features-shape value (the page now renders 2 190 × 177, the row
# and column counts the artifact reports), and scripts/check_landing.py passes 95/95 on the single Russian page.
DEFERRED: list[tuple[str, str, str]] = []


def main() -> int:
    fails = 0
    for name, cmd in CHECKS:
        parts = cmd.split()
        r = subprocess.run(
            [sys.executable] + parts, cwd=ROOT, capture_output=True, text=True, check=False
        )  # a failing checker must be reported, not raised
        ok = r.returncode == 0
        fails += not ok
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"{'PASS' if ok else 'FAIL'}  {name:34s} :: {tail[:90]}")
        if not ok and r.stderr:
            print("      stderr:", r.stderr.strip().splitlines()[-1][:120])
    for name, _cmd, reason in DEFERRED:
        print(f"DEFERRED  {name:34s} :: {reason[:120]}")
    print(
        f"\n{len(CHECKS) - fails}/{len(CHECKS)} contracts verified"
        + (f" | {len(DEFERRED)} deferred with a recorded reason" if DEFERRED else "")
    )
    print("ALL CONTRACTS VERIFIED" if fails == 0 else f"{fails} CONTRACTS FAILING")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
