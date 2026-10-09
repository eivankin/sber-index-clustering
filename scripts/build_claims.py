#!/usr/bin/env python3
"""Regenerates the claim registry so its stored values are what the artifacts say today.

The registry is what lets a reader check a public number: an id, an artifact, a field. Its stored value is metadata
only - the pages resolve live from the artifact - so a stored value that has drifted is not a harmless note, it is a
staleness alarm the landing checker reads. After the Dynamic World rebuild the shipped partition is k=5, and several
stored values still described the k=3 partition that was superseded. This script re-reads every claim from the
artifact it names and writes the current value back, so the registry and the artifacts cannot disagree.

Placements are re-pointed too. The stage-one presentation carries its own figures from its own artifacts and states
which artifact fed each one; the numbers that used to be marked as appearing on the presentation are landing-page
numbers now. That is recorded with the reason rather than left implicit, and the English placement is dropped because
the entrant asked for a Russian-only surface and the English pages are being retired.

Writes artifacts/landing/claims.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))

import check_landing  # noqa: E402

CLAIMS = ROOT / "artifacts" / "landing" / "claims.json"

PRESENTATION_CARRIES: set[str] = set()
PLACEMENT_NOTE = (
    "Placement reflects the surfaces as they are now. The stage-one presentation (web/presentation/index.html) "
    "renders its figures from its own payload and names the artifact behind each one; the numbers listed here "
    "belong to the Russian landing page. The 'en' placement was dropped on 2026-10-09 at the entrant's "
    "instruction that the submission is Russian only, and the English surfaces were removed. No claim is placed "
    "on a page that does not show it."
)


def _jsonable(value: object) -> object:
    """Turn a numeric read out of pandas or numpy into something JSON accepts.

    read_value hands back the dtype the artifact carries, so a row count can arrive as numpy int64, which json
    refuses to serialize. Converting it keeps the number identical and the file writable.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main() -> int:
    document = json.loads(CLAIMS.read_text(encoding="utf-8"))
    claims = document["claims"]

    refreshed: list[str] = []
    stale: list[str] = []
    for claim in claims:
        status, live = check_landing.read_value(
            claim["artifact"], claim["field"], claim.get("row_filter")
        )
        if status != "ok":
            stale.append(f'{claim["id"]}: artifact field unreadable ({status})')
            continue
        before = str(claim.get("value"))
        claim["value"] = _jsonable(live)
        if claim.get("second_field"):
            second_status, second_live = check_landing.read_value(
                claim["artifact"],
                claim["second_field"],
                claim.get("row_filter"),
            )
            if second_status == "ok":
                claim["second"] = _jsonable(second_live)
        if before != str(live):
            refreshed.append(f'{claim["id"]}: stored {before[:26]} -> artifact {str(live)[:26]}')

        appears = [surface for surface in claim.get("appears_on", []) if surface != "en"]
        if "presentation" in appears:
            appears.remove("presentation")
        if claim["id"] in PRESENTATION_CARRIES:
            appears.append("presentation")
        claim["appears_on"] = sorted(set(appears))

    document["placement_note"] = PLACEMENT_NOTE
    document["generated_by"] = "scripts/build_claims.py"
    document["what_this_is"] = (
        "The claim registry: every public number, the artifact and field it resolves to, and the surfaces that "
        "show it. Values here are re-read from the artifacts, so a drift is visible instead of silent."
    )
    CLAIMS.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"claims: {len(claims)}; values refreshed: {len(refreshed)}; unreadable: {len(stale)}")
    for line in refreshed[:12]:
        print(f"  refreshed {line}")
    for line in stale[:6]:
        print(f"  UNREADABLE {line}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
