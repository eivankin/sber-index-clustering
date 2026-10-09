#!/usr/bin/env python3
"""Prove the map is the centrepiece: every surface leads to the full app, and none embeds it.

The old landing never mentioned the map at all, which is how a reader ended up with a page of numbers
and no way to see the geography the numbers were about. The rule now is that the presentation's last
step and both landing languages link the full-fidelity one-screen app, and that the app is entered,
not framed inside another page — embedding it would put a second map inside a step and make the
"one surface" claim false.

Links are resolved rather than string-matched, because the presentation sits one directory down and
writes ../map/full/index.html. A resolved path is the same file, and a checker that only looked for
one spelling would fail a correct page.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web/map/full/index.html"
SURFACES = {
    "landing ru": ROOT / "index.html",
    "presentation": ROOT / "web/presentation/index.html",
}
HREF = re.compile(r"""href=["']([^"']+)["']""")


def resolves_to_app(href: str, base: Path) -> bool:
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return False
    if href.startswith("/"):
        target = str(ROOT / href.lstrip("/"))
    else:
        target = os.path.normpath(os.path.join(str(base.parent), href))
    return os.path.realpath(target) == os.path.realpath(str(APP))


def main() -> int:
    problems: list[str] = []
    if not APP.exists():
        problems.append("the full map app does not exist")
    app_bytes = APP.stat().st_size if APP.exists() else 0

    proof = {"app": str(APP.relative_to(ROOT)), "app_bytes": app_bytes, "surfaces": {}}
    for label, path in SURFACES.items():
        if not path.exists():
            problems.append(f"{label}: {path.relative_to(ROOT)} missing")
            continue
        text = path.read_text(encoding="utf-8")
        links = [h for h in HREF.findall(text) if resolves_to_app(h, path)]
        embeds = len(re.findall(r"<iframe", text, re.I))
        proof["surfaces"][label] = {
            "path": str(path.relative_to(ROOT)),
            "links_to_app": len(links),
            "spelling": sorted(set(links)),
            "iframes": embeds,
        }
        if not links:
            problems.append(f"{label}: nothing links the full map app")
        if embeds:
            problems.append(f"{label}: embeds the app in an iframe ({embeds})")

    if app_bytes < 1_000_000:
        problems.append(f"the app is only {app_bytes} B, too small to be full geometry")

    print(f"app: {proof['app']} ({app_bytes:,} B)")
    for label, record in proof["surfaces"].items():
        print(
            f"{label}: links to the app {record['links_to_app']} {record['spelling']} | "
            f"iframes {record['iframes']}"
        )
    print(f"findings: {len(problems)}")
    for p in problems[:6]:
        print(f"  FOUND {p}")

    out = ROOT / "artifacts/landing/map_centrepiece"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=1) + "\n")
    print(f"proof written to {out / 'proof.json'}")

    if problems:
        print("MAP CENTREPIECE CONTRACT FAILED")
        return 1
    print("MAP CENTREPIECE CONTRACT VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
