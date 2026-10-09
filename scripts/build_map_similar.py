#!/usr/bin/env python3
"""Build «Похожие округа» in two labelled flavours, so the novelty is visible from the page.

Flavour one is what every 2026 competitor does: nearest neighbours in the attribute space, the
similarity family in configs/edge_rules.yaml. Flavour two is ours: a partner chosen because one side
produces land and the other side consumes it. Two territories can be unlike on every measure and
still pair strongly, which is exactly the contrast the entry is about.

Edge lists come from data/processed, which is not in git, so the artifact ships what it needs to be
checked: the neighbour ids, their weights in descending order, the cluster each belongs to, and a
role for complementarity partners derived from the rules written into this same artifact. A checker
can then verify all of it against web/map/map_geom.js and artifacts/icvi/cluster_assignments.parquet,
both of which are in git.

A territory without spending data gets a reason instead of an empty list. Showing nothing and letting
the reader guess whether that means "no neighbours" or "no data" is the kind of silence this project
does not allow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "web/map/map_geom.js"
ASSIGNMENTS = ROOT / "artifacts/icvi/cluster_assignments.parquet"
DIAGNOSTICS = ROOT / "artifacts/edges/edge_diagnostics.json"
FEATURES = ROOT / "data/processed/feature_space.parquet"
SIMILARITY = ROOT / "data/processed/edges_similarity.parquet"
COMPLEMENTARITY = ROOT / "data/processed/edges_complementarity_exp.parquet"
OUT = ROOT / "artifacts/landing/map_similar.json"

TOP_K = 5

# The rule for who produces and who consumes, written here so the checker applies the artifact's
# own rule instead of keeping a copy that could drift. Supply is land the territory makes: bare
# ground and cropland. Demand is money per unit of official land.
RULE = {
    "supply_measure": "gls_fraction_bare",
    "supply_extra": "gls_fraction_crops",
    "demand_measure": "spending_per_km2_log",
    "producer_label": "производит землю",
    "consumer_label": "потребляет землю",
    "balanced_label": "и производ, и спрос близки",
}


def load_payload() -> dict:
    text = PAYLOAD.read_text(encoding="utf-8")
    start = text.find("const GEOM = ")
    body = text[start + len("const GEOM = ") :].rstrip().rstrip(";")
    return json.loads(body)


def top_neighbours(edges: pd.DataFrame, tid: int) -> list[tuple[int, float]]:
    block = edges[(edges["source_id"] == tid) | (edges["target_id"] == tid)]
    if block.empty:
        return []
    scored = block[["weight"]].copy()
    scored["other"] = block["source_id"].where(block["target_id"] == tid, block["target_id"])
    # The similarity family is fully mutual (reciprocity 1.0), so the table stores every pair twice:
    # once each way round. Reading neighbours from both endpoints without folding those two rows into
    # one meant the same district was named twice in the same list, for 1 603 of the 2 016 territories
    # the page can show. Fold first, then take the top K neighbours, so a list of five is five places.
    scored = (
        scored[scored["other"] != tid]
        .drop_duplicates(subset=["other"], keep="first")
        .sort_values(["weight", "other"], ascending=[False, True], kind="mergesort")
        .head(TOP_K)
    )
    return [(int(r.other), round(float(r.weight), 4)) for r in scored.itertuples()]


def main() -> int:
    missing = [
        p
        for p in (PAYLOAD, ASSIGNMENTS, DIAGNOSTICS, FEATURES, SIMILARITY, COMPLEMENTARITY)
        if not p.exists()
    ]
    if missing:
        print("FAIL missing input: " + ", ".join(str(m.relative_to(ROOT)) for m in missing))
        return 1

    payload = load_payload()
    assign = pd.read_parquet(ASSIGNMENTS)
    placed = dict(zip(assign["tid"].astype(int), assign["cluster"].astype(int)))
    diag = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    sim = pd.read_parquet(SIMILARITY)
    comp = pd.read_parquet(COMPLEMENTARITY)
    feats = pd.read_parquet(FEATURES).set_index("tid")

    supply = feats[RULE["supply_measure"]].fillna(0.0) + feats[RULE["supply_extra"]].fillna(0.0)
    demand = feats[RULE["demand_measure"]].fillna(0.0)

    similarity_stats = diag.get("similarity", {})
    complementarity_stats = diag.get("complementarity_exp", {})

    territories: dict[str, dict] = {}
    no_data = 0
    for prop in payload["props"]:
        tid = int(prop["i"])
        entry = {
            "name": str(prop["n"]),
            "region": str(prop["r"]),
            "cluster": prop.get("c"),
            "measured": bool(prop.get("m")),
        }
        if not entry["measured"]:
            no_data += 1
            entry["similar"] = []
            entry["partners"] = []
            entry["reason"] = (
                "Нет данных о тратах, поэтому ни похожести, ни дополнительности построить нельзя."
            )
            territories[str(tid)] = entry
            continue

        # The subject's own supply and demand are shipped once per territory so a checker can
        # recompute every partner's role from the artifact alone, without an untracked table.
        def value(series: pd.Series, who: int) -> float | None:
            if who not in series.index:
                return None
            got = series.get(who)
            return round(float(got), 4) if pd.notna(got) else None

        entry["supply"] = value(supply, tid)
        entry["demand"] = value(demand, tid)

        similar = []
        for other, weight in top_neighbours(sim, tid):
            similar.append(
                {
                    "tid": other,
                    "similarity": weight,
                    "same_cluster": (
                        placed.get(other) == placed.get(tid)
                        if placed.get(tid) is not None
                        else None
                    ),
                }
            )
        partners = []
        for other, weight in top_neighbours(comp, tid):
            partner_supply = value(supply, other)
            partner_demand = value(demand, other)
            role = RULE["balanced_label"]
            if None not in (entry["supply"], entry["demand"], partner_supply, partner_demand):
                gap = (entry["supply"] - partner_supply) - (entry["demand"] - partner_demand)
                role = (
                    RULE["producer_label"]
                    if gap > 0.5
                    else RULE["consumer_label"] if gap < -0.5 else RULE["balanced_label"]
                )
            partners.append(
                {
                    "tid": other,
                    "complementarity": weight,
                    "role": role,
                    "supply": partner_supply,
                    "demand": partner_demand,
                    "same_cluster": (
                        placed.get(other) == placed.get(tid)
                        if placed.get(tid) is not None
                        else None
                    ),
                }
            )
        entry["similar"] = similar
        entry["partners"] = partners
        if not similar and not partners:
            # A measured territory with nobody on either list is not the same as one without data.
            # It is isolated in the sparsified edge lists, and that has to be said rather than left
            # as an empty block the reader has to interpret.
            entry["isolated"] = True
            entry["reason"] = (
                "Данные о тратах есть, но в разреженных списках рёбер этот округ ни с кем не "
                "связан — ни в похожести, ни в дополнительности."
            )
        territories[str(tid)] = entry

    payload_out = {
        "generated_by": "scripts/build_map_similar.py",
        "flavours": {
            "similar": {
                "label": "похожие (базовый конкурентный способ)",
                "what": "ближайшие соседи в пространстве признаков — то же, что делает каждый",
                "family": "similarity",
                "artifact_numbers": {
                    "k": 10,
                    **similarity_stats.get("knn10", {}),
                    **similarity_stats.get("params", {}),
                    "reciprocity": similarity_stats.get("clique_hub10", {}).get("reciprocity"),
                    "clustering_coefficient": similarity_stats.get("clique_hub10", {}).get(
                        "clustering_coefficient"
                    ),
                    "degree_gini": similarity_stats.get("clique_hub10", {}).get("degree_gini"),
                },
            },
            "partners": {
                "label": "связанные (наше направление)",
                "what": "партнёр, у которого одна сторона производит землю, а другая потребляет",
                "family": "complementarity_exp",
                "artifact_numbers": {
                    "k": 25,
                    **complementarity_stats.get("knn25", {}),
                    "reciprocity": complementarity_stats.get("clique_hub10", {}).get("reciprocity"),
                    "degree_gini": complementarity_stats.get("clique_hub10", {}).get("degree_gini"),
                    "top1pct_degree_share": complementarity_stats.get("clique_hub10", {}).get(
                        "top1pct_degree_share"
                    ),
                    "contrast": {
                        "similarity_knn25_max_degree": similarity_stats.get("knn25", {}).get(
                            "max_degree"
                        ),
                        "complementarity_knn25_max_degree": complementarity_stats.get(
                            "knn25", {}
                        ).get("max_degree"),
                    },
                },
            },
        },
        "role_rule": RULE,
        "top_k": TOP_K,
        # Stated where the lists are consumed, because the fold is what makes a list of five be five
        # places rather than the same district named twice.
        "neighbour_rule": (
            "neighbours are read from both ends of an edge and folded to one row per other territory "
            "before the top K is taken; the similarity family is fully mutual, so every pair is stored "
            "twice"
        ),
        "provenance": {
            "payload": str(PAYLOAD.relative_to(ROOT)),
            "assignments": str(ASSIGNMENTS.relative_to(ROOT)),
            "edge_diagnostics": str(DIAGNOSTICS.relative_to(ROOT)),
            "edge_lists": [
                str(SIMILARITY.relative_to(ROOT)),
                str(COMPLEMENTARITY.relative_to(ROOT)),
            ],
            "per_territory_values": str(FEATURES.relative_to(ROOT)),
        },
        "counts": {
            "territories": len(territories),
            "without_spending_data": no_data,
            "with_partners": sum(1 for t in territories.values() if t["partners"]),
            "with_similar": sum(1 for t in territories.values() if t["similar"]),
            "isolated_with_data": sum(1 for t in territories.values() if t.get("isolated")),
            "absence_note": (
                "without_spending_data here means no comparison is offered: those districts have no "
                "cluster. It is not the same thing as having no spending rows - the split is in "
                "artifacts/landing/territory_measures.json, counts"
            ),
        },
        "coverage": {
            "with_similar_share": round(
                sum(1 for t in territories.values() if t["similar"]) / len(territories), 4
            ),
            "with_partners_share": round(
                sum(1 for t in territories.values() if t["partners"]) / len(territories), 4
            ),
            "without_data_share": round(no_data / len(territories), 4),
            "rule": "доля округов из counts по знаменателю territories, округлённая до четырёх знаков",
        },
        "territories": territories,
    }
    # The artifact records how big it is and how much of the country it covers, so a reader can tell
    # what fraction of districts actually gets a comparison. The byte size is written into the file it
    # describes, so the writer loops until the number it states is the number the file has.
    for _ in range(6):
        payload_out["bytes"] = OUT.stat().st_size if OUT.exists() else 0
        body = json.dumps(payload_out, ensure_ascii=False, separators=(",", ":")) + "\n"
        OUT.write_text(body)
        if (
            OUT.stat().st_size == len(body.encode("utf-8"))
            and payload_out["bytes"] == OUT.stat().st_size
        ):
            break
    print(f"wrote {OUT}: {OUT.stat().st_size:,} bytes")
    print(
        f"territories {len(territories)} | without spending data {no_data} | "
        f"with similar {payload_out['counts']['with_similar']} | with partners "
        f"{payload_out['counts']['with_partners']}"
    )
    print(
        "similarity numbers from the committed diagnostics:",
        payload_out["flavours"]["similar"]["artifact_numbers"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
