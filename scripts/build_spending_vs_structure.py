#!/usr/bin/env python3
"""Show that the five types are not a ranking of how much a district spends.

If the split were only about volume, the five clusters would sit in five neat bands of total
spending and nothing else would separate them. They do not: the bands overlap, and the clusters
differ in what the place is made of - how much of it is treed, how much is built, how green it is.
This picture puts the two statements side by side so the reader can see both at once.

Nothing is estimated here. Every value is read from the committed feature space and the shipped
partition; the numbers the figure draws are the numbers the clustering ran on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "processed" / "feature_space.parquet"
ASSIGN = ROOT / "artifacts" / "icvi" / "cluster_assignments.parquet"
OUT = ROOT / "artifacts" / "landing" / "spending_vs_structure.png"

COLOURS = {0: "#3f7d5c", 1: "#c8502f", 2: "#2f6f8f", 3: "#b08d3f", 4: "#6a5a7e"}
RUB_IN_THOUSANDS = 1e3


def main() -> int:
    features = pd.read_parquet(FEATURES)
    assign = pd.read_parquet(ASSIGN)
    joined = features.merge(assign[["tid", "cluster"]], on="tid", how="inner")
    joined = joined[joined.cluster.notna()]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    totals = joined["total_mean_rub"] / RUB_IN_THOUSANDS
    data = [totals[joined.cluster == c].dropna().to_numpy() for c in range(5)]
    positions = np.arange(1, 6)

    box = left.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
    )
    for patch, cluster in zip(box["boxes"], range(5)):
        patch.set_facecolor(COLOURS[cluster])
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLOURS[cluster])
    for part in ("whiskers", "caps"):
        for item in box[part]:
            item.set_color("#6b7480")
    medians = box["medians"]
    medians = medians if isinstance(medians, list) else [medians]
    for line in medians:
        line.set_color("#12233d")
        line.set_linewidth(1.8)

    left.set_xticks(positions)
    left.set_xticklabels([f"тип {c}" for c in range(5)], fontsize=11)
    left.set_ylabel("средние месячные траты округа, тыс. ₽ в месяц", fontsize=11)
    left.set_title("Сколько округ тратит: четыре типа из пяти лежат вперемешку", fontsize=12, loc="left")
    left.grid(axis="y", color="#dfe3e8", linewidth=0.8)
    left.set_axisbelow(True)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)

    measures = {
        "доля деревьев": "dynamic_world_trees",
        "доля застройки": "dynamic_world_built",
        "вегетация NDVI": "modis_ndvi_mean",
    }
    width = 0.26
    index = np.arange(5)
    offsets = (-width, 0.0, width)
    for offset, (label, column) in zip(offsets, measures.items()):
        values = [
            joined.loc[joined.cluster == c, column].mean() * 100.0
            for c in range(5)
        ]
        right.bar(
            index + offset,
            values,
            width=width * 0.92,
            label=label,
            color={"доля деревьев": "#2e5d3a", "доля застройки": "#b0413e",
                   "вегетация NDVI": "#7fa860"}[label],
            alpha=0.9,
        )
    right.set_xticks(index)
    right.set_xticklabels([f"тип {c}" for c in range(5)], fontsize=11)
    right.set_ylabel("значение признака, %", fontsize=11)
    right.set_title("Чем округи различаются по устройству места", fontsize=12, loc="left")
    right.legend(frameon=False, fontsize=10, loc="upper left")
    right.grid(axis="y", color="#dfe3e8", linewidth=0.8)
    right.set_axisbelow(True)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="#ffffff")
    overlap = min(
        np.percentile(data[c], 75) - np.percentile(data[other], 25)
        for c in range(5)
        for other in range(5)
        if c != other
    )
    print(
        f"figure {OUT.name} {OUT.stat().st_size:,} B · districts {len(joined):,} · "
        f"lowest overlap of the spending bands {overlap:.0f} тыс. ₽"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
