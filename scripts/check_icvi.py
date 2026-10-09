#!/usr/bin/env python
"""Sanity test of the ICVI battery on synthetic data with known answers.

Checks the *direction* of every criterion: a planted two-blob partition must beat a random
partition on SW, CH, AVU and MQ, and must be better (lower) on S_Dbw and AVI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sberindex import icvi

rng = np.random.default_rng(7)

X = np.vstack(
    [
        rng.normal([0, 0], 0.3, size=(300, 2)),
        rng.normal([8, 8], 0.3, size=(300, 2)),
        rng.normal([4, -6], 0.3, size=(300, 2)),
    ]
)
truth = np.r_[np.zeros(300, int), np.ones(300, int), np.full(300, 2)]
shuffled = rng.permutation(truth)

# planted SBM-style weighted graph: dense inside blocks, sparse between
n = 900
W = np.zeros((n, n))
for a, b in [(0, 300), (300, 600), (600, 900)]:
    blk = rng.uniform(2, 5, size=(b - a, b - a))
    blk = (blk + blk.T) / 2
    W[a:b, a:b] = blk
between = rng.uniform(0.05, 0.3, size=(n, n))
between = (between + between.T) / 2
W = np.where(W > 0, W, between)

good = icvi.score(X, truth, W)
bad = icvi.score(X, shuffled, W)
single = icvi.score(X, np.zeros(n, int), W)
print("truth   :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in good.items()})
print("shuffled:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bad.items()})
print("k=1     :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in single.items()})

checks = [
    ("SW truth > shuffled", good["SW"] > bad["SW"]),
    ("CH truth > shuffled", good["CH"] > bad["CH"]),
    ("S_Dbw truth < shuffled", good["S_Dbw"] < bad["S_Dbw"]),
    ("AVI truth < shuffled", good["AVI"] < bad["AVI"]),
    ("AVU truth > shuffled", good["AVU"] > bad["AVU"]),
    ("MQ truth > shuffled", good["MQ"] > bad["MQ"]),
    ("MQ truth > 0", good["MQ"] > 0),
    (
        "k=1 gives None for SW/CH/S_Dbw/AVU",
        all(single[c] is None for c in ("SW", "CH", "S_Dbw", "AVU")),
    ),
    ("k=1 MQ is 0", abs(single["MQ"]) < 1e-9),
]
# A self-test of the indices says the measuring stick is sound. It says nothing about the tables the
# deliverable actually ships, so the committed comparison tables are checked here too: every battery
# index must be a column, no battery cell may be null, and every row must name a method and a k. This
# is what caught the density table existing while no command in the repository produced it - a claim
# in a report resting on a table nobody could regenerate.
ROOT = Path(__file__).resolve().parents[1]
TABLES = [
    "method_comparison.csv",
    "method_comparison_land.csv",
    "method_comparison_density.csv",
]
BATTERY = ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"]
for tbl in TABLES:
    path = ROOT / "artifacts" / "icvi" / tbl
    if not path.exists():
        checks.append((f"{tbl} exists", False))
        continue
    df = pd.read_csv(path)
    present = all(c in df.columns for c in BATTERY + ["method", "k", "ICVI_avg"])
    filled = bool(present) and int(df[BATTERY].isna().sum().sum()) == 0
    checks.append((f"{tbl}: {len(df)} variants, battery complete and null-free", filled))

fails = 0
for name, ok in checks:
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
print("ICVI SELF-TEST PASSED" if fails == 0 else f"{fails} CHECKS FAILED")
sys.exit(0 if fails == 0 else 1)
