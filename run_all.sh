#!/usr/bin/env bash
# One-command reproduction of the whole entry, in the order the numbers were produced.
# Network is only needed for steps 2, 3 and 5; everything else runs from data/raw/.
set -euo pipefail
PY=${PY:-.venv/bin/python}

echo "== 1. master, geometry defect, gaps, Rosstat land area =="
$PY scripts/fetch_rostat_area.py
$PY scripts/build_master.py
$PY scripts/check_master.py
$PY scripts/check_districts.py

echo "== 2. GEE covariates (needs SBER_GEE_SA_EMAIL / SBER_GEE_SA_KEY, network) =="
$PY scripts/probe_gee.py
$PY scripts/probe_gee2.py
$PY scripts/build_covariates_gee.py --smoke
$PY scripts/build_covariates_gee.py

echo "== 3. no-auth fallback covariates (network; can run without GEE) =="
$PY scripts/build_covariates_fallback.py --smoke
$PY scripts/build_covariates_fallback.py

echo "== 4. panel and attribute space =="
$PY scripts/build_panel.py
$PY scripts/check_panel.py
$PY scripts/build_features.py

echo "== 5. edges, clustering, ICVI =="
$PY scripts/check_icvi.py
$PY scripts/build_edges.py
$PY scripts/make_edge_summary.py
$PY scripts/run_clustering.py
$PY scripts/run_clustering.py --features feature_space.parquet --groups levels+shares+land --tag land
$PY scripts/run_clustering.py --features feature_space.parquet --groups density+shares+land --tag density
$PY scripts/interpret_clusters.py
$PY scripts/type_gaps.py

echo "== 6. temporal dynamics =="
$PY scripts/run_dynamics.py
$PY scripts/check_dynamics.py
$PY scripts/corroborate_transitions.py || true

echo "== 7. land-based validation (needs covariates from step 2 or 3) =="
$PY scripts/validate_land_hypothesis.py || true

echo "== 8. contract check + artifact manifest =="
$PY scripts/check_interpretation.py
$PY scripts/check_sources_registry.py
$PY scripts/check_all.py
$PY scripts/make_manifest.py
