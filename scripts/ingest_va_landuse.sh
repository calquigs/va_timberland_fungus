#!/usr/bin/env bash
# Ingest VA parcels + harvest points, build harvested timber parcels, compute
# loblolly biomass zonal stats, and ingest protected lands.
#
# Requires: DATABASE_URL, docker compose up (PostGIS)
#
# Usage:
#   ./scripts/ingest_va_landuse.sh [--limit N]
#
# Harvests are ingested first; then only parcels within the harvest bounding box
# are fetched. Quick test: --limit 5000

set -e
cd "$(dirname "$0")/.."

export DATABASE_URL="${DATABASE_URL:-postgresql://va_woods:va_woods_dev@localhost:5432/va_woods}"

echo "=== 1. Ingest VA harvest points (VOF) ==="
python ingestion/va_harvests/ingest_va_harvests.py "$@"

echo ""
echo "=== 2. Ingest VA parcels (only those containing harvest points) ==="
python ingestion/va_parcels/ingest_va_parcels.py --parcels-containing-harvests "$@"

echo ""
echo "=== 3. Build harvested timber parcels (spatial join) ==="
python ingestion/va_parcels/ingest_va_parcels.py --harvested-only --schema staging

echo ""
echo "=== 4. Download FIA BIGMAP loblolly raster (if not cached) ==="
python ingestion/fia_bigmap/ingest_fia_loblolly.py

echo ""
echo "=== 5. Compute per-parcel loblolly biomass zonal stats ==="
python ingestion/fia_bigmap/ingest_fia_loblolly.py --zonal-stats "$@"

echo ""
echo "=== 6. Ingest protected lands (VA DCR Conservation Lands) ==="
python ingestion/protected_lands/ingest_dcr_conslands.py "$@"

echo ""
echo "=== 7. Ingest environmental rasters ==="
python ingestion/environmental/ingest_env_rasters.py "$@"

echo ""
echo "=== 8. Sample environmental values at parcel centroids ==="
python ingestion/environmental/ingest_env_rasters.py --skip-download --sample-points

echo ""
echo "=== 9. Run QA checks ==="
python qa/run_checks.py || echo "Some QA checks failed (see above)"

echo ""
echo "Ingestion complete."
