# Ingestion Layer

Scripts and small services that pull data from external sources and load them into raw/staging storage (PostGIS + object store).

## Data Sources

| Source | Script / Service | Output |
|--------|------------------|--------|
| Virginia parcels | `va_parcels/` | `raw.va_parcels` |
| Virginia harvests (VOF) | `va_harvests/` | `raw.va_harvests` |
| Harvested timber parcels | `va_parcels/ --harvested-only` | `staging.va_harvested_timber_parcels` |
| FIA BIGMAP Loblolly | `fia_bigmap/ingest_fia_loblolly.py` | Download zip from USDA + extract GeoTIFF; optional `--local-zip` if manual download |
| Environmental (DEM, PRISM) | `environmental/` | Rasters in object store; env-at-point table in PostGIS |
| OTU tables | API in `../api` (POST `/upload/otu`) | Object store + PostGIS metadata |

## Running

- **Local**: Use Python 3.10+ and a virtualenv; set `DATABASE_URL` and optional `S3_ENDPOINT` (MinIO) for local object store.
- **Docker**: Ingestion can run as one-off containers; see `docker-compose.yml` at the repo root.

## Conventions

- Write to `raw` or `staging` schemas; dbt reads from here.
- Log run timestamps and row counts for QA/QC.
- Document download URLs and licenses in `../docs/DATA_SOURCES.md`.
