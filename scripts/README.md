# Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Full pipeline: start Postgres, create venv, ingest all data, run dbt, launch app. |
| `ingest_va_landuse.sh` | Ingestion only: harvests, parcels, FIA biomass, protected lands, env rasters. Expects Postgres running and `DATABASE_URL` set. |
