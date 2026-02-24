# Distributing VA Woods

This project is designed to **run locally**. Clone the repo and follow the README.

## How evaluators access the app

1. **Docker (recommended):** `docker compose up -d postgres && docker compose up api viz` brings up everything. Map at http://localhost:8502, API at http://localhost:8000.
2. **Python venv (viz only):** Install `viz/requirements.txt`, set `DATABASE_URL`, and run `streamlit run app.py` from `viz/`. Requires a running Postgres instance.

## Data

Rasters and FIA data are large and **not stored in the repo**. Ingestion scripts (`ingestion/`) download and load them into PostGIS. Without running ingestion + dbt, the UI starts but parcel layers and derived tables will be empty.

## Environment variables

See `.env.example` at the repo root for the defaults. The app works out of the box with `docker compose`; override variables only if you change ports or run services on different hosts.
