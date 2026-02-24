# VA Woods

A **data engineering portfolio project** for predicting soil fungal communities to support inoculation of depleted forest lands in Virginia. The pipeline ingests spatial data (Virginia parcels, USFS FIA Loblolly biomass, environmental rasters), classifies parcels (wild vs timber), and trains a model on user-uploaded OTU tables + environmental variables. A web app lets you click the map to get a predicted fungal community at any point.

## Goals

- **Data engineering**: Extract → Transform (SQL + Python + dbt) → QA/QC → Serve.
- **Spatial + ML**: Parcels, FIA overlay, env vars; OTU uploads; manual retrain; predict at (lat, lon).
- **Local-first**: Docker Compose to run the full stack on your machine.

## For evaluators (quick access)

**Prerequisites:** Docker & Docker Compose, Python 3.10+, Git.
On Linux you may also need system GDAL (`sudo apt install libgdal-dev`); macOS and most pip binary wheels include it automatically.

### Full setup (data + app) — one script

```bash
git clone https://github.com/calquigs/va_timberland_fungus.git && cd va_timberland_fungus
./scripts/setup.sh
```

This starts Postgres, downloads and ingests all data sources, runs dbt, and launches the app. The first run downloads ~5 GB of raster data and takes 15–30 minutes; subsequent runs are fast (data is cached locally).

Once running, open: **Map** http://localhost:8502 · **API** http://localhost:8000

### Quick look (app only, empty DB)

```bash
docker compose up -d postgres && docker compose up api viz
```

The app starts immediately but map layers will be empty until ingestion and dbt are run.

## Repository layout

| Path | Purpose |
|------|--------|
| **docs/** | [Architecture](docs/ARCHITECTURE.md), [ADRs](docs/adr/), [data sources](docs/DATA_SOURCES.md) |
| **ingestion/** | Scripts: VA parcels, FIA BIGMAP, environmental rasters; OTU upload is in the API |
| **transform/** | dbt: staging + marts (parcels classified, env at OTU sites) |
| **qa/** | Automated QA/QC (dbt tests + custom Python checks) |
| **api/** | FastAPI + Dockerfile: OTU upload, model training, predictions, env coverage |
| **viz/** | Streamlit + Dockerfile: map (Positron), VA parcels & harvest layers |
| **scripts/** | `setup.sh` (full pipeline), `ingest_va_landuse.sh` (ingestion only) |
| **docker-compose.yml** | Orchestrates PostGIS, MinIO, Martin, API, and viz services |

## Step-by-step (manual)

If you prefer to run each stage individually:

1. **Start Postgres**
   ```bash
   docker compose up -d postgres minio
   ```

2. **Ingest data** (downloads from VA, USFS, and PRISM APIs; ~15–30 min first run)
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r ingestion/requirements.txt
   ./scripts/ingest_va_landuse.sh
   ```

3. **Transform (dbt)**
   ```bash
   pip install dbt-postgres
   export DBT_HOST=localhost DBT_USER=va_woods DBT_PASSWORD=va_woods_dev DBT_DBNAME=va_woods
   cd transform && dbt run && dbt test && cd ..
   ```

4. **Launch the app**
   ```bash
   docker compose up api viz
   ```
   - Map: http://localhost:8502
   - API: http://localhost:8000

## Architecture decisions

- **ADR-001**: PostGIS as central spatial store.  
- **ADR-002**: dbt for SQL transformations and testing.  
- **ADR-003**: Separate API service with manual retrain trigger.  
- **ADR-004**: Docker Compose for local development.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/adr/](docs/adr/) for details.

## Data sources

- **Virginia parcels**: State GIS / open data → wild vs timber classification.  
- **FIA BIGMAP**: Loblolly Pine biomass.  
- **Environmental**: Elevation, slope, precipitation, temperature (e.g. DEM + PRISM).  
- **OTU tables**: User uploads (CSV/TSV + lat/lon) via API.

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## License

MIT — see [LICENSE](LICENSE). Data sourced from Virginia state agencies, USFS FIA, and PRISM is subject to the respective providers’ terms of use.
