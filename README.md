# VA Woods

A **data engineering portfolio project** for predicting soil fungal communities to support inoculation of depleted forest lands in Virginia. The pipeline ingests spatial data (Virginia parcels, USFS FIA Loblolly biomass, environmental rasters), classifies parcels (wild vs timber), and trains a model on user-uploaded OTU tables + environmental variables. A web app lets you click the map to get a predicted fungal community at any point.

## Goals

- **Data engineering**: Extract → Transform (SQL + Python + dbt) → QA/QC → Serve.
- **Spatial + ML**: Parcels, FIA overlay, env vars; OTU uploads; manual retrain; predict at (lat, lon).
- **Deployable**: Docker Compose for local dev; same images for cloud.

## For evaluators (quick access)

- **Run the app locally (Docker):**  
  `docker compose up -d postgres && docker compose up ml viz`  
  Then open: **Map** http://localhost:8502 · **API** http://localhost:8000  
  (Ingestion + dbt are optional; without them the UI runs but parcel layers need data—see Quick start below.)
- **Live demo (if hosted):** *(After deploying to Railway, add your Viz URL here, e.g. https://viz-production-xxxx.up.railway.app)*
- **Deploy to Railway (one shareable link):** See [docs/RAILWAY.md](docs/RAILWAY.md) for step-by-step instructions.
- **Other deployment options:** See [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) for GitHub-only vs other cloud hosting.

## Repository layout

| Path | Purpose |
|------|--------|
| **docs/** | [Architecture](docs/ARCHITECTURE.md), [ADRs](docs/adr/), [data sources](docs/DATA_SOURCES.md) |
| **ingestion/** | Scripts: VA parcels, FIA BIGMAP, environmental rasters; OTU upload is in ML API |
| **transform/** | dbt: staging + marts (parcels classified, env at OTU sites) |
| **qa/** | Automated QA/QC (dbt tests + optional Great Expectations) |
| **ml/** | FastAPI: OTU upload, manual train, predict at (lat, lon) |
| **viz/** | Streamlit: map (Positron), VA parcels & harvest layers |
| **docker/** | Dockerfiles and Compose for PostGIS, MinIO, ML, viz |

## Quick start

1. **Start backend and DB**
   ```bash
   docker compose up -d postgres minio
   export DATABASE_URL=postgresql://va_woods:va_woods_dev@localhost:5432/va_woods
   ```

2. **Ingestion** (implement and run when data sources are configured)
   ```bash
   cd ingestion && pip install -r requirements.txt
   python va_parcels/ingest_va_parcels.py
   python fia_bigmap/ingest_fia_loblolly.py
   python environmental/ingest_env_rasters.py
   ```

3. **Transform**
   ```bash
   cd transform && pip install dbt-postgres
   export DBT_HOST=localhost DBT_USER=va_woods DBT_PASSWORD=va_woods_dev DBT_DBNAME=va_woods
   dbt run && dbt test
   ```

4. **ML API + Map viz**
   ```bash
   docker compose up ml viz
   ```
   - API: http://localhost:8000  
   - Map: http://localhost:8502 (Streamlit)

   Or run the viz locally (no Docker):
   ```bash
   cd viz && pip install -r requirements.txt && export DATABASE_URL=postgresql://va_woods:va_woods_dev@localhost:5432/va_woods && streamlit run app.py
   ```  

## Architecture decisions

- **ADR-001**: PostGIS as central spatial store.  
- **ADR-002**: dbt for SQL transformations and testing.  
- **ADR-003**: Separate ML service with manual retrain trigger.  
- **ADR-004**: Docker Compose locally; same images for cloud.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/adr/](docs/adr/) for details.

## Data sources

- **Virginia parcels**: State GIS / open data → wild vs timber classification.  
- **FIA BIGMAP**: Loblolly Pine biomass.  
- **Environmental**: Elevation, slope, precipitation, temperature (e.g. DEM + PRISM).  
- **OTU tables**: User uploads (CSV/TSV + lat/lon) via ML API.

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## License

Portfolio / educational use. Respect data providers’ terms (Virginia, USFS, PRISM, etc.).
