# API Service

FastAPI backend for VA Woods: OTU upload, model training, predictions, environmental coverage analysis, and parcel management.

## Endpoints

- **POST /upload/otu** — Upload OTU table (CSV/TSV) with lat, lon; store file and metadata; optionally trigger env sampling.
- **POST /train** — Manually trigger retrain: read OTU sites + env from DB, train model, save artifact.
- **GET /predict?lat=&lon=** — Return predicted community for the given point (env fetched from DB or computed).
- **GET /model/status** — Return last trained timestamp and optional metrics.
- **GET /env/coverage** — Assess parameter-space coverage at a point.
- **POST /env/refresh-gap-raster** — Regenerate the sampling-gap raster.
- **POST /parcels/{id}/target** — Mark a parcel as a target.
- **DELETE /parcels/{id}/target** — Unmark a parcel.
- **GET /target-parcels** — Paginated GeoJSON of target parcels.

## Model Design

- **Features**: elevation, slope, precip, temperature (and optionally FIA biomass at point).
- **Target**: OTU-derived community representation (e.g. diversity index, or composition vector).
- **Artifact**: Saved model (joblib) in volume or object store; loaded at startup or on first predict.

## Running

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set `DATABASE_URL` and optional `MODEL_PATH` / `S3_MODEL_BUCKET`.
