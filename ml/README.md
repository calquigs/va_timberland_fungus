# ML Service

Train a model that predicts soil fungal community (from OTU-derived features) using environmental variables at (lat, lon). Serve predictions for the web app.

## Endpoints

- **POST /upload/otu** — Upload OTU table (CSV/TSV) with lat, lon; store file and metadata; optionally trigger env sampling.
- **POST /train** — Manually trigger retrain: read OTU sites + env from DB, train model, save artifact.
- **GET /predict?lat=&lon=** — Return predicted community for the given point (env fetched from DB or computed).
- **GET /model/status** — Return last trained timestamp and optional metrics.

## Model Design

- **Features**: elevation, slope, precip, temperature (and optionally FIA biomass at point).
- **Target**: OTU-derived community representation (e.g. diversity index, or composition vector; exact target TBD).
- **Artifact**: Saved model (e.g. joblib) in volume or object store; loaded at startup or on first predict.

## Running

```bash
cd ml
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set `DATABASE_URL` and optional `MODEL_PATH` / `S3_MODEL_BUCKET`.
