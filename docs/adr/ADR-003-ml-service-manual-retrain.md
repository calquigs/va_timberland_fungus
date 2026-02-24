# ADR-003: Separate ML Service with Manual Retrain Trigger

## Status

Accepted.

## Context

We need to train a model (OTU-derived community vs environmental variables) and serve predictions at arbitrary (lat, lon). Training should incorporate new OTU uploads; we want control over when retraining runs.

## Decision

- **Separate API service** (FastAPI): endpoints for `POST /train` (manual trigger), `GET /predict?lat=&lon=`, and optionally `GET /model/status`.
- **Manual retrain**: No automatic retraining on upload; user or operator triggers retrain (e.g. button in app or CLI). Training reads: OTU samples + env features from DB (or exported snapshot), trains model, saves artifact (e.g. joblib) to object store or mounted volume.
- **Prediction**: Load model; for given (lat, lon), fetch or compute env features (from API or DB), run model, return predicted community (e.g. composition or diversity metrics).

## Consequences

- **Pros**: Reproducible training; no surprise resource spikes; easy to version and re-run; clear separation from ingestion/transform.
- **Cons**: Stale model if user forgets to retrain; need to document “retrain after N new OTU uploads.”
- **Mitigation**: Web app shows “last trained at” and a “Retrain model” button; optional CI job to retrain on schedule if desired later.
