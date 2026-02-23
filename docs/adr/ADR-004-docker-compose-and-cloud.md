# ADR-004: Docker Compose for Local Development

## Status

Accepted.

## Context

We need a consistent way to run PostGIS, ingestion jobs, dbt, ML service, and web app locally.

## Decision

- **Docker Compose** for local development: one compose file that brings up PostGIS, object store (MinIO), tile server (Martin), ML API, and Streamlit viz.
- **Secrets and config**: Use `.env` file or environment variables; defaults work out of the box (see `.env.example`).

## Consequences

- **Pros**: One-command startup (`docker compose up`); consistent environments; evaluators can run the full stack without installing dependencies individually.
- **Cons**: Requires Docker and Docker Compose installed locally.
- **Mitigation**: Document prerequisites in README; provide a non-Docker path for viz (pip + streamlit run).
