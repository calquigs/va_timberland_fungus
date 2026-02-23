# ADR-004: Docker Compose for Local Dev; Same Images for Cloud

## Status

Accepted.

## Context

We need a consistent way to run PostGIS, ingestion jobs, dbt, ML service, and web app locally and in the cloud.

## Decision

- **Docker Compose** for local development: one compose file (or override) that brings up PostGIS, object store (e.g. MinIO), ML API, web app, and optional ingestion/worker containers.
- **Same Docker images** used in cloud (e.g. ECS, GKE, or Cloud Run): build once, push to registry; orchestration (Compose vs Kubernetes) is the only difference.
- **Secrets and config**: Use env files (e.g. `.env`) for local; cloud uses managed secrets and env config.

## Consequences

- **Pros**: “Runs on my machine” matches “runs in cloud”; portfolio demonstrates containerization and deployability.
- **Cons**: Compose is not production orchestration; for production we’d add a second compose or K8s manifests.
- **Mitigation**: Document “local: docker-compose up”; “cloud: use these images with your orchestrator.”
