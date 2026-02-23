# Docker

- **docker-compose.yml** at repo root: PostGIS, MinIO (optional), ML API, web app. Use for local dev.
- **Dockerfiles** per service: build images for cloud (ECS, GKE, Cloud Run).

## Local

From repo root:

```bash
docker compose up -d postgres minio   # start DB and object store
# Run ingestion and dbt from host or one-off containers
docker compose up ml viz              # ML API + Streamlit map
```

## Images

- `ml`: Dockerfile in `ml/` — FastAPI app.
- `viz`: Dockerfile in `viz/` — Streamlit map (parcels, harvests).
- PostGIS: use official `postgis/postgis` image.
