# Docker

- **docker-compose.yml** at repo root: PostGIS, MinIO (optional), API, web app. Use for local dev.
- **Dockerfiles** per service: `api/Dockerfile` and `viz/Dockerfile`.

## Local

From repo root:

```bash
docker compose up -d postgres minio   # start DB and object store
# Run ingestion and dbt from host or one-off containers
docker compose up api viz             # API + Streamlit map
```

## Images

- `api`: Dockerfile in `api/` — FastAPI app.
- `viz`: Dockerfile in `viz/` — Streamlit map (parcels, harvests).
- PostGIS: use official `postgis/postgis` image.
