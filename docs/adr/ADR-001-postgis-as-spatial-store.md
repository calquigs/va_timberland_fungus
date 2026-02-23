# ADR-001: PostGIS as Central Spatial Store

## Status

Accepted.

## Context

We need to store and query Virginia parcels, FIA biomass (vector or raster-derived), environmental point samples, and OTU sample locations. Options include: shapefiles only, GeoPackage, PostGIS, or a mix of file-based + database.

## Decision

Use **PostGIS** (PostgreSQL with PostGIS extension) as the primary store for all vector spatial data and for derived tables (parcel classifications, FIA summaries, env-at-points). Use an object store (e.g. S3) for raw rasters and large source files; load vectorized or sampled results into PostGIS.

## Consequences

- **Pros**: Single source of truth for geometry; SQL + spatial indexes; dbt can run against it; easy to serve GeoJSON from API; same DB can store OTU metadata and training feature tables.
- **Cons**: Operational overhead (backups, migrations); need to load rasters into PostGIS or sample in Python and store points.
- **Mitigation**: Use Docker for local PostGIS; document backup/restore for cloud.
