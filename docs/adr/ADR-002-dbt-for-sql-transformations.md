# ADR-002: dbt for SQL Transformations and Testing

## Status

Accepted.

## Context

We need repeatable, testable SQL layers: parcel classification (wild vs timber), FIA biomass aggregation, and joining environmental values to points. Options: raw SQL scripts, Airflow SQL, dbt, or Spark.

## Decision

Use **dbt** for all SQL transformations that run against PostGIS/PostgreSQL. Use dbt for:
- Staging models (raw → cleaned, typed).
- Intermediate models (e.g. parcel with land-use flags).
- Marts (e.g. `parcels_classified`, `fia_biomass_by_geometry`, `env_at_otu_sites`).
- Tests: uniqueness, not null, relationships, and custom checks (e.g. bounds, value ranges).

## Consequences

- **Pros**: Documented DAG, versioned SQL, built-in testing, easy to run in CI or from Docker.
- **Cons**: dbt is SQL-centric; raster sampling or complex geometry may stay in Python and write to tables dbt then reads.
- **Mitigation**: Keep “heavy” spatial or raster logic in ingestion/transform Python; dbt only sees tables/views.
