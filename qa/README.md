# QA/QC Layer

Automated quality assurance and quality control for VA Woods data.

## Approaches

1. **dbt tests** (in `../transform/`): Uniqueness, not_null, relationships, custom SQL. Run with `dbt test`.
2. **Great Expectations (optional)** in this folder: Expectation suites for row counts, value ranges, spatial validity. Run after ingestion or after dbt.
3. **Custom checks**: Scripts that validate e.g. coordinate bounds (VA extent), biomass value ranges, OTU table schema.

## Running

- After ingestion: run `qa/run_checks.py` (or `great_expectations checkpoint run ...`) to validate raw/staging.
- After dbt: run `dbt test` from `transform/`.

## Expectations (when using Great Expectations)

- Parcels: geometry within Virginia bbox; required columns present.
- FIA: biomass >= 0; geometry not null.
- OTU sites: lat/lon in valid range; site_id unique.
- Env at points: elevation/temp/precip in reasonable ranges.
