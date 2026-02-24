# QA/QC Layer

Automated quality assurance and quality control for VA Woods data.

## Approaches

1. **dbt tests** (in `../transform/`): Uniqueness, not_null, relationships, custom SQL. Run with `dbt test`.
2. **Custom checks** (`run_checks.py`): Validate coordinate bounds (VA extent), biomass value ranges, OTU table schema, and row counts.

## Running

- After ingestion: run `python qa/run_checks.py` to validate raw/staging tables.
- After dbt: run `dbt test` from `transform/`.

## What the checks verify

- Parcels: geometry within Virginia bbox; required columns present.
- FIA: biomass >= 0; geometry not null.
- OTU sites: lat/lon in valid range; site_id unique.
- Env at points: elevation/temp/precip in reasonable ranges.
