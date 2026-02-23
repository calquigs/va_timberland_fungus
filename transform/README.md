# Transform Layer (dbt)

SQL transformations for VA Woods. Requires PostGIS and populated raw/staging tables from ingestion.

## Setup

```bash
cd transform
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install dbt-postgres
export DBT_HOST=localhost DBT_USER=va_woods DBT_PASSWORD=... DBT_DBNAME=va_woods
dbt deps   # if using packages
dbt run
dbt test
```

## Models

- **Staging**: `stg_va_parcels`, `stg_fia_loblolly` — clean and type raw data.
- **Marts**: `parcels_classified` (wild vs timber), `env_at_otu_sites` — ready for API and ML.

Classification rules in `parcels_classified` depend on Virginia land-use codes; update the `case` expression when you have the actual code list.
