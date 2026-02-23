-- Staging: Per-parcel loblolly pine biomass zonal statistics.
-- Source: raw.parcel_loblolly_stats (populated by ingest_fia_loblolly.py --zonal-stats).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    parcel_objectid,
    mean_biomass_kg_ha,
    total_biomass_kg,
    pixel_count
from {{ source('raw', 'parcel_loblolly_stats') }}
