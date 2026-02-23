-- Staging: VA parcels with consistent types and geometry.
-- Source: raw.va_parcels (populated by ingestion).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    "OBJECTID" as id,
    geom,
    "PARCELID" as parcel_id,
    "FIPS" as county_fips,
    "LOCALITY" as locality,
    "Shape__Area" / 4046.86 as area_acres,
    -- land_use from parcel not available; use va_harvested_timber_parcels for timber classification
    null::text as land_use_code,
    null::text as land_use_desc
from {{ source('raw', 'va_parcels') }}
where geom is not null
