-- Staging: FIA BIGMAP Loblolly biomass.
-- Source: raw.fia_loblolly_biomass (populated by ingest_fia_loblolly.py if vector load used).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    id,
    geom,
    biomass_loblolly,
    year_or_period
from {{ source('raw', 'fia_loblolly_biomass') }}
where geom is not null
