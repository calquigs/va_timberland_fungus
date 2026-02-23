-- Staging: Environmental variables sampled at point locations.
-- Source: raw.env_point_samples (populated by ingest_env_rasters.py --sample-points
-- and by the ML API on OTU upload).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    point_id as site_id,
    lat,
    lon,
    elevation_m,
    slope_deg,
    precip_mm as precip_mm_annual,
    (coalesce(tmax_c, 0) + coalesce(tmin_c, 0)) / 2.0 as temp_c_annual,
    tmax_c,
    tmin_c,
    point_type
from {{ source('raw', 'env_point_samples') }}
