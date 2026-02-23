-- Mart: Environmental variables at OTU sample sites.
-- Joins env data with OTU site metadata for ML training.

{{
  config(
    materialized='table',
    schema='marts'
  )
}}

select
    s.site_id,
    s.lat,
    s.lon,
    e.elevation_m,
    e.slope_deg,
    e.tmax_c,
    e.tmin_c,
    e.precip_mm_annual,
    e.temp_c_annual
from {{ ref('stg_otu_sites') }} s
inner join {{ ref('env_at_points') }} e
    on s.site_id::text = e.site_id
    and e.point_type = 'otu_site'
