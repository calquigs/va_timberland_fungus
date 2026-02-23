-- Mart: Min/max of each environmental variable across all OTU sample sites.
-- Used to assess parameter space coverage.

{{
  config(
    materialized='table',
    schema='marts'
  )
}}

select
    'elevation_m' as variable,
    min(elevation_m) as sampled_min,
    max(elevation_m) as sampled_max,
    avg(elevation_m) as sampled_mean,
    count(*) as n_sites
from {{ ref('env_at_otu_sites') }}
where elevation_m is not null

union all

select
    'slope_deg',
    min(slope_deg), max(slope_deg), avg(slope_deg), count(*)
from {{ ref('env_at_otu_sites') }}
where slope_deg is not null

union all

select
    'tmax_c',
    min(tmax_c), max(tmax_c), avg(tmax_c), count(*)
from {{ ref('env_at_otu_sites') }}
where tmax_c is not null

union all

select
    'tmin_c',
    min(tmin_c), max(tmin_c), avg(tmin_c), count(*)
from {{ ref('env_at_otu_sites') }}
where tmin_c is not null

union all

select
    'precip_mm_annual',
    min(precip_mm_annual), max(precip_mm_annual), avg(precip_mm_annual), count(*)
from {{ ref('env_at_otu_sites') }}
where precip_mm_annual is not null
