-- Staging: OTU sample sites with consistent types.
-- Source: raw.otu_sites (populated by ML API /upload/otu).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    site_id,
    lat,
    lon,
    uploaded_at,
    file_path,
    ST_SetSRID(ST_MakePoint(lon, lat), 4326) as geom
from {{ source('raw', 'otu_sites') }}
where lat is not null and lon is not null
