-- Staging: OTU observations with consistent types.
-- Source: raw.otu_observations (populated by ML API /upload/otu).

{{
  config(
    materialized='view',
    schema='staging'
  )
}}

select
    id,
    site_id,
    otu_id,
    abundance
from {{ source('raw', 'otu_observations') }}
where abundance > 0
