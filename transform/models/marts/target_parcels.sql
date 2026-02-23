-- Mart: Harvested timber parcels enriched with loblolly biomass and target status.
-- Primary mart for the Target view (Phase 2).

{{
  config(
    materialized='table',
    schema='marts'
  )
}}

select
    hp."OBJECTID",
    hp.geom,
    hp."PARCELID",
    hp."FIPS",
    hp."LOCALITY",
    hp."Shape__Area" / 4046.86 as area_acres,
    hp.harvest_number,
    hp.harvest_start_date,
    hp.harvest_status,
    hp.harvest_acres_to_date_1,
    ls.mean_biomass_kg_ha,
    ls.total_biomass_kg,
    ls.pixel_count as biomass_pixel_count,
    tp.marked_at as target_marked_at,
    (tp.parcel_objectid is not null) as is_target
from {{ source('staging', 'va_harvested_timber_parcels') }} hp
left join {{ ref('stg_parcel_loblolly_stats') }} ls
    on hp."OBJECTID" = ls.parcel_objectid
left join {{ source('raw', 'target_parcels') }} tp
    on hp."OBJECTID" = tp.parcel_objectid
