-- Mart: Parcels classified as "wild" (e.g. state forest) vs "actively harvested timberland".
-- Classification rules depend on VA data dictionary; adjust column names and logic.

{{
  config(
    materialized='table',
    schema='marts'
  )
}}

with staged as (
    select * from {{ ref('stg_va_parcels') }}
),
harvested_ids as (
    select distinct "OBJECTID" as parcel_objectid
    from {{ source('staging', 'va_harvested_timber_parcels') }}
),
classified as (
    select
        s.id,
        s.geom,
        s.parcel_id,
        s.land_use_code,
        s.land_use_desc,
        s.county_fips,
        s.area_acres,
        case
            when s.land_use_code in ('STATE_FOREST', 'WILD') then 'wild'
            when h.parcel_objectid is not null then 'timber_harvested'
            when s.land_use_code in ('TIMBER', 'HARVESTED') then 'timber_harvested'
            else 'other'
        end as parcel_type
    from staged s
    left join harvested_ids h on s.id = h.parcel_objectid
)
select * from classified
