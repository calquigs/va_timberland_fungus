# VA Woods: Architecture Overview

A data engineering portfolio project for **predicting soil fungal communities** to support microbial inoculation of loblolly pine timberland in Virginia. This document explains the system design and the reasoning behind it.

## Goals

- **Data engineering showcase**: Extract → Transform → QA/QC → Serve, with clear layers and tooling.
- **Spatial + tabular**: Virginia parcels, USFS FIA (Loblolly biomass), environmental rasters, protected lands, and user-uploaded OTU tables.
- **ML in the loop**: Train an XGBoost model on OTU + environment; predict community at any point; manual retrain trigger.
- **Operational**: Docker-based, cloud-deployable, with documented decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                    │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│ VA Parcels   │ FIA BIGMAP   │ Environmental│ Protected    │ OTU Upload API │
│ + Harvests   │ (Loblolly)   │ (elev, slope,│ Lands        │ (fungal OTU    │
│ (state GIS)  │ + Zonal Stats│ precip, temp)│ (PAD-US)     │  + lat/lon)    │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴───────┬────────┘
       │              │              │              │               │
       ▼              ▼              ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAW / STAGING STORAGE                                │
│              PostGIS (spatial) + MinIO (rasters/CSVs)                        │
│  Tables: va_parcels, va_harvests, parcel_loblolly_stats, target_parcels,   │
│          otu_sites, otu_observations, env_point_samples, protected_lands     │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMATION (dbt + Python)                              │
│  SQL: parcels_classified, target_parcels, env_at_otu_sites,                 │
│       sample_parameter_bounds                                                │
│  Python: raster sampling, zonal stats, slope derivation                     │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QA/QC LAYER                                          │
│  dbt tests: unique, not_null on staging models                              │
│  Python: row counts, null geom checks, value range validation               │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┬──────────────────────────────────────────┐
│     ML SERVICE (FastAPI)          │     TILE SERVER (Martin)                  │
│  • OTU upload + storage           │  • Vector tiles (MVT) from PostGIS       │
│  • XGBoost train (on demand)      │  • Auto-discovers spatial tables         │
│  • Predict at (lat, lon)          │  • Parcels, harvests, protected lands    │
│  • Env coverage analysis          │                                          │
│  • Target parcel management       │                                          │
└──────────────────────────────────┴──────────────────────────────────────────┘
       │                                        │
       ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAP VISUALIZATION (Streamlit)                              │
│  Three views:                                                                │
│  • Target  – Rank parcels by loblolly biomass; mark targets                 │
│  • Sample  – Env layers, coverage gaps, OTU upload, protected lands         │
│  • Predict – Train model, click-to-predict fungal community                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | Role | Key tools |
|-----------|------|-----------|
| **Ingestion** | Pull VA parcels, harvests, FIA BIGMAP, PRISM climate, PAD-US; accept OTU uploads | Python, requests, rasterio, rasterstats |
| **Storage** | Raw + staged + modeled data | PostGIS, MinIO |
| **Transform** | Parcel classification, biomass overlay, env at points, parameter bounds | dbt (SQL), Python |
| **QA/QC** | Automated validation | dbt tests, custom Python checks |
| **ML** | Train XGBoost on OTU + env; predict at (lat, lon); env coverage analysis | FastAPI, XGBoost, scikit-learn |
| **Tiles** | Vector tiles for performant map rendering | Martin (MVT from PostGIS) |
| **Map viz** | Three interactive map views | Streamlit, Folium, streamlit-folium |
| **Deploy** | Run everything in containers | Docker Compose |

## Data Flow (End-to-End)

1. **Parcels + Harvests**: Ingest VA parcels and harvest notification points → spatial join → harvested timber parcels.
2. **FIA BIGMAP**: Download Loblolly biomass raster → compute zonal statistics per parcel → rank parcels by biomass.
3. **Environmental**: Download elevation (Copernicus DEM), derive slope, download PRISM climate normals → clip to VA → sample at OTU sites and parcel centroids.
4. **Protected Lands**: Download PAD-US for Virginia → filter by GAP status → display as overlay.
5. **OTU Upload**: User uploads OTU table (CSV) + lat/lon → stored in otu_sites + otu_observations → env sampled at point.
6. **Model Training**: XGBoost MultiOutputRegressor trained on env features → OTU composition. Manual trigger via API.
7. **Prediction**: Click on map → sample env at point → predict OTU composition → display with confidence.
8. **Coverage Analysis**: Compare env at any point against sampled parameter space → identify gaps for targeted sampling.

## Database Schema

### Raw Layer
- `raw.va_parcels` – Virginia parcel boundaries (VGIN)
- `raw.va_harvests` – Timber harvest notification points (VDOF IFRIS)
- `raw.parcel_loblolly_stats` – Per-parcel biomass zonal statistics
- `raw.target_parcels` – User-marked target parcels
- `raw.otu_sites` – OTU sample site metadata
- `raw.otu_observations` – OTU abundance data per site
- `raw.env_point_samples` – Environmental variables at sample points
- `raw.protected_lands` – PAD-US protected areas
- `raw.model_runs` – Model training run metadata

### Staging Layer (dbt views)
- `staging.va_harvested_timber_parcels` – Parcels × harvests spatial join
- `staging.stg_va_parcels` – Cleaned parcel attributes
- `staging.stg_parcel_loblolly_stats` – Biomass stats
- `staging.stg_otu_sites` – OTU sites with geometry
- `staging.stg_otu_observations` – Non-zero OTU observations
- `staging.env_at_points` – Environmental data at all points

### Marts Layer (dbt tables)
- `marts.target_parcels` – Harvested parcels + biomass + target flag
- `marts.parcels_classified` – All parcels classified (wild/timber/other)
- `marts.env_at_otu_sites` – Environmental data joined to OTU sites
- `marts.sample_parameter_bounds` – Min/max env per variable across OTU sites

## Design Decisions (ADRs)

See `docs/adr/` for:

- **ADR-001**: PostGIS as central spatial store
- **ADR-002**: dbt for SQL transformations and testing
- **ADR-003**: Separate ML service with manual retrain trigger
- **ADR-004**: Docker Compose for local dev; same images for cloud

## Repository Layout

```
va_woods/
├── docs/                 # Architecture, ADRs, data sources
├── ingestion/            # Scripts per data source
│   ├── va_parcels/       #   VA parcel boundaries
│   ├── va_harvests/      #   Timber harvest points
│   ├── fia_bigmap/       #   Loblolly biomass raster
│   ├── environmental/    #   Elevation, slope, climate
│   └── protected_lands/  #   PAD-US protected areas
├── transform/            # dbt project
│   ├── models/staging/   #   Staging views
│   ├── models/marts/     #   Mart tables
│   └── macros/           #   SQL macros (simplify_geom)
├── qa/                   # QA/QC checks
├── ml/                   # FastAPI ML service
│   └── app/main.py       #   All API endpoints
├── viz/                  # Streamlit multipage app
│   ├── app.py            #   Entry point
│   ├── pages/            #   Target, Sample, Predict
│   └── lib/              #   Shared utilities
├── scripts/              # Orchestration scripts
├── docker-compose.yml    # PostGIS, MinIO, Martin, ML, Viz
└── README.md
```
