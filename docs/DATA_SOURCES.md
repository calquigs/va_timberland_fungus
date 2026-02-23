# Data Sources

Reference for all external and internal data used in VA Woods.

## Virginia Parcels

- **Purpose**: Identify actively harvested timberland parcels.
- **Source**: [VGIN VA Parcels](https://vginmaps.vdem.virginia.gov/arcgis/rest/services/VA_Base_Layers/VA_Parcels/FeatureServer/0) via ArcGIS Feature Service.
- **Format**: GeoJSON via REST API; ingest into PostGIS `raw.va_parcels`.
- **Ingestion**: `python ingestion/va_parcels/ingest_va_parcels.py --parcels-containing-harvests`

## Virginia Timber Harvests (VOF)

- **Purpose**: Harvest notification points to identify actively harvested timber parcels via spatial join.
- **Source**: [VDOF IFRIS VOF_Harvests](https://www.ifris.dof.virginia.gov/arcgis109/rest/services/Cooperators/VOF_Harvests/FeatureServer) (VOF cooperator harvests).
- **Format**: GeoJSON via REST API; geometry transformed from NAD83 Virginia Lambert (EPSG:3968) to WGS84 (EPSG:4326).
- **Ingestion**: `python ingestion/va_harvests/ingest_va_harvests.py`

## USFS FIA BIGMAP – Loblolly Pine Biomass

- **Purpose**: Quantify loblolly pine biomass per parcel for target ranking.
- **Source**: [FIA BIGMAP](https://data.fs.usda.gov/geodata/rastergateway/bigmap/index.php) (Loblolly Pine SPCD 0131, 2018 AGB).
- **Format**: GeoTIFF (30m CONUS). Downloaded and extracted to `data/fia_bigmap/`.
- **Processing**: Zonal statistics computed per parcel (mean and total biomass) via `rasterstats`.
- **Ingestion**: `python ingestion/fia_bigmap/ingest_fia_loblolly.py --zonal-stats`

## Environmental Variables

Used as model features and for sampling coverage analysis.

| Variable | Source | Resolution | Output |
|----------|--------|------------|--------|
| Elevation | Copernicus DEM GLO-90 (AWS Open Data) | 90m | `data/environmental/elevation_va.tif` |
| Slope | Derived from elevation via `numpy.gradient` | 90m | `data/environmental/slope_va.tif` |
| Annual High Temp | PRISM 30-yr normals (tmax) | 4km | `data/environmental/tmax_va.tif` |
| Annual Low Temp | PRISM 30-yr normals (tmin) | 4km | `data/environmental/tmin_va.tif` |
| Annual Precip | PRISM 30-yr normals (ppt) | 4km | `data/environmental/ppt_va.tif` |

- **Ingestion**: `python ingestion/environmental/ingest_env_rasters.py`
- **Point sampling**: `python ingestion/environmental/ingest_env_rasters.py --sample-points`
- **Storage**: Clipped GeoTIFFs in `data/environmental/`; point samples in `raw.env_point_samples`.

## Protected Lands (VA DCR Conservation Lands)

- **Purpose**: Display protected woodlands for sampling context (state/national forests, parks, conservation areas).
- **Source**: [VA DCR Conservation Lands Database](https://www.dcr.virginia.gov/natural-heritage/cldownload) — CONSLANDS shapefile (public/private protective management, excludes easements).
- **Format**: Shapefile (zipped). Downloaded, extracted, reprojected to EPSG:4326, and loaded into PostGIS `raw.protected_lands`.
- **Ingestion**: `python ingestion/protected_lands/ingest_dcr_conslands.py`

## OTU Tables (User-Uploaded)

- **Purpose**: Training labels for fungal community model; each upload = one site (lat, lon) + OTU counts.
- **Format**: CSV/TSV with OTU IDs as column headers and abundance values. Columns named `lat`, `lon`, `sample`, `site` are treated as metadata and excluded from OTU parsing.
- **Upload**: `POST /upload/otu?lat=X&lon=Y` with file attachment.
- **Storage**: Site metadata in `raw.otu_sites`; individual OTU abundances in `raw.otu_observations`; environmental values sampled automatically at upload time.
