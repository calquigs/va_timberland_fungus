"""
Ingest Virginia timber harvest notification points into PostGIS.

Data source: VDOF IFRIS VOF_Harvests FeatureServer (VOF cooperator harvests).
Geometry is returned in NAD83 Virginia Lambert (EPSG:3968) and transformed to
WGS84 (EPSG:4326) for storage. Output: raw.va_harvests with point geometry and
harvest attributes.

Usage:
  Set DATABASE_URL, then:
  python ingest_va_harvests.py [--schema raw]
"""

import argparse
import logging
import os
from urllib.parse import urlencode

import geopandas as gpd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VDOF IFRIS VOF Harvests ArcGIS Feature Server
VOF_HARVESTS_FEATURE_SERVER = (
    "https://www.ifris.dof.virginia.gov/arcgis109/rest/services"
    "/Cooperators/VOF_Harvests/FeatureServer/0"
)
BATCH_SIZE = 1000  # maxRecordCount for this service

# IFRIS returns geometry in NAD83 Virginia Lambert (EPSG:3968); we store WGS84 (4326).
SOURCE_CRS = "EPSG:3968"
TARGET_CRS = "EPSG:4326"


def fetch_harvests_from_arcgis(where: str = "1=1", limit=None) -> gpd.GeoDataFrame:
    """Fetch harvest points from VOF FeatureServer via paginated queries."""
    all_features = []
    offset = 0

    while True:
        record_count = (
            min(BATCH_SIZE, limit - offset)
            if limit is not None and (limit - offset) > 0
            else BATCH_SIZE
        )
        if limit is not None and offset >= limit:
            break

        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            # Request native SR (3968); we transform to 4326 below for storage
            "outSR": 3968,
            "resultOffset": offset,
            "resultRecordCount": record_count,
            "f": "geojson",
        }
        url = f"{VOF_HARVESTS_FEATURE_SERVER}/query?{urlencode(params)}"
        logger.info("Fetching harvests offset=%d", offset)

        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"ArcGIS error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        if len(features) < record_count or (limit is not None and len(all_features) >= limit):
            break
        offset += len(features)

    if not all_features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": all_features},
        crs=SOURCE_CRS,
    )
    gdf = gdf.to_crs(TARGET_CRS)
    logger.info("Fetched %d harvest points (3968 → 4326)", len(gdf))
    return gdf


def load_harvests_to_postgis(
    gdf: gpd.GeoDataFrame, engine, schema: str, table: str
) -> int:
    """Load GeoDataFrame to PostGIS."""
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    gdf = gdf.rename(columns={"geometry": "geom"}).set_geometry("geom")
    # Drop rows with null/invalid geometry (causes TypeError in to_postgis)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
    gdf.to_postgis(
        table,
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=500,
    )
    return len(gdf)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest VA timber harvest points into PostGIS"
    )
    parser.add_argument("--schema", default="raw", help="Target schema (raw or staging)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of harvest points to fetch (for testing; default=all)",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL is required")
        raise SystemExit(1)

    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(
            "Cannot connect to database. Is PostGIS running? Start with: docker compose up -d"
        )
        logger.error("Connection error: %s", e)
        raise SystemExit(1)
    gdf = fetch_harvests_from_arcgis(limit=args.limit)

    if gdf.empty:
        logger.warning("No harvest points fetched")
        return

    n = load_harvests_to_postgis(gdf, engine, args.schema, "va_harvests")
    logger.info("Loaded %d harvest points to %s.va_harvests", n, args.schema)


if __name__ == "__main__":
    main()
