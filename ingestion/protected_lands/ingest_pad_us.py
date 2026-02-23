"""
Ingest PAD-US (Protected Areas Database of the United States) for Virginia.

Downloads Virginia protected lands from the PAD-US ArcGIS Feature Service,
filtered to forested / relevant GAP status codes, and loads into PostGIS.

Source: USGS PAD-US 4.0 hosted Feature Service
  https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/PADUS4_0Fee/FeatureServer/0

Output: raw.protected_lands (geometry, designation, gap_status, area_acres, etc.)

Usage:
  python ingest_pad_us.py                          # download + load
  python ingest_pad_us.py --gap-status 1,2,3       # filter GAP status (default 1,2,3)
  python ingest_pad_us.py --limit 1000             # limit for testing
"""

import argparse
import json
import logging
import os
from urllib.parse import urlencode

import geopandas as gpd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PADUS_FEATURE_SERVER = (
    "https://services.arcgis.com/9Dz5quHPmab2pMtb/arcgis/rest/services"
    "/PADUS4OpenToPublic/FeatureServer/1"
)
BATCH_SIZE = 2000

VA_BBOX = {
    "xmin": -83.7, "ymin": 36.5,
    "xmax": -75.2, "ymax": 39.6,
    "spatialReference": {"wkid": 4326},
}


def fetch_protected_lands(gap_codes: list[int], limit: int | None = None) -> gpd.GeoDataFrame:
    """Fetch PAD-US features for Virginia, filtered by GAP status."""
    gap_filter = ",".join(f"'{g}'" for g in gap_codes)
    where = f"GAP_Sts IN ({gap_filter})"

    all_features = []
    offset = 0

    while True:
        record_count = BATCH_SIZE
        if limit is not None:
            remaining = limit - len(all_features)
            if remaining <= 0:
                break
            record_count = min(BATCH_SIZE, remaining)

        params = {
            "where": where,
            "outFields": "FeatClass,Mang_Name,Loc_Ds,Des_Tp,GAP_Sts,GIS_Acres,State_Nm,Unit_Nm",
            "returnGeometry": "true",
            "outSR": 4326,
            "geometry": json.dumps(VA_BBOX),
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": record_count,
            "f": "geojson",
        }
        url = f"{PADUS_FEATURE_SERVER}/query?{urlencode(params)}"
        logger.info("Fetching PAD-US offset=%d (have %d features)", offset, len(all_features))

        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                logger.warning("HTTP %d at offset %d: %s", resp.status_code, offset, resp.text[:500])
                if all_features:
                    break
                resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Request failed at offset %d: %s", offset, e)
            if all_features:
                break
            raise

        if "error" in data:
            logger.error("ArcGIS error: %s", data["error"])
            if all_features:
                break
            raise RuntimeError(f"ArcGIS error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        if len(features) < record_count:
            break
        offset += len(features)

    if not all_features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": all_features},
        crs="EPSG:4326",
    )
    logger.info("Fetched %d protected land features", len(gdf))
    return gdf


def load_to_postgis(gdf: gpd.GeoDataFrame, engine, schema: str = "raw") -> int:
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    col_map = {
        "FeatClass": "feature_class",
        "Mang_Name": "manager_name",
        "Loc_Ds": "local_designation",
        "Des_Tp": "designation_type",
        "GAP_Sts": "gap_status",
        "GIS_Acres": "area_acres",
        "State_Nm": "state",
        "Unit_Nm": "unit_name",
        "geometry": "geom",
    }
    rename = {k: v for k, v in col_map.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename)
    if "geom" in gdf.columns:
        gdf = gdf.set_geometry("geom")

    keep_cols = [v for v in col_map.values() if v in gdf.columns]
    gdf = gdf[keep_cols].copy()

    gdf.to_postgis(
        "protected_lands",
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=500,
    )
    return len(gdf)


def main():
    parser = argparse.ArgumentParser(description="Ingest PAD-US protected lands for Virginia")
    parser.add_argument("--schema", default="raw")
    parser.add_argument("--gap-status", default="1,2,3",
                        help="Comma-separated GAP status codes to include (default: 1,2,3)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL is required")
        raise SystemExit(1)

    engine = create_engine(database_url)
    gap_codes = [int(x.strip()) for x in args.gap_status.split(",")]

    gdf = fetch_protected_lands(gap_codes, limit=args.limit)
    if gdf.empty:
        logger.warning("No protected land features fetched")
        return

    n = load_to_postgis(gdf, engine, args.schema)
    logger.info("Loaded %d protected land features to %s.protected_lands", n, args.schema)


if __name__ == "__main__":
    main()
