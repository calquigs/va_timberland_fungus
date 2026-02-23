"""
Ingest Virginia parcel data into PostGIS.

Data source: VGIN (Virginia Geographic Information Network) via ArcGIS Feature Service.
Output: raw.va_parcels with geometry and attributes.

Usage:
  Set DATABASE_URL, then:
  python ingest_va_parcels.py [--schema raw]

  Fetch only parcels that contain at least one harvest point (recommended):
  python ingest_va_parcels.py --parcels-containing-harvests
  (Requires raw.va_harvests to be populated first.)

  Fetch only parcels in the harvest bounding box (may still be most of the state):
  python ingest_va_parcels.py --only-intersecting-harvests

  Build harvested timber parcels table (spatial join):
  python ingest_va_parcels.py --harvested-only
  (Requires raw.va_parcels and raw.va_harvests to be populated first.)
"""

import argparse
import json
import logging
import os
import time
from urllib.parse import urlencode

import geopandas as gpd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VGIN VA Parcels ArcGIS Feature Service
VA_PARCELS_FEATURE_SERVER = (
    "https://vginmaps.vdem.virginia.gov/arcgis/rest/services"
    "/VA_Base_Layers/VA_Parcels/FeatureServer/0"
)
BATCH_SIZE = 2000  # maxRecordCount for this service
MAX_RETRIES = 3
RETRY_BACKOFF = (5, 10, 20)  # seconds
HARVEST_EXTENT_BUFFER = 0.01  # degrees (~1 km) so edge parcels are included
MULTIPOINT_BATCH_SIZE = 100  # harvest points per VGIN POST query


def get_harvest_points_from_db(engine, limit=None):
    """Return list of (x, y) in WGS84 for each harvest point in raw.va_harvests."""
    sql = text("""
    SELECT ST_X(ST_Transform(ST_SetSRID(geom, 4326), 4326)) AS x,
           ST_Y(ST_Transform(ST_SetSRID(geom, 4326), 4326)) AS y
    FROM raw.va_harvests
    WHERE geom IS NOT NULL
    ORDER BY 1, 2
    """ + (" LIMIT :lim" if limit else ""))
    with engine.connect() as conn:
        r = conn.execute(sql, {"lim": limit} if limit else {})
        rows = r.fetchall()
    return [(float(row[0]), float(row[1])) for row in rows if row[0] is not None]


def get_harvest_extent_from_db(engine, buffer_degrees=HARVEST_EXTENT_BUFFER):
    """Return (xmin, ymin, xmax, ymax) in WGS84 for raw.va_harvests, with optional buffer."""
    sql = text("""
    SELECT
      ST_XMin(ext) - :buf AS xmin,
      ST_YMin(ext) - :buf AS ymin,
      ST_XMax(ext) + :buf AS xmax,
      ST_YMax(ext) + :buf AS ymax
    FROM (
      SELECT ST_Extent(ST_Transform(ST_SetSRID(geom, 4326), 4326)) AS ext
      FROM raw.va_harvests
      WHERE geom IS NOT NULL
        AND ST_X(ST_Transform(ST_SetSRID(geom, 4326), 4326)) BETWEEN -84.0 AND -75.0
        AND ST_Y(ST_Transform(ST_SetSRID(geom, 4326), 4326)) BETWEEN 36.0 AND 40.0
    ) t
    """)
    with engine.connect() as conn:
        r = conn.execute(sql, {"buf": buffer_degrees})
        row = r.fetchone()
    if not row or row[0] is None:
        return None
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def fetch_parcels_containing_points(points, batch_size=MULTIPOINT_BATCH_SIZE) -> gpd.GeoDataFrame:
    """Fetch parcels that contain at least one of the given (x, y) points via batched multipoint queries."""
    if not points:
        return gpd.GeoDataFrame()

    seen_oid = set()
    all_features = []
    n_batches = (len(points) + batch_size - 1) // batch_size

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info("Querying parcels containing harvest points batch %d/%d (%d points)", batch_num, n_batches, len(batch))

        # ArcGIS multipoint: points are [x,y] pairs in longitude, latitude order
        geom = {
            "points": [list(pt) for pt in batch],
            "spatialReference": {"wkid": 4326},
        }
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "geometry": json.dumps(geom),
            "geometryType": "esriGeometryMultipoint",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "resultRecordCount": BATCH_SIZE,
            "f": "geojson",
        }

        query_url = f"{VA_PARCELS_FEATURE_SERVER}/query"
        offset = 0
        batch_failed = False
        while True:
            params["resultOffset"] = offset

            resp = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.post(query_url, data=params, timeout=120)
                    resp.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    if resp is not None and resp.status_code in (500, 502, 503):
                        wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else RETRY_BACKOFF[-1]
                        logger.warning(
                            "Server %s for batch %d offset %d, retry in %ds (%d/%d)",
                            resp.status_code, batch_num, offset, wait, attempt + 1, MAX_RETRIES,
                        )
                        time.sleep(wait)
                    else:
                        raise
            else:
                logger.warning("Skipping batch %d after %d retries", batch_num, MAX_RETRIES)
                batch_failed = True
                break

            if batch_failed:
                break

            data = resp.json()
            if "error" in data:
                logger.warning("ArcGIS error on batch %d: %s — skipping", batch_num, data["error"])
                break

            features = data.get("features", [])
            for f in features:
                oid = f.get("properties", {}).get("OBJECTID") or f.get("properties", {}).get("objectid")
                if oid is not None and oid not in seen_oid:
                    seen_oid.add(oid)
                    all_features.append(f)
            if len(features) < BATCH_SIZE:
                break
            offset += len(features)

    if not all_features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": all_features},
        crs="EPSG:4326",
    )
    logger.info("Fetched %d unique parcels containing harvest points", len(gdf))
    return gdf


def fetch_parcels_from_arcgis(where: str = "1=1", limit=None, envelope=None) -> gpd.GeoDataFrame:
    """Fetch parcels from VGIN ArcGIS Feature Service via paginated queries.

    If envelope is set, only parcels intersecting (xmin, ymin, xmax, ymax) in WGS84 are requested,
    avoiding full-state pagination and server 500s at high offsets.
    """
    all_features = []
    offset = 0

    while True:
        record_count = (
            min(BATCH_SIZE, limit - offset) if limit and (limit - offset) > 0 else BATCH_SIZE
        )
        if limit is not None and offset >= limit:
            break

        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": record_count,
            "f": "geojson",
        }
        if envelope is not None:
            xmin, ymin, xmax, ymax = envelope
            params["geometry"] = f'{{"xmin":{xmin},"ymin":{ymin},"xmax":{xmax},"ymax":{ymax}}}'
            params["geometryType"] = "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = 4326

        url = f"{VA_PARCELS_FEATURE_SERVER}/query?{urlencode(params)}"
        logger.info("Fetching parcels offset=%d", offset)

        resp = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                last_error = None
                break
            except requests.exceptions.HTTPError as e:
                last_error = e
                if resp is not None and resp.status_code in (500, 502, 503):
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF[attempt]
                        logger.warning(
                            "Server error %s at offset %d, retry in %ds (%d/%d)",
                            resp.status_code,
                            offset,
                            wait,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        time.sleep(wait)
                    elif all_features:
                        logger.warning(
                            "Stopping at offset %d after server error; using %d parcels already fetched.",
                            offset,
                            len(all_features),
                        )
                        break
                    else:
                        raise
                else:
                    raise
        else:
            if all_features and last_error:
                logger.warning(
                    "Stopping at offset %d after retries; using %d parcels already fetched.",
                    offset,
                    len(all_features),
                )
                break
            if last_error:
                raise last_error

        if last_error and all_features:
            break

        data = resp.json()
        if "error" in data:
            if all_features:
                logger.warning(
                    "ArcGIS error at offset %d; using %d parcels already fetched: %s",
                    offset,
                    len(all_features),
                    data["error"],
                )
                break
            raise RuntimeError(f"ArcGIS error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        if len(features) < record_count or (limit and len(all_features) >= limit):
            break
        offset += len(features)

    if not all_features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(
        {"type": "FeatureCollection", "features": all_features},
        crs="EPSG:4326",
    )
    logger.info("Fetched %d parcels", len(gdf))
    return gdf


def load_parcels_to_postgis(gdf: gpd.GeoDataFrame, engine, schema: str, table: str) -> int:
    """Load GeoDataFrame to PostGIS. Ensures raw schema exists."""
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    gdf = gdf.rename(columns={"geometry": "geom"}).set_geometry("geom")
    gdf.to_postgis(
        table,
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=1000,
    )
    return len(gdf)


def main():
    parser = argparse.ArgumentParser(description="Ingest VA parcels into PostGIS")
    parser.add_argument("--schema", default="raw", help="Target schema (raw or staging)")
    parser.add_argument(
        "--harvested-only",
        action="store_true",
        help="Build va_harvested_timber_parcels from parcels + harvests (requires both tables)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of parcels to fetch (for testing; default=all)",
    )
    parser.add_argument(
        "--parcels-containing-harvests",
        action="store_true",
        help="Fetch only parcels that contain at least one harvest point (use after ingesting harvests)",
    )
    parser.add_argument(
        "--only-intersecting-harvests",
        action="store_true",
        help="Fetch only parcels in the harvest bounding box (alternative to --parcels-containing-harvests)",
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

    if args.harvested_only:
        _build_harvested_timber_parcels(engine, args.schema)
        return

    envelope = None
    if args.parcels_containing_harvests:
        points = get_harvest_points_from_db(engine, limit=args.limit)
        if not points:
            logger.error("No harvest points in raw.va_harvests. Ingest harvests first.")
            raise SystemExit(1)
        logger.info("Fetching parcels that contain %d harvest points", len(points))
        gdf = fetch_parcels_containing_points(points)
    elif args.only_intersecting_harvests:
        envelope = get_harvest_extent_from_db(engine)
        if envelope is None:
            logger.error("No harvest extent (raw.va_harvests empty or no geometry). Ingest harvests first.")
            raise SystemExit(1)
        logger.info("Harvest extent (buffered): %s", envelope)
        gdf = fetch_parcels_from_arcgis(limit=args.limit, envelope=envelope)
    else:
        gdf = fetch_parcels_from_arcgis(limit=args.limit)
    if gdf.empty:
        logger.warning("No parcels fetched")
        return

    n = load_parcels_to_postgis(gdf, engine, args.schema, "va_parcels")
    logger.info("Loaded %d parcels to %s.va_parcels", n, args.schema)


def _build_harvested_timber_parcels(engine, schema: str) -> None:
    """Build staging.va_harvested_timber_parcels from parcels intersecting harvest points."""
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    # Check how many parcel–harvest overlaps exist (diagnostic)
    intersect_sql = """
    SELECT COUNT(*) FROM raw.va_parcels p
    JOIN raw.va_harvests h ON ST_Intersects(
      COALESCE(ST_SetSRID(p.geom, 4326), p.geom),
      COALESCE(ST_SetSRID(h.geom, 4326), h.geom)
    )
    """
    with engine.connect() as conn:
        try:
            r = conn.execute(text(intersect_sql))
            overlap_count = r.scalar()
        except Exception as e:
            logger.warning("Could not count overlaps (check geom column names): %s", e)
            overlap_count = None

    if overlap_count == 0:
        logger.warning(
            "No parcel–harvest overlaps found. With --limit 5000 the first parcels may not "
            "cover areas where harvests exist. Try a larger limit, e.g. --limit 100000"
        )

    sql = f"""
    DROP TABLE IF EXISTS {schema}.va_harvested_timber_parcels;
    CREATE TABLE {schema}.va_harvested_timber_parcels AS
    SELECT DISTINCT ON (p."OBJECTID")
        p."OBJECTID",
        p."PARCELID",
        p."FIPS",
        p."LOCALITY",
        p."PTM_ID",
        p."Shape__Area",
        p."Shape__Length",
        p.geom,
        h.harvest_number,
        h.harvest_start_date,
        h.harvest_status,
        h.harvest_acres_to_date_1
    FROM raw.va_parcels p
    JOIN raw.va_harvests h ON ST_Intersects(
      COALESCE(ST_SetSRID(p.geom, 4326), p.geom),
      COALESCE(ST_SetSRID(h.geom, 4326), h.geom)
    )
    ORDER BY p."OBJECTID", h.harvest_start_date DESC NULLS LAST;
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()

    with engine.connect() as conn:
        r = conn.execute(
            text(f"SELECT COUNT(*) FROM {schema}.va_harvested_timber_parcels")
        )
        n = r.scalar()
    logger.info("Built %s.va_harvested_timber_parcels with %d parcels", schema, n)


if __name__ == "__main__":
    main()
