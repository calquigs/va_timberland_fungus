"""
Ingest Virginia Conservation Lands from DCR (Dept. of Conservation and Recreation).

Downloads the CONSLANDS shapefile from the DCR Natural Heritage program and
loads it into PostGIS.

Source: VA DCR Conservation Lands Database
  https://www.dcr.virginia.gov/natural-heritage/cldownload
  https://www.dcr.virginia.gov/natural-heritage/document/conslands.zip

Output: raw.protected_lands (geom, name, manager, designation_type, acres, etc.)

Usage:
  python ingest_dcr_conslands.py                    # download + load
  python ingest_dcr_conslands.py --inspect           # download + print schema only
  python ingest_dcr_conslands.py --keep-download     # don't delete zip after loading
"""

import argparse
import io
import logging
import os
import tempfile
import zipfile

import geopandas as gpd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSLANDS_URL = "https://www.dcr.virginia.gov/natural-heritage/document/conslands.zip"
TARGET_CRS = "EPSG:4326"
TABLE_NAME = "protected_lands"


def download_and_extract(url: str, dest_dir: str) -> str:
    """Download a zip and extract; return the path to the extracted directory."""
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            logger.info("  %.1f MB / %.1f MB", downloaded / 1e6, total / 1e6)

    logger.info("Download complete (%.1f MB)", downloaded / 1e6)

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)
        logger.info("Extracted %d files to %s", len(zf.namelist()), dest_dir)

    return dest_dir


def find_shapefile(directory: str) -> str:
    """Walk a directory and return the first .shp path found."""
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".shp"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .shp file found in {directory}")


def load_shapefile(shp_path: str) -> gpd.GeoDataFrame:
    """Read shapefile and reproject to WGS-84."""
    logger.info("Reading shapefile: %s", shp_path)
    gdf = gpd.read_file(shp_path)
    logger.info("Read %d features with CRS %s", len(gdf), gdf.crs)
    logger.info("Columns: %s", list(gdf.columns))

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        logger.info("Reprojecting from %s to %s", gdf.crs, TARGET_CRS)
        gdf = gdf.to_crs(TARGET_CRS)

    return gdf


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename DCR CONSLANDS shapefile columns to our standard schema.

    Source columns (as of July 2025 release):
      LABEL, MANAME, MATYPE, MAAGENCY, MALEVEL, OWNER, PUBACCESS,
      TRACTACRE, TOTALACRE, GISACRE, ACQDATE, BNDACCURAC, BNDORIGIN,
      LANDLINK, Shape_Leng, Shape_Area
    """
    col_map = {
        "LABEL": "local_designation",
        "MANAME": "manager_name",
        "MATYPE": "designation_type",
        "MAAGENCY": "manager_agency",
        "MALEVEL": "manager_level",
        "OWNER": "owner_name",
        "PUBACCESS": "public_access",
        "TRACTACRE": "tract_acres",
        "TOTALACRE": "total_acres",
        "GISACRE": "area_acres",
        "ACQDATE": "acquisition_date",
        "LANDLINK": "source_url",
    }

    drop_cols = {"Shape_Leng", "Shape_Area", "BNDACCURAC", "BNDORIGIN"}
    gdf = gdf.drop(columns=[c for c in gdf.columns if c in drop_cols], errors="ignore")

    rename = {}
    for orig_col in gdf.columns:
        upper = orig_col.upper()
        if upper in col_map:
            rename[orig_col] = col_map[upper]
        elif orig_col != "geometry":
            rename[orig_col] = orig_col.lower()

    gdf = gdf.rename(columns=rename)

    if "geometry" in gdf.columns:
        gdf = gdf.rename(columns={"geometry": "geom"}).set_geometry("geom")
    elif gdf.geometry.name != "geom":
        gdf = gdf.rename_geometry("geom")

    gdf = gdf[gdf.geom.notna() & gdf.geom.is_valid].copy()
    logger.info("After normalization: %d features, columns: %s", len(gdf), list(gdf.columns))
    return gdf


def load_to_postgis(gdf: gpd.GeoDataFrame, engine, schema: str = "raw") -> int:
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    gdf.to_postgis(
        TABLE_NAME,
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=500,
    )
    return len(gdf)


def main():
    parser = argparse.ArgumentParser(description="Ingest VA DCR Conservation Lands")
    parser.add_argument("--schema", default="raw")
    parser.add_argument("--inspect", action="store_true",
                        help="Download and print schema info without loading to DB")
    parser.add_argument("--keep-download", action="store_true",
                        help="Keep the extracted shapefile after loading")
    parser.add_argument("--local-zip", type=str, default=None,
                        help="Path to an already-downloaded conslands.zip")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="conslands_") as tmpdir:
        if args.local_zip:
            logger.info("Using local zip: %s", args.local_zip)
            with zipfile.ZipFile(args.local_zip) as zf:
                zf.extractall(tmpdir)
        else:
            download_and_extract(CONSLANDS_URL, tmpdir)

        shp_path = find_shapefile(tmpdir)
        gdf = load_shapefile(shp_path)

        if args.inspect:
            print("\n=== CONSLANDS Shapefile Schema ===")
            print(f"Features:  {len(gdf)}")
            print(f"CRS:       {gdf.crs}")
            print(f"Bounds:    {gdf.total_bounds}")
            print(f"\nColumns ({len(gdf.columns)}):")
            for col in gdf.columns:
                if col == "geometry":
                    print(f"  {col:30s}  geometry")
                else:
                    dtype = gdf[col].dtype
                    nulls = gdf[col].isna().sum()
                    sample = gdf[col].dropna().iloc[0] if not gdf[col].dropna().empty else "N/A"
                    print(f"  {col:30s}  {str(dtype):12s}  nulls={nulls:5d}  sample={sample}")
            print()
            return

        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL is required")
            raise SystemExit(1)

        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection OK")

        gdf = normalize_columns(gdf)
        n = load_to_postgis(gdf, engine, args.schema)
        logger.info("Loaded %d features to %s.%s", n, args.schema, TABLE_NAME)


if __name__ == "__main__":
    main()
