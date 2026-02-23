"""
Ingest USFS FIA BIGMAP Loblolly Pine biomass data.

Source: FIA BIGMAP (see docs/DATA_SOURCES.md).
  Direct zip: https://data.fs.usda.gov/geodata/rastergateway/bigmap/index.php
  Loblolly SPCD 0131, 2018 AGB layer.
Output: Downloaded zip extracted to a directory (GeoTIFF + aux files). Optionally
        compute per-parcel zonal statistics and write to PostGIS.

Usage:
  # Automated: download from USDA and extract to FIA_BIGMAP_DATA (or --output-dir)
  python ingest_fia_loblolly.py

  # Manual: use a zip you already downloaded from the BIGMAP page
  python ingest_fia_loblolly.py --local-zip /path/to/BIGMAP_AGB_2018_SPCD0131_LOBLOLLY_PINE.zip

  # Optional: set output directory (default: env FIA_BIGMAP_DATA or ./data/fia_bigmap)
  python ingest_fia_loblolly.py --output-dir ./data/fia_bigmap

  # Compute per-parcel zonal stats from existing GeoTIFF and write to PostGIS:
  python ingest_fia_loblolly.py --zonal-stats
  python ingest_fia_loblolly.py --zonal-stats --raster /path/to/loblolly.tif
"""

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Direct download URL for Loblolly Pine (SPCD 0131), 2018 AGB (same as BIGMAP page link)
BIGMAP_LOBLOLLY_ZIP_URL = (
    "https://data.fs.usda.gov/geodata/rastergateway/bigmap/2018/"
    "BIGMAP_AGB_2018_SPCD0131_LOBLOLLY_PINE.zip"
)
# Some servers expect path with %5F for underscore
BIGMAP_LOBLOLLY_ZIP_URL_ENCODED = (
    "https://data.fs.usda.gov/geodata/rastergateway/bigmap/2018/"
    "BIGMAP%5FAGB%5F2018%5FSPCD0131%5FLOBLOLLY%5FPINE.zip"
)
DOWNLOAD_CHUNK_BYTES = 2 * 1024 * 1024  # 2 MiB
REQUEST_TIMEOUT = 600  # 10 min for large file
MAX_RETRIES = 3
USER_AGENT = (
    "Mozilla/5.0 (compatible; VA-Woods-FIA-Ingest/1.0; +https://github.com/va-woods)"
)


def get_output_dir(args) -> Path:
    out = args.output_dir or os.environ.get("FIA_BIGMAP_DATA")
    if not out:
        # Default relative to ingestion dir or cwd
        out = Path(__file__).resolve().parent.parent.parent / "data" / "fia_bigmap"
    return Path(out).resolve()


def download_loblolly_zip(dest_path: Path) -> bool:
    """Download BIGMAP Loblolly zip from USDA; return True on success."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Try unencoded URL first; some servers redirect or require encoded
    for url in (BIGMAP_LOBLOLLY_ZIP_URL, BIGMAP_LOBLOLLY_ZIP_URL_ENCODED):
        try:
            logger.info("Downloading from %s ...", url.split("/")[-1].replace("%5F", "_"))
            headers = {"User-Agent": USER_AGENT}
            with requests.get(
                url,
                stream=True,
                timeout=30,
                headers=headers,
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                size = resp.headers.get("Content-Length")
                total = int(size) if size else None
                written = 0
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                            if total and total > 0:
                                pct = min(100, 100 * written / total)
                                if written % (20 * DOWNLOAD_CHUNK_BYTES) < len(chunk):
                                    logger.info("  %.1f%% (%s / %s)", pct, written, total)
                logger.info("Downloaded %s bytes to %s", written, dest_path)
                return True
        except requests.RequestException as e:
            logger.warning("Download failed for %s: %s", url[:60], e)
            if dest_path.exists():
                dest_path.unlink()
            continue
    return False


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract zip into out_dir; return list of extracted paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Avoid path traversal
            safe_name = Path(name).name if "/" in name or "\\" in name else name
            dest = out_dir / safe_name
            if not safe_name or safe_name.startswith("."):
                continue
            with zf.open(name) as src:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(src, f)
            extracted.append(dest)
    return extracted


VA_BBOX = {
    "west": -83.675,
    "south": 36.540,
    "east": -75.242,
    "north": 39.466,
}


def find_geotiff(out_dir: Path) -> Path | None:
    """Find the first .tif file in out_dir."""
    for f in sorted(out_dir.iterdir()):
        if f.suffix.lower() == ".tif":
            return f
    return None


def run_zonal_stats(raster_path: Path, database_url: str) -> int:
    """Compute per-parcel loblolly biomass zonal stats and write to PostGIS.

    Clips the CONUS raster to Virginia, loads harvested timber parcels from
    staging.va_harvested_timber_parcels, computes zonal_stats, and writes
    results to raw.parcel_loblolly_stats.

    Returns the number of parcels processed.
    """
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from rasterstats import zonal_stats
    from shapely.geometry import box
    from sqlalchemy import create_engine, text

    engine = create_engine(database_url)

    logger.info("Loading harvested timber parcels from staging.va_harvested_timber_parcels")
    parcels = gpd.read_postgis(
        'SELECT "OBJECTID", geom FROM staging.va_harvested_timber_parcels WHERE geom IS NOT NULL',
        engine,
        geom_col="geom",
    )
    if parcels.empty:
        logger.error("No parcels found in staging.va_harvested_timber_parcels. Run parcel ingestion first.")
        raise SystemExit(1)
    logger.info("Loaded %d parcels", len(parcels))

    logger.info("Opening raster: %s", raster_path)
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

        va_box = box(VA_BBOX["west"], VA_BBOX["south"], VA_BBOX["east"], VA_BBOX["north"])
        va_geom = gpd.GeoDataFrame(geometry=[va_box], crs="EPSG:4326").to_crs(raster_crs)

        logger.info("Clipping raster to Virginia bounding box")
        clipped, clipped_transform = rasterio_mask(
            src, va_geom.geometry, crop=True, nodata=src.nodata
        )
        clipped = clipped[0]  # single band
        nodata = src.nodata

    parcels_proj = parcels.to_crs(raster_crs)

    logger.info("Computing zonal statistics for %d parcels (pixel area: %.1f m²)", len(parcels_proj), pixel_area_m2)
    stats = zonal_stats(
        parcels_proj.geometry,
        clipped,
        affine=clipped_transform,
        stats=["mean", "sum", "count"],
        nodata=nodata,
    )

    # BIGMAP AGB units: kg/ha at 30m pixels
    records = []
    for i, row in enumerate(stats):
        mean_val = row.get("mean")
        sum_val = row.get("sum")
        count_val = row.get("count", 0)
        if mean_val is None or count_val == 0:
            continue
        pixel_area_ha = pixel_area_m2 / 10_000.0
        records.append({
            "parcel_objectid": int(parcels.iloc[i]["OBJECTID"]),
            "mean_biomass_kg_ha": float(np.round(mean_val, 4)),
            "total_biomass_kg": float(np.round(sum_val * pixel_area_ha, 4)),
            "pixel_count": int(count_val),
        })

    if not records:
        logger.warning("No valid zonal stats computed (parcels may not overlap raster data)")
        return 0

    import pandas as pd
    df = pd.DataFrame(records)

    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS raw'))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.parcel_loblolly_stats (
                parcel_objectid INTEGER PRIMARY KEY,
                mean_biomass_kg_ha DOUBLE PRECISION,
                total_biomass_kg DOUBLE PRECISION,
                pixel_count INTEGER
            )
        """))
        conn.execute(text("TRUNCATE raw.parcel_loblolly_stats"))
        conn.commit()

    df.to_sql(
        "parcel_loblolly_stats",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
    )
    logger.info("Wrote %d rows to raw.parcel_loblolly_stats", len(df))
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest FIA BIGMAP Loblolly biomass (download + extract, or compute zonal stats)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to extract GeoTIFF and aux files (default: FIA_BIGMAP_DATA or ./data/fia_bigmap)",
    )
    parser.add_argument(
        "--local-zip",
        type=str,
        default=None,
        metavar="PATH",
        help="Use this zip file instead of downloading (e.g. manually from BIGMAP page)",
    )
    parser.add_argument(
        "--zonal-stats",
        action="store_true",
        help="Compute per-parcel zonal stats from GeoTIFF and write to raw.parcel_loblolly_stats",
    )
    parser.add_argument(
        "--raster",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to GeoTIFF for --zonal-stats (default: auto-detect in output dir)",
    )
    args = parser.parse_args()

    out_dir = get_output_dir(args)

    if args.zonal_stats:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL is required for --zonal-stats")
            raise SystemExit(1)

        if args.raster:
            raster_path = Path(args.raster).resolve()
        else:
            raster_path = find_geotiff(out_dir)
        if not raster_path or not raster_path.exists():
            logger.error(
                "No GeoTIFF found. Run without --zonal-stats first to download/extract, "
                "or specify --raster /path/to/loblolly.tif"
            )
            raise SystemExit(1)

        n = run_zonal_stats(raster_path, database_url)
        logger.info("Zonal stats complete: %d parcels processed", n)
        return

    if args.local_zip:
        zip_path = Path(args.local_zip).resolve()
        if not zip_path.exists():
            logger.error("Local zip not found: %s", zip_path)
            raise SystemExit(1)
        logger.info("Using local zip: %s", zip_path)
    else:
        zip_path = out_dir / "BIGMAP_AGB_2018_SPCD0131_LOBLOLLY_PINE.zip"
        if not download_loblolly_zip(zip_path):
            logger.error(
                "Automated download failed. Download the Loblolly Pine zip manually from:\n  %s\n"
                "Then run: python ingest_fia_loblolly.py --local-zip /path/to/BIGMAP_AGB_2018_SPCD0131_LOBLOLLY_PINE.zip",
                "https://data.fs.usda.gov/geodata/rastergateway/bigmap/index.php",
            )
            raise SystemExit(1)

    files = extract_zip(zip_path, out_dir)
    tifs = [f for f in files if str(f).lower().endswith(".tif")]
    logger.info("Extracted %d files to %s (GeoTIFFs: %s)", len(files), out_dir, [f.name for f in tifs])
    if tifs:
        logger.info("Primary raster: %s", tifs[0])
    logger.info("Run with --zonal-stats to compute per-parcel biomass statistics.")


if __name__ == "__main__":
    main()
