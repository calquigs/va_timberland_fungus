"""
Ingest environmental rasters: elevation, slope, precipitation, temperature.

Downloads PRISM 30-year normals and Copernicus DEM 90m tiles, clips to Virginia,
derives slope from elevation, and optionally samples values at OTU sites and
parcel centroids for storage in PostGIS.

Sources:
  Elevation   – Copernicus DEM GLO-90 (90m, 3 arc-second) from AWS Open Data
  Slope       – Derived from elevation via numpy gradient
  Precip      – PRISM 30-yr normals (1991-2020), annual ppt, 4km
  Tmax        – PRISM 30-yr normals, annual tmax, 4km
  Tmin        – PRISM 30-yr normals, annual tmin, 4km

Usage:
  python ingest_env_rasters.py                        # download + process
  python ingest_env_rasters.py --local-dir /path      # use pre-downloaded files
  python ingest_env_rasters.py --sample-points        # also sample at DB points
"""

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.merge import merge
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VA_BBOX = (-83.7, 36.5, -75.2, 39.6)
VA_RES_DEG = 0.001  # ~100m target resolution for clipped outputs
DOWNLOAD_TIMEOUT = 300

PRISM_BASE_URL = "https://services.nacse.org/prism/data/get/us/4km"
PRISM_YEAR = "2024"
PRISM_VARS = {
    "ppt": "precip_mm",
    "tmax": "tmax_c",
    "tmin": "tmin_c",
}

COPERNICUS_DEM_BASE = (
    "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
)


def get_output_dir(args) -> Path:
    out = args.local_dir or os.environ.get("ENV_RASTER_DIR")
    if not out:
        out = Path(__file__).resolve().parent.parent.parent / "data" / "environmental"
    return Path(out).resolve()


# ---------------------------------------------------------------------------
# Copernicus DEM download + mosaic
# ---------------------------------------------------------------------------

def _copernicus_tile_url(lat: int, lon_abs: int) -> str:
    tile_name = f"Copernicus_DSM_COG_30_N{lat:02d}_00_W{lon_abs:03d}_00_DEM"
    return f"{COPERNICUS_DEM_BASE}/{tile_name}/{tile_name}.tif"


def download_elevation(out_dir: Path) -> Path | None:
    """Download Copernicus DEM 90m tiles for Virginia and mosaic them."""
    mosaic_path = out_dir / "elevation_va.tif"
    if mosaic_path.exists():
        logger.info("Elevation mosaic already exists: %s", mosaic_path)
        return mosaic_path

    tile_dir = out_dir / "dem_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = VA_BBOX
    lat_range = range(int(np.floor(south)), int(np.ceil(north)))
    lon_range = range(int(np.floor(abs(east))), int(np.ceil(abs(west))) + 1)

    tile_paths = []
    for lat in lat_range:
        for lon_abs in lon_range:
            tile_path = tile_dir / f"N{lat:02d}_W{lon_abs:03d}.tif"
            if tile_path.exists():
                tile_paths.append(tile_path)
                continue
            url = _copernicus_tile_url(lat, lon_abs)
            logger.info("Downloading DEM tile N%02d W%03d ...", lat, lon_abs)
            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
                resp.raise_for_status()
                with open(tile_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tile_paths.append(tile_path)
            except requests.RequestException as e:
                logger.warning("Failed to download tile N%02d W%03d: %s", lat, lon_abs, e)

    if not tile_paths:
        logger.error("No DEM tiles downloaded")
        return None

    logger.info("Mosaicking %d DEM tiles ...", len(tile_paths))
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_data, mosaic_transform = merge(datasets)
    finally:
        for ds in datasets:
            ds.close()

    profile = {
        "driver": "GTiff",
        "dtype": mosaic_data.dtype,
        "width": mosaic_data.shape[2],
        "height": mosaic_data.shape[1],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": mosaic_transform,
        "compress": "deflate",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic_data[0], 1)

    shutil.rmtree(tile_dir, ignore_errors=True)
    logger.info("Elevation mosaic saved: %s", mosaic_path)
    return mosaic_path


# ---------------------------------------------------------------------------
# PRISM download
# ---------------------------------------------------------------------------

def download_prism_var(var: str, out_dir: Path) -> Path | None:
    """Download PRISM annual data (COG) for a climate variable."""
    tif_path = out_dir / f"prism_{var}_annual.tif"
    if tif_path.exists():
        logger.info("PRISM %s already exists: %s", var, tif_path)
        return tif_path

    zip_path = out_dir / f"prism_{var}_annual.zip"
    if not zip_path.exists():
        url = f"{PRISM_BASE_URL}/{var}/{PRISM_YEAR}"
        logger.info("Downloading PRISM %s from %s ...", var, url)
        try:
            resp = requests.get(
                url, timeout=DOWNLOAD_TIMEOUT, stream=True,
                headers={"User-Agent": "VA-Woods-Ingest/1.0"},
            )
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "text/html" in ct:
                logger.error("PRISM returned HTML for %s — check URL: %s", var, resp.text[:300])
                return None
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        except requests.RequestException as e:
            logger.warning("PRISM %s download failed: %s", var, e)
            if zip_path.exists():
                zip_path.unlink()
            return None

    extract_dir = out_dir / f"prism_{var}_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        logger.error("Bad zip file for PRISM %s — download manually", var)
        zip_path.unlink()
        return None

    tif_files = list(extract_dir.glob("*.tif"))
    bil_files = list(extract_dir.glob("*.bil"))
    src_file = (tif_files or bil_files or [None])[0]
    if not src_file:
        logger.error("No raster file found in PRISM %s archive", var)
        return None

    with rasterio.open(src_file) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", compress="deflate")
        data = src.read()
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(data)

    shutil.rmtree(extract_dir, ignore_errors=True)
    logger.info("PRISM %s saved: %s", var, tif_path)
    return tif_path


# ---------------------------------------------------------------------------
# Clip + resample to VA extent
# ---------------------------------------------------------------------------

def clip_to_va(src_path: Path, dst_path: Path, nodata=np.nan) -> Path:
    """Reproject and clip a raster to VA bbox at ~100m resolution."""
    if dst_path.exists():
        logger.info("VA clip already exists: %s", dst_path)
        return dst_path

    west, south, east, north = VA_BBOX
    dst_width = int((east - west) / VA_RES_DEG)
    dst_height = int((north - south) / VA_RES_DEG)
    dst_transform = transform_from_bounds(west, south, east, north, dst_width, dst_height)

    dst_data = np.empty((1, dst_height, dst_width), dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            dst_nodata=float(nodata) if np.isnan(nodata) else nodata,
        )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": dst_width,
        "height": dst_height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "compress": "deflate",
        "nodata": float("nan"),
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_data)

    logger.info("Clipped to VA: %s (%dx%d)", dst_path, dst_width, dst_height)
    return dst_path


# ---------------------------------------------------------------------------
# Slope derivation
# ---------------------------------------------------------------------------

def derive_slope(elev_va_path: Path, out_dir: Path) -> Path:
    """Compute slope in degrees from a clipped elevation raster."""
    slope_path = out_dir / "slope_va.tif"
    if slope_path.exists():
        logger.info("Slope raster already exists: %s", slope_path)
        return slope_path

    with rasterio.open(elev_va_path) as src:
        elev = src.read(1).astype(np.float64)
        transform = src.transform
        profile = src.profile.copy()

    cell_size_x = abs(transform.a)
    cell_size_y = abs(transform.e)
    meters_per_deg = 111_320
    dx = cell_size_x * meters_per_deg
    dy = cell_size_y * meters_per_deg

    elev[np.isnan(elev)] = 0
    grad_y, grad_x = np.gradient(elev, dy, dx)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    profile.update(dtype="float32", compress="deflate", nodata=float("nan"))
    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope_deg, 1)

    logger.info("Slope raster saved: %s", slope_path)
    return slope_path


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_rasters_at_points(
    points: list[tuple[float, float]],
    raster_paths: dict[str, Path],
) -> pd.DataFrame:
    """Sample raster values at (lon, lat) points. Returns a DataFrame."""
    results = {"lon": [], "lat": []}
    for name in raster_paths:
        results[name] = []

    datasets = {name: rasterio.open(p) for name, p in raster_paths.items()}

    try:
        for lon, lat in points:
            results["lon"].append(lon)
            results["lat"].append(lat)
            for name, ds in datasets.items():
                try:
                    vals = list(ds.sample([(lon, lat)]))
                    val = float(vals[0][0]) if vals else np.nan
                    if val < -9000 or val > 1e30:
                        val = np.nan
                except Exception:
                    val = np.nan
                results[name].append(val)
    finally:
        for ds in datasets.values():
            ds.close()

    return pd.DataFrame(results)


def load_sample_points(engine) -> list[dict]:
    """Load OTU sites and parcel centroids from PostGIS as sample points."""
    points = []

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT site_id::text, ST_X(ST_SetSRID(ST_MakePoint(lon, lat), 4326)) AS lon,
                       ST_Y(ST_SetSRID(ST_MakePoint(lon, lat), 4326)) AS lat
                FROM raw.otu_sites WHERE lat IS NOT NULL AND lon IS NOT NULL
            """)).fetchall()
            for r in rows:
                points.append({"point_id": r[0], "point_type": "otu_site", "lon": r[1], "lat": r[2]})
    except Exception as e:
        logger.warning("Could not load OTU sites: %s", e)

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT "OBJECTID"::text,
                       ST_X(ST_Centroid(geom)) AS lon,
                       ST_Y(ST_Centroid(geom)) AS lat
                FROM staging.va_harvested_timber_parcels
                WHERE geom IS NOT NULL
            """)).fetchall()
            for r in rows:
                points.append({"point_id": r[0], "point_type": "parcel_centroid", "lon": r[1], "lat": r[2]})
    except Exception as e:
        logger.warning("Could not load parcel centroids: %s", e)

    logger.info("Loaded %d sample points (%d OTU sites, %d parcel centroids)",
                len(points),
                sum(1 for p in points if p["point_type"] == "otu_site"),
                sum(1 for p in points if p["point_type"] == "parcel_centroid"))
    return points


def write_env_samples(df: pd.DataFrame, engine):
    """Write sampled environmental values to raw.env_point_samples."""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.env_point_samples (
                point_id TEXT NOT NULL,
                point_type TEXT NOT NULL,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION,
                elevation_m DOUBLE PRECISION,
                slope_deg DOUBLE PRECISION,
                tmax_c DOUBLE PRECISION,
                tmin_c DOUBLE PRECISION,
                precip_mm DOUBLE PRECISION,
                PRIMARY KEY (point_id, point_type)
            )
        """))
        conn.execute(text("TRUNCATE raw.env_point_samples"))
        conn.commit()

    df.to_sql(
        "env_point_samples",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info("Wrote %d rows to raw.env_point_samples", len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest environmental rasters (elevation, slope, PRISM climate)"
    )
    parser.add_argument(
        "--local-dir", type=str, default=None,
        help="Directory with pre-downloaded rasters (skip download step)",
    )
    parser.add_argument(
        "--sample-points", action="store_true",
        help="Sample raster values at OTU sites and parcel centroids, write to DB",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (use existing files in output dir)",
    )
    args = parser.parse_args()

    out_dir = get_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    raster_paths: dict[str, Path] = {}

    if not args.skip_download and not args.local_dir:
        elev_raw = download_elevation(out_dir)
        if elev_raw:
            elev_va = clip_to_va(elev_raw, out_dir / "elevation_va.tif")
            raster_paths["elevation_m"] = elev_va
            slope_va = derive_slope(elev_va, out_dir)
            raster_paths["slope_deg"] = slope_va

        for prism_var, col_name in PRISM_VARS.items():
            raw_path = download_prism_var(prism_var, out_dir)
            if raw_path:
                va_path = clip_to_va(raw_path, out_dir / f"{prism_var}_va.tif")
                raster_paths[col_name] = va_path
    else:
        elev_va = out_dir / "elevation_va.tif"
        if elev_va.exists():
            raster_paths["elevation_m"] = elev_va
        slope_va = out_dir / "slope_va.tif"
        if slope_va.exists():
            raster_paths["slope_deg"] = slope_va
        elif elev_va.exists():
            slope_va = derive_slope(elev_va, out_dir)
            raster_paths["slope_deg"] = slope_va

        for prism_var, col_name in PRISM_VARS.items():
            va_path = out_dir / f"{prism_var}_va.tif"
            if va_path.exists():
                raster_paths[col_name] = va_path

    logger.info("Available rasters: %s", list(raster_paths.keys()))
    if not raster_paths:
        logger.error(
            "No rasters available. Download them or provide --local-dir.\n"
            "Expected files: elevation_va.tif, slope_va.tif, ppt_va.tif, tmax_va.tif, tmin_va.tif"
        )
        raise SystemExit(1)

    if args.sample_points:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.error("DATABASE_URL is required for --sample-points")
            raise SystemExit(1)

        engine = create_engine(database_url)
        sample_points = load_sample_points(engine)
        if not sample_points:
            logger.warning("No sample points found in DB")
            return

        coords = [(p["lon"], p["lat"]) for p in sample_points]
        sampled = sample_rasters_at_points(coords, raster_paths)

        sampled["point_id"] = [p["point_id"] for p in sample_points]
        sampled["point_type"] = [p["point_type"] for p in sample_points]

        col_order = ["point_id", "point_type", "lat", "lon",
                     "elevation_m", "slope_deg", "tmax_c", "tmin_c", "precip_mm"]
        for c in col_order:
            if c not in sampled.columns:
                sampled[c] = np.nan
        sampled = sampled[col_order]

        write_env_samples(sampled, engine)

    logger.info("Environmental raster ingestion complete.")


if __name__ == "__main__":
    main()
