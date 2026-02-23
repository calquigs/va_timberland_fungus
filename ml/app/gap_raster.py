"""
Generate a sampling-gap raster that highlights regions of Virginia whose
environmental conditions are not yet represented by OTU sample sites.

For each pixel and each environmental variable the "gap" is:
  - 0 when the pixel value falls within [sampled_min, sampled_max]
  - distance_to_nearest_bound / state_range  otherwise

The per-variable gaps are averaged into a single composite score. A score
of 0 means the pixel is inside the sampled parameter space for every
variable; a score of 1.0 means the pixel is, on average, one full
state-wide range away from the nearest sampled value.

Outputs:
  sampling_gap.tif          – Float32 GeoTIFF at the same resolution as env layers
  sampling_gap_overlay.png  – RGBA PNG for Folium/Streamlit map overlays
  sampling_gap_bounds.json  – [[south, west], [north, east]]
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

VA_BBOX = (-83.7, 36.5, -75.2, 39.6)
VA_RES_DEG = 0.001

RASTER_FILES = {
    "elevation_m": "elevation_va.tif",
    "slope_deg": "slope_va.tif",
    "tmax_c": "tmax_va.tif",
    "tmin_c": "tmin_va.tif",
    "precip_mm": "ppt_va.tif",
}

OVERLAY_MAX_WIDTH = 1500


def get_sample_bounds(engine) -> dict[str, tuple[float, float]] | None:
    """Query min/max of each env variable across OTU sample sites."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                MIN(elevation_m), MAX(elevation_m),
                MIN(slope_deg),   MAX(slope_deg),
                MIN(tmax_c),      MAX(tmax_c),
                MIN(tmin_c),      MAX(tmin_c),
                MIN(precip_mm),   MAX(precip_mm),
                COUNT(*)
            FROM raw.env_point_samples
            WHERE point_type = 'otu_site'
        """)).fetchone()

    if not row or row[10] == 0:
        return None

    return {
        "elevation_m": (row[0], row[1]),
        "slope_deg":   (row[2], row[3]),
        "tmax_c":      (row[4], row[5]),
        "tmin_c":      (row[6], row[7]),
        "precip_mm":   (row[8], row[9]),
    }


def _read_to_common_grid(raster_path, dst_height, dst_width, dst_transform):
    """Read a raster and reproject it onto the common VA grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.empty((dst_height, dst_width), dtype=np.float32)

    with rasterio.open(raster_path) as src:
        if (src.height == dst_height and src.width == dst_width
                and src.transform.almost_equals(dst_transform)):
            return src.read(1).astype(np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return dst


def generate_gap_raster(
    env_dir: Path,
    engine=None,
    database_url: str | None = None,
) -> Path | None:
    """Build the sampling-gap GeoTIFF and PNG overlay.

    Returns the path to the GeoTIFF, or None on failure.
    """
    import rasterio
    from PIL import Image
    from rasterio.transform import from_bounds as transform_from_bounds

    if engine is None:
        database_url = database_url or os.environ.get(
            "DATABASE_URL",
            "postgresql://va_woods:va_woods_dev@localhost:5432/va_woods",
        )
        engine = create_engine(database_url)

    bounds = get_sample_bounds(engine)
    if bounds is None:
        logger.warning("No OTU sample sites with env data — cannot generate gap raster")
        return None

    west, south, east, north = VA_BBOX
    dst_width = int((east - west) / VA_RES_DEG)
    dst_height = int((north - south) / VA_RES_DEG)
    dst_transform = transform_from_bounds(west, south, east, north, dst_width, dst_height)

    raster_arrays: dict[str, np.ndarray] = {}
    for var_name, filename in RASTER_FILES.items():
        raster_path = env_dir / filename
        if not raster_path.exists():
            logger.warning("Missing raster %s — skipping variable %s", raster_path, var_name)
            continue
        data = _read_to_common_grid(raster_path, dst_height, dst_width, dst_transform)
        raster_arrays[var_name] = data
        logger.info("Loaded %s: shape=%s", var_name, data.shape)

    if not raster_arrays:
        logger.error("No environmental rasters found in %s", env_dir)
        return None

    height, width = dst_height, dst_width
    gap_sum = np.zeros((height, width), dtype=np.float64)
    gap_count = np.zeros((height, width), dtype=np.float64)

    for var_name, data in raster_arrays.items():
        if var_name not in bounds:
            continue

        sampled_min, sampled_max = bounds[var_name]
        if sampled_min is None or sampled_max is None:
            continue

        valid = ~(np.isnan(data) | (data < -9000) | (data > 1e30))

        valid_vals = data[valid]
        if valid_vals.size == 0:
            continue

        state_min = float(np.nanmin(valid_vals))
        state_max = float(np.nanmax(valid_vals))
        state_range = state_max - state_min
        if state_range <= 0:
            continue

        pixel_gap = np.zeros_like(data, dtype=np.float64)
        below = valid & (data < sampled_min)
        above = valid & (data > sampled_max)
        pixel_gap[below] = (sampled_min - data[below]) / state_range
        pixel_gap[above] = (data[above] - sampled_max) / state_range

        gap_sum[valid] += pixel_gap[valid]
        gap_count[valid] += 1.0

    has_data = gap_count > 0
    mean_gap = np.full((height, width), np.nan, dtype=np.float32)
    mean_gap[has_data] = (gap_sum[has_data] / gap_count[has_data]).astype(np.float32)
    mean_gap = np.clip(mean_gap, 0, None)

    # ---- Write GeoTIFF ----
    tif_path = env_dir / "sampling_gap.tif"
    out_profile = {
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
    with rasterio.open(tif_path, "w", **out_profile) as dst:
        dst.write(mean_gap, 1)
    logger.info("Wrote sampling gap GeoTIFF: %s (%dx%d)", tif_path, width, height)

    # ---- Write PNG overlay ----
    factor = max(1, width // OVERLAY_MAX_WIDTH)
    small_w = width // factor
    small_h = height // factor
    with rasterio.open(tif_path) as src:
        small = src.read(1, out_shape=(small_h, small_w))
        b = src.bounds

    overlay_bounds = [[b.bottom, b.left], [b.top, b.right]]
    nodata_mask = np.isnan(small)
    covered = (~nodata_mask) & (small == 0)
    uncovered = (~nodata_mask) & (small > 0)

    rgba = np.zeros((small_h, small_w, 4), dtype=np.uint8)

    rgba[covered, 0] = 44   # green: #2ca02c
    rgba[covered, 1] = 160
    rgba[covered, 2] = 44
    rgba[covered, 3] = 170

    if np.any(uncovered):
        vals = np.clip(small[uncovered], 0, 1).astype(np.float64)
        rgba[uncovered, 0] = (255 * vals + 255 * (1 - vals)).astype(np.uint8)  # white→red
        rgba[uncovered, 1] = (255 * (1 - vals)).astype(np.uint8)               # white→0
        rgba[uncovered, 2] = (255 * (1 - vals)).astype(np.uint8)               # white→0
        rgba[uncovered, 3] = (100 + 120 * vals).astype(np.uint8)               # more opaque as gap grows

    png_path = env_dir / "sampling_gap_overlay.png"
    Image.fromarray(rgba).save(png_path, optimize=True)

    bounds_path = env_dir / "sampling_gap_bounds.json"
    with open(bounds_path, "w") as f:
        json.dump(overlay_bounds, f)

    logger.info("Wrote sampling gap overlay: %s", png_path)
    return tif_path
