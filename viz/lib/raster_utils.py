"""Utilities for loading and rendering raster overlays."""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

LOBLOLLY_RASTER = os.environ.get(
    "LOBLOLLY_RASTER",
    str(Path(__file__).resolve().parents[2] / "data" / "fia_bigmap"
        / "Hosted_AGB_0131_2018_LOBLOLLY_PINE_06062023002031.tif"),
)
OVERLAY_DIR = Path(__file__).resolve().parents[2] / "data" / "fia_bigmap"
VA_CLIP_PATH = OVERLAY_DIR / "loblolly_va_4326_100m.tif"
OVERLAY_PNG = OVERLAY_DIR / "loblolly_va_overlay_greens.png"
OVERLAY_BOUNDS = OVERLAY_DIR / "loblolly_va_bounds.json"

VA_BBOX_WGS84 = (-83.7, 36.5, -75.2, 39.6)
TARGET_RES_DEG = 0.001  # ~100 m — used only when generating the GeoTIFF clip
OVERLAY_MAX_WIDTH = 2500  # PNG overlay target width (downsampled from the GeoTIFF)


def _ensure_va_clip() -> Path | None:
    """Reproject + clip the CONUS raster to a WGS84 GeoTIFF. Runs once."""
    if VA_CLIP_PATH.exists():
        return VA_CLIP_PATH

    if not Path(LOBLOLLY_RASTER).exists():
        return None

    try:
        import rasterio
        from rasterio.transform import from_bounds as transform_from_bounds
        from rasterio.warp import Resampling, reproject
    except ImportError:
        st.warning("rasterio not installed — cannot generate VA loblolly clip")
        return None

    west, south, east, north = VA_BBOX_WGS84
    dst_width = int((east - west) / TARGET_RES_DEG)
    dst_height = int((north - south) / TARGET_RES_DEG)
    dst_transform = transform_from_bounds(west, south, east, north, dst_width, dst_height)

    dst_data = np.empty((1, dst_height, dst_width), dtype=np.float32)

    with rasterio.open(LOBLOLLY_RASTER) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
            dst_nodata=np.nan,
        )

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
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
    with rasterio.open(VA_CLIP_PATH, "w", **profile) as dst:
        dst.write(dst_data)

    return VA_CLIP_PATH


def _ensure_overlay_png() -> tuple[Path, Path] | None:
    """Generate a pre-rendered RGBA PNG from the VA clip. Runs once."""
    if OVERLAY_PNG.exists() and OVERLAY_BOUNDS.exists():
        return OVERLAY_PNG, OVERLAY_BOUNDS

    clip_path = _ensure_va_clip()
    if clip_path is None:
        return None

    try:
        import rasterio
    except ImportError:
        return None
    from PIL import Image

    with rasterio.open(clip_path) as src:
        factor = max(1, src.width // OVERLAY_MAX_WIDTH)
        out_w = src.width // factor
        out_h = src.height // factor
        data = src.read(1, out_shape=(out_h, out_w))
        b = src.bounds

    bounds = [[b.bottom, b.left], [b.top, b.right]]

    nodata_mask = np.isnan(data) | (data <= 0) | (data >= 3e+38)
    valid = data[~nodata_mask]
    if valid.size == 0:
        return None

    vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if vmin == vmax:
        vmax = vmin + 1

    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    norm[nodata_mask] = 0

    colors = np.array([
        [247, 252, 245],  # #f7fcf5
        [229, 245, 224],  # #e5f5e0
        [199, 233, 192],  # #c7e9c0
        [161, 217, 155],  # #a1d99b
        [116, 196, 118],  # #74c476
        [65, 171, 93],    # #41ab5d
        [35, 139, 69],    # #238b45
        [0, 109, 44],     # #006d2c
        [0, 68, 27],      # #00441b
    ], dtype=np.float64)

    indices = norm * (len(colors) - 1)
    low = np.floor(indices).astype(int)
    high = np.minimum(low + 1, len(colors) - 1)
    frac = indices - low

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    for c in range(3):
        rgba[:, :, c] = (colors[low, c] * (1 - frac) + colors[high, c] * frac).astype(np.uint8)
    rgba[:, :, 3] = 180
    rgba[nodata_mask, 3] = 0

    Image.fromarray(rgba).save(OVERLAY_PNG, optimize=True)
    with open(OVERLAY_BOUNDS, "w") as f:
        json.dump(bounds, f)

    return OVERLAY_PNG, OVERLAY_BOUNDS


@st.cache_data(ttl=3600, show_spinner="Loading loblolly raster…")
def get_loblolly_overlay() -> tuple[str, list[list[float]]] | None:
    """Return (data_uri, [[south, west], [north, east]]).

    Returns a base64 data URI string (not a numpy array) so Folium's
    ImageOverlay can embed it directly without re-encoding every render.
    """
    result = _ensure_overlay_png()
    if result is None:
        return None

    png_path, bounds_path = result

    with open(bounds_path) as f:
        bounds = json.load(f)

    png_bytes = png_path.read_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()

    return data_uri, bounds
