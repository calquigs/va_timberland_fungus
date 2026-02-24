"""
Sample view – environmental coverage and OTU sampling analysis.

Features:
- Environmental raster overlays (elevation, slope, tmax, tmin, precip)
- Parameter-space coverage gap layer (highlights under-sampled regions)
- Protected woodlands overlay (PAD-US)
- Target parcels overlay
- OTU sample site markers with popups
- "Add New Sample" form for OTU table upload
- Click-to-query coverage assessment popup
"""

import io
import os
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import requests
import streamlit as st
from branca.element import MacroElement
from jinja2 import Template
from sqlalchemy import text
from streamlit_folium import st_folium

from lib.constants import (
    API_URL,
    POSITRON_ATTR,
    POSITRON_TILES,
    SATELLITE_ATTR,
    SATELLITE_TILES,
    VA_CENTER,
    VA_ZOOM,
)
from lib.db import get_engine

st.set_page_config(page_title="Sample | VA Woods", layout="wide")
st.title("Sample – Environmental Coverage & OTU Sampling")

API_BASE = os.environ.get("API_URL", API_URL)
ENV_RASTER_DIR = Path(
    os.environ.get("ENV_RASTER_DIR",
                    Path(__file__).resolve().parents[2] / "data" / "environmental")
)

ENV_LAYERS = {
    "elevation_m": {"file": "elevation_va.tif", "label": "Elevation (m)", "cmap": "terrain"},
    "slope_deg": {"file": "slope_va.tif", "label": "Slope (deg)", "cmap": "oranges"},
    "tmax_c": {"file": "tmax_va.tif", "label": "Annual High Temp (C)", "cmap": "reds"},
    "tmin_c": {"file": "tmin_va.tif", "label": "Annual Low Temp (C)", "cmap": "blues"},
    "precip_mm": {"file": "ppt_va.tif", "label": "Annual Precip (mm)", "cmap": "greens"},
}

GAP_LAYER_KEY = "sampling_gap"

COLORMAPS = {
    "terrain": ["#194d33", "#2d8659", "#80cc80", "#e6e6b3", "#c8a060", "#8c6239", "#ffffff"],
    "oranges": ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"],
    "reds":    ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"],
    "blues":   ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"],
    "greens":  ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#238b45", "#00441b"],
    "rdylgn_r": ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"],
}


# ---------------------------------------------------------------------------
# Raster overlay generation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _load_raster_overlay(var_name: str) -> tuple[str, list[list[float]], float, float] | None:
    """Load a clipped VA raster and return (data_uri, bounds, vmin, vmax)."""
    info = ENV_LAYERS.get(var_name)
    if not info:
        return None
    raster_path = ENV_RASTER_DIR / info["file"]
    if not raster_path.exists():
        return None

    try:
        import rasterio
        from PIL import Image
    except ImportError:
        return None

    cache_png = ENV_RASTER_DIR / f"{var_name}_overlay.png"
    cache_meta = ENV_RASTER_DIR / f"{var_name}_overlay.json"

    with rasterio.open(raster_path) as src:
        max_w = 1500
        factor = max(1, src.width // max_w)
        out_w = src.width // factor
        out_h = src.height // factor
        data = src.read(1, out_shape=(out_h, out_w))
        b = src.bounds

    bounds = [[b.bottom, b.left], [b.top, b.right]]
    nodata_mask = np.isnan(data) | (data < -9000) | (data > 1e30)
    valid = data[~nodata_mask]
    if valid.size == 0:
        return None

    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmin == vmax:
        vmax = vmin + 1

    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    norm[nodata_mask] = 0

    cmap_name = info.get("cmap", "greens")
    colors = np.array(COLORMAPS.get(cmap_name, COLORMAPS["greens"]), dtype=object)
    rgb_colors = np.array([
        [int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors
    ], dtype=np.float64)

    indices = norm * (len(rgb_colors) - 1)
    low = np.floor(indices).astype(int)
    high = np.minimum(low + 1, len(rgb_colors) - 1)
    frac = indices - low

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    for c in range(3):
        rgba[:, :, c] = (rgb_colors[low, c] * (1 - frac) + rgb_colors[high, c] * frac).astype(np.uint8)
    rgba[:, :, 3] = 160
    rgba[nodata_mask, 3] = 0

    import base64
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return data_uri, bounds, vmin, vmax


@st.cache_data(ttl=120, show_spinner=False)
def _load_gap_overlay() -> tuple[str, list[list[float]]] | None:
    """Load the pre-generated sampling-gap PNG overlay."""
    import base64

    png_path = ENV_RASTER_DIR / "sampling_gap_overlay.png"
    bounds_path = ENV_RASTER_DIR / "sampling_gap_bounds.json"
    if not png_path.exists() or not bounds_path.exists():
        return None

    import json as _json

    with open(bounds_path) as f:
        bounds = _json.load(f)

    png_bytes = png_path.read_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    return data_uri, bounds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def load_otu_sites(_engine) -> gpd.GeoDataFrame | None:
    try:
        sql = text("""
            SELECT s.site_id::text, s.lat, s.lon, s.uploaded_at, s.file_path,
                   COUNT(o.id) AS n_observations,
                   ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326) AS geom
            FROM raw.otu_sites s
            LEFT JOIN raw.otu_observations o ON s.site_id = o.site_id
            GROUP BY s.site_id, s.lat, s.lon, s.uploaded_at, s.file_path
        """)
        gdf = gpd.read_postgis(sql, _engine, geom_col="geom", crs="EPSG:4326")
        return gdf if not gdf.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_protected_lands(_engine, limit: int = 8000) -> gpd.GeoDataFrame | None:
    try:
        sql = text("""
            SELECT ST_Simplify(geom, 0.001) AS geom,
                   local_designation, manager_name, designation_type,
                   manager_agency, public_access, area_acres
            FROM raw.protected_lands
            WHERE geom IS NOT NULL
            LIMIT :lim
        """)
        gdf = gpd.read_postgis(sql, _engine, geom_col="geom", crs="EPSG:4326", params={"lim": limit})
        gdf = gdf[gdf.geom.notna() & ~gdf.geom.is_empty]
        return gdf if not gdf.empty else None
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def load_target_parcels_simple(_engine, limit: int = 2000) -> gpd.GeoDataFrame | None:
    """Load only parcels that are in the target shortlist (raw.target_parcels)."""
    try:
        # Prefer marts for geometry + attributes; join raw.target_parcels so we only get shortlisted parcels
        queries = [
            (
                "marts + shortlist",
                text("""
                    SELECT h.geom, h."OBJECTID", h."LOCALITY"
                    FROM marts.target_parcels h
                    INNER JOIN raw.target_parcels t ON h."OBJECTID" = t.parcel_objectid
                    WHERE h.geom IS NOT NULL
                    LIMIT :lim
                """),
            ),
            (
                "staging + shortlist",
                text("""
                    SELECT h.geom, h."OBJECTID", h."LOCALITY"
                    FROM staging.va_harvested_timber_parcels h
                    INNER JOIN raw.target_parcels t ON h."OBJECTID" = t.parcel_objectid
                    WHERE h.geom IS NOT NULL
                    LIMIT :lim
                """),
            ),
        ]
        for _label, sql in queries:
            try:
                gdf = gpd.read_postgis(
                    sql, _engine, geom_col="geom", crs="EPSG:4326", params={"lim": limit}
                )
                if not gdf.empty:
                    return gdf
            except Exception:
                continue
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_sample_map(
    env_overlay: tuple[str, list, float, float] | None,
    env_label: str | None,
    env_cmap_name: str | None,
    otu_sites: gpd.GeoDataFrame | None,
    protected: gpd.GeoDataFrame | None,
    target_parcels: gpd.GeoDataFrame | None,
    gap_overlay: tuple[str, list[list[float]]] | None = None,
) -> folium.Map:

    m = folium.Map(
        location=VA_CENTER, zoom_start=VA_ZOOM,
        tiles=None, control_scale=True,
    )
    folium.TileLayer(tiles=POSITRON_TILES, attr=POSITRON_ATTR, name="Positron", show=True).add_to(m)
    folium.TileLayer(tiles=SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite", show=False).add_to(m)

    if gap_overlay is not None:
        data_uri, bounds = gap_overlay
        folium.raster_layers.ImageOverlay(
            image=data_uri,
            bounds=bounds,
            name="Sampling Gap",
            opacity=0.75,
            interactive=False,
            show=True,
        ).add_to(m)

        gap_cmap = cm.LinearColormap(
            colors=["#2ca02c", "#ffffff", "#ff0000"],
            vmin=0, vmax=1,
            caption="Sampling Gap (0 = covered, 1 = 1 full range away)",
            index=[0, 0.01, 1],
        )
        gap_cmap.add_to(m)

    elif env_overlay is not None and env_label and env_cmap_name:
        data_uri, bounds, vmin, vmax = env_overlay
        folium.raster_layers.ImageOverlay(
            image=data_uri,
            bounds=bounds,
            name=env_label,
            opacity=0.65,
            interactive=False,
            show=True,
        ).add_to(m)

        colors = COLORMAPS.get(env_cmap_name, COLORMAPS["greens"])
        colormap = cm.LinearColormap(
            colors=colors, vmin=vmin, vmax=vmax, caption=env_label,
        )
        colormap.add_to(m)

    if protected is not None and not protected.empty:
        folium.GeoJson(
            protected.__geo_interface__,
            name="Protected Lands",
            show=True,
            style_function=lambda _: {
                "color": "#2ca02c",
                "weight": 1,
                "fillColor": "#2ca02c",
                "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["local_designation", "manager_agency", "designation_type", "public_access"],
                aliases=["Name", "Manager", "Type", "Public Access"],
            ),
        ).add_to(m)

    if target_parcels is not None and not target_parcels.empty:
        folium.GeoJson(
            target_parcels.__geo_interface__,
            name="Target Parcels",
            show=False,
            style_function=lambda _: {
                "color": "#e31a1c",
                "weight": 1.5,
                "fillColor": "#e31a1c",
                "fillOpacity": 0.15,
            },
        ).add_to(m)

    if otu_sites is not None and not otu_sites.empty:
        site_fg = folium.FeatureGroup(name="OTU Sample Sites")
        for _, row in otu_sites.iterrows():
            popup_html = (
                f"<b>Site: {row.get('site_id', '?')[:8]}...</b><br>"
                f"Lat: {row.get('lat', 0):.4f}<br>"
                f"Lon: {row.get('lon', 0):.4f}<br>"
                f"OTU observations: {row.get('n_observations', 0)}<br>"
                f"File: {row.get('file_path', '—')}"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7,
                color="#ff7800",
                fill=True,
                fillColor="#ff7800",
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(site_fg)
        site_fg.add_to(m)

    ClickCoveragePopup(api_url=BROWSER_API_URL).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Click-to-query coverage popup (runs in-browser via Leaflet JS)
# ---------------------------------------------------------------------------

BROWSER_API_URL = os.environ.get("BROWSER_API_URL", "http://localhost:8000")


class ClickCoveragePopup(MacroElement):
    """Leaflet click handler that fetches /env/coverage and shows a popup."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            var apiBase = {{ this.api_url|tojson }};

            map.on('click', function(e) {
                var lat = e.latlng.lat.toFixed(4);
                var lon = e.latlng.lng.toFixed(4);
                var popup = L.popup({maxWidth: 360, minWidth: 240})
                    .setLatLng(e.latlng)
                    .setContent('<div style="text-align:center;padding:8px"><i>Loading coverage…</i></div>')
                    .openOn(map);

                fetch(apiBase + '/env/coverage?lat=' + lat + '&lon=' + lon)
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        var html = '<div style="font-family:sans-serif;font-size:13px;line-height:1.5">';
                        html += '<b>Coverage at (' + lat + ', ' + lon + ')</b>';
                        html += '<hr style="margin:6px 0">';
                        var vars = data.variables || {};
                        for (var key in vars) {
                            var v = vars[key];
                            var icon = v.status === 'covered' ? '&#9989;' : '&#9888;&#65039;';
                            var val = v.value !== null && v.value !== undefined ? v.value + ' ' + v.unit : 'N/A';
                            var rng = (v.sampled_min !== null && v.sampled_max !== null)
                                ? '[' + v.sampled_min + ' – ' + v.sampled_max + ' ' + v.unit + ']'
                                : '[no data]';
                            html += icon + ' <b>' + v.label + '</b>: ' + val + '<br>';
                            html += '<span style="color:#666;margin-left:20px">Sampled: ' + rng + '</span><br>';
                        }
                        html += '</div>';
                        popup.setContent(html);
                    })
                    .catch(function(err) {
                        popup.setContent(
                            '<div style="font-family:sans-serif;color:#c00">'
                            + '<b>Error</b>: Could not fetch coverage.<br>'
                            + err.message + '</div>'
                        );
                    });
            });
        })();
        {% endmacro %}
    """)

    def __init__(self, api_url: str):
        super().__init__()
        self.api_url = api_url


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

engine = get_engine()

# Sidebar controls
with st.sidebar:
    st.subheader("Sample Controls")

    _layer_options = [None, GAP_LAYER_KEY] + list(ENV_LAYERS.keys())

    def _layer_label(key):
        if key is None:
            return "None"
        if key == GAP_LAYER_KEY:
            return "Sampling Gap"
        return ENV_LAYERS[key]["label"]

    env_layer = st.radio(
        "Environmental Layer",
        options=_layer_options,
        format_func=_layer_label,
        index=0,
    )

    st.divider()

    if st.button("Refresh data"):
        load_otu_sites.clear()
        load_protected_lands.clear()
        load_target_parcels_simple.clear()
        _load_raster_overlay.clear()
        _load_gap_overlay.clear()
        for key in list(st.session_state.keys()):
            if key.startswith("sample_"):
                del st.session_state[key]
        st.rerun()

    if st.button("Regenerate Gap Raster"):
        with st.spinner("Regenerating sampling gap raster..."):
            try:
                r = requests.post(f"{API_BASE}/env/refresh-gap-raster", timeout=120)
                if r.status_code == 200:
                    _load_gap_overlay.clear()
                    st.success("Gap raster regenerated.")
                    st.rerun()
                else:
                    st.error(f"Failed: {r.text}")
            except requests.ConnectionError:
                st.error(f"Cannot connect to ML API at {API_BASE}")
            except Exception as e:
                st.error(f"Error: {e}")

# Load data
with st.spinner("Loading data..."):
    otu_sites = load_otu_sites(engine)
    protected = load_protected_lands(engine)
    targets = load_target_parcels_simple(engine)

    env_overlay = None
    env_label = None
    env_cmap_name = None
    gap_overlay = None
    if env_layer == GAP_LAYER_KEY:
        gap_overlay = _load_gap_overlay()
    elif env_layer:
        env_overlay = _load_raster_overlay(env_layer)
        env_label = ENV_LAYERS[env_layer]["label"]
        env_cmap_name = ENV_LAYERS[env_layer]["cmap"]

n_sites = len(otu_sites) if otu_sites is not None else 0
n_protected = len(protected) if protected is not None else 0
st.caption(f"OTU sites: {n_sites} | Protected lands: {n_protected}")

# Map
map_obj = build_sample_map(
    env_overlay=env_overlay,
    env_label=env_label,
    env_cmap_name=env_cmap_name,
    otu_sites=otu_sites,
    protected=protected,
    target_parcels=targets,
    gap_overlay=gap_overlay,
)

st_folium(map_obj, width="stretch", height=550, returned_objects=[])

# Add New Sample form
st.divider()
st.subheader("Add New OTU Sample")

with st.form("upload_otu"):
    col1, col2 = st.columns(2)
    with col1:
        sample_lat = st.number_input("Latitude", value=37.5, min_value=36.0, max_value=40.0, step=0.01)
    with col2:
        sample_lon = st.number_input("Longitude", value=-79.0, min_value=-84.0, max_value=-75.0, step=0.01)

    otu_file = st.file_uploader(
        "OTU Table (CSV/TSV)",
        type=["csv", "tsv", "txt"],
        help="CSV or TSV with two columns: OTU_ID and relative abundance "
             "(normalised to 1). Each row is one OTU.",
    )

    submitted = st.form_submit_button("Upload Sample", type="primary")

    if submitted:
        if otu_file is None:
            st.error("Please select an OTU table file.")
        else:
            try:
                files = {"file": (otu_file.name, otu_file.getvalue(), "text/csv")}
                r = requests.post(
                    f"{API_BASE}/upload/otu",
                    files=files,
                    params={"lat": sample_lat, "lon": sample_lon},
                    timeout=30,
                )
                if r.status_code in (200, 201):
                    result = r.json()
                    st.success(
                        f"Uploaded! Site {result.get('site_id', '?')[:8]}... — "
                        f"{result.get('n_otus', 0)} OTUs, "
                        f"{result.get('n_observations', 0)} observations"
                    )
                    load_otu_sites.clear()
                    _load_gap_overlay.clear()
                    st.rerun()
                else:
                    st.error(f"Upload failed ({r.status_code}): {r.text}")
            except requests.ConnectionError:
                st.error(f"Cannot connect to ML API at {API_BASE} — is it running?")
            except Exception as e:
                st.error(f"Upload error: {e}")

# OTU Sites table
if otu_sites is not None and not otu_sites.empty:
    st.divider()
    st.subheader("OTU Sample Sites")
    display_df = otu_sites.drop(columns=["geom"], errors="ignore").copy()
    display_df["uploaded_at"] = display_df["uploaded_at"].astype(str)
    st.dataframe(display_df, width="stretch", hide_index=True)
