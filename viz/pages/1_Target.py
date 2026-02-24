"""
Target view – harvested timber parcels ranked by loblolly biomass.

Features:
- Choropleth map colored by mean loblolly biomass (kg/ha) using QGIS Greens ramp
- All layers (parcels, harvest points, loblolly raster) loaded once; toggled via
  in-map LayerControl — no sidebar reloads
- Select a table row to highlight (magenta) and zoom to the parcel on the map —
  driven by a JS bridge between st.fragment and the Leaflet map, no map reload
- Table runs inside st.fragment so row selection never reloads the map
- Harvest point popups with full attribute data
- Mark / unmark parcel as inoculation target
"""

import html as html_lib
import json
import os
import re

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as st_components
from branca.element import MacroElement
from jinja2 import Template
from sqlalchemy import text
from streamlit_folium import st_folium

from lib.constants import (
    HARVEST_POINT,
    HARVESTED_FILL,
    API_URL,
    PARCEL_BORDER,
    POSITRON_ATTR,
    POSITRON_TILES,
    SATELLITE_ATTR,
    SATELLITE_TILES,
    VA_CENTER,
    VA_ZOOM,
)
from lib.db import get_engine
from lib.raster_utils import get_loblolly_overlay

st.set_page_config(page_title="Target | VA Woods", layout="wide")
st.title("Target Parcels")

API_BASE = os.environ.get("API_URL", API_URL)
TABLE_PAGE_SIZE = 50

_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\\]")


def _safe_html(val) -> str:
    """Sanitise a value for safe embedding in Folium popup HTML/JS."""
    s = _CTRL_RE.sub("", str(val))
    return html_lib.escape(s)


# ---------------------------------------------------------------------------
# Leaflet JavaScript helpers (run client-side, no Python reruns)
# ---------------------------------------------------------------------------

class _PersistMapView(MacroElement):
    """Save/restore map center+zoom via browser sessionStorage.

    On mount the saved view is restored *before* tile layers initialise,
    so the user never sees the default VA-wide view flash.  Every pan/zoom
    silently updates sessionStorage for the next mount.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            var key = 'va_target_map_view';
            try {
                var s = JSON.parse(sessionStorage.getItem(key));
                if (s && s.lat != null && s.lng != null && s.z != null) {
                    map.setView([s.lat, s.lng], s.z);
                }
            } catch(_) {}
            map.on('moveend', function() {
                try {
                    var c = map.getCenter();
                    sessionStorage.setItem(key, JSON.stringify({
                        lat: c.lat, lng: c.lng, z: map.getZoom()
                    }));
                } catch(_) {}
            });
        })();
        {% endmacro %}
    """)


class _SelectionBridge(MacroElement):
    """Expose a ``_vaTargetSelect(oid)`` function on ``window.parent``.

    The table fragment calls this function (via ``st.html``) whenever the
    selected row changes.  Because the function lives in the st_folium
    iframe and manipulates the Leaflet map directly, it highlights + zooms
    without any Python rerun or map reload.

    Passing ``null`` clears the current highlight.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var parcelLayer = {{ this._parent.get_name() }};
            var map = {{ this._parent._parent.get_name() }};
            var hlLayer = null;

            function selectParcel(objectId) {
                if (hlLayer) { map.removeLayer(hlLayer); hlLayer = null; }
                if (objectId == null) return;

                parcelLayer.eachLayer(function(layer) {
                    var props = layer.feature && layer.feature.properties;
                    if (props && props.OBJECTID == objectId) {
                        hlLayer = L.geoJSON(layer.feature, {
                            style: {
                                color: '#ff00ff',
                                weight: 4,
                                fillColor: '#ff00ff',
                                fillOpacity: 0.08,
                                interactive: false
                            }
                        }).addTo(map);
                        map.fitBounds(layer.getBounds().pad(0.5), {maxZoom: 15});
                    }
                });
            }

            try { window.parent._vaTargetSelect = selectParcel; } catch(_) {}
            window._vaTargetSelect = selectParcel;
        })();
        {% endmacro %}
    """)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _marts_table_exists(engine) -> bool:
    """Check whether the marts.target_parcels table has been materialised."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'marts' AND table_name = 'target_parcels'
            )
        """)).scalar()
    return bool(row)


@st.cache_data(ttl=120, show_spinner=False)
def load_target_parcels(
    _engine,
    bbox: tuple | None = None,
    sort_by: str = "total_biomass_kg",
    sort_dir: str = "desc",
    limit: int = 500,
    offset: int = 0,
    use_mart: bool = True,
) -> gpd.GeoDataFrame | None:
    """Load parcels from marts.target_parcels (preferred) or staging fallback."""
    if use_mart:
        base_table = "marts.target_parcels"
        cols = """
            "OBJECTID", geom, "PARCELID", "FIPS", "LOCALITY",
            area_acres, harvest_number, harvest_start_date, harvest_status,
            harvest_acres_to_date_1,
            mean_biomass_kg_ha, total_biomass_kg, biomass_pixel_count,
            target_marked_at, is_target
        """
    else:
        base_table = "staging.va_harvested_timber_parcels"
        cols = """
            "OBJECTID", geom, "PARCELID", "FIPS", "LOCALITY",
            "Shape__Area" / 4046.86 AS area_acres,
            harvest_number, harvest_start_date, harvest_status,
            harvest_acres_to_date_1,
            NULL::double precision AS mean_biomass_kg_ha,
            NULL::double precision AS total_biomass_kg,
            NULL::integer AS biomass_pixel_count,
            NULL::timestamptz AS target_marked_at,
            false AS is_target
        """

    where_clauses = ["geom IS NOT NULL"]
    params: dict = {}

    if bbox:
        minx, miny, maxx, maxy = bbox
        where_clauses.append(
            "ST_Intersects(geom, ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))"
        )
        params.update(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    where_sql = " AND ".join(where_clauses)

    allowed_sort = {
        "total_biomass_kg", "mean_biomass_kg_ha", "area_acres",
        "harvest_start_date", "OBJECTID", "LOCALITY",
    }
    if sort_by not in allowed_sort:
        sort_by = "total_biomass_kg"
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"
    order_sql = f'"{sort_by}"' if sort_by in ("OBJECTID", "LOCALITY") else sort_by
    nulls = "NULLS LAST" if direction == "DESC" else "NULLS FIRST"

    sql = text(f"""
        SELECT {cols}
        FROM {base_table}
        WHERE {where_sql}
        ORDER BY {order_sql} {direction} {nulls}
        LIMIT :lim OFFSET :off
    """)
    params.update(lim=limit, off=offset)

    try:
        gdf = gpd.read_postgis(sql, _engine, geom_col="geom", crs="EPSG:4326", params=params)
        return gdf if not gdf.empty else None
    except Exception as e:
        st.warning(f"Could not load target parcels: {e}")
        return None


@st.cache_data(ttl=30, show_spinner=False)
def get_target_parcel_ids(_engine) -> set:
    """Return set of OBJECTIDs that are in the target shortlist (raw.target_parcels)."""
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT parcel_objectid FROM raw.target_parcels"
            )).fetchall()
        return {r[0] for r in rows} if rows else set()
    except Exception:
        return set()


@st.cache_data(ttl=120, show_spinner=False)
def get_total_count(_engine, use_mart: bool) -> int:
    table = "marts.target_parcels" if use_mart else "staging.va_harvested_timber_parcels"
    with _engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE geom IS NOT NULL")).scalar() or 0


@st.cache_data(ttl=300, show_spinner=False)
def load_harvests(_engine, limit: int = 5000) -> gpd.GeoDataFrame | None:
    try:
        sql = text("""
            SELECT * FROM raw.va_harvests
            WHERE geom IS NOT NULL
            LIMIT :lim
        """)
        gdf = gpd.read_postgis(sql, _engine, geom_col="geom", crs="EPSG:4326", params={"lim": limit})
        return gdf if not gdf.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_target_map(
    gdf: gpd.GeoDataFrame,
    harvests: gpd.GeoDataFrame | None = None,
    raster_overlay: tuple | None = None,
) -> folium.Map:
    """Build a choropleth map of harvested parcels.

    Always initialises at VA_CENTER / VA_ZOOM so the generated JS is
    identical across reruns (stable hash → st_folium keeps the component
    mounted).  _PersistMapView restores the user's last view from
    sessionStorage before tiles start loading.  _SelectionBridge exposes
    a JS function that the table fragment calls to highlight + zoom.
    """
    m = folium.Map(
        location=VA_CENTER,
        zoom_start=VA_ZOOM,
        tiles=None,
        control_scale=True,
    )

    # Restore saved view before any tiles are requested
    _PersistMapView().add_to(m)

    folium.TileLayer(
        tiles=POSITRON_TILES, attr=POSITRON_ATTR, name="Positron",
        control=True, show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles=SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite",
        control=True, show=False,
    ).add_to(m)

    if raster_overlay is not None:
        data_uri, bounds = raster_overlay
        folium.raster_layers.ImageOverlay(
            image=data_uri,
            bounds=bounds,
            name="Loblolly Biomass (raster)",
            opacity=0.7,
            interactive=False,
            show=False,
        ).add_to(m)

    if gdf is None or gdf.empty:
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    has_biomass = gdf["mean_biomass_kg_ha"].notna().any()

    if has_biomass:
        vmin = float(gdf["mean_biomass_kg_ha"].min())
        vmax = float(gdf["mean_biomass_kg_ha"].max())
        if vmin == vmax:
            vmax = vmin + 1
        colormap = cm.LinearColormap(
            colors=[
                "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476",
                "#41ab5d", "#238b45", "#006d2c", "#00441b",
            ],
            vmin=vmin,
            vmax=vmax,
            caption="Mean Loblolly Biomass (kg/ha)",
        )
        colormap.add_to(m)

    def style_fn(feature):
        props = feature.get("properties", {})
        biomass = props.get("mean_biomass_kg_ha")
        is_target = props.get("is_target", False)

        if has_biomass and biomass is not None:
            fill = colormap(biomass)
        else:
            fill = HARVESTED_FILL

        if is_target:
            border = "#e31a1c"
            weight = 2.5
            opacity = 0.55
        else:
            border = PARCEL_BORDER
            weight = 1.2
            opacity = 0.55

        return {
            "color": border,
            "weight": weight,
            "fillColor": fill,
            "fillOpacity": opacity,
        }

    def highlight_fn(_feature):
        return {"weight": 3, "fillOpacity": 0.75}

    ts_cols = gdf.select_dtypes(include=["datetime", "datetimetz"]).columns
    if len(ts_cols):
        gdf = gdf.copy()
        for c in ts_cols:
            gdf[c] = gdf[c].dt.strftime("%Y-%m-%d").fillna("")
    geojson_data = json.loads(gdf.to_json())

    for feature in geojson_data["features"]:
        props = feature["properties"]
        oid = props.get("OBJECTID", "?")
        locality = props.get("LOCALITY", "—")
        acres = props.get("area_acres")
        acres_str = f"{acres:,.1f}" if acres else "—"
        biomass_mean = props.get("mean_biomass_kg_ha")
        biomass_total = props.get("total_biomass_kg")
        harvest_date = props.get("harvest_start_date", "—")
        is_target = props.get("is_target", False)

        popup_html = (
            f"<b>Parcel {_safe_html(oid)}</b><br>"
            f"Locality: {_safe_html(locality)}<br>"
            f"Area: {acres_str} ac<br>"
            f"Harvest date: {_safe_html(harvest_date)}<br>"
            f"Mean biomass: {f'{biomass_mean:,.1f} kg/ha' if biomass_mean else '—'}<br>"
            f"Total biomass: {f'{biomass_total:,.0f} kg' if biomass_total else '—'}<br>"
            f"Target: {'Yes' if is_target else 'No'}"
        )
        feature["properties"]["__popup"] = popup_html

    geojson_layer = folium.GeoJson(
        geojson_data,
        name="Harvested parcels",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["OBJECTID", "LOCALITY", "area_acres", "mean_biomass_kg_ha"],
            aliases=["ID", "Locality", "Acres", "Biomass (kg/ha)"],
            localize=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["__popup"],
            aliases=[""],
            labels=False,
            parse_html=True,
        ),
    )
    _SelectionBridge().add_to(geojson_layer)
    geojson_layer.add_to(m)

    if harvests is not None and not harvests.empty:
        harvest_fg = folium.FeatureGroup(name="Harvest points", show=False)
        geom_col = harvests.geometry.name
        skip_cols = {geom_col, "index", "level_0"}
        attr_cols = [c for c in harvests.columns if c not in skip_cols]
        for _, row in harvests.iterrows():
            geom = row[geom_col]
            if geom is None or geom.is_empty:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            lines = []
            for col in attr_cols:
                val = row.get(col)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                label = col.replace("_", " ").title()
                lines.append(f"<b>{_safe_html(label)}</b>: {_safe_html(val)}")
            popup_html = "<br>".join(lines) if lines else "(no attributes)"
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=3,
                color=HARVEST_POINT,
                fill=True,
                fillColor=HARVEST_POINT,
                fillOpacity=0.8,
                weight=1,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(harvest_fg)
        harvest_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_mark_target(parcel_id: int) -> bool:
    try:
        r = requests.post(f"{API_BASE}/parcels/{parcel_id}/target", timeout=5)
        return r.status_code in (200, 201)
    except Exception:
        return False


def _api_unmark_target(parcel_id: int) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/parcels/{parcel_id}/target", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

engine = get_engine()
use_mart = _marts_table_exists(engine)

if not use_mart:
    st.caption("*marts.target_parcels not found — using staging fallback (no biomass data)*")

# Sidebar controls
with st.sidebar:
    st.subheader("Target Controls")
    if st.button("Refresh data", help="Reload from database (use after running dbt or loading new loblolly data)"):
        load_target_parcels.clear()
        get_total_count.clear()
        load_harvests.clear()
        st.rerun()
    sort_col = st.selectbox(
        "Sort by",
        options=["total_biomass_kg", "mean_biomass_kg_ha", "area_acres", "harvest_start_date"],
        format_func=lambda x: {
            "total_biomass_kg": "Total Biomass",
            "mean_biomass_kg_ha": "Mean Biomass",
            "area_acres": "Area (acres)",
            "harvest_start_date": "Harvest Date",
        }.get(x, x),
    )
    sort_dir = st.radio("Order", ["desc", "asc"], format_func=lambda x: "Descending" if x == "desc" else "Ascending")
    row_limit = st.slider("Parcels to load", min_value=50, max_value=2000, value=200, step=50)

total_count = get_total_count(engine, use_mart)
st.caption(f"Total harvested parcels in database: **{total_count:,}**")

# Load data
with st.spinner("Loading parcels…"):
    gdf = load_target_parcels(
        engine,
        sort_by=sort_col,
        sort_dir=sort_dir,
        limit=row_limit,
        use_mart=use_mart,
    )

with st.spinner("Loading map layers…"):
    harvests_gdf = load_harvests(engine)
    raster_data = get_loblolly_overlay()

if gdf is None or gdf.empty:
    st.info(
        "No harvested parcels found. Make sure ingestion has been run "
        "(va_parcels --harvested-only) and dbt models are built."
    )
    st.stop()

target_ids = get_target_parcel_ids(engine)
gdf = gdf.copy()
gdf["is_target"] = gdf["OBJECTID"].isin(target_ids)

# ---------------------------------------------------------------------------
# Map (outside fragment — only re-executes on full app reruns)
# ---------------------------------------------------------------------------
st.caption(f"Showing **{len(gdf):,}** parcels (sorted by {sort_col} {sort_dir})")
st.caption(
    "Select a row in the table to highlight and zoom to the parcel on the map. "
    "Toggle layers with the legend (top-right)."
)

map_obj = build_target_map(gdf, harvests=harvests_gdf, raster_overlay=raster_data)
st_folium(map_obj, width="stretch", height=550, returned_objects=[])

# ---------------------------------------------------------------------------
# Table + detail panel (inside fragment — row selection never reloads the map)
# ---------------------------------------------------------------------------
display_cols = [
    "OBJECTID", "LOCALITY", "area_acres",
    "mean_biomass_kg_ha", "total_biomass_kg",
    "harvest_start_date", "is_target",
]
available_cols = [c for c in display_cols if c in gdf.columns]
table_df = (
    gdf[available_cols]
    .copy()
    .rename(columns={
        "OBJECTID": "ID",
        "LOCALITY": "Locality",
        "area_acres": "Acres",
        "mean_biomass_kg_ha": "Mean Biomass (kg/ha)",
        "total_biomass_kg": "Total Biomass (kg)",
        "harvest_start_date": "Harvest Date",
        "is_target": "Target",
    })
)


@st.fragment
def _parcel_table():
    st.subheader("Parcel Table")

    event = st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=400,
    )

    selected_rows = event.selection.rows if event.selection else []
    if not selected_rows:
        st_components.html(
            "<script>"
            "try{(window.parent._vaTargetSelect||window._vaTargetSelect||function(){})(null)}"
            "catch(e){}"
            "</script>",
            height=0,
        )
        return

    idx = selected_rows[0]
    selected_parcel = gdf.iloc[idx]
    oid = int(selected_parcel["OBJECTID"])

    # Tell the Leaflet map to highlight + zoom to this parcel
    st_components.html(
        f"<script>"
        f"try{{(window.parent._vaTargetSelect||window._vaTargetSelect||function(){{}})({oid})}}"
        f"catch(e){{}}"
        f"</script>",
        height=0,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### Parcel {oid}")
        detail_items = {
            "Locality": selected_parcel.get("LOCALITY", "—"),
            "Area": f"{selected_parcel.get('area_acres', 0):,.1f} acres",
            "Harvest Date": selected_parcel.get("harvest_start_date", "—"),
            "Harvest Status": selected_parcel.get("harvest_status", "—"),
            "Mean Biomass": f"{selected_parcel.get('mean_biomass_kg_ha', 0) or 0:,.1f} kg/ha",
            "Total Biomass": f"{selected_parcel.get('total_biomass_kg', 0) or 0:,.0f} kg",
        }
        for label, val in detail_items.items():
            st.markdown(f"**{label}:** {val}")

    with col2:
        is_targeted = bool(selected_parcel.get("is_target", False))
        if is_targeted:
            if st.button("Remove Target", type="secondary", width="stretch"):
                if _api_unmark_target(oid):
                    st.success(f"Parcel {oid} removed from targets")
                    get_target_parcel_ids.clear()
                    load_target_parcels.clear()
                    st.rerun(scope="app")
                else:
                    st.error("Failed to update — is the ML API running?")
        else:
            if st.button("Mark as Target", type="primary", width="stretch"):
                if _api_mark_target(oid):
                    st.success(f"Parcel {oid} marked as target")
                    get_target_parcel_ids.clear()
                    load_target_parcels.clear()
                    st.rerun(scope="app")
                else:
                    st.error("Failed to update — is the ML API running?")


_parcel_table()
