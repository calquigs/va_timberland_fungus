"""
Predict view – train model and click-to-predict fungal community.

Features:
- "Train Model" button that triggers XGBoost training via ML API
- Model status indicator with training metrics
- Click-to-predict: click on map to see predicted OTU composition
- Target parcels overlay
- Predicted OTU composition displayed as bar chart and table
- Confidence indicator based on environmental distance from training data
"""

import json
import os

import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text
from streamlit_folium import st_folium

from lib.constants import (
    ML_API_URL,
    POSITRON_ATTR,
    POSITRON_TILES,
    SATELLITE_ATTR,
    SATELLITE_TILES,
    VA_CENTER,
    VA_ZOOM,
)
from lib.db import get_engine

st.set_page_config(page_title="Predict | VA Woods", layout="wide")
st.title("Predict – Fungal Community Composition")

API_BASE = os.environ.get("ML_API_URL", ML_API_URL)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def load_target_parcels(_engine, limit: int = 2000) -> gpd.GeoDataFrame | None:
    """Load only parcels that are in the target shortlist (raw.target_parcels)."""
    queries = [
        (
            "marts + shortlist",
            text("""
                SELECT h.geom, h."OBJECTID", h."LOCALITY",
                       COALESCE(h.mean_biomass_kg_ha, 0) AS mean_biomass_kg_ha
                FROM marts.target_parcels h
                INNER JOIN raw.target_parcels t ON h."OBJECTID" = t.parcel_objectid
                WHERE h.geom IS NOT NULL
                LIMIT :lim
            """),
        ),
        (
            "staging + shortlist",
            text("""
                SELECT h.geom, h."OBJECTID", h."LOCALITY", 0 AS mean_biomass_kg_ha
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


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_model_status() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/model/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _train_model() -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/train", timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def _predict_at(lat: float, lon: float) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/predict", params={"lat": lat, "lon": lon}, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_predict_map(
    target_parcels: gpd.GeoDataFrame | None,
) -> folium.Map:

    m = folium.Map(
        location=VA_CENTER, zoom_start=VA_ZOOM,
        tiles=None, control_scale=True,
    )
    folium.TileLayer(tiles=POSITRON_TILES, attr=POSITRON_ATTR, name="Positron", show=True).add_to(m)
    folium.TileLayer(tiles=SATELLITE_TILES, attr=SATELLITE_ATTR, name="Satellite", show=False).add_to(m)

    if target_parcels is not None and not target_parcels.empty:
        folium.GeoJson(
            target_parcels.__geo_interface__,
            name="Target Parcels",
            style_function=lambda _: {
                "color": "#e31a1c",
                "weight": 1.5,
                "fillColor": "#e31a1c",
                "fillOpacity": 0.15,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["OBJECTID", "LOCALITY"],
                aliases=["Parcel ID", "Locality"],
            ),
        ).add_to(m)

    if "predict_marker" in st.session_state:
        lat, lon = st.session_state["predict_marker"]
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
            popup=f"Prediction point: ({lat:.4f}, {lon:.4f})",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

engine = get_engine()

# Sidebar: model controls
with st.sidebar:
    st.subheader("Model Controls")

    status = _get_model_status()
    if status:
        is_loaded = status.get("model_loaded", False)
        if is_loaded:
            st.success("Model is trained and loaded")
            metrics = status.get("metrics") or {}
            if metrics:
                st.metric("Training R2", f"{metrics.get('train_r2', 0):.3f}" if metrics.get("train_r2") is not None else "N/A")
                st.metric("Training Sites", metrics.get("n_sites", 0))
                st.metric("OTU Count", metrics.get("n_otus", 0))
            st.caption(f"Last trained: {status.get('last_trained_at', 'unknown')}")
        else:
            st.warning("No model trained yet")
    else:
        st.error("Cannot reach ML API")

    st.divider()

    if st.button("Train Model", type="primary", help="Train XGBoost on all OTU + environmental data"):
        with st.spinner("Training model..."):
            result = _train_model()
        if result and "error" not in result:
            st.success("Model trained successfully!")
            st.json(result.get("metrics", {}))
            st.rerun()
        else:
            error_msg = result.get("error", "Unknown error") if result else "No response"
            st.error(f"Training failed: {error_msg}")

    st.divider()
    if st.button("Refresh data"):
        load_target_parcels.clear()
        for key in list(st.session_state.keys()):
            if key.startswith("predict_"):
                del st.session_state[key]
        st.rerun()

st.markdown(
    "Click on the map to predict fungal community composition at that location. "
    "Target parcels are shown in red. Train a model first using the sidebar."
)

# Load target parcels
with st.spinner("Loading target parcels..."):
    targets = load_target_parcels(engine)

# Map
map_obj = build_predict_map(targets)
map_result = st_folium(map_obj, width="stretch", height=550, returned_objects=["last_clicked"])

# Click-to-predict: detect new clicks, fetch prediction, then rerun to show marker
last_clicked = map_result.get("last_clicked") if map_result else None
if last_clicked:
    click_lat = last_clicked["lat"]
    click_lon = last_clicked["lng"]

    current_marker = st.session_state.get("predict_marker")
    is_new_click = current_marker is None or (
        round(current_marker[0], 8) != round(click_lat, 8)
        or round(current_marker[1], 8) != round(click_lon, 8)
    )

    if is_new_click:
        st.session_state["predict_marker"] = (click_lat, click_lon)
        with st.spinner("Running prediction..."):
            st.session_state["predict_result"] = _predict_at(click_lat, click_lon)
        st.rerun()

# Display stored prediction results (persisted across reruns)
if "predict_marker" in st.session_state and "predict_result" in st.session_state:
    click_lat, click_lon = st.session_state["predict_marker"]
    pred = st.session_state["predict_result"]

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Prediction at ({click_lat:.4f}, {click_lon:.4f})")

        if pred is None:
            st.error("Could not get prediction — is the ML API running?")
        elif pred.get("prediction") is None:
            st.warning(pred.get("message", "Model not trained yet."))
        else:
            prediction = pred["prediction"]
            env = pred.get("env", {})
            confidence = pred.get("confidence", {})

            conf_status = confidence.get("status", "unknown")
            conf_colors = {"high": "green", "medium": "orange", "low": "red"}
            st.markdown(
                f"**Confidence**: :{conf_colors.get(conf_status, 'gray')}[{conf_status.upper()}] "
                f"(env distance: {confidence.get('distance', 'N/A')})"
            )

            st.markdown("**Environmental Variables:**")
            env_df = pd.DataFrame([
                {"Variable": k.replace("_", " ").title(), "Value": f"{v:.2f}" if v is not None else "N/A"}
                for k, v in env.items()
            ])
            st.dataframe(env_df, width="stretch", hide_index=True, height=220)

    with col2:
        if pred and pred.get("prediction"):
            st.subheader("Predicted OTU Composition")

            prediction = pred["prediction"]
            pred_df = pd.DataFrame([
                {"OTU": k, "Relative Abundance": round(v, 6)}
                for k, v in sorted(prediction.items(), key=lambda x: -x[1])
            ])

            st.bar_chart(
                pred_df.set_index("OTU"),
                width="stretch",
                height=400,
            )

            st.dataframe(pred_df, width="stretch", hide_index=True, height=300)
