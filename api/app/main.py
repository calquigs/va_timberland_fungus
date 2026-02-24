"""
ML API: OTU upload, train, predict, environmental coverage, target parcels.

Endpoints:
  POST /upload/otu           – upload OTU CSV with lat/lon
  POST /train                – train XGBoost model on OTU + env
  GET  /predict              – predict fungal community at (lat, lon)
  GET  /model/status         – model info and metrics
  GET  /env/coverage         – parameter space coverage at a point
  POST /env/refresh-gap-raster – regenerate the sampling-gap raster
  POST /parcels/{id}/target  – mark parcel as target
  DELETE /parcels/{id}/target – unmark parcel
  GET  /target-parcels       – paginated GeoJSON of target parcels
  GET  /otu-sites            – all OTU sample sites as GeoJSON
"""

import csv
import io
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://va_woods:va_woods_dev@localhost:5432/va_woods",
)
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

ENV_RASTER_DIR = Path(
    os.environ.get("ENV_RASTER_DIR", Path(__file__).resolve().parents[2] / "data" / "environmental")
)

RASTER_FILES = {
    "elevation_m": "elevation_va.tif",
    "slope_deg": "slope_va.tif",
    "tmax_c": "tmax_va.tif",
    "tmin_c": "tmin_va.tif",
    "precip_mm": "ppt_va.tif",
}


# ---------------------------------------------------------------------------
# Raster point sampling (used by /env/coverage and /predict)
# ---------------------------------------------------------------------------

def _sample_env_at_point(lon: float, lat: float) -> dict:
    """Sample all environmental rasters at a single (lon, lat)."""
    result = {}
    for var_name, filename in RASTER_FILES.items():
        raster_path = ENV_RASTER_DIR / filename
        if not raster_path.exists():
            result[var_name] = None
            continue
        try:
            import rasterio
            with rasterio.open(raster_path) as ds:
                vals = list(ds.sample([(lon, lat)]))
                val = float(vals[0][0]) if vals else None
                if val is not None and (val < -9000 or val > 1e30 or np.isnan(val)):
                    val = None
                result[var_name] = val
        except Exception:
            result[var_name] = None
    return result


# ---------------------------------------------------------------------------
# Lifespan: create tables on startup
# ---------------------------------------------------------------------------

def _ensure_tables():
    if not engine:
        return
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.target_parcels (
                parcel_objectid INTEGER PRIMARY KEY,
                marked_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.otu_sites (
                site_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                lat DOUBLE PRECISION NOT NULL,
                lon DOUBLE PRECISION NOT NULL,
                uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                file_path TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.otu_observations (
                id SERIAL PRIMARY KEY,
                site_id UUID NOT NULL REFERENCES raw.otu_sites(site_id) ON DELETE CASCADE,
                otu_id TEXT NOT NULL,
                abundance DOUBLE PRECISION NOT NULL DEFAULT 0
            )
        """))
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.model_runs (
                run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                n_sites INTEGER,
                n_otus INTEGER,
                metrics JSONB,
                artifact_path TEXT
            )
        """))
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_tables()
    yield


app = FastAPI(
    title="VA Woods API",
    description="Train and predict soil fungal community from environmental variables.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model state
_model_state: dict = {
    "model": None,
    "otu_ids": None,
    "last_trained_at": None,
    "metrics": None,
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Target parcel management
# ---------------------------------------------------------------------------

@app.post("/parcels/{parcel_id}/target")
def mark_parcel_as_target(parcel_id: int = PathParam(...)):
    if not engine:
        raise HTTPException(503, "Database not configured")
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO raw.target_parcels (parcel_objectid)
            VALUES (:pid)
            ON CONFLICT (parcel_objectid) DO UPDATE SET marked_at = now()
        """), {"pid": parcel_id})
        conn.commit()
    return {"parcel_objectid": parcel_id, "status": "targeted"}


@app.delete("/parcels/{parcel_id}/target")
def unmark_parcel_as_target(parcel_id: int = PathParam(...)):
    if not engine:
        raise HTTPException(503, "Database not configured")
    with engine.connect() as conn:
        result = conn.execute(text(
            "DELETE FROM raw.target_parcels WHERE parcel_objectid = :pid"
        ), {"pid": parcel_id})
        conn.commit()
    if result.rowcount == 0:
        raise HTTPException(404, f"Parcel {parcel_id} was not targeted")
    return {"parcel_objectid": parcel_id, "status": "untargeted"}


# ---------------------------------------------------------------------------
# OTU upload
# ---------------------------------------------------------------------------

@app.post("/upload/otu")
async def upload_otu(
    file: UploadFile = File(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Upload an OTU table (CSV/TSV) with lat/lon. Stores site + observations in DB.

    Expected format: two columns — an OTU ID column and a relative abundance column
    (values normalised to 1). Each row is one OTU.
    """
    if not engine:
        raise HTTPException(503, "Database not configured")

    content = await file.read()
    text_content = content.decode("utf-8-sig")

    dialect = csv.Sniffer().sniff(text_content[:2048], delimiters=",\t")
    reader = csv.DictReader(io.StringIO(text_content), delimiter=dialect.delimiter)

    if not reader.fieldnames:
        raise HTTPException(422, "CSV has no columns")

    fields = [c.strip() for c in reader.fieldnames]

    ID_NAMES = {"otu_id", "otu", "id", "taxon_id", "taxon"}
    id_col = None
    for c in fields:
        if c.lower() in ID_NAMES:
            id_col = c
            break
    if id_col is None:
        raise HTTPException(
            422, "No OTU ID column found. Expected a column named 'OTU_ID'."
        )

    abundance_cols = [c for c in fields if c != id_col]
    if not abundance_cols:
        raise HTTPException(422, "No abundance column found alongside the OTU ID column.")
    abundance_col = abundance_cols[0]

    rows = list(reader)
    if not rows:
        raise HTTPException(422, "CSV has no data rows")

    site_id = str(uuid.uuid4())

    observations = []
    for row in rows:
        otu_id = (row.get(id_col) or "").strip()
        if not otu_id:
            continue
        raw_val = (row.get(abundance_col) or "0").strip()
        try:
            abundance = float(raw_val) if raw_val else 0.0
        except ValueError:
            abundance = 0.0
        if abundance > 0:
            observations.append({"otu_id": otu_id, "abundance": abundance})

    if not observations:
        raise HTTPException(422, "No non-zero OTU abundances found")

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO raw.otu_sites (site_id, lat, lon, file_path)
            VALUES (:sid, :lat, :lon, :fp)
        """), {"sid": site_id, "lat": lat, "lon": lon, "fp": file.filename})

        for obs in observations:
            conn.execute(text("""
                INSERT INTO raw.otu_observations (site_id, otu_id, abundance)
                VALUES (:sid, :otu_id, :abundance)
            """), {"sid": site_id, "otu_id": obs["otu_id"], "abundance": obs["abundance"]})

        env_vals = _sample_env_at_point(lon, lat)
        conn.execute(text("""
            INSERT INTO raw.env_point_samples
                (point_id, point_type, lat, lon, elevation_m, slope_deg, tmax_c, tmin_c, precip_mm)
            VALUES (:pid, 'otu_site', :lat, :lon, :elev, :slope, :tmax, :tmin, :precip)
            ON CONFLICT (point_id, point_type) DO UPDATE SET
                elevation_m = EXCLUDED.elevation_m,
                slope_deg = EXCLUDED.slope_deg,
                tmax_c = EXCLUDED.tmax_c,
                tmin_c = EXCLUDED.tmin_c,
                precip_mm = EXCLUDED.precip_mm
        """), {
            "pid": site_id, "lat": lat, "lon": lon,
            "elev": env_vals.get("elevation_m"),
            "slope": env_vals.get("slope_deg"),
            "tmax": env_vals.get("tmax_c"),
            "tmin": env_vals.get("tmin_c"),
            "precip": env_vals.get("precip_mm"),
        })
        conn.commit()

    threading.Thread(target=_regenerate_gap_raster_background, daemon=True).start()

    return JSONResponse(status_code=201, content={
        "site_id": site_id,
        "lat": lat,
        "lon": lon,
        "filename": file.filename,
        "n_otus": len(observations),
        "n_observations": len(observations),
        "env": env_vals,
    })


# ---------------------------------------------------------------------------
# OTU sites listing
# ---------------------------------------------------------------------------

@app.get("/otu-sites")
def list_otu_sites():
    """Return all OTU sample sites as GeoJSON."""
    if not engine:
        return {"type": "FeatureCollection", "features": []}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT s.site_id::text, s.lat, s.lon, s.uploaded_at,
                       s.file_path, COUNT(o.id) AS n_observations
                FROM raw.otu_sites s
                LEFT JOIN raw.otu_observations o ON s.site_id = o.site_id
                GROUP BY s.site_id, s.lat, s.lon, s.uploaded_at, s.file_path
                ORDER BY s.uploaded_at DESC
            """)).fetchall()

        features = []
        for r in rows:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r[2], r[1]]},
                "properties": {
                    "site_id": r[0],
                    "lat": r[1],
                    "lon": r[2],
                    "uploaded_at": r[3].isoformat() if r[3] else None,
                    "file_path": r[4],
                    "n_observations": r[5],
                },
            })
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.exception("Failed to list OTU sites")
        return {"type": "FeatureCollection", "features": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Environmental coverage analysis
# ---------------------------------------------------------------------------

@app.get("/env/coverage")
def env_coverage(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Assess how well a point's environment is covered by OTU sampling."""
    if not engine:
        raise HTTPException(503, "Database not configured")

    env_at_point = _sample_env_at_point(lon, lat)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                MIN(elevation_m) AS elev_min, MAX(elevation_m) AS elev_max,
                MIN(slope_deg) AS slope_min, MAX(slope_deg) AS slope_max,
                MIN(tmax_c) AS tmax_min, MAX(tmax_c) AS tmax_max,
                MIN(tmin_c) AS tmin_min, MAX(tmin_c) AS tmin_max,
                MIN(precip_mm) AS precip_min, MAX(precip_mm) AS precip_max,
                COUNT(*) AS n_sites
            FROM raw.env_point_samples
            WHERE point_type = 'otu_site'
        """)).fetchone()

    n_sites = rows[10] if rows else 0
    if n_sites == 0:
        return {
            "lat": lat, "lon": lon,
            "n_sample_sites": 0,
            "variables": {},
            "message": "No OTU sites have been sampled yet.",
        }

    var_map = {
        "elevation_m": (rows[0], rows[1]),
        "slope_deg": (rows[2], rows[3]),
        "tmax_c": (rows[4], rows[5]),
        "tmin_c": (rows[6], rows[7]),
        "precip_mm": (rows[8], rows[9]),
    }

    LABELS = {
        "elevation_m": "Elevation",
        "slope_deg": "Slope",
        "tmax_c": "Annual High Temp",
        "tmin_c": "Annual Low Temp",
        "precip_mm": "Annual Precip",
    }
    UNITS = {
        "elevation_m": "m",
        "slope_deg": "deg",
        "tmax_c": "C",
        "tmin_c": "C",
        "precip_mm": "mm",
    }

    variables = {}
    max_gap = 0.0
    for var, (smin, smax) in var_map.items():
        val = env_at_point.get(var)
        if val is None or smin is None or smax is None:
            variables[var] = {
                "label": LABELS[var], "unit": UNITS[var],
                "value": val, "sampled_min": smin, "sampled_max": smax,
                "status": "no_data",
            }
            continue

        if val < smin:
            status = "below"
            gap = smin - val
        elif val > smax:
            status = "above"
            gap = val - smax
        else:
            status = "covered"
            gap = 0.0

        rng = (smax - smin) if smax > smin else 1.0
        norm_gap = gap / rng
        max_gap = max(max_gap, norm_gap)

        variables[var] = {
            "label": LABELS[var], "unit": UNITS[var],
            "value": round(val, 2),
            "sampled_min": round(smin, 2),
            "sampled_max": round(smax, 2),
            "status": status,
            "gap": round(gap, 2),
            "normalized_gap": round(norm_gap, 3),
        }

    return {
        "lat": lat, "lon": lon,
        "n_sample_sites": n_sites,
        "max_normalized_gap": round(max_gap, 3),
        "overall_status": "covered" if max_gap == 0 else "gap",
        "variables": variables,
    }


# ---------------------------------------------------------------------------
# Sampling-gap raster generation
# ---------------------------------------------------------------------------

def _regenerate_gap_raster_background():
    """Run gap raster generation in a background thread (fire-and-forget)."""
    try:
        from app.gap_raster import generate_gap_raster
        generate_gap_raster(env_dir=ENV_RASTER_DIR, engine=engine)
    except Exception:
        logger.exception("Background gap raster generation failed")


@app.post("/env/refresh-gap-raster")
def refresh_gap_raster(background: bool = Query(False)):
    """Regenerate the sampling-gap raster from current OTU sample bounds."""
    if not engine:
        raise HTTPException(503, "Database not configured")

    from app.gap_raster import generate_gap_raster

    if background:
        threading.Thread(target=_regenerate_gap_raster_background, daemon=True).start()
        return {"status": "started", "message": "Gap raster regeneration started in background."}

    result = generate_gap_raster(env_dir=ENV_RASTER_DIR, engine=engine)
    if result is None:
        raise HTTPException(
            400,
            "Could not generate gap raster — no OTU sample sites or missing env rasters.",
        )
    return {
        "status": "complete",
        "raster_path": str(result),
        "overlay_path": str(result.parent / "sampling_gap_overlay.png"),
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@app.get("/model/status")
def model_status():
    return {
        "model_loaded": _model_state["model"] is not None,
        "last_trained_at": _model_state["last_trained_at"],
        "metrics": _model_state["metrics"],
        "n_otus": len(_model_state["otu_ids"]) if _model_state["otu_ids"] is not None else 0,
    }


@app.post("/train")
def train():
    """Train XGBoost multi-output model on OTU + environmental data."""
    if not engine:
        raise HTTPException(503, "Database not configured")

    with engine.connect() as conn:
        sites = pd.read_sql(text("""
            SELECT s.site_id, e.elevation_m, e.slope_deg, e.tmax_c, e.tmin_c, e.precip_mm
            FROM raw.otu_sites s
            JOIN raw.env_point_samples e
                ON s.site_id::text = e.point_id AND e.point_type = 'otu_site'
        """), conn)

        if sites.empty:
            raise HTTPException(400, "No OTU sites with environmental data found. Upload OTU data first.")

        obs = pd.read_sql(text("""
            SELECT site_id::text, otu_id, abundance
            FROM raw.otu_observations
        """), conn)

    if obs.empty:
        raise HTTPException(400, "No OTU observations found.")

    sites["site_id"] = sites["site_id"].astype(str)
    otu_wide = obs.pivot_table(index="site_id", columns="otu_id", values="abundance", fill_value=0)

    merged = sites.set_index("site_id").join(otu_wide, how="inner")
    if merged.empty or len(merged) < 2:
        raise HTTPException(400, f"Need at least 2 sites with both env and OTU data (have {len(merged)})")

    feature_cols = ["elevation_m", "slope_deg", "tmax_c", "tmin_c", "precip_mm"]
    otu_cols = [c for c in merged.columns if c not in feature_cols]

    X = merged[feature_cols].fillna(0).values
    Y = merged[otu_cols].fillna(0).values

    try:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBRegressor

        base = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1,
        )
        model = MultiOutputRegressor(base)
        model.fit(X, Y)

        n_cv = min(5, len(X))
        if n_cv >= 2:
            from sklearn.metrics import r2_score
            Y_pred = model.predict(X)
            train_r2 = float(r2_score(Y, Y_pred, multioutput="variance_weighted"))
        else:
            train_r2 = None

    except ImportError as e:
        raise HTTPException(500, f"Missing ML dependency: {e}. Install xgboost and scikit-learn.")

    _model_state["model"] = model
    _model_state["otu_ids"] = list(otu_cols)
    from datetime import datetime, timezone
    _model_state["last_trained_at"] = datetime.now(timezone.utc).isoformat()
    _model_state["metrics"] = {
        "n_sites": len(merged),
        "n_otus": len(otu_cols),
        "n_features": len(feature_cols),
        "train_r2": train_r2,
    }

    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO raw.model_runs (n_sites, n_otus, metrics)
                    VALUES (:ns, :no, :m)
                """), {
                    "ns": len(merged),
                    "no": len(otu_cols),
                    "m": json.dumps(_model_state["metrics"]),
                })
                conn.commit()
        except Exception as e:
            logger.warning("Could not save model run to DB: %s", e)

    return {
        "message": "Training complete",
        "last_trained_at": _model_state["last_trained_at"],
        "metrics": _model_state["metrics"],
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.get("/predict")
def predict(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Predict fungal community composition at (lat, lon)."""
    model = _model_state["model"]
    if model is None:
        return {
            "lat": lat, "lon": lon,
            "prediction": None,
            "message": "Model not trained yet. Click 'Train Model' first.",
        }

    env = _sample_env_at_point(lon, lat)
    feature_cols = ["elevation_m", "slope_deg", "tmax_c", "tmin_c", "precip_mm"]
    X = np.array([[env.get(c) or 0.0 for c in feature_cols]])

    Y_pred = model.predict(X)[0]
    Y_pred = np.clip(Y_pred, 0, None)
    total = Y_pred.sum()
    if total > 0:
        Y_rel = (Y_pred / total).round(6)
    else:
        Y_rel = Y_pred

    otu_ids = _model_state["otu_ids"]
    composition = {otu_ids[i]: float(Y_rel[i]) for i in range(len(otu_ids))}

    env_dist = _compute_env_distance(env)

    return {
        "lat": lat, "lon": lon,
        "env": env,
        "prediction": composition,
        "all_otus": len(composition),
        "confidence": env_dist,
    }


def _compute_env_distance(env: dict) -> dict:
    """Compute distance from training data in feature space."""
    if not engine:
        return {"distance": None, "status": "no_db"}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    MIN(elevation_m), MAX(elevation_m),
                    MIN(slope_deg), MAX(slope_deg),
                    MIN(tmax_c), MAX(tmax_c),
                    MIN(tmin_c), MAX(tmin_c),
                    MIN(precip_mm), MAX(precip_mm),
                    COUNT(*)
                FROM raw.env_point_samples
                WHERE point_type = 'otu_site'
            """)).fetchone()

        if not rows or rows[10] == 0:
            return {"distance": None, "status": "no_training_data"}

        var_ranges = [
            ("elevation_m", rows[0], rows[1]),
            ("slope_deg", rows[2], rows[3]),
            ("tmax_c", rows[4], rows[5]),
            ("tmin_c", rows[6], rows[7]),
            ("precip_mm", rows[8], rows[9]),
        ]

        max_gap = 0.0
        for var, smin, smax in var_ranges:
            val = env.get(var)
            if val is None or smin is None:
                continue
            rng = max(smax - smin, 0.001)
            if val < smin:
                max_gap = max(max_gap, (smin - val) / rng)
            elif val > smax:
                max_gap = max(max_gap, (val - smax) / rng)

        if max_gap == 0:
            status = "high"
        elif max_gap < 0.5:
            status = "medium"
        else:
            status = "low"

        return {"distance": round(max_gap, 3), "status": status}
    except Exception:
        return {"distance": None, "status": "error"}


# ---------------------------------------------------------------------------
# Target parcels GeoJSON
# ---------------------------------------------------------------------------

def _resolve_target_table() -> tuple[str, list[str]]:
    if not engine:
        return "staging.va_harvested_timber_parcels", []
    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'marts' AND table_name = 'target_parcels'
            )
        """)).scalar()
    if exists:
        return "marts.target_parcels", [
            "OBJECTID", "PARCELID", "FIPS", "LOCALITY",
            "area_acres", "harvest_number", "harvest_start_date",
            "harvest_status", "harvest_acres_to_date_1",
            "mean_biomass_kg_ha", "total_biomass_kg", "biomass_pixel_count",
            "target_marked_at", "is_target",
        ]
    return "staging.va_harvested_timber_parcels", [
        "OBJECTID", "PARCELID", "FIPS", "LOCALITY",
        "Shape__Area", "harvest_number", "harvest_start_date",
        "harvest_status", "harvest_acres_to_date_1",
    ]


@app.get("/target-parcels")
def target_parcels_geojson(
    bbox: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("total_biomass_kg"),
    sort_dir: str = Query("desc", regex="^(asc|desc)$"),
):
    if not engine:
        raise HTTPException(503, "Database not configured")

    table, prop_cols = _resolve_target_table()
    allowed_sort = {
        "total_biomass_kg", "mean_biomass_kg_ha", "area_acres",
        "harvest_start_date", "OBJECTID", "LOCALITY", "Shape__Area",
    }
    if sort_by not in allowed_sort:
        sort_by = "OBJECTID"

    where_clauses = ["geom IS NOT NULL"]
    params: dict = {"lim": limit, "off": offset}

    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            minx, miny, maxx, maxy = parts
        except (ValueError, TypeError):
            raise HTTPException(422, "bbox must be minx,miny,maxx,maxy")
        where_clauses.append(
            "ST_Intersects(geom, ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326))"
        )
        params.update(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    where_sql = " AND ".join(where_clauses)
    direction = "ASC" if sort_dir == "asc" else "DESC"
    order_col = f'"{sort_by}"' if sort_by in ("OBJECTID", "LOCALITY", "Shape__Area") else sort_by
    nulls = "NULLS LAST" if direction == "DESC" else "NULLS FIRST"
    prop_select = ", ".join(f'"{c}"' for c in prop_cols) if prop_cols else "1 AS _dummy"

    count_sql = text(f"SELECT COUNT(*) FROM {table} WHERE {where_sql}")
    data_sql = text(f"""
        SELECT ST_AsGeoJSON(geom)::text AS geojson, {prop_select}
        FROM {table}
        WHERE {where_sql}
        ORDER BY {order_col} {direction} {nulls}
        LIMIT :lim OFFSET :off
    """)

    try:
        with engine.connect() as conn:
            total = conn.execute(count_sql, params).scalar() or 0
            rows = conn.execute(data_sql, params).fetchall()

        features = []
        for row in rows:
            geom_json = row[0]
            if not geom_json:
                continue
            props = {}
            for i, col in enumerate(prop_cols):
                val = row[i + 1] if (i + 1) < len(row) else None
                if hasattr(val, "isoformat"):
                    val = val.isoformat()
                props[col] = val
            features.append({
                "type": "Feature",
                "geometry": json.loads(geom_json),
                "properties": props,
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "total_count": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.exception("target-parcels query failed")
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Generic GeoJSON helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def _table_to_geojson(schema: str, table: str, geom_col: str, limit: int = 5000) -> dict:
    if not engine:
        return {"type": "FeatureCollection", "features": []}
    try:
        with engine.connect() as conn:
            cols_result = conn.execute(text("""
                SELECT array_agg(column_name ORDER BY ordinal_position)
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                AND column_name != :geom_col
            """), {"schema": schema, "table": table, "geom_col": geom_col})
            col_row = cols_result.fetchone()
        prop_cols = col_row[0] if col_row and col_row[0] else []

        geom_select = f"ST_AsGeoJSON({geom_col})::text"
        prop_select = ", ".join(f'"{c}"' for c in prop_cols) if prop_cols else "1 as _dummy"
        sql = text(f"""
            SELECT {geom_select}, {prop_select}
            FROM {schema}.{table}
            WHERE {geom_col} IS NOT NULL
            LIMIT :limit
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"limit": limit}).fetchall()
        features = []
        for row in rows:
            geom_json = row[0]
            if not geom_json:
                continue
            props = {}
            for i, col in enumerate(prop_cols):
                if i + 1 < len(row):
                    val = row[i + 1]
                    if hasattr(val, "isoformat"):
                        val = val.isoformat()
                    props[col] = val
            features.append({
                "type": "Feature",
                "geometry": json.loads(geom_json),
                "properties": props,
            })
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.exception("GeoJSON query failed for %s.%s", schema, table)
        return {"type": "FeatureCollection", "features": [], "error": str(e)}


@app.get("/parcels")
def parcels_geojson(limit: int = Query(5000, ge=1, le=50000)):
    return _table_to_geojson("raw", "va_parcels", "geom", limit)


@app.get("/harvests")
def harvests_geojson(limit: int = Query(2000, ge=1, le=10000)):
    return _table_to_geojson("raw", "va_harvests", "geom", limit)


@app.get("/harvested-parcels")
def harvested_parcels_geojson(limit: int = Query(5000, ge=1, le=50000)):
    return _table_to_geojson("staging", "va_harvested_timber_parcels", "geom", limit)
