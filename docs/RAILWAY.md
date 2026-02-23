# Deploying VA Woods on Railway

This guide gets you to **one shareable link** for the Streamlit app (and optional API link) using [Railway](https://railway.app). You’ll have: Postgres, ML API, and Viz all in one project.

## Prerequisites

- A [Railway](https://railway.app) account (GitHub login is enough).
- Your repo pushed to **GitHub** (public or connected to Railway).
- (Optional) Railway CLI if you want to run ingestion from the terminal; otherwise you can run ingestion from your laptop using the hosted Postgres URL.

## 1. Create the project and add Postgres

1. Go to [railway.app/dashboard](https://railway.app/dashboard) → **New Project**.
2. Choose **Empty project** (we’ll add services manually).
3. In the project, click **+ New** → **Database** → **PostgreSQL**.
4. Wait for Postgres to deploy. Open the Postgres service → **Variables** and confirm you see `DATABASE_URL` (and optionally `PGHOST`, `PGPORT`, etc.). You’ll use `DATABASE_URL` for the app services.
5. (Optional) Rename the Postgres service to something like `postgres` so it’s easy to reference.

## 2. Add the ML service

1. In the same project, click **+ New** → **GitHub Repo** (or **Empty Service** if you prefer to connect the repo later).
2. If you chose GitHub Repo, select your `va_woods` repo. If you chose Empty Service, open the service → **Settings** → **Source** and connect the repo.
3. Open the new service → **Settings**:
   - **Root Directory:** set to `ml` (so Railway builds from the `ml/` folder and uses `ml/Dockerfile`).
   - **Watch Paths:** set to `ml/**` so only changes under `ml/` trigger redeploys.
4. **Variables:**
   - Add a variable and use **Reference** (or “Add variable from another service”):
     - Name: `DATABASE_URL`
     - Value: reference your Postgres service’s `DATABASE_URL` (e.g. `${{postgres.DATABASE_URL}}` — the exact name depends on how you named the Postgres service).
   - Railway sets `PORT`; the ML Dockerfile is set up to use it (default 8000).
5. **Networking:** open **Settings** → **Networking** → **Generate Domain** so the ML API gets a public URL (e.g. `https://ml-production-xxxx.up.railway.app`). Copy this URL; the Viz service will need it as `ML_API_URL`.
6. Deploy (or trigger a deploy). Fix any build errors (e.g. missing system deps in `ml/Dockerfile`).

## 3. Add the Viz (Streamlit) service

1. In the same project, click **+ New** → **GitHub Repo** again (or **Empty Service** and connect the same repo).
2. Open the new service → **Settings**:
   - **Root Directory:** set to `viz`.
   - **Watch Paths:** set to `viz/**`.
3. **Variables:**
   - `DATABASE_URL`: reference the same Postgres `DATABASE_URL` as in step 2 (e.g. `${{postgres.DATABASE_URL}}`).
   - `ML_API_URL`: set to the **ML service’s public URL** from step 2 (e.g. `https://ml-production-xxxx.up.railway.app`). No trailing slash.
4. **Networking:** **Generate Domain** for this service. This is the **link you share** (e.g. `https://viz-production-xxxx.up.railway.app`).
5. Deploy. The Streamlit app should be available at the generated domain.

## 4. Load data (so the app isn’t empty)

The hosted Postgres starts empty. To see parcels and use the app as intended, run ingestion and dbt **once** against the hosted database.

1. **Get the public Postgres URL**  
   Postgres service → **Variables** (or **Connect**). Use the URL that’s reachable from your machine (often labeled for “Public network” or “External”). It should look like:
   `postgresql://postgres:...@...railway.app:5432/railway`

2. **From your laptop** (with the repo and Python/venv set up):
   ```bash
   export DATABASE_URL="postgresql://..."   # paste the Railway Postgres URL

   # Ingestion (run from repo root or ingestion dir)
   cd ingestion && pip install -r requirements.txt
   python va_parcels/ingest_va_parcels.py
   python fia_bigmap/ingest_fia_loblolly.py
   python environmental/ingest_env_rasters.py

   # Transform
   cd ../transform && pip install dbt-postgres
   export DBT_HOST=... DBT_USER=... DBT_PASSWORD=... DBT_DBNAME=...  # from same URL
   dbt run && dbt test
   ```
   Use the host/user/password/dbname from `DATABASE_URL` for the dbt profile.

3. After this, refresh the Streamlit app; maps and data should appear.

If you don’t run ingestion/dbt, the app will still run but parcel layers and some features may be empty or show errors until the DB has the expected tables.

## 5. Share the link

- **App (share this):**  
  Viz service → **Settings** → **Networking** → your generated domain, e.g. `https://viz-production-xxxx.up.railway.app`
- **API (optional):**  
  ML service domain, e.g. `https://ml-production-xxxx.up.railway.app`  
  You can add both to your README “For evaluators” section.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Viz shows “connection refused” or ML errors | Ensure `ML_API_URL` on the Viz service is exactly the ML service’s public URL (https, no trailing slash). |
| Empty maps / no parcels | Run ingestion and dbt against the Railway `DATABASE_URL` (see step 4). |
| Build fails (e.g. GDAL) | The `ml/Dockerfile` installs `libgdal-dev`; if something else is missing, add it in the Dockerfile and redeploy. |
| App sleeps / cold start | On the free tier, services may sleep; first load can be slow. Upgrade or use a cron ping if you need always-on. |

## Cost

Railway’s free tier usually includes a small monthly credit. Postgres + two web services can stay within free limits for a demo; monitor usage in the dashboard. For production you’d tune resources and consider paid plans.

---

Once this is done, put your **Viz URL** in the main README under “For evaluators” so anyone can open the app with one click.
