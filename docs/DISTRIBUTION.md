# Distributing & Showcasing VA Woods

This guide walks through options for sharing the project—from a GitHub repo with clear instructions to fully hosted demos so that evaluators or collaborators can access it with minimal friction.

---

## Option 1: GitHub repo with install instructions (minimum viable)

**Best for:** Code review, technical evaluators who are comfortable with Docker.

You already have most of this. The goal is to make the “evaluator path” obvious and one-command where possible.

### What you have

- **README.md** with Quick start, architecture, and data sources.
- **docker-compose.yml** that brings up Postgres, ML API, and Streamlit viz (and optionally MinIO, Martin).
- **Dockerfiles** in `ml/` and `viz/` for consistent runs.

### Recommended additions

1. **“For evaluators” section in README**  
   - Single place that says: “To run the full app: `docker compose up -d postgres && docker compose up ml viz` then open Map: http://localhost:8502, API: http://localhost:8000.”
   - Note that ingestion + dbt are optional for a “view the app” demo if you have pre-loaded data or can document “with empty DB you’ll see the UI but no parcel data until ingestion is run.”

2. **Prerequisites**  
   - Docker & Docker Compose; Git. Optional: Python 3.10+ if they want to run ingestion/dbt/viz locally without Docker.

3. **Optional: `.env.example`**  
   - If you ever add env vars (e.g. `DATABASE_URL` overrides), list them in `.env.example` so people know what can be configured.

**Pros:** No hosting cost; full control; shows you can document and containerize.  
**Cons:** Evaluators must clone and run Docker; no “click one link” demo.

---

## Option 2: Free / low-friction cloud hosting (live link for recruiters)

**Best for:** “Click this link to try the app” with minimal setup for the company.

Your app has **three parts**: Postgres (data), ML API (predictions), Streamlit (UI). To get a single URL that “just works,” you need to host all three (or a simplified variant).

### 2a. Full stack on a single platform (easiest to reason about)

Host Postgres + ML API + Streamlit on one provider so networking is simple.

| Platform | Notes | Free tier |
|----------|--------|-----------|
| **Railway** | Run Postgres + multiple services from one project; supports Dockerfile and `docker-compose`-style services. Good for “one link to app.” | Yes (usage limits). |
| **Render** | Postgres + Web Services (build from Dockerfile or repo). Separate services for API and Streamlit; all get URLs. | Yes (Postgres and services spin down on free). |
| **Fly.io** | Run Postgres + `ml` + `viz` as separate apps; one org, multiple URLs. More config than Railway but very flexible. | Yes (enough for a demo). |

**Step-by-step for Railway:** See **[docs/RAILWAY.md](RAILWAY.md)** for the full walkthrough. Summary:

1. Create a project; add Postgres.
2. Add ML service from repo (root directory `ml`), set `DATABASE_URL` from Postgres, generate domain.
3. Add Viz service (root directory `viz`), set `DATABASE_URL` and `ML_API_URL` to the ML service’s public URL, generate domain.
4. Run ingestion + dbt once against the hosted DB (from your laptop or a one-off job).
5. Share the Viz service URL (e.g. `https://viz-production-xxxx.up.railway.app`).

**Pros:** One shareable link; no local install.  
**Cons:** You must run ingestion/dbt at least once; free tiers may sleep or have limits.

### 2b. Streamlit Community Cloud (Streamlit-only demo)

**Streamlit Community Cloud** can host only the **viz** app. It does **not** run your Postgres or ML API by default.

- **Option A – Viz only, no backend:**  
  - Strip the app to a “showcase” mode: static maps, sample screenshots, or pre-generated outputs. No DB, no ML API. Easiest, but not the real app.

- **Option B – Viz + external backend:**  
  - Host Postgres elsewhere (e.g. **Neon**, **Supabase**, **Railway**) and ML API elsewhere (e.g. **Railway**, **Render**).  
  - Deploy the Streamlit app to Community Cloud and set secrets: `DATABASE_URL`, `ML_API_URL`.  
  - Streamlit’s free tier can talk to external URLs; you’ll need to run ingestion/dbt against the hosted DB and have the ML API publicly reachable (HTTPS).

**Pros:** Free, official Streamlit hosting; easy “Try the app” link.  
**Cons:** Either a reduced demo (no real backend) or you must host and wire DB + API yourself.

### 2c. Hugging Face Spaces

Similar idea to Streamlit Community Cloud: you can run a Streamlit (or Gradio) app. Same tradeoff: either a simplified “demo” app or you connect to externally hosted Postgres + ML API and document that in the Space description.

---

## Option 3: Hybrid (recommended for “as easy as possible for the company”)

Combine **Option 1** and **Option 2** so you have both a link and the repo.

1. **README (Option 1)**  
   - “For evaluators” at the top:  
     - **Live demo:** [Link to your hosted app]  
     - **Run locally:** `git clone … && docker compose up -d postgres && docker compose up ml viz` → Map: http://localhost:8502, API: http://localhost:8000.

2. **Hosted app (Option 2a or 2b)**  
   - Deploy the full stack (e.g. Railway) **or** a Streamlit-only demo with external DB + API.  
   - Put the live URL in the README and in your resume/cover letter.

3. **Docs**  
   - Keep `docs/ARCHITECTURE.md`, `docs/DATA_SOURCES.md`, and this `docs/DISTRIBUTION.md` so interested evaluators can see how you think about deployment and distribution.

Result: **One-click** for non-technical reviewers (live link) and **one-command** for technical reviewers (Docker), with the repo as the single source of truth.

---

## Checklist before sharing

- [ ] README has a clear “For evaluators” or “Quick demo” section with live link (if any) and local Docker command.
- [ ] Correct port in README: Streamlit app at **8502** when using repo’s `docker-compose.yml` (viz is mapped 8502→8501).
- [ ] If hosting: `DATABASE_URL` and `ML_API_URL` point at the hosted Postgres and ML API (not localhost).
- [ ] Ingestion + dbt have been run at least once against the DB that the hosted app uses (so maps/data aren’t empty).
- [ ] Repo is public (or shared with the company); no secrets in the repo, use platform secrets for `DATABASE_URL` etc.

---

## Summary

| Goal | Approach |
|------|----------|
| Easiest for company to *see* the app | Host full stack (e.g. Railway/Render) or Streamlit + external DB/API; put link in README. |
| Easiest for company to *run* the app | Polish README with one-command Docker; optional `.env.example`. |
| Best of both | Live link + “Run locally” in README; keep `docs/DISTRIBUTION.md` for your own reference and for evaluators who want deployment details. |

If you tell me which option you want to implement first (e.g. “README only” vs “Railway full stack”), I can outline concrete steps or edit the README and compose for you.
