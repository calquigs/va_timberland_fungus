#!/usr/bin/env bash
# Full setup: start services, ingest data, run dbt, and launch the app.
#
# Prerequisites: Docker, Docker Compose, Python 3.10+, pip
#
# Usage:
#   ./scripts/setup.sh           # full pipeline then start app
#   ./scripts/setup.sh --no-app  # full pipeline only (don't start api+viz)

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

NO_APP=false
for arg in "$@"; do
  case "$arg" in
    --no-app) NO_APP=true ;;
  esac
done

echo "╔══════════════════════════════════════════╗"
echo "║         VA Woods – Full Setup            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# 1. Start Postgres + MinIO
# ------------------------------------------------------------------
echo ">>> Starting Postgres and MinIO..."
docker compose up -d postgres minio

echo ">>> Waiting for Postgres to become healthy..."
until docker compose exec postgres pg_isready -U va_woods -d va_woods -q 2>/dev/null; do
  sleep 2
done
echo "    Postgres is ready."
echo ""

# ------------------------------------------------------------------
# 2. Set up Python venv with ingestion + dbt deps
# ------------------------------------------------------------------
VENV_DIR="$ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo ">>> Creating Python virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">>> Installing Python dependencies (ingestion + dbt)..."
pip install -q -r ingestion/requirements.txt dbt-postgres 2>&1 | tail -1
echo ""

# ------------------------------------------------------------------
# 3. Run ingestion pipeline
# ------------------------------------------------------------------
export DATABASE_URL="${DATABASE_URL:-postgresql://va_woods:va_woods_dev@localhost:5432/va_woods}"

echo ">>> Running ingestion pipeline (this downloads data and may take 15–30 min)..."
echo ""
bash scripts/ingest_va_landuse.sh
echo ""

# ------------------------------------------------------------------
# 4. Run dbt
# ------------------------------------------------------------------
echo ">>> Running dbt models..."
export DBT_HOST=localhost DBT_USER=va_woods DBT_PASSWORD=va_woods_dev DBT_DBNAME=va_woods DBT_PORT=5432
cd "$ROOT/transform"
dbt run
echo ""
echo ">>> Running dbt tests..."
dbt test
cd "$ROOT"
echo ""

# ------------------------------------------------------------------
# 5. Launch app
# ------------------------------------------------------------------
if [ "$NO_APP" = true ]; then
  echo "══════════════════════════════════════════"
  echo "Setup complete. Start the app with:"
  echo "  docker compose up api viz"
  echo "  Map: http://localhost:8502  API: http://localhost:8000"
  echo "══════════════════════════════════════════"
else
  echo ">>> Starting ML API and Streamlit app..."
  echo "    Map: http://localhost:8502"
  echo "    API: http://localhost:8000"
  echo "    Press Ctrl+C to stop."
  echo ""
  docker compose up api viz
fi
