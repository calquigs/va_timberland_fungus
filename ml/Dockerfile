FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG RASTER_URL=https://github.com/calquigs/va_timberland_fungus/releases/download/data-v1/va_rasters.tar.gz
RUN curl -fSL "$RASTER_URL" | tar xz -C /app

COPY app/ ./app/

ENV ENV_RASTER_DIR=/app/data/environmental
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
