FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG RASTER_URL=https://github.com/calquigs/va_timberland_fungus/releases/download/data-v1/va_rasters.tar.gz
RUN curl -fSL "$RASTER_URL" | tar xz -C /app

COPY . .

ENV ENV_RASTER_DIR=/app/data/environmental
ENV LOBLOLLY_DATA_DIR=/app/data/fia_bigmap
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
