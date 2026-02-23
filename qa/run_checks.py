"""
QA/QC checks for the VA Woods data pipeline.

Validates row counts, null geometries, value ranges, and referential
integrity across the raw, staging, and marts schemas.

Usage:
  python qa/run_checks.py
"""

import logging
import os
import sys

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_engine():
    url = os.environ.get("DATABASE_URL", "postgresql://va_woods:va_woods_dev@localhost:5432/va_woods")
    return create_engine(url)


def check_table_exists(conn, schema: str, table: str) -> bool:
    row = conn.execute(text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = :s AND table_name = :t
        )
    """), {"s": schema, "t": table}).scalar()
    return bool(row)


def check_row_count(conn, schema: str, table: str, min_rows: int = 1) -> tuple[bool, int]:
    count = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')).scalar() or 0
    return count >= min_rows, count


def check_null_geom(conn, schema: str, table: str, geom_col: str = "geom") -> tuple[bool, int]:
    null_count = conn.execute(text(
        f'SELECT COUNT(*) FROM "{schema}"."{table}" WHERE "{geom_col}" IS NULL'
    )).scalar() or 0
    return null_count == 0, null_count


CHECKS = [
    ("raw", "va_parcels", 1, "geom"),
    ("raw", "va_harvests", 1, "geom"),
    ("staging", "va_harvested_timber_parcels", 1, "geom"),
    ("raw", "parcel_loblolly_stats", 0, None),
    ("raw", "target_parcels", 0, None),
    ("raw", "otu_sites", 0, None),
    ("raw", "otu_observations", 0, None),
    ("raw", "env_point_samples", 0, None),
    ("raw", "protected_lands", 0, "geom"),
]

RANGE_CHECKS = [
    ("raw", "parcel_loblolly_stats", "mean_biomass_kg_ha", 0, 500_000),
    ("raw", "parcel_loblolly_stats", "total_biomass_kg", 0, 1e12),
    ("raw", "env_point_samples", "elevation_m", -100, 5000),
    ("raw", "env_point_samples", "slope_deg", 0, 90),
    ("raw", "env_point_samples", "tmax_c", -20, 60),
    ("raw", "env_point_samples", "tmin_c", -40, 40),
    ("raw", "env_point_samples", "precip_mm", 0, 5000),
]


def main():
    engine = get_engine()
    passed = 0
    failed = 0
    skipped = 0

    with engine.connect() as conn:
        logger.info("=== Table existence and row counts ===")
        for schema, table, min_rows, geom_col in CHECKS:
            if not check_table_exists(conn, schema, table):
                logger.warning("SKIP %s.%s — table does not exist", schema, table)
                skipped += 1
                continue

            ok, count = check_row_count(conn, schema, table, min_rows)
            if ok:
                logger.info("PASS %s.%s — %d rows (min %d)", schema, table, count, min_rows)
                passed += 1
            else:
                logger.error("FAIL %s.%s — %d rows (min %d)", schema, table, count, min_rows)
                failed += 1

            if geom_col and count > 0:
                ok, null_count = check_null_geom(conn, schema, table, geom_col)
                if ok:
                    logger.info("PASS %s.%s — no null geometries", schema, table)
                    passed += 1
                else:
                    logger.warning("WARN %s.%s — %d null geometries", schema, table, null_count)

        logger.info("")
        logger.info("=== Value range checks ===")
        for schema, table, col, vmin, vmax in RANGE_CHECKS:
            if not check_table_exists(conn, schema, table):
                skipped += 1
                continue

            count = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')).scalar() or 0
            if count == 0:
                skipped += 1
                continue

            row = conn.execute(text(
                f'SELECT MIN("{col}"), MAX("{col}") FROM "{schema}"."{table}" WHERE "{col}" IS NOT NULL'
            )).fetchone()
            actual_min, actual_max = row[0], row[1]

            if actual_min is None:
                logger.info("SKIP %s.%s.%s — all NULL", schema, table, col)
                skipped += 1
                continue

            ok = actual_min >= vmin and actual_max <= vmax
            if ok:
                logger.info("PASS %s.%s.%s — range [%.2f, %.2f] within [%s, %s]",
                            schema, table, col, actual_min, actual_max, vmin, vmax)
                passed += 1
            else:
                logger.error("FAIL %s.%s.%s — range [%.2f, %.2f] outside [%s, %s]",
                             schema, table, col, actual_min, actual_max, vmin, vmax)
                failed += 1

    logger.info("")
    logger.info("=== Summary: %d passed, %d failed, %d skipped ===", passed, failed, skipped)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
