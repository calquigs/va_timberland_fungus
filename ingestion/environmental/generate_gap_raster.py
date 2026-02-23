"""
CLI wrapper for generating the sampling-gap raster.

The core logic lives in ml/app/gap_raster.py; this script provides a
standalone entry point for running outside Docker.

Usage:
  python generate_gap_raster.py                        # defaults
  python generate_gap_raster.py --env-dir /path        # custom raster dir
  python generate_gap_raster.py --database-url ...     # custom DB
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_env_dir() -> Path:
    env_dir = os.environ.get("ENV_RASTER_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return _PROJECT_ROOT / "data" / "environmental"


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate sampling-gap raster")
    parser.add_argument("--env-dir", type=str, default=None, help="Directory with VA env rasters")
    parser.add_argument("--database-url", type=str, default=None, help="PostgreSQL URL")
    args = parser.parse_args()

    env_dir = Path(args.env_dir) if args.env_dir else _default_env_dir()

    sys.path.insert(0, str(_PROJECT_ROOT / "ml"))
    from app.gap_raster import generate_gap_raster

    result = generate_gap_raster(env_dir=env_dir, database_url=args.database_url)
    if result:
        print(f"Gap raster generated: {result}")
    else:
        print("Failed to generate gap raster.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
