"""
Utility helpers for preparing and managing Unity Catalog storage locations
used by the RAG chatbot application.

This module:
- Selects the target Unity Catalog and schema
- Ensures required directories exist inside a Unity Catalog Volume
- Is safe to run in Databricks Serverless and non-serverless environments

Intended usage:
- Run once during environment setup
- Import and reuse `ensure_storage()` in notebooks, jobs, or pipelines

Assumptions:
- The Unity Catalog volume already exists
- The executing principal has CREATE and READ/WRITE permissions
- Databricks runtime provides SparkSession and DBUtils

Directory structure created under the volume:
- pdf/                 → raw PDF files
- vector_search/dev/   → vector index artifacts
- logs/                → application logs
- temp/                → temporary or intermediate files
"""

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from typing import List

# ---------------------------------------------------------------------
# Spark & DBUtils initialization (serverless-compatible)
# ---------------------------------------------------------------------

spark: SparkSession = SparkSession.builder.getOrCreate()
dbutils: DBUtils = DBUtils(spark)

# ---------------------------------------------------------------------
# Unity Catalog configuration
# ---------------------------------------------------------------------

CATALOG: str = "databricks_vishal"  # Must already exist
SCHEMA: str = "chatbot"
VOLUME: str = "rag_data"

BASE_VOLUME_PATH: str = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

DIRECTORIES: List[str] = [
    "pdf",
    "vector_search/dev",
    "logs",
    "temp",
]

# ---------------------------------------------------------------------
# Catalog & schema preparation
# ---------------------------------------------------------------------

def _prepare_catalog() -> None:
    """
    Ensure the correct Unity Catalog and schema are selected and available.

    This function:
    - Switches to the configured catalog
    - Creates the schema if it does not exist
    """
    spark.sql(f"USE CATALOG `{CATALOG}`")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{SCHEMA}`")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ensure_storage() -> None:
    """
    Ensure required directories exist inside the Unity Catalog volume.

    This function is idempotent and safe to run multiple times.
    It creates all configured directories if they do not already exist.

    Raises:
        RuntimeError: If directory creation fails due to permissions or
                      missing volume.
    """
    _prepare_catalog()

    for directory in DIRECTORIES:
        path = f"{BASE_VOLUME_PATH}/{directory}"
        dbutils.fs.mkdirs(path)

    print("Unity Catalog storage directories are ready.")


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ensure_storage()