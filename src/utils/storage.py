# src/utils/storage.py
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# Initialize Spark and dbutils (serverless-compatible)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# ----------------- Configuration -----------------
CATALOG = "databricks_vishal"   # Must exist
SCHEMA = "chatbot"
VOLUME = "rag_data"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

DIRS = [
    "pdfs",
    "vector_search/dev",
    "logs",
    "temp"
]

# ----------------- Prepare Unity Catalog -----------------
# Ensure catalog context is selected first
spark.sql(f"USE CATALOG `{CATALOG}`")

# Create schema if it does not exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{SCHEMA}`")

# ----------------- Function to create storage folders -----------------
def ensure_storage() -> None:
    for d in DIRS:
        path = f"{BASE_VOLUME_PATH}/{d}"
        dbutils.fs.mkdirs(path)  # serverless-safe
    print(" Unity Catalog folders created successfully")

# ----------------- Run if executed directly -----------------
if __name__ == "__main__":
    ensure_storage()