# src/utils/storage.py
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

CATALOG = "databricks_vishal"   # MUST exist
SCHEMA = "chatbot"
VOLUME = "rag_data"


BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

DIRS = [
    "pdfs",
    "vector_search/dev",
    "vector_search/prod",
    "logs",
    "temp"
]

for d in DIRS:
    spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils.fs.mkdirs(
        f"{BASE_VOLUME_PATH}/{d}"
    )

print("Unity Catalog storage initialized")


if __name__ == "__main__":
    ensure_storage()