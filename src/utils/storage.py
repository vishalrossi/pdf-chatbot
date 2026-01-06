# src/utils/storage.py
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

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

# Function to create storage folders
def ensure_storage():
    for d in DIRS:
        path = f"{BASE_VOLUME_PATH}/{d}"
        dbutils.fs.mkdirs(path)
    print(" Unity Catalog folders created successfully")


if __name__ == "__main__":
    ensure_storage()