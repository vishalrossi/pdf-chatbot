# src/utils/storage.py
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
def get_dbutils(spark):
        try:
            from pyspark.dbutils import DBUtils
            dbutils = DBUtils(spark)
        except ImportError:
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
        return dbutils

dbutils = get_dbutils(spark)

#BASE_VOLUME_PATH = "/Volumes/vishal/chatbot/rag_data"
BASE_VOLUME_PATH = "/Volumes/databricks_vishal"

REQUIRED_DIRS = [
    "pdfs",
    "vector_search/dev",
    "vector_search/prod",
    "logs",
    "temp",
]

def ensure_storage():
    for d in REQUIRED_DIRS:
        path = f"{BASE_VOLUME_PATH}/{d}"
        dbutils.fs.mkdirs(path)
    print("✅ Unity Catalog folders created successfully")

if __name__ == "__main__":
    ensure_storage()