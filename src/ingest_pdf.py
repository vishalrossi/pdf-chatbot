import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from databricks.vector_search.client import VectorSearchClient
from utils.storage import BASE_VOLUME_PATH
from dotenv import load_dotenv
import pandas as pd
from pyspark.sql import SparkSession


# -----------------------------
# Config
# -----------------------------
ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key=os.getenv('OPENAI_API_KEY')

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")

PDF_PATH = f"{BASE_VOLUME_PATH}/pdf/About_Dogs.pdf"
VECTOR_PATH = f"{BASE_VOLUME_PATH}/vector_search/{ENV}"

# -----------------------------
# 1️⃣ Load PDF & Chunk
# -----------------------------

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

print(f"Loaded {len(docs)} page(s) from the PDF.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                          chunk_overlap=50, 
                                          separators=["\n\n", "\n", ".", " "])

relevant_pages = docs[132:140]
chunks = splitter.split_documents(relevant_pages)

print(f"Total Splits after chunking: {len(chunks)}")

texts = [c.page_content for c in chunks]

# -----------------------------
# 2️⃣ Create Embeddings
# -----------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key).embed_documents(texts)


# -----------------------------
# 3️⃣ Initialize Vector Search
# -----------------------------

client = VectorSearchClient()
ENDPOINT_NAME=f"pdf_chatbot_endpoint_{ENV}"
CATALOG = "databricks_vishal"          # or your UC catalog
SCHEMA = "default" # or any schema you use

INDEX_NAME = f"{CATALOG}.{SCHEMA}.pdf_chatbot_{ENV}"
#INDEX_NAME = f"pdf_chatbot_{ENV}"


# Check if endpoint exists
endpoints = [e["name"] for e in client.list_endpoints().get("endpoints", [])]

if ENDPOINT_NAME in endpoints:
    print(f"Vector Search endpoint '{ENDPOINT_NAME}' already exists. Skipping creation.")
else:
    client.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print(f"Vector Search endpoint '{ENDPOINT_NAME}' created.")


spark = SparkSession.builder.getOrCreate()

df = pd.DataFrame({
    "id": range(len(texts)),
    "text": texts,
    "embedding": embeddings
})

print(df.head)

spark_df = spark.createDataFrame(df)

TABLE_NAME = f"pdf_chatbot_embeddings_{ENV}"
SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.pdf_chatbot_embeddings_{ENV}"


spark_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_NAME)

client.create_delta_sync_index(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME,
    source_table_name=SOURCE_TABLE_NAME,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=len(embeddings[0]),  # 1536 for text-embedding-3-small
    embedding_vector_column="embedding"
)

'''
index = client.create_delta_sync_index(
  endpoint_name="ENDPOINT_NAME",
  source_table_name="vector_search_demo.vector_search.en_wiki",
  index_name="vector_search_demo.vector_search.en_wiki_index",
  pipeline_type="TRIGGERED",
  primary_key="id",
  embedding_dimension=1024,
  embedding_vector_column="text_vector"
)
'''