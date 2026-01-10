"""
PDF to Databricks Vector Search pipeline

Steps:
1. Load PDF pages
2. Select relevant pages
3. Split pages into chunks
4. Generate embeddings using OpenAI (safe 2D list)
5. Save chunks + embeddings to Delta
6. Create or sync Vector Search index in Databricks

Safe for repeated runs and production deployment.
"""

import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from databricks.vector_search.client import VectorSearchClient

from utils.storage import BASE_VOLUME_PATH


# -----------------------------
# Configuration
# -----------------------------
ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")

PDF_PATH = f"{BASE_VOLUME_PATH}/pdf/About_Dogs.pdf"
CATALOG = "databricks_vishal"
SCHEMA = "default"
ENDPOINT_NAME = f"pdf_chatbot_endpoint_{ENV}"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.pdf_chatbot_{ENV}"
TABLE_NAME = f"pdf_chatbot_embeddings_{ENV}"
SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.pdf_chatbot_embeddings_{ENV}"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PAGE_START = 132
PAGE_END = 140


# -----------------------------
# Helper Functions
# -----------------------------
def load_pdf_pages(pdf_path: str) -> List:
    """
    Load pages from a PDF using PyPDFLoader.

    Args:
        pdf_path: Path to PDF.

    Returns:
        List of page documents.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")
    return docs


def select_pages(docs: List, start: int, end: int) -> List:
    """
    Select relevant pages from PDF.

    Args:
        docs: List of document pages.
        start: Start page (inclusive, 0-indexed).
        end: End page (exclusive).

    Returns:
        List of selected pages.
    """
    return docs[start:end]


def chunk_documents(docs: List, chunk_size: int, chunk_overlap: int) -> List:
    """
    Split documents into chunks.

    Args:
        docs: List of document pages.
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def generate_embeddings_safe(chunks: List, model: str, api_key: str) -> Tuple[List[List[float]], List[str]]:
    """
    Generate embeddings for chunks, ensuring a 2D list.

    Args:
        chunks: List of chunk documents.
        model: OpenAI embedding model name.
        api_key: OpenAI API key.

    Returns:
        Tuple of embeddings (2D list) and chunk texts.
    """
    texts = [c.page_content for c in chunks]
    embeddings_raw = OpenAIEmbeddings(model=model, api_key=api_key).embed_documents(texts)

    # Wrap single embedding into a list if needed
    if len(texts) == 1 and isinstance(embeddings_raw[0], float):
        embeddings = [embeddings_raw]
    else:
        embeddings = embeddings_raw

    print(f"Generated {len(embeddings)} embeddings for {len(texts)} chunks")
    return embeddings, texts


def initialize_vector_search(endpoint_name: str) -> VectorSearchClient:
    """
    Initialize Databricks Vector Search client and ensure endpoint exists.

    Args:
        endpoint_name: Name of the vector search endpoint.

    Returns:
        VectorSearchClient instance.
    """
    client = VectorSearchClient()
    endpoints = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint_name not in endpoints:
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"Created Vector Search endpoint '{endpoint_name}'")
    else:
        print(f"Vector Search endpoint '{endpoint_name}' already exists")
    return client


def save_to_delta(texts: List[str], embeddings: List[List[float]], spark: SparkSession, table_name: str) -> None:
    """
    Save chunks and embeddings to a Delta table.

    Args:
        texts: List of chunk texts.
        embeddings: Corresponding embeddings.
        spark: SparkSession instance.
        table_name: Name of Delta table.
    """
    df = pd.DataFrame({
        "id": range(len(texts)),
        "text": texts,
        "embedding": embeddings
    })
    spark_df = spark.createDataFrame(df)
    spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"Saved embeddings to Delta table '{table_name}'")


def create_or_sync_index_safe(client: VectorSearchClient, endpoint_name: str, index_name: str,
                              source_table_name: str, embeddings: List[List[float]]) -> None:
    """
    Create or sync a Vector Search index safely.

    Args:
        client: VectorSearchClient instance.
        endpoint_name: Endpoint name.
        index_name: Fully qualified index name.
        source_table_name: Delta table source.
        embeddings: List of embeddings (2D list).
    """
    existing_indexes = [i["name"] for i in client.list_indexes().get("indexes", [])]
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Skipping creation.")
        return

    embedding_dim = len(embeddings[0])
    client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table_name,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_dimension=embedding_dim,
        embedding_vector_column="embedding"
    )
    print(f"Vector Search index '{index_name}' created with dimension {embedding_dim}")


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # 1️⃣ Load PDF pages
    docs = load_pdf_pages(PDF_PATH)
    selected_pages = select_pages(docs, PAGE_START, PAGE_END)

    # 2️⃣ Split pages into chunks
    chunks = chunk_documents(selected_pages, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3️⃣ Generate embeddings safely
    embeddings, texts = generate_embeddings_safe(chunks, EMBEDDING_MODEL, API_KEY)

    # 4️⃣ Initialize Vector Search client
    client = initialize_vector_search(ENDPOINT_NAME)

    # 5️⃣ Save to Delta
    spark = SparkSession.builder.getOrCreate()
    save_to_delta(texts, embeddings, spark, TABLE_NAME)

    # 6️⃣ Create or sync index
    create_or_sync_index_safe(client, ENDPOINT_NAME, INDEX_NAME, SOURCE_TABLE_NAME, embeddings)


if __name__ == "__main__":
    main()


'''
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