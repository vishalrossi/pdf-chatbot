import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.models import VectorIndexType
from utils.storage import BASE_VOLUME_PATH
from dotenv import load_dotenv, dotenv_values


# -----------------------------
# Config
# -----------------------------
ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key=os.getenv('OPENAI_API_KEY')

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")
print("env is", ENV)

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

'''
vsc = VectorSearchClient()
index_name = f"pdf_chatbot_{ENV}"

vsc.create_delta_sync_index(
    endpoint_name="pdf-vector-search",
    index_name=index_name,
    source_table_name=None,
    embeddings=embeddings,
    texts=texts
)
'''

# -----------------------------
# 3️⃣ Initialize Vector Search
# -----------------------------

vsc = VectorSearchClient()
INDEX_NAME = f"pdf_chatbot_{ENV}"

try:
    # Try to get existing index
    index = vsc.get_index(INDEX_NAME)
    print(f"Connected to existing index: {INDEX_NAME}")
except Exception:
    # Create a new index if it doesn't exist
    index = vsc.create_index(
        name=INDEX_NAME,
        index_type=VectorIndexType.DENSE_VECTOR,
        dimension=1536,  # OpenAI text-embedding-ada-002 dimension
        metric="cosine"
    )
    print(f"Created new index: {INDEX_NAME}")

try:
    index.upsert(vectors=embeddings)
    print(f"Inserted {len(embeddings)} embeddings into {INDEX_NAME}")
except Exception as e:
    print(f"Error inserting embeddings: {e}")

'''
index_name = f"pdf_chatbot_{ENV}"

def index_exists(vsc, index_name):
    try:
        vsc.list_indexes(name=index_name)
        return True
    except Exception:
        return False

if not index_exists(vsc, index_name):
    vsc.create_index(
        endpoint_name="pdf-vector-search",
        index_name=index_name,
        dimension=len(embeddings[0]),
        metric_type="COSINE"
    )
'''
# -----------------------------
# 4️⃣ Prepare Metadata & IDs
# -----------------------------

ids = [f"pdf_{c.metadata.get('page')}_{i}" for i, c in enumerate(chunks)]

metadata = [
    {
        "text": c.page_content,
        "source": "pdf",
        "page": c.metadata.get("page"),
        "chunk_id": i
    }
    for i, c in enumerate(chunks)
]


# -----------------------------
# 5️⃣ Upsert to Vector Index
# -----------------------------
'''
vsc.upsert(
    index_name=index_name,
    vectors=embeddings,
    ids=ids,
    metadata=metadata
)
'''