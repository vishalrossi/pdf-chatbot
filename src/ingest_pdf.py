import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from databricks.vector_search.client import VectorSearchClient
from utils.storage import BASE_VOLUME_PATH

#ENV = os.environ["ENV"]
ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")
print("env is", ENV)

PDF_PATH = f"{BASE_VOLUME_PATH}/pdfs/About_Dogs.pdf"
VECTOR_PATH = f"{BASE_VOLUME_PATH}/vector_search/{ENV}"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                          chunk_overlap=50, 
                                          separators=["\n\n", "\n", ".", " "])
chunks = splitter.split_documents(docs)

texts = [c.page_content for c in chunks]
embeddings = OpenAIEmbeddings().embed_documents(texts)

vsc = VectorSearchClient()
index_name = f"pdf_chatbot_{ENV}"

vsc.create_delta_sync_index(
    endpoint_name="pdf-vector-search",
    index_name=index_name,
    source_table_name=None,
    embeddings=embeddings,
    texts=texts
)