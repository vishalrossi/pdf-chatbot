import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
#from model import PDFRAGModel
from dotenv import load_dotenv

from pathlib import Path

#PROJECT_ROOT = Path(__file__).resolve().parents[1]

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve()
ENV_PATH = PROJECT_ROOT / ".env"

print("DEBUG: Loading env from:", ENV_PATH)
print("DEBUG: .env exists:", ENV_PATH.exists())

load_dotenv(dotenv_path=ENV_PATH)

print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))


#ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

#load_dotenv(dotenv_path=ENV_PATH, override=True)

#api_key=os.getenv('OPENAI_API_KEY')

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")
# -----------------------------
# Config
# -----------------------------
API_KEY = os.environ.get("OPENAI_API_KEY")  # store in Databricks secrets
INDEX_NAME = "databricks_vishal.default.pdf_chatbot_dev"
ENDPOINT_NAME = "pdf_chatbot_endpoint_dev"
EMBEDDING_MODEL = "text-embedding-3-small"
RAG_MODEL_NAME = "gpt-4o-mini"  # can adjust

# -----------------------------
# Initialize RAG model
# -----------------------------
#pdf_model = PDFRAGModel(INDEX_NAME, ENDPOINT_NAME, API_KEY, model_name=RAG_MODEL_NAME)

from model import PDFRAGModel

# Initialize RAG model (once)
pdf_model = PDFRAGModel(
    index_name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    model_name=RAG_MODEL_NAME,
)

# Initialize embeddings (once)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot")

user_question = st.text_input(
    "Ask a question about the PDF",
    placeholder="What does this document say about X?"
)

if user_question:
    with st.spinner("Searching PDF and generating answer..."):
        query_embedding = embeddings.embed_query(user_question)
        result = pdf_model.ask_pdf(query_embedding, user_question)

    st.markdown("### 🤖 Answer")
    st.write(result["answer"])

    with st.expander("📎 Sources"):
        for c in result["citations"]:
            st.write(f"**Chunk {c['id']}**")
            st.text(c["score"])
