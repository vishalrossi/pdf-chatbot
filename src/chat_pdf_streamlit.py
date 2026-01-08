import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
#from model import PDFRAGModel
from dotenv import load_dotenv

ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

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

pdf_model = PDFRAGModel(
    index_name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    model_name=RAG_MODEL_NAME,
)

st.title("PDF Chatbot (RAG)")

# User input
question = st.text_input("Ask a question about your PDF:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving answer..."):
        # Compute query embedding
        query_embedding = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=API_KEY
        ).embed_query(question)

        # Ask PDF via Vector Search + LLM
        result = pdf_model.ask_pdf(query_embedding, question)

        # Display
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Citations")
        st.json(result["citations"])