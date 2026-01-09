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
# Initialize embeddings & RAG model
# -----------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from model import PDFRAGModel

# Initialize RAG model (once)
pdf_model = PDFRAGModel(
    index_name=INDEX_NAME,
    endpoint_name=ENDPOINT_NAME,
    model_name=RAG_MODEL_NAME,
)

# ---------
# Streamlit
# ---------

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chat Assistant")

# ------------------------------
# SESSION STATE FOR CHAT HISTORY
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------
for entry in st.session_state.chat_history:
    # User message
    st.markdown(f"<div style='text-align:right; background-color:#a8d5ba; color:black; padding:12px; border-radius:12px; margin-bottom:5px;'>{entry['question']}</div>",
                unsafe_allow_html=True)
    # Bot response
    st.markdown(f"<div style='text-align:left; background-color:#d9d9d9; color:black; padding:12px; border-radius:12px; margin-bottom:5px;'>{entry['answer']}</div>",
        unsafe_allow_html=True)
    
    # Sources / citations
    if entry.get("citations"):
        with st.expander("📎 Sources"):
            for c in entry["citations"]:
                chunk_id = c["id"]
                chunk_text = c["score"]
                # Highlight country if present
                country_lines = [
                    f"➡ {line}" if any(country.lower() in line.lower() for country in pdf_model.COUNTRY_LIST)
                    else line
                    for line in chunk_text.split("\n")
                ]
                st.text("\n".join(country_lines))
    st.markdown("---")

# ------------------------------
# INPUT FORM
# ------------------------------
with st.form(key="question_form"):
    user_question = st.text_input(
        "Ask a question about the PDF",
        placeholder="What does this document say about X?"
    )
    submit_button = st.form_submit_button("Send")

# ------------------------------
# HANDLE SUBMISSION
# ------------------------------
if submit_button and user_question:
    with st.spinner("Searching PDF and generating answer..."):
        # Generate embedding for user question
        query_embedding = embeddings.embed_query(user_question)
        # Get answer + sources
        result = pdf_model.ask_pdf(query_embedding, user_question, k=2)
        # Append to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": result["answer"],
            "citations": result["citations"]
        })
    st.rerun()  # Rerun to show updated chat
