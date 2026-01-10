"""
Run the web interface using straemlit.

Ask questions with the cahtbot, receive answers and look at the sources
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

print("DEBUG: __file__ =", globals().get("__file__"))
print("DEBUG: os.getcwd() =", os.getcwd())

st.write("DEBUG (Streamlit): __file__ =", globals().get("__file__"))
st.write("DEBUG (Streamlit): os.getcwd() =", os.getcwd())

#from model import PDFRAGModel


# ------------------------------------------------------------------
# Environment & config
# ------------------------------------------------------------------

'''
def load_environment(env_path: str) -> None:
    """
    Load environment variables from the given .env file.

    Args:
        env_path: Absolute path to the .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set")
        st.stop()

    # CRITICAL: make it explicit
    os.environ["OPENAI_API_KEY"] = api_key
    
load_environment(env_path="/Workspace/vishal/pdf-chatbot/.env")
'''

def load_environment() -> None:
    """
    Load OPENAI_API_KEY for Streamlit using cwd-based resolution.
    """

    cwd = os.getcwd()
    st.write("DEBUG: cwd =", cwd)

    # Move up from /files/src → project root
    project_root = os.path.dirname(os.path.dirname(cwd))
    env_path = os.path.join(project_root, ".env")

    st.write("DEBUG: project_root =", project_root)
    st.write("DEBUG: .env path =", env_path)
    st.write("DEBUG: .env exists =", os.path.exists(env_path))

    if not os.path.exists(env_path):
        st.error(f".env file not found at {env_path}")
        st.stop()

    load_dotenv(env_path, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    st.write("DEBUG: OPENAI_API_KEY loaded =", bool(api_key))

    if not api_key:
        st.error("OPENAI_API_KEY is not set in .env")
        st.stop()

    # Ensure cached resources can access it
    os.environ["OPENAI_API_KEY"] = api_key


load_environment()
st.write(f"OPENAI_API_KEY loaded? {bool(os.getenv('OPENAI_API_KEY'))}")

#os.environ["OPENAI_API_KEY"]
#os.getenv("OPENAI_API_KEY")

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")

INDEX_NAME = f"databricks_vishal.default.pdf_chatbot_{ENV}"
ENDPOINT_NAME = f"pdf_chatbot_endpoint_{ENV}"
EMBEDDING_MODEL = "text-embedding-3-small"
RAG_MODEL_NAME = "gpt-4o-mini"

COUNTRY_CSV_PATH = (
    "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
)


from model import PDFRAGModel

# ------------------------------------------------------------------
# Cached initialization (VERY important for Streamlit)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Initialize and cache the OpenAI embeddings model for Streamlit.

    Returns:
        OpenAIEmbeddings: An instance of the embeddings model.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key,)


@st.cache_resource(show_spinner=False)
def get_pdf_rag_model() -> PDFRAGModel:
    """
    Initialize and cache the PDFRAGModel for Streamlit.

    Returns:
        PDFRAGModel: An instance of the RAG model with vector search and OpenAI integration.
    """
    return PDFRAGModel(
        index_name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        model_name=RAG_MODEL_NAME,
    )

#api_key = os.environ.get("OPENAI_API_KEY")
embeddings = get_embeddings()
pdf_model = get_pdf_rag_model()


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered",
)

st.title("📄 PDF Chat Assistant")
st.caption("Ask questions grounded strictly in the PDF content")


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, object]] = []


# ------------------------------------------------------------------
# Chat history rendering
# ------------------------------------------------------------------

def render_chat_entry(entry: Dict[str, object]) -> None:
    """
    Render a single chat turn.
    """
    st.markdown(
        f"""
        <div style="
            text-align:right;
            background-color:#a8d5ba;
            color:black;
            padding:12px;
            border-radius:12px;
            margin-bottom:5px;
        ">
        {entry['question']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            text-align:left;
            background-color:#d9d9d9;
            color:black;
            padding:12px;
            border-radius:12px;
            margin-bottom:5px;
        ">
        {entry['answer']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    citations = entry.get("citations") or []
    if citations:
        with st.expander("📎 Sources"):
            for citation in citations:
                text = citation.get("text", "")
                highlighted_lines = [
                    f"➡ {line}"
                    if any(
                        country.lower() in line.lower()
                        for country in pdf_model.COUNTRY_LIST
                    )
                    else line
                    for line in text.split("\n")
                ]
                st.text("\n".join(highlighted_lines))


for entry in st.session_state.chat_history:
    render_chat_entry(entry)
    st.markdown("---")


# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------

with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input(
        "Ask a question about the PDF",
        placeholder="What does this document say about dogs in Sweden?",
    )
    submitted = st.form_submit_button("Send")


# ------------------------------------------------------------------
# Submission handler
# ------------------------------------------------------------------

if submitted and user_question:
    with st.spinner("Searching the document and generating answer..."):
        query_embedding = embeddings.embed_query(user_question)

        result = pdf_model.ask_pdf(
            query_embedding=query_embedding,
            question=user_question,
            k=2,
        )

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "answer": result["answer"],
                "citations": result.get("citations", []),
            }
        )

    st.rerun()
