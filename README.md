#  PDF Chatbot (Databricks)

This project implements a production-grade Retrieval Augmented Generation (RAG)
chatbot using Databricks.

## Features
- PDF-based contextual chatbot
- Databricks Vector Search
- MLflow model registry
- Databricks Model Serving
- CI/CD with GitHub Actions
- Dev / Prod isolation

## RAG Architecture
PDF → Embeddings → Vector Search → LLM → API

## Deployment
- Push to `dev` → auto deploy to dev
- Push to `prod` → approval → prod deploy


# epassi pdf chatbot architecture

Databricks Workspace
└── vishal
    └── chatbot
        ├── src/
        │   ├── ingest_pdf.py
        │   ├── get_names.py.py
        │   ├── model.py
        │   ├── register_model.py
        │   ├── evaluate.py
        │   └── chat_pdf_streamlit.py
        │ 
        ├── utils/
        │   ├── storage.py
        │
        ├── databricks.yml
        ├── requirements.txt
        ├── README.md