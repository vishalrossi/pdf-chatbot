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

## Architecture
PDF → Embeddings → Vector Search → LLM → API

## Deployment
- Push to `dev` → auto deploy to dev
- Push to `prod` → approval → prod deploy

## Workspace
/vishal/chatbot

# epassi pdf chatbot architecture

Databricks Workspace
└── vishal
    └── chatbot
        ├── src/
        │   ├── ingest_pdf.py
        │   ├── rag_chain.py
        │   ├── model.py
        │   └── config/
        │       ├── dev.yaml
        │       └── prod.yaml
        │
        ├── notebooks/
        │   ├── pdf_ingestion.py
        │   └── chatbot_inference.py
        │
        ├── databricks.yml
        ├── requirements.txt
        ├── README.md
        └── tests/