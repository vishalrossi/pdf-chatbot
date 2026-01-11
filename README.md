#  PDF Chatbot (Databricks)

This repository implements an **internal, production-grade Retrieval-Augmented Generation (RAG) chatbot**
built on **Databricks**.

The system enables users to ask natural-language questions over PDF documents and receive
**answers grounded strictly in the document content**, with **citations**.

This project is designed for:
- Internal knowledge assistants
- Document-based Q&A
- Databricks-native deployment and governance


# Features
- PDF-based contextual chatbot
- Databricks Vector Search
- MLflow model registry
- Databricks Model Serving
- CI/CD with GitHub Actions
- Dev / Prod isolation (TODO)

# RAG Architecture
PDF → Embeddings → Vector Search → LLM → API

# Deployment
- Push to `dev` → auto deploy to dev
- Push to `prod` → approval → prod deploy (TODO)


# Pdf chatbot architecture

Databricks Workspace
└── vishal
    └── chatbot
        ├── src/
        │   ├── ingest_pdf.py
        │   ├── get_names.py
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


---

# Environment & Secrets (Internal)

### Required Environment Variables

```env
OPENAI_API_KEY=<stored via Databricks secrets or local .env>
DATABRICKS_BUNDLE_TARGET=dev
```

# Running the pipeline

1. Deploy resources to Databricks (Sync Databricks Asset Bundles with Databricks resources)
databricks bundle deploy --target dev

2. Run the end-to-end RAG pipeline
databricks bundle run rag_pipeline --target dev

3. Run the Streamlit UI pipeline
databricks bundle run pdf_chat --target dev

4. If you want to run only one module, for ex. storage_init or pdf_ingestion (optional)
databricks bundle run storage_init --target dev
databricks bundle run pdf_ingestion --target dev

# Testing the results through UI

python -m streamlit run src/chat_pdf_streamlit.py
