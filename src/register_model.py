from model import PDFRAGModel
import os
from dotenv import load_dotenv

ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

#api_key=os.getenv('OPENAI_API_KEY')

ENV = os.getenv("DATABRICKS_BUNDLE_TARGET", "dev")

API_KEY = os.environ.get("OPENAI_API_KEY")
ENV = "dev"
CATALOG = "databricks_vishal"
SCHEMA = "default"

INDEX_NAME = f"{CATALOG}.{SCHEMA}.pdf_chatbot_{ENV}"
ENDPOINT_NAME = f"pdf_chatbot_endpoint_{ENV}"

pdf_model = PDFRAGModel(INDEX_NAME, ENDPOINT_NAME, API_KEY)

# Register in Databricks Model Registry
pdf_model.register_model(f"pdf_rag_model_{ENV}")
print("Model registered in Databricks Model Registry.")
