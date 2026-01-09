from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import pandas as pd
import re
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid
from pyspark.sql import SparkSession

# -----------------------------
# Config
# -----------------------------
ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set")

os.environ["OPENAI_API_KEY"] = api_key


# -----------------------------
# Paths
# -----------------------------
input_countries_path = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
delta_path = "/Volumes/databricks_vishal/chatbot/rag_data/eval_results/"  # Change if needed


# -----------------------------
# Load countries
# -----------------------------
input_counties_df = pd.read_csv(input_countries_path)

predicted_countries = input_counties_df["Country"].dropna().tolist()


# -----------------------------
# Initiate Q&a
# -----------------------------
question = "Name some of the countries mentioned in the document."

llm_answer = """
The document mentions Brazil, Canada, China, Czech Republic, Denmark, Finland, Norway, South Africa and others .
"""

retrieval_context = [
    "\n".join(predicted_countries)
]

test_case = LLMTestCase(
    input=question,
    actual_output=llm_answer,
    retrieval_context=retrieval_context
)

faithfulness = FaithfulnessMetric(threshold=0.7)
faithfulness.measure(test_case)

print("Faithfulness:", faithfulness.score)
print("Reason:", faithfulness.reason)

relevancy = AnswerRelevancyMetric(threshold=0.7)
relevancy.measure(test_case)

print("Relevancy:", relevancy.score)
print("Reason:", relevancy.reason)



results = [{
    "run_id": str(uuid.uuid4()),
    "timestamp":  datetime.now(timezone.utc).isoformat(),
    "document": "About_dogs.pdf",
    "page_range": "132-139",
    "question": question,
    "answer": llm_answer,
    "faithfulness_score": faithfulness.score,
    "faithfulness_reason": faithfulness.reason,
    "relevancy_score": relevancy.score,
    "relevancy_reason": relevancy.reason
}]

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(results)

df.write.format("delta") \
  .mode("append") \
  .save(delta_path)
  