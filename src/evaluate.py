"""
RAG evaluation script using DeepEval metrics.

This module:
1. Loads environment variables and validates required configuration
2. Reads extracted country data used as retrieval context
3. Constructs an LLM test case
4. Evaluates the response using Faithfulness and Answer Relevancy metrics
5. Persists evaluation results to a Delta table for analysis

The code is structured into small, reusable functions for clarity,
maintainability, and easier extension to additional metrics or questions.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

INPUT_COUNTRIES_PATH = (
    "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
)
DELTA_OUTPUT_PATH = "/Volumes/databricks_vishal/chatbot/rag_data/eval_results/"

DOCUMENT_NAME = "About_dogs.pdf"
PAGE_RANGE = "132-139"

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

def load_environment(env_path: str) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Absolute path to the .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)


def ensure_openai_api_key() -> None:
    """
    Ensure OPENAI_API_KEY is present in the environment.

    Raises:
        RuntimeError: If the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    # Explicitly re-export for downstream libraries
    os.environ["OPENAI_API_KEY"] = api_key


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def load_predicted_countries(csv_path: str) -> List[str]:
    """
    Load predicted countries from a CSV file.

    Args:
        csv_path: Path to the CSV containing extracted countries.

    Returns:
        A list of country names.
    """
    df = pd.read_csv(csv_path)
    return df["Country"].dropna().tolist()


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def build_test_case(
    *,
    question: str,
    answer: str,
    retrieval_context: List[str],
) -> LLMTestCase:
    """
    Build a DeepEval LLMTestCase.

    Args:
        question: Input question to the LLM.
        answer: LLM-generated answer.
        retrieval_context: Retrieved context passed to the LLM.

    Returns:
        An initialized LLMTestCase.
    """
    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )


def evaluate_test_case(test_case: LLMTestCase) -> Dict[str, Dict[str, float | str]]:
    """
    Evaluate a test case using DeepEval metrics.

    Args:
        test_case: The LLM test case to evaluate.

    Returns:
        A dictionary containing scores and reasons for each metric.
    """
    faithfulness = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD)
    faithfulness.measure(test_case)

    relevancy = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD)
    relevancy.measure(test_case)

    return {
        "faithfulness": {
            "score": faithfulness.score,
            "reason": faithfulness.reason,
        },
        "relevancy": {
            "score": relevancy.score,
            "reason": relevancy.reason,
        },
    }


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------

def persist_results_to_delta(
    spark: SparkSession,
    results: List[Dict[str, object]],
    output_path: str,
) -> None:
    """
    Persist evaluation results to a Delta table.

    Args:
        spark: Active SparkSession.
        results: List of evaluation result dictionaries.
        output_path: Delta table storage path.
    """
    df: DataFrame = spark.createDataFrame(results)

    (
        df.write.format("delta")
        .mode("append")
        .save(output_path)
    )


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main execution routine for RAG evaluation.
    """
    load_environment(ENV_PATH)
    ensure_openai_api_key()

    # Load retrieval context
    predicted_countries = load_predicted_countries(INPUT_COUNTRIES_PATH)
    retrieval_context = ["\n".join(predicted_countries)]

    # Define Q&A
    question = "Name some of the countries mentioned in the document."
    llm_answer = (
        "The document mentions Brazil, Canada, China, Czech Republic, "
        "Denmark, Finland, Norway, South Africa and others."
    )

    # Build and evaluate test case
    test_case = build_test_case(
        question=question,
        answer=llm_answer,
        retrieval_context=retrieval_context,
    )

    metrics = evaluate_test_case(test_case)

    print("Faithfulness:", metrics["faithfulness"]["score"])
    print("Reason:", metrics["faithfulness"]["reason"])
    print("Relevancy:", metrics["relevancy"]["score"])
    print("Reason:", metrics["relevancy"]["reason"])

    # Prepare result record
    results = [
        {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document": DOCUMENT_NAME,
            "page_range": PAGE_RANGE,
            "question": question,
            "answer": llm_answer,
            "faithfulness_score": metrics["faithfulness"]["score"],
            "faithfulness_reason": metrics["faithfulness"]["reason"],
            "relevancy_score": metrics["relevancy"]["score"],
            "relevancy_reason": metrics["relevancy"]["reason"],
        }
    ]

    # Persist to Delta
    spark = SparkSession.builder.getOrCreate()
    persist_results_to_delta(spark, results, DELTA_OUTPUT_PATH)


if __name__ == "__main__":
    main()



'''
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
'''