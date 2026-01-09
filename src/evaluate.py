import pdfplumber
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import pandas as pd
import re
import os
from dotenv import load_dotenv

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
# 1️⃣ Paths
# -----------------------------
pdf_path = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/About_Dogs.pdf"
output_csv = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
delta_path = "/mnt/delta/evaluation_results"  # Change if needed


# -----------------------------
# 2️⃣ Extract countries (pages 132-138)
# -----------------------------


NON_COUNTRY_KEYWORDS = {
    "see", "list", "dog", "dogs", "breed", "breeds", "country", "countries"
}

def extract_countries(pdf_path, start_page=132, end_page=139):
    countries = []

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page):
            text = pdf.pages[i].extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()

                # Skip empty
                if not line:
                    continue

                # Skip bullets / breed entries
                if line.startswith(("•", "o")):
                    continue

                # Skip page numbers
                if line.isdigit():
                    continue

                # Skip headers / ALL CAPS
                if line.isupper():
                    continue

                # Only alphabetic + spaces
                if not re.match(r"^[A-Za-z ]+$", line):
                    continue

                # Skip single letters
                if len(line) == 1:
                    continue

                # Skip section headers using keywords
                tokens = set(line.lower().split())
                if tokens & NON_COUNTRY_KEYWORDS:
                    continue

                countries.append(line)

    return list(dict.fromkeys(countries))  # preserve order

predicted_countries = extract_countries(pdf_path)

# -----------------------------
# 3️⃣ Save extracted countries to CSV
# -----------------------------
df = pd.DataFrame(predicted_countries, columns=["Country"])
df.to_csv(output_csv, index=False)

print(f"Extracted countries saved to: {output_csv}")
print("Predicted countries:", predicted_countries)


question = "Which countries are mentioned in the document?"

llm_answer = """
The document mentions Brazil, Canada, China, Czech Republic, Denmark, and Finland.
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