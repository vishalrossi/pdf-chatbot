import pdfplumber
from judge import Judge
import pandas as pd

# -----------------------------
# 1️⃣ Paths
# -----------------------------
pdf_path = "/dbfs/databricks_vishal/chatbot/rag_data/pdf/About_dogs.pdf"
output_csv = "/dbfs/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
delta_path = "/mnt/delta/evaluation_results"  # Change if needed

# -----------------------------
# 2️⃣ Extract countries (pages 132-138)
# -----------------------------
def extract_countries(pdf_path, start_page=132, end_page=138):
    countries = []
    with pdfplumber.open(pdf_path) as pdf:
        start_idx = start_page - 1
        end_idx = end_page - 1
        for i in range(start_idx, end_idx + 1):
            page = pdf.pages[i]
            text = page.extract_text()
            if not text:
                continue
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("•"):
                    countries.append(line)
    return list(dict.fromkeys(countries))  # remove duplicates

predicted_countries = extract_countries(pdf_path)

# -----------------------------
# 3️⃣ Save extracted countries to CSV
# -----------------------------
df = pd.DataFrame(predicted_countries, columns=["Country"])
df.to_csv(output_csv, index=False)

print(f"Extracted countries saved to: {output_csv}")
print("Predicted countries:", predicted_countries)


'''
# -----------------------------
# 3️⃣ Ground truth (replace with actual list)
# -----------------------------
ground_truth = [
    "Brazil", "Canada", "China", "Czech Republic", "Denmark", "Finland"
    # Add all countries that should appear in pages 25-37
]

# -----------------------------
# 4️⃣ Evaluate with JudgeLLM
# -----------------------------
judge = Judge()
data = [{"prediction": pred, "reference": ref} 
        for pred, ref in zip(predicted_countries, ground_truth)]

results = []
for item in data:
    result = judge.evaluate(
        completion=item["prediction"],
        reference=item["reference"],
        criteria="Correctness"
    )
    result.update({
        "prediction": item["prediction"],
        "reference": item["reference"]
    })
    results.append(result)

# -----------------------------
# 5️⃣ Aggregate metrics
# -----------------------------
accuracy = sum(r['score'] for r in results) / len(results)
print(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# 6️⃣ Store results
# -----------------------------
# Convert to DataFrame
df = pd.DataFrame(results)

# Save as Delta table
df.to_parquet(delta_path, index=False)
print(f"Results saved to Delta path: {delta_path}")

# Optionally also save JSON or CSV
df.to_csv("/dbfs/FileStore/evaluation_results.csv", index=False)
df.to_json("/dbfs/FileStore/evaluation_results.json", orient="records", indent=2)
'''