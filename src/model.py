# model.py

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc

class PDFRAGModel:
    """
    Independent PDF RAG model:
    - Uses an existing Databricks vector search index
    - Accepts precomputed query embeddings
    - Calls OpenAI LLM for chat
    - Optional: register model in Databricks Model Registry
    """
    def __init__(self, index_name, endpoint_name, api_key, model_name="gpt-4o-mini"):
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.api_key = api_key

        self.vsc = VectorSearchClient()
        self.client = OpenAI(api_key=api_key)

    # -----------------------------
    # 1️⃣ Similarity search
    # -----------------------------
    def retrieve_context(self, query_embedding, k=5):
        """
        Retrieve top-k chunks from the existing vector index
        """
        results = self.vsc.similarity_search(
            index_name=self.index_name,
            query_vector=query_embedding,
            num_results=k,
            endpoint_name=self.endpoint_name
        )
        return results

    # -----------------------------
    # 2️⃣ Build context + citations
    # -----------------------------
    @staticmethod
    def build_cited_context(results):
        contexts = []
        citations = []
        for i, r in enumerate(results):
            meta = r["metadata"]
            contexts.append(f"[{i+1}] (Page {meta.get('page')}) {meta.get('text')}")
            citations.append({"citation_id": i+1, "page": meta.get("page")})
        return "\n\n".join(contexts), citations

    # -----------------------------
    # 3️⃣ Build LLM prompt
    # -----------------------------
    @staticmethod
    def build_prompt(context, question):
        return f"""
                You are a PDF-based assistant.

                Answer the question using only the context below.
                If the answer is not explicitly stated, say:
                "I could not find this information in the provided document."

                Always include citations like: (Page X).

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

    # -----------------------------
    # 4️⃣ Ask PDF
    # -----------------------------
    def ask_pdf(self, query_embedding, question, k=5):
        """
        Perform similarity search + LLM completion
        """
        results = self.retrieve_context(query_embedding, k=k)
        context, citations = self.build_cited_context(results)
        prompt = self.build_prompt(context, question)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content,
            "citations": citations
        }

    # -----------------------------
    # 5️⃣ Register model in Databricks
    # -----------------------------
    def register_model(self, model_name):
        """
        Register this PDF RAG model in Databricks Model Registry using MLflow
        """

        class PDFRAGWrapper(pyfunc.PythonModel):
            def load_context(self, context):
                self.pdf_rag = self

            def predict(self, context, model_input):
                # model_input should contain: {"query_embedding": [...], "question": "..."}
                return self.pdf_rag.ask_pdf(
                    query_embedding=model_input["query_embedding"],
                    question=model_input["question"]
                )

        pyfunc_model_path = f"/tmp/{model_name}_pyfunc"
        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=pyfunc_model_path,
            registered_model_name=model_name
        )
        print(f"Model registered as '{model_name}' in Databricks Model Registry")