# model.py

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import json
import os

class PDFRAGModel:
    """
    Independent PDF RAG model:
    - Uses an existing Databricks vector search index
    - Accepts precomputed query embeddings
    - Calls OpenAI LLM for chat
    - Optional: register model in Databricks Model Registry
    """
    '''
    def __init__(self, index_name, endpoint_name, api_key, model_name="gpt-4o-mini"):
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.api_key = api_key

        self.vsc = VectorSearchClient()
        self.client = OpenAI(api_key=api_key)
    '''
        
    def __init__(self, index_name, endpoint_name, model_name= "gpt-4o-mini"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. "
                "Set it via environment variable or .env file."
            )
        

        self.client = OpenAI()  # reads from env
        self.vs_client = VectorSearchClient()

        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name

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
        
    def register_model(self, model_name, experiment_name="/Shared/pdf_rag_experiment"):
        """
        Register this PDF RAG model in Databricks Model Registry (Unity Catalog compliant)

        Args:
            model_name (str): Name of the registered model in Databricks Model Registry
            experiment_name (str): MLflow experiment path for logging the run
        """
        class PDFRAGWrapper(pyfunc.PythonModel):
            def load_context(self, context):
                self.pdf_rag = self

            def predict(self, context, model_input):
                """
                model_input: dict with keys:
                    - "question": str
                    - "query_embedding": JSON string of embedding list
                Returns:
                    dict with keys:
                        - "answer": str
                        - "citations": JSON string
                """
                query_embedding = json.loads(model_input["query_embedding"])
                result = self.pdf_rag.ask_pdf(query_embedding=query_embedding,
                                            question=model_input["question"])

                # Convert citations to JSON string
                result["citations"] = json.dumps(result.get("citations", []))
                return result

        # -----------------------------
        # Ensure experiment exists
        # -----------------------------
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # -----------------------------
        # Define MLflow signature (Unity Catalog compliant)
        # -----------------------------
        input_schema = Schema([
            ColSpec(DataType.string, "question"),
            ColSpec(DataType.string, "query_embedding")  # JSON string of floats
        ])
        output_schema = Schema([
            ColSpec(DataType.string, "answer"),
            ColSpec(DataType.string, "citations")       # JSON string
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # -----------------------------
        # Log and register the model
        # -----------------------------
        artifact_path = f"{model_name}_pyfunc"  # simple name, no slashes or periods

        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=artifact_path,
            registered_model_name=model_name,
            signature=signature
        )

        print(f"Model '{model_name}' registered successfully in Databricks Model Registry")