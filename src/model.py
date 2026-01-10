"""
This module implements a Retrieval-Augmented Generation (RAG) workflow
for question answering over PDF documents using Databricks Vector Search
and OpenAI chat models.

High-level flow:
1. A query embedding is generated outside this module.
2. The embedding is used to retrieve relevant PDF text chunks from an
   existing Databricks Vector Search index.
3. Retrieved chunks are filtered and assembled into a context window
   (optionally scoped to a detected country).
4. An OpenAI chat model generates an answer strictly based on the
   retrieved context.
5. The model returns both the answer and chunk-level citations.

Key characteristics:
- Assumes the vector index already exists and is populated.
- Does NOT generate embeddings (handled upstream).
- Designed for interactive use (e.g., Streamlit) and batch inference.
- Can be registered as an MLflow PyFunc model (Unity Catalog compatible).

Intended usage:
- As a standalone RAG inference component
- As a registered MLflow model for Databricks model serving
- As a backend for document Q&A applications

External dependencies:
- Databricks Vector Search
- OpenAI Python SDK
- MLflow (for optional model registration)

Security & configuration:
- Requires OPENAI_API_KEY to be set as an environment variable.
- Does not persist user queries or retrieved content.

"""

"""
Retrieval-Augmented Generation (RAG) model for PDF-based Q&A using
Databricks Vector Search and OpenAI.

This implementation is:
- MLflow PyFunc compatible
- Unity Catalog compatible
- Safe for Databricks Model Serving
- Free of pickle / threading errors
"""

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import json
import os
from typing import List, Dict, Tuple, Optional, Any


# ============================================================================
# MLflow PyFunc Wrapper (TOP-LEVEL — REQUIRED)
# ============================================================================
class PDFRAGWrapper(pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper that reconstructs PDFRAGModel
    from a serialized configuration.
    """

    def load_context(self, context) -> None:
        """
        Load model configuration and recreate PDFRAGModel.
        """
        with open(context.artifacts["model_config"], "r") as f:
            config = json.load(f)

        self.pdf_rag = PDFRAGModel.from_config(config)

    def predict(self, context, model_input: dict) -> dict:
        """
        Run inference using the reconstructed PDFRAGModel.
        """
        query_embedding = json.loads(model_input["query_embedding"])

        result = self.pdf_rag.ask_pdf(
            query_embedding=query_embedding,
            question=model_input["question"],
        )

        result["citations"] = json.dumps(result.get("citations", []))
        return result


# ============================================================================
# Main RAG Model
# ============================================================================
class PDFRAGModel:
    """
    Retrieval-Augmented Generation (RAG) model over PDF content.
    """

    COUNTRY_LIST: List[str] = [
        "Australia", "Brazil", "Canada", "China", "Czech Republic",
        "Denmark", "Finland", "France", "Germany", "Hungary", "Iceland",
        "India", "Ireland", "Japan", "Korea", "Mexico", "Norway",
        "South Africa", "Spain", "Sweden", "Thailand", "Turkey",
        "United Kingdom", "United States", "Zimbabwe",
    ]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        index_name: str,
        endpoint_name: str,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set")

        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name

        # IMPORTANT: clients are created at runtime (NOT serialized)
        self.client = OpenAI()
        self.vsc = VectorSearchClient()

        self.index = self.vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name,
        )

    # ------------------------------------------------------------------
    # Serialization helpers (CRITICAL)
    # ------------------------------------------------------------------
    def to_config(self) -> dict:
        """
        Serialize lightweight model configuration only.
        """
        return {
            "index_name": self.index_name,
            "endpoint_name": self.endpoint_name,
            "model_name": self.model_name,
        }

    @classmethod
    def from_config(cls, config: dict) -> "PDFRAGModel":
        """
        Reconstruct model from serialized configuration.
        """
        return cls(
            index_name=config["index_name"],
            endpoint_name=config["endpoint_name"],
            model_name=config["model_name"],
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_context(
        self,
        query_embedding: List[float],
        k: int = 3,
    ) -> List[str]:
        response = self.index.similarity_search(
            query_vector=query_embedding,
            columns=["text"],
            num_results=k,
        )
        return [
            row[0] for row in response["result"]["data_array"]
            if row and row[0]
        ]

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def build_cited_context(
        self,
        results: List[str],
        country: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:

        context_lines: List[str] = []
        citations: List[Dict[str, Any]] = []

        for idx, text in enumerate(results):
            lines = [l.strip() for l in str(text).split("\n") if l.strip()]
            filtered: List[str] = []

            for line in lines:
                if country:
                    if country.lower() in line.lower() or line.startswith("•"):
                        filtered.append(line)
                else:
                    filtered.append(line)

            if not filtered:
                continue

            context_lines.extend(filtered)
            citations.append({
                "id": idx + 1,
                "text": "\n".join(filtered),
            })

        return "\n".join(context_lines), citations

    # ------------------------------------------------------------------
    # RAG inference
    # ------------------------------------------------------------------
    def ask_pdf(
        self,
        query_embedding: List[float],
        question: str,
        k: int = 2,
    ) -> Dict[str, Any]:

        country = next(
            (c for c in self.COUNTRY_LIST if c.lower() in question.lower()),
            None,
        )

        results = self.retrieve_context(query_embedding, k)
        context, citations = self.build_cited_context(results, country)

        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY "
            "the provided context. "
            "If the country is not found, respond exactly:\n"
            "'I could not find this information in the provided document, so I can't answer!'\n"
        )

        user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "citations": citations,
        }

    # ------------------------------------------------------------------
    # MLflow registration (FIXED)
    # ------------------------------------------------------------------
    def register_model(
        self,
        model_name: str,
        experiment_name: str = "/Shared/pdf_rag_experiment",
    ) -> None:

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        input_schema = Schema([
            ColSpec(DataType.string, "question"),
            ColSpec(DataType.string, "query_embedding"),
        ])
        output_schema = Schema([
            ColSpec(DataType.string, "answer"),
            ColSpec(DataType.string, "citations"),
        ])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Save ONLY config (never pickle clients)
        with open("model_config.json", "w") as f:
            json.dump(self.to_config(), f)

        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=f"{model_name}_pyfunc",
            registered_model_name=model_name,
            signature=signature,
            artifacts={
                "model_config": "model_config.json"
            },
        )



    '''
    # ------------------------------------------------------------------
    # MLflow registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        experiment_name: str = "/Shared/pdf_rag_experiment",
    ) -> None:
        """
        Register this PDF RAG model in Databricks Model Registry.

        Args:
            model_name: Name of the registered model.
            experiment_name: MLflow experiment path.
        """

        pdf_rag_instance = self

        class PDFRAGWrapper(pyfunc.PythonModel):
            """
            MLflow PyFunc wrapper for PDFRAGModel.
            """

            def load_context(self, context):
                self.pdf_rag = pdf_rag_instance

            def predict(self, context, model_input):
                """
                Args:
                    model_input: Dict with keys:
                        - question (str)
                        - query_embedding (JSON string)

                Returns:
                    Dict with keys:
                        - answer (str)
                        - citations (JSON string)
                """
                query_embedding = json.loads(model_input["query_embedding"])

                result = self.pdf_rag.ask_pdf(
                    query_embedding=query_embedding,
                    question=model_input["question"],
                )

                result["citations"] = json.dumps(result.get("citations", []))
                return result

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        input_schema = Schema(
            [
                ColSpec(DataType.string, "question"),
                ColSpec(DataType.string, "query_embedding"),
            ]
        )
        output_schema = Schema(
            [
                ColSpec(DataType.string, "answer"),
                ColSpec(DataType.string, "citations"),
            ]
        )

        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
        )

        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=f"{model_name}_pyfunc",
            registered_model_name=model_name,
            signature=signature,
        )
    '''
