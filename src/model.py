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


from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import json
import os
from typing import List, Dict, Tuple, Optional, Any


# ----------------------------------------------------------------------
# MLflow PyFunc wrapper (top-level for serialization)
# ----------------------------------------------------------------------
class PDFRAGWrapper(pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for PDFRAGModel.

    This wrapper allows a PDFRAGModel instance to be registered
    and served via MLflow Model Registry.
    """

    def load_context(self, context: dict) -> None:
        """
        Load the PDFRAGModel instance into the wrapper.

        Args:
            context: Dictionary containing 'pdf_rag_instance'
        """
        self.pdf_rag: PDFRAGModel = context["pdf_rag_instance"]

    def predict(self, context, model_input: dict) -> dict:
        """
        Predict using PDFRAGModel.

        Args:
            model_input: Dictionary with keys:
                - question (str)
                - query_embedding (JSON string)

        Returns:
            Dictionary with keys:
                - answer (str)
                - citations (JSON string)
        """
        query_embedding = json.loads(model_input["query_embedding"])
        result = self.pdf_rag.ask_pdf(
            query_embedding=query_embedding,
            question=model_input["question"]
        )
        result["citations"] = json.dumps(result.get("citations", []))
        return result


# ----------------------------------------------------------------------
# Main PDF RAG Model
# ----------------------------------------------------------------------
class PDFRAGModel:
    """
    Retrieval-Augmented Generation (RAG) model over PDF content.

    This model:
    - Uses an existing Databricks Vector Search index
    - Accepts precomputed query embeddings
    - Retrieves relevant PDF chunks
    - Generates answers using an OpenAI chat model
    - Can be registered as an MLflow PyFunc model (Unity Catalog compliant)
    """

    COUNTRY_LIST: List[str] = [
        "Australia", "Brazil", "Canada", "China", "Czech Republic",
        "Denmark", "Finland", "France", "Germany", "Hungary", "Iceland",
        "India", "Ireland", "Japan", "Korea", "Mexico", "Norway",
        "South Africa", "Spain", "Sweden", "Thailand", "Turkey",
        "United Kingdom", "United States", "Zimbabwe",
    ]

    def __init__(
        self,
        index_name: str,
        endpoint_name: str,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """
        Initialize the PDF RAG model.

        Args:
            index_name: Fully qualified Databricks Vector Search index name.
            endpoint_name: Databricks Vector Search endpoint name.
            model_name: OpenAI chat model name.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. "
                "Please set it as an environment variable."
            )

        self.client = OpenAI()
        self.vsc = VectorSearchClient()

        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name

        # Retrieve an existing vector search index
        self.index = self.vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query_embedding: List[float],
        k: int = 3,
    ) -> List[str]:
        """
        Retrieve top-k text chunks from the vector search index.

        Args:
            query_embedding: Precomputed embedding vector.
            k: Number of chunks to retrieve.

        Returns:
            List of retrieved text chunks.
        """
        response = self.index.similarity_search(
            query_vector=query_embedding,
            columns=["text"],
            num_results=k,
        )

        # Databricks returns rows as arrays aligned with `columns`
        return [
            row[0]
            for row in response["result"]["data_array"]
            if row and row[0]
        ]

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def build_cited_context(
        self,
        results: List[str],
        country: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build a filtered context string and citation metadata.

        Args:
            results: Retrieved text chunks.
            country: Optional country name to filter relevant lines.

        Returns:
            Tuple of:
                - Combined context string
                - List of citation dictionaries
        """
        context_lines: List[str] = []
        citations: List[Dict[str, Any]] = []

        for idx, text in enumerate(results):
            text = str(text).strip()
            if not text:
                continue

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            relevant_lines: List[str] = []

            for line in lines:
                if country:
                    if country.lower() in line.lower() or line.startswith("•"):
                        relevant_lines.append(line)
                else:
                    relevant_lines.append(line)

            if not relevant_lines:
                continue

            context_lines.extend(relevant_lines)
            citations.append(
                {
                    "id": idx + 1,
                    "text": "\n".join(relevant_lines),
                }
            )

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
        """
        Perform retrieval-augmented generation over the PDF.

        Args:
            query_embedding: Precomputed query embedding.
            question: User question.
            k: Number of retrieved chunks.

        Returns:
            Dictionary with:
                - "answer": Generated answer
                - "citations": List of citation metadata
        """
        country_in_question: Optional[str] = next(
            (c for c in self.COUNTRY_LIST if c.lower() in question.lower()),
            None,
        )

        results = self.retrieve_context(query_embedding, k=k)
        context, citations = self.build_cited_context(
            results,
            country=country_in_question,
        )

        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY "
            "the provided context. "
            "If the country is present, list all relevant breeds naturally. "
            "If the country is not found, respond exactly:\n"
            "'I could not find this information in the provided document, so I can't answer!'\n"
            "Preserve chunk-level citations."
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
        """
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

        # Use top-level wrapper and pass the PDFRAGModel instance
        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=f"{model_name}_pyfunc",
            registered_model_name=model_name,
            signature=signature,
            python_model_context={"pdf_rag_instance": self}  # <-- pass self
        )