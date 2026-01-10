'''
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import json
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class PDFRAGModel:
    """
    PDF Retrieval-Augmented Generation (RAG) model.

    Features:
    - Query a Databricks Vector Search index using precomputed embeddings
    - Build citation-aware context for the LLM
    - Generate grounded answers using OpenAI Chat Completions
    - Register the model in Databricks Model Registry via MLflow
    """

    def __init__(self, index_name: str, endpoint_name: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the PDF RAG model.

        Args:
            index_name: Fully qualified Vector Search index name.
            endpoint_name: Databricks Vector Search endpoint name.
            model_name: OpenAI chat model name.

        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI()
        self.vsc = VectorSearchClient()
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.index = self.vsc.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)

        self.COUNTRY_LIST = self._load_country_list()

    # -----------------------------
    # Helper methods
    # -----------------------------

    @staticmethod
    def _load_country_list(csv_path: str = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv") -> List[str]:
        df = pd.read_csv(csv_path)
        return df["Country"].dropna().tolist()

    @staticmethod
    def _detect_country(question: str, countries: List[str]) -> Optional[str]:
        question_lower = question.lower()
        for country in countries:
            if country.lower() in question_lower:
                return country
        return None

    # -----------------------------
    # Retrieval & context
    # -----------------------------

    def retrieve_context(self, query_embedding: List[float], k: int = 3) -> List[str]:
        response = self.index.similarity_search(query_vector=query_embedding, columns=["text"], num_results=k)
        return [row[0] for row in response["result"]["data_array"]]

    def build_cited_context(self, chunks: List[str], country: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        context_lines = []
        citations = []
        for idx, chunk in enumerate(chunks, start=1):
            text = str(chunk).strip()
            if not text:
                continue
            relevant_lines = [line.strip() for line in text.split("\n") if line.strip() and (not country or country.lower() in line.lower() or line.startswith("•"))]
            if not relevant_lines:
                continue
            joined = "\n".join(relevant_lines)
            context_lines.append(joined)
            citations.append({"id": idx, "text": joined})
        return "\n".join(context_lines), citations

    # -----------------------------
    # RAG query
    # -----------------------------

    def ask_pdf(self, query_embedding: List[float], question: str, k: int = 2) -> Dict[str, Any]:
        country = self._detect_country(question, self.COUNTRY_LIST)
        chunks = self.retrieve_context(query_embedding, k=k)
        context, citations = self.build_cited_context(chunks, country)
        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY the provided context.\n"
            "The context may contain countries and lists of dog breeds.\n"
            "If the country is not found, respond exactly: 'I could not find this information in the provided document.'\n"
            "Always preserve chunk-level citations."
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
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return {"answer": response.choices[0].message.content.strip(), "citations": citations}

    # -----------------------------
    # MLflow registration
    # -----------------------------

    def register_model(self, model_name: str, experiment_name: str = "/Shared/pdf_rag_experiment") -> None:
        """
        Register PDF RAG model in Databricks Model Registry.

        Args:
            model_name: Name for the MLflow registered model.
            experiment_name: MLflow experiment path.
        """
        pdf_instance = self

        class PDFRAGWrapper(pyfunc.PythonModel):
            def load_context(self, context):
                self.model = pdf_instance

            def predict(self, context, model_input):
                embedding = json.loads(model_input["query_embedding"])
                result = self.model.ask_pdf(query_embedding=embedding, question=model_input["question"])
                return {"answer": result["answer"], "citations": json.dumps(result.get("citations", []))}

        # Ensure experiment exists
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        signature = ModelSignature(
            inputs=Schema([ColSpec(DataType.string, "question"), ColSpec(DataType.string, "query_embedding")]),
            outputs=Schema([ColSpec(DataType.string, "answer"), ColSpec(DataType.string, "citations")]),
        )

        # Log and register the model
        mlflow.pyfunc.log_model(
            python_model=PDFRAGWrapper(),
            artifact_path=f"{model_name}_pyfunc",
            registered_model_name=model_name,
            signature=signature,
        )

        print(f"Model '{model_name}' registered successfully")
'''


from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import json
import os
import pandas as pd

class PDFRAGModel:
    """
    Independent PDF RAG model:
    - Uses an existing Databricks vector search index
    - Accepts precomputed query embeddings
    - Calls OpenAI LLM for chat
    - Optional: register model in Databricks Model Registry
    """
    
    input_countries_path = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"
    input_counties_df = pd.read_csv(input_countries_path)

    COUNTRY_LIST = input_counties_df["Country"].dropna().tolist()
    '''
    # List of countries appearing in your PDF
    COUNTRY_LIST = [
        "Australia", "Brazil", "Canada", "China", "Czech Republic",
        "Denmark", "Finland", "France", "Germany", "Hungary", "Iceland",
        "India", "Ireland", "Japan", "Korea", "Mexico", "Norway",
        "South Africa", "Spain", "Sweden", "Thailand", "Turkey", 
        "United Kingdom", "United States", "Zimbabwe"
        ]
    '''
    def __init__(self, index_name, endpoint_name, model_name= "gpt-4o-mini"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. "
                "Set it via environment variable or .env file."
            )
        

        self.client = OpenAI()  # reads from env
        self.vsc = VectorSearchClient()

        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        
        # THIS IS THE IMPORTANT PART
        self.index = self.vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name
        )

    # -----------------------------
    # 1️⃣ Similarity search
    # -----------------------------
    
    def retrieve_context(self, query_embedding, k=3):
        """
        Retrieve top-k chunks from the existing vector index
        """
        response = self.index.similarity_search(
            query_vector=query_embedding,
            columns=["text"],
            num_results=k,
        )
        return response["result"]["data_array"]


    def build_cited_context(self, results, country=None):
        """
        Build context string and citations safely.
        Handles results as a list of strings.
        """
        context_lines = []
        citations = []

        for idx, text in enumerate(results):
            # Ensure text is string
            text = str(text).strip()
            if not text:
                continue

            # Split into lines
            lines = text.split("\n")
            relevant_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if country:
                    # Include only lines with the country or bullet points
                    if country.lower() in line.lower() or line.startswith("•"):
                        relevant_lines.append(line)
                else:
                    relevant_lines.append(line)

            if not relevant_lines:
                continue

            # Add to context and citations
            context_lines.extend(relevant_lines)
            citations.append({
                "id": idx + 1,  # simple chunk id
                "score": "\n".join(relevant_lines)
            })

        # Combine all relevant lines into a single string for the LLM
        context = "\n".join(context_lines)
        return context, citations

    def ask_pdf(self, query_embedding, question, k=2):
        """
        Perform similarity search + LLM completion with country filtering.
        Returns answer and citations for Streamlit display.
        """
        # 1️⃣ Detect country mentioned in the question
        country_in_question = None
        for c in self.COUNTRY_LIST:
            if c.lower() in question.lower():
                country_in_question = c
                break

        # 2️⃣ Retrieve top-k chunks from vector search
        results = self.retrieve_context(query_embedding, k=k)

        # 3️⃣ Build context and citations safely
        context, citations = self.build_cited_context(results, country=country_in_question)

        # 4️⃣ System prompt for the LLM
        system_prompt = """
    You are a helpful assistant that answers questions using ONLY the provided context.
    The context may contain countries and lists of dog breeds.
    If the country is in the context, list all breeds in a natural sentence.
    If the country is not found, respond exactly:
    'I could not find this information in the provided document.'
    Always preserve chunk-level citations.
    """

        # 5️⃣ User prompt including context and question
        user_prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        # 6️⃣ Call the LLM
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        # 7️⃣ Return the answer + citations
        return {
            "answer": response.choices[0].message.content.strip(),
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