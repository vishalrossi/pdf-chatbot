"""
Entry point for registering the PDF RAG model in Databricks Model Registry.

This script:
1. Loads environment variables from a .env file
2. Resolves Databricks- and environment-specific configuration
3. Initializes the PDFRAGModel
4. Ensures an MLflow experiment exists
5. Registers the model in Databricks Model Registry

The script is intentionally structured into small, testable functions
for maintainability and reuse.
"""

from __future__ import annotations

import os
from typing import Optional

import mlflow
from dotenv import load_dotenv

from model import PDFRAGModel


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def load_environment(env_path: str) -> None:
    """
    Load environment variables from the given .env file.

    Args:
        env_path: Absolute path to the .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)


def get_env(default: str = "dev") -> str:
    """
    Resolve the current deployment environment.

    Args:
        default: Fallback environment name if none is set.

    Returns:
        The resolved environment name (e.g., dev, staging, prod).
    """
    return os.getenv("DATABRICKS_BUNDLE_TARGET", default)


def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from environment variables.

    Raises:
        RuntimeError: If the API key is not set.

    Returns:
        OpenAI API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return api_key


# -----------------------------------------------------------------------------
# MLflow helpers
# -----------------------------------------------------------------------------

def ensure_mlflow_experiment(experiment_name: str) -> None:
    """
    Ensure that an MLflow experiment exists and set it as active.

    Args:
        experiment_name: Name of the MLflow experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def create_pdf_rag_model(
    *,
    catalog: str,
    schema: str,
    env: str,
    api_key: str,
) -> PDFRAGModel:
    """
    Create and configure a PDFRAGModel instance.

    Args:
        catalog: Databricks catalog name.
        schema: Databricks schema name.
        env: Deployment environment.
        api_key: OpenAI API key.

    Returns:
        An initialized PDFRAGModel instance.
    """
    index_name = f"{catalog}.{schema}.pdf_chatbot_{env}"
    endpoint_name = f"pdf_chatbot_endpoint_{env}"

    return PDFRAGModel(index_name, endpoint_name, api_key)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main(
    env_path: str,
    catalog: str,
    schema: str,
    experiment_name: str,
) -> None:
    """
    Main execution routine.

    Args:
        env_path: Path to the .env file.
        catalog: Databricks catalog name.
        schema: Databricks schema name.
        experiment_name: MLflow experiment name.
    """
    # Load environment variables
    load_environment(env_path)

    # Resolve runtime configuration
    env = get_env()
    api_key = get_openai_api_key()

    # Initialize model
    pdf_model = create_pdf_rag_model(
        catalog=catalog,
        schema=schema,
        env=env,
        api_key=api_key,
    )

    # Ensure MLflow experiment exists
    ensure_mlflow_experiment(experiment_name)

    # Register model
    model_name = f"pdf_rag_model_{env}"
    pdf_model.register_model(model_name)

    print(f"Model '{model_name}' registered in Databricks Model Registry.")


if __name__ == "__main__":
    ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"
    CATALOG = "databricks_vishal"
    SCHEMA = "default"
    EXPERIMENT_NAME = "/Shared/pdf_rag_experiment"

    main(
        env_path=ENV_PATH,
        catalog=CATALOG,
        schema=SCHEMA,
        experiment_name=EXPERIMENT_NAME,
    )
