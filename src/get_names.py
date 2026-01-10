"""
Utility for extracting country names from a PDF document and saving them to CSV.

This script:
1. Loads environment variables from a .env file
2. Validates required configuration (OpenAI API key)
3. Extracts country-like entities from a specified page range in a PDF
4. Applies multiple heuristics to filter out non-country text
5. Saves the extracted countries to a CSV file

The code is structured into small, reusable functions for clarity,
maintainability, and testability.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Set

import pdfplumber
import pandas as pd
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ENV_PATH = "/Workspace/vishal/pdf-chatbot/.env"

PDF_PATH = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/About_Dogs.pdf"
OUTPUT_CSV = "/Volumes/databricks_vishal/chatbot/rag_data/pdf/extracted_countries.csv"

NON_COUNTRY_KEYWORDS: Set[str] = {
    "see",
    "list",
    "dog",
    "dogs",
    "breed",
    "breeds",
    "country",
    "countries",
}


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

    # Re-export explicitly for downstream libraries if needed
    os.environ["OPENAI_API_KEY"] = api_key


# -----------------------------------------------------------------------------
# Text filtering helpers
# -----------------------------------------------------------------------------

def is_valid_country_candidate(line: str, banned_keywords: Set[str]) -> bool:
    """
    Determine whether a line of text is a valid country candidate.

    Args:
        line: A single line of extracted PDF text.
        banned_keywords: Keywords indicating non-country content.

    Returns:
        True if the line passes all heuristics, False otherwise.
    """
    # Empty or whitespace
    if not line:
        return False

    # Bullets or list markers
    if line.startswith(("•", "o")):
        return False

    # Page numbers
    if line.isdigit():
        return False

    # Headers / ALL CAPS
    if line.isupper():
        return False

    # Only alphabetic characters and spaces
    if not re.fullmatch(r"[A-Za-z ]+", line):
        return False

    # Single-character tokens
    if len(line.strip()) == 1:
        return False

    # Keyword-based exclusion
    tokens = set(line.lower().split())
    if tokens & banned_keywords:
        return False

    return True


# -----------------------------------------------------------------------------
# Core extraction logic
# -----------------------------------------------------------------------------

def extract_countries_from_pdf(
    pdf_path: str,
    *,
    start_page: int,
    end_page: int,
    banned_keywords: Set[str],
) -> List[str]:
    """
    Extract country names from a PDF within a page range.

    Args:
        pdf_path: Path to the PDF file.
        start_page: First page number to process (1-based, inclusive).
        end_page: Last page number to process (1-based, exclusive).
        banned_keywords: Keywords used to filter out non-country lines.

    Returns:
        A list of unique country names, preserving original order.
    """
    countries: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index in range(start_page - 1, end_page):
            page = pdf.pages[page_index]
            text = page.extract_text()
            if not text:
                continue

            for raw_line in text.split("\n"):
                line = raw_line.strip()

                if is_valid_country_candidate(line, banned_keywords):
                    countries.append(line)

    # Deduplicate while preserving order
    return list(dict.fromkeys(countries))


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------

def save_countries_to_csv(countries: Iterable[str], output_path: str) -> None:
    """
    Save a list of countries to a CSV file.

    Args:
        countries: Iterable of country names.
        output_path: Destination CSV file path.
    """
    df = pd.DataFrame(countries, columns=["Country"])
    df.to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main execution routine.
    """
    load_environment(ENV_PATH)
    ensure_openai_api_key()

    countries = extract_countries_from_pdf(
        PDF_PATH,
        start_page=132,
        end_page=139,
        banned_keywords=NON_COUNTRY_KEYWORDS,
    )

    save_countries_to_csv(countries, OUTPUT_CSV)

    print(f"Extracted countries saved to: {OUTPUT_CSV}")
    print("Predicted countries:", countries)


if __name__ == "__main__":
    main()
