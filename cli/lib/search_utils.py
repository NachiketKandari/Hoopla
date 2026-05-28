import json
import os
from typing import Optional

from dotenv import load_dotenv

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_ALPHA_VALUE = 0.5
DEFAULT_K_VALUE = 60.0
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIE_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
TESTCASES_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PKL_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PKL_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
EMBEDDING_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDING_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

load_dotenv()


def _load_from_streamlit_secrets(key: str) -> Optional[str]:
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None

    try:
        return st.secrets.get(key)
    except StreamlitSecretNotFoundError:
        return None


def get_gemini_api_key() -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key

    api_key = _load_from_streamlit_secrets("GEMINI_API_KEY")
    if api_key:
        return api_key

    return None


def get_deepseek_api_key() -> Optional[str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key

    api_key = _load_from_streamlit_secrets("DEEPSEEK_API_KEY")
    if api_key:
        return api_key

    return None


def get_llm_client():
    """Return an LLM client based on LLM_PROVIDER env var (default: deepseek).
    Returns (None, None, None) if no API key is configured."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()

    try:
        if provider == "gemini":
            from google import genai
            api_key = get_gemini_api_key()
            if not api_key:
                return None, None, None
            return genai.Client(api_key=api_key), "gemini-2.0-flash", "gemini"
        else:
            from openai import OpenAI
            api_key = get_deepseek_api_key()
            if not api_key:
                return None, None, None
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            return client, "deepseek-v4-flash", "deepseek"
    except ImportError:
        return None, None, None


def generate_text(prompt: str, client, model: str, provider: str) -> str:
    """Unified text generation across Gemini and DeepSeek providers."""
    if not client:
        return ""
    if provider == "gemini":
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or ""
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""


def load_movies() -> list[dict]:
    with open(MOVIE_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]


def load_testcases() -> list[dict]:
    with open(TESTCASES_PATH, "r") as file:
        data = json.load(file)
    return data["test_cases"]


def read_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as file:
        data = file.read().splitlines()
    return data
