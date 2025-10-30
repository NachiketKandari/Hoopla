import json
import os

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
