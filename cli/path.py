import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MOVIE_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PKL_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PKL_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


def load_movies() -> list[dict]:
    with open(MOVIE_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]

def read_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as file:
        data = file.read().splitlines()
    return data
