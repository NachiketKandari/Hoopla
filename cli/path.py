import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MOVIE_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(MOVIE_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]

def read_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as file:
        data = file.read().splitlines()
    return data
