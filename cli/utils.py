from path import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,read_stopwords,
)
import string
from nltk.stem import PorterStemmer

def search_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        for word in pre_process(query) :
            for title in pre_process(movie['title']) :
                if word in title and len(results)<DEFAULT_SEARCH_LIMIT and movie not in results:
                        results.append(movie)
    return results

def pre_process(input: str) -> list:
    return stem_words(remove_stopwords(tokenize(remove_punctuation(convert_to_lower(input)))))

def convert_to_lower(input: str) -> str:
    return input.lower()

def remove_punctuation(input: str) -> str:
    return input.translate(str.maketrans('','', string.punctuation))

def tokenize(input: str)-> list[str]:
    return input.strip().split()

def remove_stopwords(input: list[str]) -> list[str]:
    stopwords = read_stopwords()
    for stopword in stopwords:
        if stopword in input:
            input.remove(stopword)
    return input

def stem_words(input: list[str])->list[str]:
    stemmer = PorterStemmer()
    stemmed_list = []
    for token in input:
        stemmed_word = stemmer.stem(token)
        if stemmed_word not in stemmed_list:
            stemmed_list.append(stemmed_word)
    return stemmed_list
    