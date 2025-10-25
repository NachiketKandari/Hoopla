from path import (
    DEFAULT_SEARCH_LIMIT,CACHE_DIR,
    load_movies,read_stopwords,
)
import os
import pickle
import string
from collections import defaultdict, Counter
import math
from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    
    def __add_documents(self, doc_id: int, text: str) -> None:
        tokenized_text = pre_process(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokenized_text)
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_text = pre_process(term)
        if len(tokenized_text) != 1:
            raise Exception("Should be single token")
        return self.term_frequencies[doc_id][tokenized_text[0]]
    
    def build(self) -> None:
        movies = load_movies()
        results = []
        for movie in movies:
            doc_id = movie['id']
            self.docmap[doc_id]=movie
            self.__add_documents(doc_id,f"{movie['title']} {movie['description']}")
        
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
    
    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

def tf_command(doc_id: int, term: str) -> int:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.get_tf(doc_id,term)

def idf_command(term:str) -> float:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    tokenized_term = pre_process(term)
    doc_count = len(invertedIdx.docmap)
    term_doc_count = len(invertedIdx.get_documents(tokenized_term[0]))
    idf = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf

def build_command() -> int:
    invertedIdx = InvertedIndex()
    invertedIdx.build()
    invertedIdx.save()

def search_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    query_tokenized = pre_process(query)
    seen = set()
    results = []
    for query_token in query_tokenized: 
        doc_ids = invertedIdx.get_documents(query_token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = invertedIdx.docmap[doc_id]
            if not doc:
                continue
            if len(results)<limit:
                results.append(doc)
            
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
    new_list = []
    for word in input:
        if word not in stopwords:
            new_list.append(word)
    return new_list

def stem_words(input: list[str])->list[str]:
    stemmer = PorterStemmer()
    stemmed_list = []
    for token in input:
        stemmed_word = stemmer.stem(token)
        stemmed_list.append(stemmed_word)
    return stemmed_list
    