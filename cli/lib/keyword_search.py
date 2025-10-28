from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,CACHE_DIR, BM25_K1,BM25_B,
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
        self.doc_lengths: dict = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.avg_doc_length: float = 0.0
        self.avg_doc_length_path = os.path.join(CACHE_DIR, "avg_doc_length.pkl")
        self.avg_doc_length_path = os.path.join(CACHE_DIR, "avg_doc_length.pkl")
    
    def __add_documents(self, doc_id: int, text: str) -> None:
        tokenized_text = pre_process(text)
        self.doc_lengths[doc_id] = len(tokenized_text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokenized_text)

    def __get_avg_doc_length(self) -> float:
        total = 0
        for _, value in self.doc_lengths.items():
            total+=value
        return total/len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))

########### TF - IDF ###########
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_term = pre_process(term)
        if len(tokenized_term) != 1:
            raise Exception("Should be single token")
        tf = self.term_frequencies[doc_id][tokenized_term[0]]
        return tf
    
    def get_idf(self, term: str) -> float:
        tokenized_term = pre_process(term)
        if len(tokenized_term) != 1:
            raise Exception("Should be single token")
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(tokenized_term[0]))
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf
    
########### BM25 TF - IDF ###########

    # helps in more stable idf scoring viz a viz tf-idf
    # term-freq saturation : prevents terms from dominating by appearing too often
    # document length normalization : accounts for longer vs shorter docs
    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = pre_process(term)
        if len(tokenized_term) != 1:
            raise Exception("Should be single token")
        N = len(self.docmap)
        df = len(self.get_documents(tokenized_term[0]))
        bm_25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return max(bm_25, 0.0)
        return max(bm_25, 0.0)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float: 
        tf = self.get_tf(doc_id, term)
        movie = self.docmap[doc_id]
        doc_length = self.doc_lengths[doc_id]
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.avg_doc_length
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf*bm25_idf

########### Search ###########

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        # self.avg_doc_length = self.__get_avg_doc_length() 
        # self.avg_doc_length = self.__get_avg_doc_length() 
        
        tokens = pre_process(query)
        scores = defaultdict(float)

        # using eligible movies to speed up the search
        eligible_movies = retrieve_documents(query, self, len(self.docmap)).copy()
        for movie in eligible_movies:
            doc_id = movie['id']
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id,token)
            scores[doc_id] = score

        # assigning 0 to the rest.
        for doc_id, _ in self.docmap.items():
            if doc_id not in scores:
                scores[doc_id] = 0.0

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            document = self.docmap[doc_id].copy()
            document["score"] = score
            results.append(document)

        return results
    
    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            self.docmap[doc_id]=movie
            self.__add_documents(doc_id,f"{movie['title']} {movie['description']}")
        self.avg_doc_length = self.__get_avg_doc_length()
        self.avg_doc_length = self.__get_avg_doc_length()
        
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)
        with open(self.avg_doc_length_path, "wb") as file:
            pickle.dump(self.avg_doc_length, file)
        
        with open(self.avg_doc_length_path, "wb") as file:
            pickle.dump(self.avg_doc_length, file)
        
    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
        with open(self.avg_doc_length_path, "rb") as file:
            self.avg_doc_length = pickle.load(file)
        with open(self.avg_doc_length_path, "rb") as file:
            self.avg_doc_length = pickle.load(file)

def tf_command(doc_id: int, term: str) -> int:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.get_tf(doc_id,term)

def idf_command(term: str) -> float:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.get_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.get_bm25_tf(doc_id,term)

def bm25_idf_command(term: str) -> float:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.get_bm25_idf(term)

def bm25_search_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    return invertedIdx.bm25_search(query, limit)

def search_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    invertedIdx = InvertedIndex()
    invertedIdx.load()
    results = retrieve_documents(query, invertedIdx, limit)
    return results
    
def retrieve_documents(query: str, invertedIdx: InvertedIndex, limit: int) -> list[dict]:
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

def build_command() -> int:
    invertedIdx = InvertedIndex()
    invertedIdx.build()
    invertedIdx.save()

########### Pre-Processing ###########

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
    