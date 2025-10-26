from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import (
    EMBEDDING_PATH, DEFAULT_SEARCH_LIMIT,
    load_movies,
)
import os

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if not os.path.exists(EMBEDDING_PATH):
            return self.build_embeddings(documents)
        with open(EMBEDDING_PATH, "rb") as f:
            self.embeddings = np.load(f)
        return self.embeddings

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list,show_progress_bar=True)
        with open(EMBEDDING_PATH, 'wb') as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def generate_embeddings(self, text: str):
        if text.isspace() or len(text) == 0:
            raise ValueError("Empty string")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def search(self, query: str, limit:int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if len(self.embeddings) == 0 :
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embeddings(query)
        score_list = []
        for i in range(len(self.documents)):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            document = self.documents[i].copy()
            del document['id']
            document['score'] = similarity_score
            score_list.append(document)
        sorted_list = sorted(score_list, key=lambda item: item['score'], reverse=True)
        return sorted_list[:limit]

def search_command(query: str, limit:int = DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model() -> None:
    semantic_search = SemanticSearch()
    model = semantic_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)