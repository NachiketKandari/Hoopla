from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json
from .search_utils import (
    DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE, EMBEDDING_PATH, DEFAULT_SEARCH_LIMIT,DEFAULT_MAX_CHUNK_SIZE,CHUNK_EMBEDDING_PATH, CHUNK_METADATA_PATH,
    load_movies,
)
import os

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embeddings(self, text: str):
        if text.isspace() or len(text) == 0:
            raise ValueError("Empty string")
        embedding = self.model.encode([text])
        return embedding[0]
    
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
        
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if not os.path.exists(EMBEDDING_PATH):
            return self.build_embeddings(documents)
        with open(EMBEDDING_PATH, "rb") as f:
            self.embeddings = np.load(f)
        return self.embeddings
    
    def search(self, query: str, limit:int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if len(self.embeddings) == 0 or self.embeddings is None:
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        doc_list = []

        chunk_list : list[str]= []
        chunk_metadata = []

        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
            
            if doc['description'].isspace() or len(doc['description']) == 0:
                continue
            chunks = semantic_chunk_command(doc['description'],4,1)
            chunk_len = len(chunks)

            for chunk in chunks:
                chunk_list.append(chunk)
                chunk_metadata.append({
                    'movie_idx' : doc['id'],
                    'chunk_idx' : chunks.index(chunk),
                    'total_chunks' : chunk_len
                })

        self.chunk_embeddings = self.model.encode(chunk_list,show_progress_bar=True)
        self.chunk_metadata = {
            "chunks": chunk_metadata,
            "total_chunks": len(chunk_list),
        }

        with open(CHUNK_EMBEDDING_PATH, 'wb') as f:
            np.save(f, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc['id']] = doc

        if not os.path.exists(CHUNK_EMBEDDING_PATH) or not os.path.exists(CHUNK_METADATA_PATH):
            return self.build_chunk_embeddings(documents)
        
        with open(CHUNK_EMBEDDING_PATH, "rb") as f:
            self.chunk_embeddings = np.load(f)
        with open(CHUNK_METADATA_PATH, "r") as f:
            self.chunk_metadata = json.load(f)
            
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if len(self.chunk_embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embeddings(query)
        chunk_score_list = []

        for i in range(len(self.chunk_embeddings)):
            embedding = self.chunk_embeddings[i]
            metadata = self.chunk_metadata['chunks'][i]
            score = cosine_similarity(embedding, query_embedding)
            chunk_score_list.append({
                'chunk_idx': metadata['chunk_idx'],
                'movie_idx': metadata['movie_idx'],
                'score': score,
            })
        
        movie_to_score_dict = {}
        for chunk_score in chunk_score_list:
            movie_index = chunk_score['movie_idx']
            if movie_index not in movie_to_score_dict or (chunk_score['score']> movie_to_score_dict[movie_index]):
                movie_to_score_dict[movie_index] = chunk_score['score']
        
        movie_scores_sorted = sorted(movie_to_score_dict.items(), key= lambda item : item[1], reverse=True)

        results = []
        for key, value in movie_scores_sorted:
            if len(results) == limit:
                break
            movie = self.document_map[key]
            results.append({
                'id'    : movie['id'],
                'title' : movie['title'],
                'description'  : movie['description'][:100],
                'score' : value
                })
        return results


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    documents = load_movies()
    chunked_ss = ChunkedSemanticSearch()
    chunk_embeddings = chunked_ss.load_or_create_chunk_embeddings(documents)
    results = chunked_ss.search_chunks(query, limit)
    return results

def embed_chunks_command():
    chunked_semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_command(query: str, limit:int = DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")

def chunk_command(query: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP_SIZE) -> list[str]:
    query_split = query.split(" ")
    results = []
    for i in range(0, len(query_split)-overlap, chunk_size):  
        if i == 0:
            results.append(" ".join(query_split[i:i+chunk_size]))  
            continue
        results.append(" ".join(query_split[i-overlap:i+chunk_size-overlap])) 
    return results

def semantic_chunk_command(input: str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP_SIZE) -> list[str]:
    input = input.strip()
    if len(input) == 0:
        return []
    input_split = re.split(r'(?<=[.!?])\s+', input)
    results = []
    for i in range(0, len(input_split) - overlap, max_chunk_size-overlap):  
        results.append(" ".join(input_split[i:i+max_chunk_size])) 
    return results

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