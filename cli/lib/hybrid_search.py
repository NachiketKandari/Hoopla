import os

from regex import R

from lib.search_utils import (
    DEFAULT_ALPHA_VALUE, DEFAULT_SEARCH_LIMIT,DEFAULT_K_VALUE,
    load_movies,
    )

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        keyword_res = self._bm25_search(query, limit= limit*500)
        semantic_res = self.semantic_search.search_chunks(query, limit= limit*500)

        keyword_res_sorted = sorted(keyword_res, key= lambda x: x['id'])
        semantic_res_sorted = sorted(semantic_res, key= lambda x: x['id'])

        bm25_scores = []
        for res in keyword_res_sorted:
            bm25_scores.append(res['score'])
        semantic_scores =[]
        for res in semantic_res_sorted:
            semantic_scores.append(res['score'])
        normalized_key_scores = normalize_command(bm25_scores)
        normalized_sem_scores = normalize_command(semantic_scores)

        documents = self.documents

        for i in range(0,len(normalized_key_scores)):
            documents[i]['hybrid_score'] = hybrid_score(normalized_key_scores[i], normalized_sem_scores[i], alpha)

        sorted_documents = sorted(documents, key= lambda x: x['hybrid_score'], reverse=True)
        return sorted_documents[:limit]

    def rrf_search(self, query, k: int = DEFAULT_K_VALUE, limit=10):
        keyword_res = self._bm25_search(query, limit= limit*500)
        semantic_res = self.semantic_search.search_chunks(query, limit= limit*500)

        for i, res in enumerate(keyword_res, 1):
            res['bm25_rank'] = i 

        for i, res in enumerate(semantic_res, 1):
            res['semantic_rank'] = i 
        
        documents = self.documents

        keyword_res_sorted = sorted(keyword_res, key= lambda x: x['id'])
        semantic_res_sorted = sorted(semantic_res, key= lambda x: x['id'])

        for i in range(0,len(keyword_res_sorted)):
            documents[i]['rrf_score'] = rrf_score(keyword_res_sorted[i]['bm25_rank'], k) + rrf_score(semantic_res_sorted[i]['semantic_rank'], k) 

            documents[i]['bm25_rank'] = keyword_res_sorted[i]['bm25_rank'] 
            documents[i]['semantic_rank'] = semantic_res_sorted[i]['semantic_rank'] 

        return sorted(documents, key=lambda x : x['rrf_score'], reverse=True)[:limit]
    
def weighted_search_command(query:str, alpha:float = DEFAULT_ALPHA_VALUE, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tHybrid Score: {res['hybrid_score']:.3f} \n{res['description'][:100]}...")

def rrf_search_command(query: str, k: int, limit: int) :
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n{res['description'][:100]}...\n")

def rrf_score(rank, k: int = DEFAULT_K_VALUE):
    return 1/(k + rank)
    
def hybrid_score(bm25_score, semantic_score, alpha=DEFAULT_ALPHA_VALUE):
    return alpha * bm25_score + (1 - alpha) * semantic_score
    
def normalize_command(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    sorted_scores = sorted(scores)
    max_score = sorted_scores[-1]
    min_score = sorted_scores[0]
    results =[]
    for score in scores:
        if min_score == max_score:
            return [1]
        results.append((score-min_score)/(max_score-min_score))
    return results
