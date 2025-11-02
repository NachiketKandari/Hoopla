import os
from collections import defaultdict
from regex import R
import heapq
import logging

from lib.search_utils import (
    DEFAULT_ALPHA_VALUE, DEFAULT_SEARCH_LIMIT,DEFAULT_K_VALUE,
    load_movies,
    )

from lib.query_enhancement import (
    enhance_query,
    )

from lib.reranking import (
    re_rank,evaluate_results,
)

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

logger = logging.getLogger(__name__)
class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.doc_map = {doc['id']: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
        keyword_res = self._bm25_search(query, limit=limit * 500)
        semantic_res = self.semantic_search.search_chunks(query, limit=limit * 500)

        bm25_scores_dict = {doc['id']: doc['score'] for doc in keyword_res}
        semantic_scores_dict = {doc['id']: doc['score'] for doc in semantic_res}

        norm_bm25_scores = normalize_command(list(bm25_scores_dict.values()))
        norm_semantic_scores = normalize_command(list(semantic_scores_dict.values()))
        
        norm_bm25_dict = {doc_id: score for doc_id, score in zip(bm25_scores_dict.keys(), norm_bm25_scores)}
        norm_semantic_dict = {doc_id: score for doc_id, score in zip(semantic_scores_dict.keys(), norm_semantic_scores)}

        all_ids = set(norm_bm25_dict.keys()).union(norm_semantic_dict.keys())
        hybrid_scores = {}
        for doc_id in all_ids:
            bm_score = norm_bm25_dict.get(doc_id, 0.0)
            sem_score = norm_semantic_dict.get(doc_id, 0.0)
            hybrid_scores[doc_id] = hybrid_score(bm_score, sem_score, alpha)

        top_doc_ids = heapq.nlargest(limit, hybrid_scores, key=hybrid_scores.get)

        results = []
        for doc_id in top_doc_ids:
            document = self.doc_map[doc_id].copy()
            document['hybrid_score'] = hybrid_scores[doc_id]
            results.append(document)

        return results
    
    def rrf_search(self, query, k: int = DEFAULT_K_VALUE, limit: int = 2 * DEFAULT_SEARCH_LIMIT):
        
        logging.info(f"Original Query: {query}")

        keyword_res = self._bm25_search(query, limit=limit * 500)
        semantic_res = self.semantic_search.search_chunks(query, limit=limit * 500)

        rrf_scores = defaultdict(float)
        
        doc_ranks_bm25 = {}
        doc_ranks_semantic = {}

        for i, doc in enumerate(keyword_res, 1):
            doc_id = doc['id']
            rank = i
            rrf_scores[doc_id] += rrf_score(rank, k)
            doc_ranks_bm25[doc_id] = rank

        for i, doc in enumerate(semantic_res, 1):
            doc_id = doc['id']
            rank = i
            rrf_scores[doc_id] += rrf_score(rank, k)
            doc_ranks_semantic[doc_id] = rank

        top_doc_ids = heapq.nlargest(limit, rrf_scores, key=rrf_scores.get)

        results = []
        for doc_id in top_doc_ids:
            document = self.doc_map[doc_id].copy()
            document["rrf_score"] = rrf_scores[doc_id]
            
            document["bm25_rank"] = doc_ranks_bm25.get(doc_id)
            document["semantic_rank"] = doc_ranks_semantic.get(doc_id)
            
            results.append(document)
        
        # logging.info(f"RRF_SEARCH Results: {results}")
        return results
    
def weighted_search_command(query:str, alpha:float = DEFAULT_ALPHA_VALUE, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tHybrid Score: {res['hybrid_score']:.3f} \n\t{res['description'][:100]}...\n")

def rrf_search_command(query: str, k: int, limit: int, enhance: str, rerank: str, evaluate: bool) :
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    if enhance:
        enhanced_query = enhance_query(query,method=enhance)
        query = enhanced_query
    
    if rerank:
        results = hybrid_search.rrf_search(query, k, limit*5)
        results = re_rank(query,results,limit,rerank)
        return
    else:
        results = hybrid_search.rrf_search(query, k, limit)

    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")

    if evaluate:
        evaluate_results(query, results)

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
            return [1.0] * len(scores)
        results.append((score-min_score)/(max_score-min_score))
    return results
