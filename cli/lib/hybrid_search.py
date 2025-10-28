import os
from dotenv import load_dotenv
from google import genai

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

    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
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

    def rrf_search(self, query, k: int = DEFAULT_K_VALUE, limit: int = 2*DEFAULT_SEARCH_LIMIT):
        keyword_res = self._bm25_search(query, limit= limit*500)
        semantic_res = self.semantic_search.search_chunks(query, limit= limit*500)

        for i, res in enumerate(keyword_res, 1):
            res['bm25_rank'] = i 

        for i, res in enumerate(semantic_res, 1):
            res['semantic_rank'] = i 
        
        documents = self.documents

        for i in range(0, len(documents)):
            documents[i]['rrf_score'] = 0.0

        keyword_res_sorted = sorted(keyword_res, key= lambda x: x['id'])
        semantic_res_sorted = sorted(semantic_res, key= lambda x: x['id'])

        for i in range(0,len(keyword_res_sorted)):
            id = keyword_res_sorted[i]['id']-1
            documents[id]['rrf_score'] = rrf_score(keyword_res_sorted[i]['bm25_rank'], k) + rrf_score(semantic_res_sorted[i]['semantic_rank'], k) 

            documents[id]['bm25_rank'] = keyword_res_sorted[i]['bm25_rank'] 
            documents[id]['semantic_rank'] = semantic_res_sorted[i]['semantic_rank'] 

        return sorted(documents, key=lambda x : x['rrf_score'], reverse=True)[:limit]
    
def weighted_search_command(query:str, alpha:float = DEFAULT_ALPHA_VALUE, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tHybrid Score: {res['hybrid_score']:.3f} \n{res['description'][:100]}...")

def rrf_search_command(query: str, k: int, limit: int, enhance: str) :
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    match enhance:
        case "spell":
            response = spell_check(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text.strip('"')
        case "rewrite":
            response = rewrite(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text.strip('"')
        case "expand":
            response = expand(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
            query = response.text.strip('"')
        case _:
            pass
        
    results = hybrid_search.rrf_search(query, k, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")

def spell_check(query: str) :
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=f"""Fix any spelling errors in this movie search query.

                    Only correct obvious typos. Don't change correctly spelled words.

                    Query: "{query}"

                    If no errors, return the original query.
                    Corrected:"""
    )
    return response

def rewrite(query: str):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=f"""Rewrite this movie search query to be more specific and searchable.

                    Original: "{query}"

                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep it concise (under 10 words)
                    - It should be a google style search query that's very specific
                    - Don't use boolean logic

                    Examples:

                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                    Rewritten query:"""
    )
    return response

def expand(query: str):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=f"""Expand this movie search query with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    This will be appended to the original query.

                    Examples:

                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"

                    Query: "{query}"
                    """
    )
    return response

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
