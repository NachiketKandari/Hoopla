from collections import defaultdict
import os
import json
from typing import Optional
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import logging

from .search_utils import get_llm_client, generate_text

logger = logging.getLogger(__name__)

load_dotenv()
_client, _model, _provider = get_llm_client()


def _generate(prompt: str) -> str:
    return generate_text(prompt, _client, _model, _provider)


cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def rerank_individual(query: str, results: list[dict], limit: int) -> None:
    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:"""

        response_text = _generate(prompt)
        doc['score'] = int(response_text.strip().strip('"'))

    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

    for i, res in enumerate(sorted_results, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")


def rerank_batch(query: str, results: list[dict], limit: int) -> None:

    prompt = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {results}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing     else. Don't put it inside a json markdown. For example, the output should be like :

    [75, 12, 34, 2, 1]
    """

    json_response = _generate(prompt)
    batch_results = json.loads(json_response)
    docs = []
    for result in batch_results[:limit]:
        for doc in results:
            if doc['id'] == result:
                docs.append(doc)

    for i, res in enumerate(docs, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")


def rerank_cross_encoder(query: str, results: list[dict], limit: int) -> None:

    pairs = []
    for doc in results:
        text = f"{doc.get('title', '')} - {doc.get('document', '')}"
        pairs.append([query, text])

    if not pairs:
        print("No results to rerank.")
        return

    scores = cross_encoder.predict(pairs)
    scored_results = list(zip(scores, results))

    sorted_scored_results = sorted(scored_results, key=lambda x: x[0], reverse=True)

    docs = []
    for score, result in sorted_scored_results[:limit]:
        result['cross-encoder-score'] = score
        docs.append(result)

    for i, res in enumerate(docs, 1):
        print(f"{i}.\t{res['title']} \n\tCross Encoder Score: {res['cross-encoder-score']}\n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")


def format_results(results: list[dict]) -> str:
    formatted: str = ''
    for res in results:
        formatted += f"Title: {res['title']} Description: {res['description']}\n"
    return formatted


def evaluate_results(query: str, results: list[dict]):

    formatted_results = format_results(results)

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"


    Results:

    {chr(10).join(formatted_results)}


    Scale:

    - 3: Highly relevant

    - 2: Relevant

    - 1: Marginally relevant

    - 0: Not relevant


    Do NOT give any numbers out of 0, 1, 2 or 3.


    Return ONLY the scores in teh same order you were given the documents. Return a valid JSON
    list, nothing else. Don't use markdown in your response. For example:

    [2, 0, 3, 2, 0, 1]

    """

    response_text = _generate(prompt)

    res_list = json.loads(response_text)
    for i in range(0, len(res_list)):
        print(f"{results[i]['title']} : {res_list[i]}/3\n")


def re_rank(query: str, results: list[dict], limit: int, method: Optional[str] = None) -> str:
    match method:
        case "individual":
            return rerank_individual(query, results, limit)
        case "batch":
            return rerank_batch(query, results, limit)
        case "cross_encoder":
            return rerank_cross_encoder(query, results, limit)
        case _:
            return
