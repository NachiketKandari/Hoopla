from collections import defaultdict
import os
import json
from typing import Optional
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

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

    response = client.models.generate_content(model=model, contents=prompt)
    rewritten = (response.text or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    response = client.models.generate_content(model=model, contents=prompt)
    expanded_terms = (response.text or "").strip().strip('"')

    return f"{query} {expanded_terms}"

def rerank_individual(query: str, results: list[dict],limit: int) -> None:
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

        response = client.models.generate_content(model=model, contents=prompt)
        doc['score'] = int((response.text or "").strip().strip('"'))
    
    sorted_results = sorted(results, key =lambda x:x['score'], reverse=True)[:limit] 

    for i, res in enumerate(sorted_results, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")

def rerank_batch(query: str, results: list[dict],limit: int) -> None:

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{results}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. Don't put it inside a json markdown. For example, the output should be like :

[75, 12, 34, 2, 1]
"""

    json_response = client.models.generate_content(model=model, contents=prompt)
    batch_results = json.loads(json_response.text)
    docs = []
    for result in batch_results[:limit]:
        for doc in results:
            if doc['id'] == result:
                docs.append(doc)

    for i, res in enumerate(docs, 1):
        print(f"{i}.\t{res['title']} \n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")

def rerank_cross_encoder(query: str, results: list[dict],limit: int) -> None:
    
    pairs = []
    for doc in results:
        text = f"{doc.get('title', '')} - {doc.get('document', '')}"
        pairs.append([query, text])
    
    if not pairs:
        print("No results to rerank.")
        return

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    scored_results = list(zip(scores, results))    
        
    sorted_scored_results = sorted(scored_results, key=lambda x: x[0], reverse=True)

    docs = []
    for score, result in sorted_scored_results[:limit]:
        result['cross-encoder-score'] = score
        docs.append(result)

    for i, res in enumerate(docs, 1):
        print(f"{i}.\t{res['title']} \n\tCross Encoder Score: {res['cross-encoder-score']}\n\tRRF Score: {res['rrf_score']:.3f} \n\tBM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}\n\t{res['description'][:100]}...\n")




def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query

def re_rank(query: str, results: list[dict], limit: int ,method: Optional[str] = None) -> str:
    match method:
        case "individual":
            return rerank_individual(query, results, limit)
        case "batch":
            return rerank_batch(query, results, limit)
        case "cross_encoder":
            return rerank_cross_encoder(query, results, limit)
        case _:
            return