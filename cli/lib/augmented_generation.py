import os
from dotenv import load_dotenv
from google import genai
import logging

from lib.hybrid_search import HybridSearch
from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def generate_response(query: str, results: list[dict]) -> str:
    
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service. Respond without any bolding, italics, or other markdown. Just the text in points if neccessary.

    Query: {query}

    Documents:
    {results}

    Provide a comprehensive answer that addresses the query:""" 

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def generate_multidoc_summary(query: str, results: list[dict]) -> str:
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Respond without any bolding, italics, or other markdown. Just the text in points if neccessary.
    Query: {query}
    Search Results:
    {results}
    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def generate_citations(query: str, results: list[dict]) -> str:
    prompt = prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the  sources you have.

    Query: {query}

    Documents:
    {results}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information. The number inside the braces should be the id field for that result.
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative
    - Respond without any bolding, italics, or other markdown. Just the text in points if neccessary.


    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def generate_answer(query: str, results: list[dict]) -> str:
    prompt = f"""Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {results}

    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def get_results(query: str) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    results = hybrid_search.rrf_search(query, limit=DEFAULT_SEARCH_LIMIT)
    return results

def rag_command(query: str) -> None:
    results = get_results(query)
    response = generate_response(query, results)

    print("Search Results: ")
    for res in results:
        print(f"\n\t-{res['title']}")
    
    print(f"\n\nRAG Response:\n{response}")

def summarize_command(query: str) -> None:
    results = get_results(query)
    response = generate_multidoc_summary(query, results)

    print("Search Results: ")
    for res in results:
        print(f"\n\t-{res['title']}")
    
    print(f"\n\nLLM Summary:\n{response}")

def citations_command(query: str) -> None:
    results = get_results(query)
    response = generate_citations(query, results)

    print("Search Results: ")
    for res in results:
        print(f"\n\t-{res['title']}")
    
    print(f"\n\nLLM Answer:\n{response}")

def question_command(query: str) -> None:
    results = get_results(query)
    response = generate_answer(query, results)

    print("Search Results: ")
    for res in results:
        print(f"\n\t-{res['title']}")
    
    print(f"\n\nAnswer:\n{response}")