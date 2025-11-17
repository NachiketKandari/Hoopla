import os
import requests
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

def get_gemini_client():
    api_key = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found in environment variables")
    client = genai.Client(api_key=api_key)
    return client

def generate_with_gemini(prompt: str) -> str:
    client = get_gemini_client()
    model = "gemini-2.0-flash"
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def generate_with_ollama(prompt: str, model_name: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    result = response.json()
    return result.get("response", "")

def generate_response(query: str, results: list[dict], model_type: str = "API", ollama_model: str = None) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service. Respond without any bolding, italics, or other markdown. Just the text in points if neccessary.

    Query: {query}

    Documents:
    {results}

    Provide a comprehensive answer that addresses the query:"""
    
    if model_type == "local" and ollama_model:
        return generate_with_ollama(prompt, ollama_model)
    else:
        return generate_with_gemini(prompt)

def generate_multidoc_summary(query: str, results: list[dict], model_type: str = "API", ollama_model: str = None) -> str:
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
    
    if model_type == "local" and ollama_model:
        return generate_with_ollama(prompt, ollama_model)
    else:
        return generate_with_gemini(prompt)

def generate_citations(query: str, results: list[dict], model_type: str = "API", ollama_model: str = None) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents.

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
    
    if model_type == "local" and ollama_model:
        return generate_with_ollama(prompt, ollama_model)
    else:
        return generate_with_gemini(prompt)

def generate_answer(query: str, results: list[dict], model_type: str = "API", ollama_model: str = None) -> str:
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
    
    if model_type == "local" and ollama_model:
        return generate_with_ollama(prompt, ollama_model)
    else:
        return generate_with_gemini(prompt)

