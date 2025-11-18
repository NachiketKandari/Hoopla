import streamlit as st
import os
import sys
import json
from pathlib import Path
from typing import Tuple

# Adding Parent directory to path to import cli modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.hybrid_search import HybridSearch, rrf_search_command, weighted_search_command
from cli.lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE
from cli.lib.augmented_generation import get_results
from app.model_handler import generate_response, generate_multidoc_summary, generate_citations, generate_answer
from cli.lib.multimodal_search import MultiModalSearch
from cli.lib.semantic_search import search_chunked_command
from cli.lib.keyword_search import InvertedIndex
from app.auth import (
    register_user, authenticate_user, check_rate_limit, consume_rate_limit,
    get_current_user_id, is_user_logged_in, login_user, logout_user
)
from app.database import add_chat_history, image_file_to_base64
import requests

st.set_page_config(page_title="Hoopla", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def get_readme_content() -> str:
    readme_path = Path(__file__).parent.parent / "README.md"
    try:
        return readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "README.md not found."


@st.cache_data(show_spinner=False)
def get_movies_dataset() -> list[dict]:
    return load_movies()

def render_alt_page(title: str, content_renderer, total_content_sections: int = 1):
    st.title(title)
    col = st.columns([1])[0]
    if col.button("⬅ Back to main app", type="primary"):
        st.session_state.show_readme_panel = False
        st.session_state.show_dataset_panel = False
        st.rerun()
    st.divider()
    content_renderer()
    st.divider()
    col_bottom = st.columns([1])[0]
    if col_bottom.button("⬅ Back", key=f"back_bottom_{title}", type="primary"):
        st.session_state.show_readme_panel = False
        st.session_state.show_dataset_panel = False
        st.rerun()
    st.stop()

# Initializing session state
if 'model_type' not in st.session_state:
    st.session_state.model_type = "API"
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'selected_ollama_model' not in st.session_state:
    st.session_state.selected_ollama_model = None
if 'show_readme_panel' not in st.session_state:
    st.session_state.show_readme_panel = False
if 'show_dataset_panel' not in st.session_state:
    st.session_state.show_dataset_panel = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return []

def load_ollama_models():
    if st.session_state.model_type == "local":
        models = get_ollama_models()
        st.session_state.ollama_models = models
        if models and not st.session_state.selected_ollama_model:
            st.session_state.selected_ollama_model = models[0] if models else None

def check_and_consume_rate_limit() -> Tuple[bool, str]:
    """
    Check and consume rate limit for current user.
    Returns (allowed, message)
    """
    user_id = get_current_user_id()
    if not user_id:
        return False, "User not logged in"
    
    is_system_api = st.session_state.model_type == "API"
    allowed, requests_left = check_rate_limit(user_id, is_system_api)
    
    if not allowed:
        return False, f"Daily limit reached. You have used all 50 requests for today. Please try again tomorrow or use local API (Ollama)."
    
    # Consume the rate limit
    if is_system_api:
        success = consume_rate_limit(user_id, is_system_api)
        if not success:
            return False, "Failed to consume rate limit. Please try again."
    
    return True, ""

def save_chat_history(query_type: str, query_text: str = None, query_image_base64: str = None, 
                      response_text: str = None, response_results: str = None):
    """Save chat history for current user."""
    user_id = get_current_user_id()
    if not user_id:
        return
    
    model_type = st.session_state.model_type
    if model_type == "local" and st.session_state.selected_ollama_model:
        model_type = f"local:{st.session_state.selected_ollama_model}"
    
    add_chat_history(
        user_id=user_id,
        query_type=query_type,
        query_text=query_text,
        query_image_base64=query_image_base64,
        response_text=response_text,
        response_results=response_results,
        model_type=model_type
    )

def check_rate_limit_for_api_call() -> bool:
    """Check rate limit before making a SYSTEM API call. Returns True if allowed."""
    user_id = get_current_user_id()
    if not user_id:
        return False
    
    is_system_api = st.session_state.model_type == "API"
    if not is_system_api:
        return True  # No limit for local API
    
    allowed, _ = check_rate_limit(user_id, is_system_api)
    if allowed:
        consume_rate_limit(user_id, is_system_api)
    return allowed

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    
    model_type = st.radio("Select Model Type", ["API", "local"], index=0 if st.session_state.model_type == "API" else 1)
    st.session_state.model_type = model_type
    
    if model_type == "local":
        load_ollama_models()
        if st.session_state.ollama_models:
            st.session_state.selected_ollama_model = st.selectbox(
                "Select Ollama Model",
                st.session_state.ollama_models,
                index=0 if st.session_state.selected_ollama_model in st.session_state.ollama_models else 0
            )
        else:
            st.warning("No Ollama models found. Make sure Ollama is running on localhost:11434")
            st.session_state.selected_ollama_model = None
    else:
        st.info("Using Gemini API from .env file")
    
    st.divider()

    st.header("Docs & Data")

    if st.button("Open README", use_container_width=True):
        st.session_state.show_readme_panel = True
        st.session_state.show_dataset_panel = False

    if st.button("Open Dataset Viewer", use_container_width=True):
        st.session_state.show_dataset_panel = True
        st.session_state.show_readme_panel = False
    
    st.divider()
    
    # Authentication section in sidebar
    st.header("Authentication")
    if is_user_logged_in():
        st.success(f"Logged in as: **{st.session_state.get('username', 'Unknown')}**")
        user_id = get_current_user_id()
        if user_id:
            is_system_api = st.session_state.model_type == "API"
            allowed, requests_left = check_rate_limit(user_id, is_system_api)
            if is_system_api:
                if allowed:
                    st.info(f"Requests left today: **{requests_left}**")
                else:
                    st.error("Daily limit reached (50 requests/day)")
            else:
                st.info("Using local API (unlimited)")
        
        if st.button("Logout", use_container_width=True, type="secondary"):
            logout_user()
            st.rerun()
    else:
        st.warning("Please login to use the app")
        if st.button("Login", use_container_width=True):
            st.session_state.show_login = True
            st.session_state.show_register = False
            st.rerun()
        if st.button("Register", use_container_width=True):
            st.session_state.show_register = True
            st.session_state.show_login = False
            st.rerun()

# Authentication UI (Login/Register)
if not is_user_logged_in():
    st.title("🎬 Hoopla - Login Required")
    
    if st.session_state.show_login:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user_id, message = authenticate_user(username, password)
                if user_id:
                    login_user(user_id)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button("Don't have an account? Register"):
            st.session_state.show_register = True
            st.session_state.show_login = False
            st.rerun()
    
    elif st.session_state.show_register:
        st.header("Register")
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                success, message = register_user(username, password)
                if success:
                    st.success(message)
                    st.session_state.show_login = True
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button("Already have an account? Login"):
            st.session_state.show_login = True
            st.session_state.show_register = False
            st.rerun()
    
    st.stop()

if st.session_state.get("show_readme_panel"):
    def _render_readme():
        st.markdown(get_readme_content())
    render_alt_page("Project README", _render_readme)

if st.session_state.get("show_dataset_panel"):
    def _render_dataset():
        st.caption("Full contents of data/movies.json")
        st.json({"movies": get_movies_dataset()})
    render_alt_page("Dataset Viewer", _render_dataset)

st.title("🎬 Hoopla")
st.caption("Unified UI for Hoopla’s hybrid search, RAG workflows, semantic retrieval, keyword tools, reranking helpers, and multimodal experiments powered by Gemini/Ollama.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["RAG", "Hybrid Search", "Semantic Search", "Keyword Search", "Multimodal Search"])

# RAG Tab
with tab1:
    st.header("Retrieval Augmented Generation")
    
    rag_type = st.selectbox("Select RAG Type", ["rag", "summarize", "citations", "question"])
    query = st.text_input("Enter your query", key="rag_query", placeholder="movies about action and dinosaurs")
    
    if st.button("Generate", key="rag_button"):
        if query:
            # Check rate limit
            allowed, message = check_and_consume_rate_limit()
            if not allowed:
                st.error(message)
            else:
                with st.spinner("Searching and generating response..."):
                    try:
                        results = get_results(query)
                        
                        st.subheader("Search Results")
                        for i, res in enumerate(results, 1):
                            st.write(f"{i}. {res['title']}")
                        
                        st.subheader("Generated Response")
                        
                        if rag_type == "rag":
                            response = generate_response(query, results, st.session_state.model_type, st.session_state.selected_ollama_model)
                        elif rag_type == "summarize":
                            response = generate_multidoc_summary(query, results, st.session_state.model_type, st.session_state.selected_ollama_model)
                        elif rag_type == "citations":
                            response = generate_citations(query, results, st.session_state.model_type, st.session_state.selected_ollama_model)
                        elif rag_type == "question":
                            response = generate_answer(query, results, st.session_state.model_type, st.session_state.selected_ollama_model)
                        
                        st.write(response)
                        
                        # Save chat history
                        save_chat_history(
                            query_type=f"rag_{rag_type}",
                            query_text=query,
                            response_text=response,
                            response_results=json.dumps([{"title": r.get("title", ""), "id": r.get("id", "")} for r in results])
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Hybrid Search Tab
with tab2:
    st.header("Hybrid Search")
    
    search_type = st.selectbox("Select Search Type", ["rrf-search", "weighted-search"])
    
    query = st.text_input("Enter your query", key="hybrid_query", placeholder="friendship transformation magic with bears")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if search_type == "rrf-search":
            k_value = st.number_input("K Value", min_value=1.0, max_value=100.0, value=DEFAULT_K_VALUE, step=1.0)
        else:
            alpha_value = st.number_input("Alpha Value", min_value=0.0, max_value=1.0, value=DEFAULT_ALPHA_VALUE, step=0.1)
    
    with col2:
        limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1)
    
    if search_type == "rrf-search":
        enhance = st.selectbox("Query Enhancement (optional)", [None, "spell", "rewrite", "expand"])
        rerank = st.selectbox("Re-ranking Method (optional)", [None, "individual", "batch", "cross_encoder"])
        evaluate = st.checkbox("Evaluate Results")
    
    if st.button("Search", key="hybrid_button"):
        if query:
            # Check rate limit for initial search (only if using SYSTEM API for reranking/evaluation)
            # The search itself doesn't use API, but reranking and evaluation do
            with st.spinner("Searching..."):
                try:
                    documents = get_movies_dataset()
                    hybrid_search = HybridSearch(documents)
                    
                    if search_type == "rrf-search":
                        if enhance:
                            from cli.lib.query_enhancement import enhance_query
                            enhanced_query = enhance_query(query, method=enhance)
                            query = enhanced_query
                            st.info(f"Enhanced query: {query}")
                        
                        if rerank:
                            results = hybrid_search.rrf_search(query, k=k_value, limit=limit*5)
                            from cli.lib.reranking import rerank_individual, rerank_batch, rerank_cross_encoder
                            
                            if rerank == "individual":
                                # Check rate limit for each API call
                                for doc in results:
                                    if st.session_state.model_type == "API":
                                        if not check_rate_limit_for_api_call():
                                            st.error("Rate limit reached. Cannot complete reranking.")
                                            break
                                    
                                    from app.model_handler import generate_with_gemini, generate_with_ollama
                                    prompt = f"""Rate how well this movie matches the search query.

                                    Query: "{query}"
                                    Movie: {doc.get("title", "")} - {doc.get("description", "")}

                                    Consider:
                                    - Direct relevance to query
                                    - User intent (what they're looking for)
                                    - Content appropriateness

                                    Rate 0-10 (10 = perfect match).
                                    Give me ONLY the number in your response, no other text or explanation.

                                    Score:"""
                                    
                                    if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                                        response_text = generate_with_ollama(prompt, st.session_state.selected_ollama_model)
                                    else:
                                        response_text = generate_with_gemini(prompt)
                                    doc['score'] = int((response_text or "").strip().strip('"'))
                                results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
                            elif rerank == "batch":
                                # Check rate limit before API call
                                if st.session_state.model_type == "API":
                                    if not check_rate_limit_for_api_call():
                                        st.error("Rate limit reached. Cannot complete reranking.")
                                    else:
                                        from app.model_handler import generate_with_gemini, generate_with_ollama
                                        prompt = f"""Rank these movies by relevance to the search query.

                                        Query: "{query}"

                                        Movies:
                                        {results}

                                        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. Don't put it inside a json markdown. For example, the output should be like :

                                        [75, 12, 34, 2, 1]
                                        """
                                        
                                        if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                                            json_response_text = generate_with_ollama(prompt, st.session_state.selected_ollama_model)
                                        else:
                                            json_response_text = generate_with_gemini(prompt)
                                else:
                                    from app.model_handler import generate_with_gemini, generate_with_ollama
                                    prompt = f"""Rank these movies by relevance to the search query.

                                    Query: "{query}"

                                    Movies:
                                    {results}

                                    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. Don't put it inside a json markdown. For example, the output should be like :

                                    [75, 12, 34, 2, 1]
                                    """
                                    
                                    if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                                        json_response_text = generate_with_ollama(prompt, st.session_state.selected_ollama_model)
                                    else:
                                        json_response_text = generate_with_gemini(prompt)
                                
                                batch_results = json.loads(json_response_text)
                                docs = []
                                for result in batch_results[:limit]:
                                    for doc in results:
                                        if doc['id'] == result:
                                            docs.append(doc)
                                results = docs
                            elif rerank == "cross_encoder":
                                from cli.lib.reranking import cross_encoder
                                pairs = []
                                for doc in results:
                                    text = f"{doc.get('title', '')} - {doc.get('description', '')}"
                                    pairs.append([query, text])
                                
                                if pairs:
                                    scores = cross_encoder.predict(pairs)
                                    scored_results = list(zip(scores, results))
                                    sorted_scored_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
                                    docs = []
                                    for score, result in sorted_scored_results[:limit]:
                                        result['cross-encoder-score'] = score
                                        docs.append(result)
                                    results = docs
                        else:
                            results = hybrid_search.rrf_search(query, k=k_value, limit=limit)
                        
                        for i, res in enumerate(results, 1):
                            st.write(f"{i}. **{res['title']}**")
                            st.write(f"   RRF Score: {res['rrf_score']:.3f}")
                            if 'cross-encoder-score' in res:
                                st.write(f"   Cross Encoder Score: {res['cross-encoder-score']:.3f}")
                            st.write(f"   BM25 Rank: {res.get('bm25_rank', 'N/A')}, Semantic Rank: {res.get('semantic_rank', 'N/A')}")
                            st.write(f"   {res['description'][:200]}...")
                            st.divider()
                        
                        # Save chat history for rrf search (before evaluation if it happens)
                        save_chat_history(
                            query_type=f"hybrid_{search_type}",
                            query_text=query,
                            response_results=json.dumps([{"title": r.get("title", ""), "id": r.get("id", ""), "score": r.get("rrf_score", 0)} for r in results])
                        )
                        
                        if evaluate:
                            # Check rate limit before evaluation API call
                            if st.session_state.model_type == "API":
                                if not check_rate_limit_for_api_call():
                                    st.error("Rate limit reached. Cannot complete evaluation.")
                                else:
                                    from cli.lib.reranking import format_results
                                    from app.model_handler import generate_with_gemini, generate_with_ollama
                                    import json
                                    
                                    with st.expander("Evaluation Results"):
                                        formatted_results = format_results(results)
                                        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
                                        
                                        Query: "{query}"

                                        
                                        Results: 
                                        
                                        {chr(10).join(formatted_results.split(chr(10)))}

                                        
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
                                        
                                        if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                                            response_text = generate_with_ollama(prompt, st.session_state.selected_ollama_model)
                                        else:
                                            response_text = generate_with_gemini(prompt)
                            else:
                                from cli.lib.reranking import format_results
                                from app.model_handler import generate_with_gemini, generate_with_ollama
                                import json
                                
                                with st.expander("Evaluation Results"):
                                    formatted_results = format_results(results)
                                    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
                                    
                                    Query: "{query}"

                                    
                                    Results: 
                                    
                                    {chr(10).join(formatted_results.split(chr(10)))}

                                    
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
                                    
                                    if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                                        response_text = generate_with_ollama(prompt, st.session_state.selected_ollama_model)
                                    else:
                                        response_text = generate_with_gemini(prompt)
                                
                                if not response_text:
                                    st.error("Failed to get evaluation response from the model. Please try again.")
                                else:
                                    try:
                                        # Clean the response text - remove markdown code blocks if present
                                        cleaned_text = response_text.strip()
                                        if cleaned_text.startswith("```"):
                                            # Remove markdown code block markers
                                            lines = cleaned_text.split("\n")
                                            cleaned_text = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_text
                                            cleaned_text = cleaned_text.strip()
                                        
                                        res_list = json.loads(cleaned_text)
                                        if isinstance(res_list, list) and len(res_list) == len(results):
                                            for i in range(0, len(res_list)):
                                                st.write(f"{results[i]['title']} : {res_list[i]}/3")
                                        else:
                                            st.error(f"Invalid evaluation format. Expected a list of {len(results)} scores, but got: {response_text}")
                                    except json.JSONDecodeError as e:
                                        st.error(f"Failed to parse evaluation response as JSON: {response_text}")
                                        st.error(f"Error: {str(e)}")
                    else:
                        results = hybrid_search.weighted_search(query, alpha=alpha_value, limit=limit)
                        for i, res in enumerate(results, 1):
                            st.write(f"{i}. **{res['title']}**")
                            st.write(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                            st.write(f"   {res['description'][:200]}...")
                            st.divider()
                    
                    # Save chat history for hybrid search
                    save_chat_history(
                        query_type=f"hybrid_{search_type}",
                        query_text=query,
                        response_results=json.dumps([{"title": r.get("title", ""), "id": r.get("id", ""), "score": r.get("rrf_score", r.get("hybrid_score", 0))} for r in results])
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Semantic Search Tab
with tab3:
    st.header("Semantic Search")
    
    query = st.text_input("Enter your query", key="semantic_query", placeholder="funny bear movies")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="semantic_limit")
    
    if st.button("Search", key="semantic_button"):
        if query:
            # Semantic search doesn't use API, so no rate limiting needed
            with st.spinner("Searching..."):
                try:
                    results = search_chunked_command(query, limit)
                    for i, res in enumerate(results, 1):
                        st.write(f"{i}. **{res['title']}** (score: {res['score']:.4f})")
                        st.write(f"   {res['description'][:200]}...")
                        st.divider()
                    
                    # Save chat history
                    save_chat_history(
                        query_type="semantic_search",
                        query_text=query,
                        response_results=json.dumps([{"title": r.get("title", ""), "id": r.get("id", ""), "score": r.get("score", 0)} for r in results])
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Keyword Search Tab
with tab4:
    st.header("Keyword Search (BM25)")
    
    query = st.text_input("Enter your query", key="keyword_query", placeholder="animated family")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="keyword_limit")
    
    if st.button("Search", key="keyword_button"):
        if query:
            # Keyword search doesn't use API, so no rate limiting needed
            with st.spinner("Searching..."):
                try:
                    idx = InvertedIndex()
                    idx.load()
                    results = idx.bm25_search(query, limit)
                    documents = get_movies_dataset()
                    doc_map = {doc['id']: doc for doc in documents}
                    
                    for i, res in enumerate(results, 1):
                        doc = doc_map[res['id']]
                        st.write(f"{i}. **{doc['title']}** (score: {res['score']:.4f})")
                        st.write(f"   {doc['description'][:200]}...")
                        st.divider()
                    
                    # Save chat history
                    save_chat_history(
                        query_type="keyword_search",
                        query_text=query,
                        response_results=json.dumps([{"title": doc_map[r['id']].get("title", ""), "id": r.get("id", ""), "score": r.get("score", 0)} for r in results])
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Multimodal Search Tab
with tab5:
    st.header("Multimodal Image Search")
    
    uploaded_file = st.file_uploader("Upload an image to search by image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if st.button("Search", key="multimodal_button"):
            # Multimodal search doesn't use API for generation, so no rate limiting needed
            with st.spinner("Searching with image..."):
                try:
                    documents = get_movies_dataset()
                    multimodal_search = MultiModalSearch(documents)
                    multimodal_search.load_or_create_embeddings()
                    results = multimodal_search.search_with_image(tmp_path)
                    
                    st.subheader("Search Results")
                    for i, res in enumerate(results, 1):
                        st.write(f"{i}. **{res['title']}** (score: {res['score']:.4f})")
                        st.write(f"   {res['description'][:200]}...")
                        st.divider()
                    
                    # Convert image to base64 and save chat history
                    image_base64 = image_file_to_base64(uploaded_file)
                    save_chat_history(
                        query_type="multimodal_search",
                        query_image_base64=image_base64,
                        response_results=json.dumps([{"title": r.get("title", ""), "id": r.get("id", ""), "score": r.get("score", 0)} for r in results])
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

