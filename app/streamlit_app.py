import streamlit as st
import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple

# Adding Parent directory to path to import cli modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.hybrid_search import HybridSearch, rrf_search_command, weighted_search_command
from cli.lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE
from cli.lib.augmented_generation import get_results
from app.model_handler import generate_response, generate_multidoc_summary, generate_citations, generate_answer, InvalidAPIKeyError
from cli.lib.multimodal_search import MultiModalSearch
from cli.lib.semantic_search import search_chunked_command
from cli.lib.keyword_search import InvertedIndex
from cli.lib.codebase_rag import CodebaseRAG, rewrite_query
from app.auth import (
    register_user, authenticate_user, check_rate_limit, consume_rate_limit,
    get_current_user_id, is_user_logged_in, login_user, logout_user, is_admin
)
from app.database import (
    add_chat_history, image_file_to_base64, add_conversation, get_recent_chat_messages, 
    mark_conversations_as_deleted, get_all_users, get_user_conversations, get_db_stats
)
import requests

st.set_page_config(page_title="Hoopla", page_icon="🎬", layout="wide")

logger = logging.getLogger("hoopla_streamlit_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event: str, level: str = "info", **details) -> None:
    """Log structured events for server-side debugging."""
    user = st.session_state.get('username', 'anonymous')
    payload = {"event": event, "user": user}
    payload.update({k: v for k, v in details.items() if v is not None})
    message = " | ".join(f"{key}={value}" for key, value in payload.items())
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)


@st.cache_data(show_spinner=False)
def get_readme_content() -> str:
    readme_path = Path(__file__).parent.parent / "README.md"
    try:
        return readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "README.md not found."


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour to reduce file I/O
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
if 'custom_api_key' not in st.session_state:
    st.session_state.custom_api_key = ""
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
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'thinking_mode' not in st.session_state:
    st.session_state.thinking_mode = False

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
    
    # No rate limit for custom Gemini API or local models
    if st.session_state.model_type in ["custom_gemini", "local"]:
        return True, ""
    
    is_system_api = st.session_state.model_type == "API"
    allowed, requests_left = check_rate_limit(user_id, is_system_api)
    
    if not allowed:
        return False, f"Daily limit reached. You have used all 50 requests for today. Please try again tomorrow or use local API (Ollama) or your own Gemini API key."
    
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
    elif model_type == "custom_gemini":
        model_type = "gemini:custom"
    
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
    
    # No rate limit for custom Gemini API or local models
    if st.session_state.model_type in ["custom_gemini", "local"]:
        return True

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
    
    model_options = {
        "System (Limited)": "API",
        "Gemini API (Custom Key)": "custom_gemini",
        "Ollama (Local)": "local"
    }
    
    # Reverse mapping to find index
    current_label = "System (Limited)"
    for label, value in model_options.items():
        if value == st.session_state.model_type:
            current_label = label
            break
            
    selected_label = st.selectbox("Select Model Source", list(model_options.keys()), index=list(model_options.keys()).index(current_label))
    # Model Selection
    st.sidebar.subheader("🤖 Model Selection")
    model_type_label = st.sidebar.selectbox(
        "Choose Model",
        ["API (Gemini 2.0 Flash)", "Custom Gemini API", "Local (Ollama)"],
        index=0 if st.session_state.model_type == "API" else (1 if st.session_state.model_type == "custom_gemini" else 2)
    )
    
    # Update session state based on selection
    if "Custom" in model_type_label:
        st.session_state.model_type = "custom_gemini"
    elif "Local" in model_type_label:
        st.session_state.model_type = "local"
    else:
        st.session_state.model_type = "API"

    if st.session_state.model_type == "local":
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
    elif st.session_state.model_type == "custom_gemini":
        st.session_state.custom_api_key = st.text_input("Enter Gemini API Key", type="password", value=st.session_state.custom_api_key)
        if not st.session_state.custom_api_key:
            st.warning("Please enter a valid API key")
    else:
        st.info("Using System API (50 requests/day)")
    
    st.divider()

    st.divider()

    st.header("Docs & Data")

    if st.button("Open README", width="stretch"):
        log_event("open_readme_clicked")
        st.session_state.show_readme_panel = True
        st.session_state.show_dataset_panel = False

    if st.button("Open Dataset Viewer", width="stretch"):
        log_event("open_dataset_viewer_clicked")
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
            elif st.session_state.model_type == "custom_gemini":
                st.info("Using custom Gemini API Key (unlimited)")
            else:
                st.info("Using local API (unlimited)")
        
        if st.button("Logout", width="stretch", type="secondary"):
            log_event("user_logout")
            logout_user()
            st.rerun()
    else:
        st.warning("Please login to use the app")
        if st.button("Login", width="stretch"):
            st.session_state.show_login = True
            st.session_state.show_register = False
            st.rerun()
        if st.button("Register", width="stretch"):
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
                log_event("login_attempt", attempt_username=username)
                user_id, message = authenticate_user(username, password)
                if user_id:
                    log_event("login_success", username=username)
                    login_user(user_id)
                    st.success(message)
                    st.rerun()
                else:
                    log_event("login_failed", username=username)
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
                log_event("register_attempt", attempt_username=username)
                success, message = register_user(username, password)
                if success:
                    log_event("register_success", username=username)
                    st.success(message)
                    st.session_state.show_login = True
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    log_event("register_failed", username=username)
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

# Load chat history from database ONCE when user logs in (before tabs to avoid repeated loads)
user_id = get_current_user_id()
if user_id and 'chat_loaded' not in st.session_state:
    try:
        db_messages = get_recent_chat_messages(user_id, limit=20)
        for msg in db_messages:
            st.session_state.chat_messages.append({"role": "user", "content": msg['query']})
            st.session_state.chat_messages.append({"role": "assistant", "content": msg['response'], "results": []})
        st.session_state.chat_loaded = True
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        st.session_state.chat_loaded = True  # Set anyway to prevent repeated attempts

# Create tabs - include admin panel only for admin users
if is_admin():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Chat", "RAG", "Hybrid Search", "Semantic Search", "Keyword Search", "Multimodal Search", "🔐 Admin Panel"])
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chat", "RAG", "Hybrid Search", "Semantic Search", "Keyword Search", "Multimodal Search"])
    tab7 = None  # Placeholder

# Chat Tab
with tab1:
    st.header("💬 Chat with Hoopla")
    st.caption("Ask questions about the Hoopla codebase. Toggle Thinking Mode for more accurate results using AI re-ranking.")
    
    # Thinking Mode toggle below caption
    st.session_state.thinking_mode = st.checkbox(
        "🧠 Thinking Mode",
        value=st.session_state.thinking_mode,
        help="Enable re-ranking for more accurate results (slower)"
    )
    
    # Clear chat button - right aligned
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("🗑️ Clear Chat", width="stretch"):
            # Soft delete conversations in database
            user_id = get_current_user_id()
            if user_id:
                mark_conversations_as_deleted(user_id, mode='chat')
            # Clear session state
            st.session_state.chat_messages = []
            st.session_state.chat_loaded = False
            st.success("Chat cleared!")
            st.rerun()
    
    # Search Mode Toggle
    st.caption("🔍 Search Mode")
    search_mode_display = st.radio(
        "Search Mode",
        ["HyDE (Concept Search)", "Code (Exact Search)"],
        horizontal=True,
        label_visibility="collapsed",
        help="HyDE matches concepts/descriptions. Code matches variable names and structure."
    )
    search_mode = "hyde" if "HyDE" in search_mode_display else "code"
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "results" in message and message["results"]:
                with st.expander("📄 Relevant Code Chunks"):
                    for i, res in enumerate(message["results"], 1):
                        st.markdown(f"**{i}. {res['filename']}:{res['name']}** (Score: {res['score']:.4f})")
                        st.code(res['content'], language='python')
    
    # Chat input
    if prompt := st.chat_input("Ask me about the codebase..."):
        log_event("chat_message_sent", has_query=bool(prompt))
        
        # Check rate limit
        allowed, rate_message = check_and_consume_rate_limit()
        if not allowed:
            st.error(rate_message)
        else:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching codebase..."):
                    try:
                        # Build context from chat history
                        chat_context = ""
                        if len(st.session_state.chat_messages) > 1:
                            recent_messages = st.session_state.chat_messages[-4:-1]  # Last 3 messages (excluding current)
                            chat_context = "Previous conversation:\n"
                            for msg in recent_messages:
                                role = "User" if msg["role"] == "user" else "Assistant"
                                chat_context += f"{role}: {msg['content']}\n"
                            chat_context += "\n"
                        
                        # Codebase RAG Logic
                        rag = CodebaseRAG()
                        
                        # Rewrite query for better search
                        api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
                        search_query = rewrite_query(prompt, api_key)
                        if search_query != prompt:
                            with st.expander("🔍 Search Details"):
                                st.caption(f"Rewritten Query: {search_query}")
                        
                        results = rag.search(
                            search_query,
                            limit=10,
                            score_threshold=0.01,  # RRF score threshold
                            use_reranking=st.session_state.thinking_mode,
                            mode=search_mode
                        )
                        
                        # Format results for prompt
                        context_str = ""
                        for res in results:
                            # Safely get score, defaulting to 0.0 if missing
                            score = res.get('score', 0.0)
                            context_str += f"File: {res['filename']}\nFunction: {res['name']}\nScore: {score:.4f}\nCode:\n{res['content']}\n\n"
                        
                        enhanced_prompt = f"""{chat_context}
Original Question: {prompt}
Rewritten Query (for technical retrieval): {search_query}

You are an expert coding assistant for the Hoopla codebase. 
I have retrieved relevant code chunks based on the rewritten query above, which expands the original question into technical terms.

Use the following code chunks to answer the user's ORIGINAL QUESTION.
- The "Score" indicates relevance (higher is better).
- Cite the file and function name when explaining code.
- If the code chunks don't contain the answer, say so, but try to be helpful based on the function names and descriptions.

Code Chunks:
{context_str}

Your response:"""
                        
                        # Generate response
                        api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
                        if st.session_state.model_type == "local" and st.session_state.selected_ollama_model:
                            from app.model_handler import generate_with_ollama
                            response = generate_with_ollama(enhanced_prompt, st.session_state.selected_ollama_model)
                        else:
                            from app.model_handler import generate_with_gemini
                            response = generate_with_gemini(enhanced_prompt, api_key=api_key)
                        
                        st.write(response)
                        
                        # Show code suggestions
                        if results:
                            with st.expander("📄 Relevant Code Chunks"):
                                for i, res in enumerate(results, 1):
                                    st.markdown(f"**{i}. {res['filename']}:{res['name']}** (Score: {res['score']:.4f})")
                                    st.code(res['content'], language='python')

                        # Add assistant message to chat
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response,
                            "results": results # Store chunks as results
                        })
                        
                        # Save to database
                        user_id = get_current_user_id()
                        if user_id:
                            model_type_str = st.session_state.model_type
                            if model_type_str == "local" and st.session_state.selected_ollama_model:
                                model_type_str = f"local:{st.session_state.selected_ollama_model}"
                            elif model_type_str == "custom_gemini":
                                model_type_str = "gemini:custom"
                            
                            add_conversation(
                                user_id=user_id,
                                mode="chat",
                                query=prompt,
                                response=response,
                                model_type=model_type_str
                            )
                        
                        log_event("chat_message_completed", result_count=len(results))
                            
                    except InvalidAPIKeyError as e:
                        logger.error(f"Invalid API Key in chat: {str(e)}")
                        st.error("⚠️ **Invalid API Key**: Please check your Gemini API key in the sidebar configuration.")
                    except Exception as e:
                        logger.exception("Chat generation failed")
                        st.error(f"Error generating response: {str(e)}")

# RAG Tab
with tab2:
    st.header("Retrieval Augmented Generation")
    
    rag_type = st.selectbox("Select RAG Type", ["rag", "summarize", "citations", "question"])
    query = st.text_input("Enter your query", key="rag_query", placeholder="movies about action and dinosaurs")
    
    if st.button("Generate", key="rag_button"):
        log_event("rag_generate_requested", rag_type=rag_type, has_query=bool(query))
        if query:
            # Check rate limit
            allowed, message = check_and_consume_rate_limit()
            if not allowed:
                st.error(message)
            else:
                with st.spinner("Searching and generating response..."):
                    try:
                        log_event("rag_generate_started", rag_type=rag_type)
                        results = get_results(query)
                        
                        st.subheader("Search Results")
                        for i, res in enumerate(results, 1):
                            st.write(f"{i}. {res['title']}")
                        
                        st.subheader("Generated Response")
                        
                        api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
                        
                        if rag_type == "rag":
                            response = generate_response(query, results, st.session_state.model_type, st.session_state.selected_ollama_model, api_key=api_key)
                        elif rag_type == "summarize":
                            response = generate_multidoc_summary(query, results, st.session_state.model_type, st.session_state.selected_ollama_model, api_key=api_key)
                        elif rag_type == "citations":
                            response = generate_citations(query, results, st.session_state.model_type, st.session_state.selected_ollama_model, api_key=api_key)
                        elif rag_type == "question":
                            response = generate_answer(query, results, st.session_state.model_type, st.session_state.selected_ollama_model, api_key=api_key)
                        
                        st.write(response)
                        
                        # Save to database
                        user_id = get_current_user_id()
                        if user_id:
                            model_type_str = st.session_state.model_type
                            if model_type_str == "local" and st.session_state.selected_ollama_model:
                                model_type_str = f"local:{st.session_state.selected_ollama_model}"
                            elif model_type_str == "custom_gemini":
                                model_type_str = "gemini:custom"
                            
                            # Determine mode based on RAG type
                            mode_map = {
                                "rag": "rag",
                                "summarize": "rag_summarize",
                                "citations": "rag_citations",
                                "question": "rag_question"
                            }
                            
                            add_conversation(
                                user_id=user_id,
                                mode=mode_map.get(rag_type, "rag"),
                                query=query,
                                response=response,
                                model_type=model_type_str
                            )
                        
                        log_event("rag_generate_completed", rag_type=rag_type, result_count=len(results))
                        log_event("rag_generate_completed", rag_type=rag_type, result_count=len(results))
                    except InvalidAPIKeyError as e:
                        logger.error(f"Invalid API Key: {str(e)}")
                        st.error("⚠️ **Invalid API Key**: The provided Gemini API key is invalid or has expired. Please check your key in the sidebar configuration.")
                    except Exception as e:
                        logger.exception("RAG generation failed")
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")
            log_event("rag_generate_missing_query", level="warning", rag_type=rag_type)

# Hybrid Search Tab
with tab3:
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
    
    enhance = None
    rerank = None
    evaluate = False
    if search_type == "rrf-search":
        enhance = st.selectbox("Query Enhancement (optional)", [None, "spell", "rewrite", "expand"])
        rerank = st.selectbox("Re-ranking Method (optional)", [None, "individual", "batch", "cross_encoder"])
        evaluate = st.checkbox("Evaluate Results")
    
    if st.button("Search", key="hybrid_button"):
        log_event("hybrid_search_requested", search_type=search_type, rerank=rerank, enhancement=enhance, limit=limit)
        if query:
            # Check rate limit for initial search (only if using SYSTEM API for reranking/evaluation)
            # The search itself doesn't use API, but reranking and evaluation do
            with st.spinner("Searching..."):
                try:
                    log_event("hybrid_search_started", search_type=search_type)
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
                                    
                                    from app.model_handler import generate_with_gemini, generate_with_ollama, InvalidAPIKeyError
                                    api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
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
                                        try:
                                            response_text = generate_with_gemini(prompt, api_key=api_key)
                                        except InvalidAPIKeyError as e:
                                            logger.error(f"Invalid API Key during reranking: {str(e)}")
                                            st.error("⚠️ **Invalid API Key**: Cannot rerank results. Please check your Gemini API key.")
                                            response_text = "0"
                                    doc['score'] = int((response_text or "").strip().strip('"'))
                                results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
                            elif rerank == "batch":
                                # Check rate limit before API call
                                if st.session_state.model_type == "API":
                                    if not check_rate_limit_for_api_call():
                                        st.error("Rate limit reached. Cannot complete reranking.")
                                    else:
                                        from app.model_handler import generate_with_gemini, generate_with_ollama, InvalidAPIKeyError
                                        api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
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
                                            try:
                                                json_response_text = generate_with_gemini(prompt, api_key=api_key)
                                            except InvalidAPIKeyError as e:
                                                logger.error(f"Invalid API Key during batch reranking: {str(e)}")
                                                st.error("⚠️ **Invalid API Key**: Cannot rerank results. Please check your Gemini API key.")
                                                json_response_text = "[]"
                                else:
                                    from app.model_handler import generate_with_gemini, generate_with_ollama, InvalidAPIKeyError
                                    api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
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
                                        try:
                                            json_response_text = generate_with_gemini(prompt, api_key=api_key)
                                        except InvalidAPIKeyError as e:
                                            logger.error(f"Invalid API Key during batch reranking: {str(e)}")
                                            st.error("⚠️ **Invalid API Key**: Cannot rerank results. Please check your Gemini API key.")
                                            json_response_text = "[]"
                                
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
                                    from app.model_handler import generate_with_gemini, generate_with_ollama, InvalidAPIKeyError
                                    import json
                                    api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
                                    
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
                                            try:
                                                response_text = generate_with_gemini(prompt, api_key=api_key)
                                            except InvalidAPIKeyError as e:
                                                logger.error(f"Invalid API Key during evaluation: {str(e)}")
                                                st.error("⚠️ **Invalid API Key**: Cannot evaluate results. Please check your Gemini API key.")
                                                response_text = None
                            else:
                                from cli.lib.reranking import format_results
                                from app.model_handler import generate_with_gemini, generate_with_ollama, InvalidAPIKeyError
                                import json
                                api_key = st.session_state.custom_api_key if st.session_state.model_type == "custom_gemini" else None
                                
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
                                        try:
                                            response_text = generate_with_gemini(prompt, api_key=api_key)
                                        except InvalidAPIKeyError as e:
                                            logger.error(f"Invalid API Key during evaluation: {str(e)}")
                                            st.error("⚠️ **Invalid API Key**: Cannot evaluate results. Please check your Gemini API key.")
                                            response_text = None
                                
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
                    log_event("hybrid_search_completed", search_type=search_type, result_count=len(results))
                except InvalidAPIKeyError as e:
                    logger.error(f"Invalid API Key: {str(e)}")
                    st.error("⚠️ **Invalid API Key**: The provided Gemini API key is invalid or has expired. Please check your key in the sidebar configuration.")
                except Exception as e:
                    logger.exception("Hybrid search failed")
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")
            log_event("hybrid_search_missing_query", level="warning", search_type=search_type)

# Semantic Search Tab
with tab4:
    st.header("Semantic Search")
    
    query = st.text_input("Enter your query", key="semantic_query", placeholder="funny bear movies")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="semantic_limit")
    
    if st.button("Search", key="semantic_button"):
        log_event("semantic_search_requested", limit=limit)
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
                        response_results=json.dumps([
                            {
                                "title": r.get("title", ""),
                                "id": r.get("id", ""),
                                "score": float(r.get("score", 0) or 0)
                            } for r in results
                        ])
                    )
                    log_event("semantic_search_completed", result_count=len(results))
                except Exception as e:
                    logger.exception("Semantic search failed")
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")
            log_event("semantic_search_missing_query", level="warning")

# Keyword Search Tab
with tab5:
    st.header("Keyword Search (BM25)")
    
    query = st.text_input("Enter your query", key="keyword_query", placeholder="animated family")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="keyword_limit")
    
    if st.button("Search", key="keyword_button"):
        log_event("keyword_search_requested", limit=limit)
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
                    log_event("keyword_search_completed", result_count=len(results))
                except Exception as e:
                    logger.exception("Keyword search failed")
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")
            log_event("keyword_search_missing_query", level="warning")

# Multimodal Search Tab
with tab6:
    st.header("Multimodal Image Search")
    
    uploaded_file = st.file_uploader("Upload an image to search by image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        log_event("multimodal_image_uploaded", file_name=uploaded_file.name)
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if st.button("Search", key="multimodal_button"):
            log_event("multimodal_search_requested", file_name=uploaded_file.name)
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
                    log_event("multimodal_search_completed", result_count=len(results))
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.exception("Multimodal search failed")
                    st.error(f"Error: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

# Admin Panel Tab
if tab7 is not None and is_admin():
    with tab7:
        st.header("🔐 Admin Panel")
        st.caption("Administrator dashboard for viewing users, conversations, and database statistics")
        
        # Database Statistics
        st.subheader("📊 Database Statistics")
        stats = get_db_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", stats.get('total_users', 0))
        with col2:
            st.metric("Total Conversations", stats.get('total_conversations', 0))
        with col3:
            st.metric("Deleted Conversations", stats.get('deleted_conversations', 0))
        
        # Conversations by mode
        if stats.get('conversations_by_mode'):
            st.write("**Conversations by Mode:**")
            mode_data = stats['conversations_by_mode']
            mode_cols = st.columns(len(mode_data))
            for idx, (mode, count) in enumerate(mode_data.items()):
                with mode_cols[idx]:
                    st.metric(mode.capitalize(), count)
        
        st.divider()
        
        # View All Users
        st.subheader("👥 All Users")
        users = get_all_users()
        
        if users:
            # Create DataFrame for better display
            import pandas as pd
            users_df = pd.DataFrame(users)
            users_df = users_df[['id', 'username', 'requests_left', 'is_admin', 'created_at']]
            st.dataframe(users_df, width="stretch", hide_index=True)
        else:
            st.info("No users found")
        
        st.divider()
        
        # View User Conversations
        st.subheader("💬 View User Conversations")
        
        if users:
            # User selector
            user_options = {f"{u['username']} (ID: {u['id']})": u['id'] for u in users}
            selected_user = st.selectbox("Select a user to view their conversations", options=list(user_options.keys()))
            
            include_deleted = st.checkbox("Include deleted conversations", value=False)
            
            if st.button("Load Conversations"):
                user_id = user_options[selected_user]
                conversations = get_user_conversations(user_id, include_deleted=include_deleted)
                
                if conversations:
                    st.write(f"**Found {len(conversations)} conversation(s)**")
                    
                    for conv in conversations:
                        status = "🗑️ DELETED" if conv.get('deleted', 0) == 1 else "✅ Active"
                        with st.expander(f"{status} | {conv['mode'].upper()} | {conv['timestamp']}"):
                            st.write(f"**Query:** {conv['query']}")
                            st.write(f"**Response:** {conv['response'][:500]}..." if len(conv['response']) > 500 else f"**Response:** {conv['response']}")
                            st.write(f"**Model:** {conv.get('model_type', 'N/A')}")
                            st.write(f"**ID:** {conv['id']}")
                else:
                    st.info("No conversations found for this user")
        else:
            st.warning("No users available")
