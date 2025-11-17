import streamlit as st
import os
import sys
from pathlib import Path

# Adding Parent directory to path to import cli modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.hybrid_search import HybridSearch, rrf_search_command, weighted_search_command
from cli.lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE
from cli.lib.augmented_generation import get_results
from app.model_handler import generate_response, generate_multidoc_summary, generate_citations, generate_answer
from cli.lib.multimodal_search import MultiModalSearch
from cli.lib.semantic_search import search_chunked_command
from cli.lib.keyword_search import InvertedIndex
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
    query = st.text_input("Enter your query", key="rag_query")
    
    if st.button("Generate", key="rag_button"):
        if query:
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
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Hybrid Search Tab
with tab2:
    st.header("Hybrid Search")
    
    search_type = st.selectbox("Select Search Type", ["rrf-search", "weighted-search"])
    
    query = st.text_input("Enter your query", key="hybrid_query")
    
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
                                for doc in results:
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
                                
                                import json
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
                        
                        if evaluate:
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
                                
                                res_list = json.loads(response_text)
                                for i in range(0, len(res_list)):
                                    st.write(f"{results[i]['title']} : {res_list[i]}/3")
                    else:
                        results = hybrid_search.weighted_search(query, alpha=alpha_value, limit=limit)
                        for i, res in enumerate(results, 1):
                            st.write(f"{i}. **{res['title']}**")
                            st.write(f"   Hybrid Score: {res['hybrid_score']:.3f}")
                            st.write(f"   {res['description'][:200]}...")
                            st.divider()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Semantic Search Tab
with tab3:
    st.header("Semantic Search")
    
    query = st.text_input("Enter your query", key="semantic_query")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="semantic_limit")
    
    if st.button("Search", key="semantic_button"):
        if query:
            with st.spinner("Searching..."):
                try:
                    results = search_chunked_command(query, limit)
                    for i, res in enumerate(results, 1):
                        st.write(f"{i}. **{res['title']}** (score: {res['score']:.4f})")
                        st.write(f"   {res['description'][:200]}...")
                        st.divider()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Keyword Search Tab
with tab4:
    st.header("Keyword Search (BM25)")
    
    query = st.text_input("Enter your query", key="keyword_query")
    limit = st.number_input("Limit", min_value=1, max_value=50, value=DEFAULT_SEARCH_LIMIT, step=1, key="keyword_limit")
    
    if st.button("Search", key="keyword_button"):
        if query:
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
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")

# Multimodal Search Tab
with tab5:
    st.header("Multimodal Image Search")
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if st.button("Search", key="multimodal_button"):
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
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

