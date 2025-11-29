import ast
import os
import pickle
import json
import fnmatch
import time
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import heapq
from .search_utils import PROJECT_ROOT, CACHE_DIR, DEFAULT_K_VALUE
from .keyword_search import InvertedIndex
from google import genai

# Constants
CODEBASE_INDEX_PATH = os.path.join(CACHE_DIR, "codebase_index.pkl")
CODEBASE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "codebase_embeddings.npy")
CODEBASE_EMBEDDINGS_CODE_PATH = os.path.join(CACHE_DIR, "codebase_embeddings_code.npy")
CODEBASE_KEYWORD_INDEX_PATH = os.path.join(CACHE_DIR, "codebase_keyword_index.pkl")

def rrf_score(rank, k: int = DEFAULT_K_VALUE):
    """Calculate RRF score for a given rank."""
    return 1 / (k + rank)

def rewrite_query(query: str, api_key: str = None) -> str:
    """
    Rewrites a user query to be more suitable for searching function descriptions.
    Focus on expanding the query with related concepts and technical terms.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return query # Fallback to original query if no key
        
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""You are a search query expert. Your task is to rewrite user questions into comprehensive search queries that will match against technical function descriptions.

The search system uses function descriptions (not code or function names), so focus on:
- Key concepts and technical terms related to the question
- Related operations, processes, or workflows
- Alternative phrasings and synonyms
- Domain-specific terminology

Examples:

User: "How do I log in?"
Rewritten: "user authentication login session management credential verification password check login process user authorization"

User: "Where is the database connection handled?"
Rewritten: "database connection management session creation database initialization connection pooling database setup establish connection"

User: "Show me the code for searching movies"
Rewritten: "movie search functionality search implementation query processing search results retrieval movie lookup find movies"

User: "How are images processed?"
Rewritten: "image processing image manipulation encoding decoding image transformation image handling base64 conversion"

User: "{query}"
Rewritten:"""
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        rewritten = response.text.strip().replace('"', '')
        return rewritten
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query

class CodebaseChunker:
    def __init__(self, root_dir: str, api_key: str = None):
        self.root_dir = root_dir
        self.ignore_patterns = self._load_gitignore()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._rate_limit_hit = False  # Circuit breaker flag
        self.cached_descriptions = self._load_cached_descriptions()
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: No Gemini API key found. AI descriptions will be disabled.")

    def _load_cached_descriptions(self) -> Dict[str, str]:
        """Load existing descriptions from cache/codebase_data.json to avoid re-generation."""
        cache_path = os.path.join(CACHE_DIR, "codebase_data.json")
        descriptions = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        # Key by filename:function_name
                        key = f"{item['filename']}:{item['name']}"
                        descriptions[key] = item['description']
                print(f"Loaded {len(descriptions)} cached descriptions.")
            except Exception as e:
                print(f"Error loading cached descriptions: {e}")
        return descriptions

    def _load_gitignore(self) -> List[str]:
        gitignore_path = os.path.join(self.root_dir, ".gitignore")
        patterns = []
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        # Add default ignores
        patterns.extend([".git", "__pycache__", "*.pyc", ".DS_Store", ".venv", ".env"])
        return patterns

    def _is_ignored(self, path: str) -> bool:
        rel_path = os.path.relpath(path, self.root_dir)
        filename = os.path.basename(path)
        
        # Explicitly ignore admin_panel_ui.py
        if filename == "admin_panel_ui.py":
            return True
            
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
            # Handle directory matching
            if os.path.isdir(path) and fnmatch.fnmatch(rel_path + "/", pattern):
                return True
        return False
    
    def generate_description(self, code: str, function_name: str) -> str:
        """Generate an AI description for a function using Gemini with retry logic."""
        # Check circuit breaker
        if self._rate_limit_hit:
            return f"Function {function_name}"
            
        if not self.client:
            return f"Function {function_name}"
        
        max_retries = 5
        base_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                prompt = f"""You are a technical documentation expert. Write a concise 50-100 word description of what this Python function does. Focus on:
- What the function accomplishes
- Key parameters and return values
- Important logic or algorithms used

Function name: {function_name}

Code:
```python
{code}
```

Description:"""
                
                response = self.client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
                description = response.text.strip()
                
                # Add a small delay between successful requests to avoid hitting rate limits
                time.sleep(0.5)
                
                return description
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2, 4, 8, 16, 32 seconds
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit for {function_name}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Max retries reached for {function_name}. Disabling AI descriptions for this session.")
                        self._rate_limit_hit = True  # Trip circuit breaker
                        return f"Function {function_name}"
                else:
                    # Non-rate-limit error, fail immediately
                    print(f"Error generating description for {function_name}: {e}")
                    return f"Function {function_name}"
        
        return f"Function {function_name}"

    def chunk_file(self, filepath: str) -> List[Dict[str, Any]]:
        chunks = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            rel_path = os.path.relpath(filepath, self.root_dir)
            filename = os.path.basename(filepath)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip functions with "admin" in the name
                    if "admin" in node.name.lower():
                        continue

                    # Extract function/method
                    start_line = node.lineno
                    end_line = node.end_lineno
                    code_segment = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    # Generate AI description
                    # Check cache first
                    cache_key = f"{filename}:{node.name}"
                    if cache_key in self.cached_descriptions and len(self.cached_descriptions[cache_key].split()) > 5:
                        ai_description = self.cached_descriptions[cache_key]
                        # print(f"Using cached description for {node.name}")
                    else:
                        ai_description = self.generate_description(code_segment, node.name)
                    
                    chunk = {
                        "type": "function",
                        "name": node.name,
                        "filepath": rel_path,
                        "filename": filename,
                        "content": code_segment,
                        "start_line": start_line,
                        "end_line": end_line,
                        "description": ai_description
                    }
                    chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                     # We might want to chunk the class definition itself (docstring + signature)
                     # but usually methods are more useful. 
                     # Let's add a chunk for the class docstring/signature if needed.
                     # For now, focusing on methods/functions as requested.
                     pass
                     
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            # Fallback or skip? Skip for now.
        
        return chunks

    def walk_and_chunk(self) -> List[Dict[str, Any]]:
        all_chunks = []
        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(os.path.join(root, d))]
            
            for file in files:
                filepath = os.path.join(root, file)
                if self._is_ignored(filepath):
                    continue
                if not file.endswith(".py"): # Only chunk python files for AST
                    continue
                
                chunks = self.chunk_file(filepath)
                all_chunks.extend(chunks)
        return all_chunks

class CodebaseRAG:
    def __init__(self, root_dir: str = PROJECT_ROOT, api_key: str = None):
        self.root_dir = root_dir
        self.api_key = api_key
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.chunks = []
        self.embeddings = None  # Description embeddings (HyDE)
        self.code_embeddings = None  # Code content embeddings
        self.keyword_index = None

    def build_index(self):
        chunker = CodebaseChunker(self.root_dir, api_key=self.api_key)
        self.chunks = chunker.walk_and_chunk()
        
        if not self.chunks:
            print("No chunks found.")
            return

        # Embed descriptions (HyDE mode)
        texts = [c['description'] for c in self.chunks]
        print(f"Generating description embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Embed code content (Code mode)
        # Include metadata to help with context
        code_texts = [f"File: {c['filename']}\nFunction: {c['name']}\n{c['content']}" for c in self.chunks]
        print(f"Generating code embeddings for {len(code_texts)} chunks...")
        self.code_embeddings = self.model.encode(code_texts, show_progress_bar=True)
        
        # Build keyword index
        print("Building keyword index...")
        # Convert chunks to documents format for InvertedIndex
        docs_for_index = []
        for idx, chunk in enumerate(self.chunks):
            docs_for_index.append({
                'id': idx,
                'title': chunk['name'],
                'description': chunk['description']
            })
        
        self.keyword_index = InvertedIndex()
        self.keyword_index = InvertedIndex()
        self.keyword_index.build_from_documents(docs_for_index)
        
        self.keyword_index.build_from_documents(docs_for_index)
        
        print(f"DEBUG: docs_for_index size: {len(docs_for_index)}")
        print(f"DEBUG: keyword_index.docmap size: {len(self.keyword_index.docmap)}")
        
        self.save_index()

    def save_index(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CODEBASE_INDEX_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(CODEBASE_EMBEDDINGS_PATH, self.embeddings)
        np.save(CODEBASE_EMBEDDINGS_CODE_PATH, self.code_embeddings)
        
        # Save keyword index
        if self.keyword_index:
            with open(CODEBASE_KEYWORD_INDEX_PATH, "wb") as f:
                pickle.dump(self.keyword_index, f)
        
        # Also save as JSON for transparency
        json_path = os.path.join(CACHE_DIR, "codebase_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
            
        print(f"Index saved to {CODEBASE_INDEX_PATH}")
        print(f"Metadata saved to {json_path}")

    def load_index(self):
        if not os.path.exists(CODEBASE_INDEX_PATH) or not os.path.exists(CODEBASE_EMBEDDINGS_PATH):
            print("Index not found. Building new index...")
            self.build_index()
            return

        with open(CODEBASE_INDEX_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.embeddings = np.load(CODEBASE_EMBEDDINGS_PATH)
        
        if os.path.exists(CODEBASE_EMBEDDINGS_CODE_PATH):
            self.code_embeddings = np.load(CODEBASE_EMBEDDINGS_CODE_PATH)
        else:
            self.code_embeddings = None
            print("Warning: Code embeddings not found. Code search mode will not work until rebuild.")
        
        # Load keyword index
        if os.path.exists(CODEBASE_KEYWORD_INDEX_PATH):
            with open(CODEBASE_KEYWORD_INDEX_PATH, "rb") as f:
                self.keyword_index = pickle.load(f)
        else:
            print("Keyword index not found. Building keyword index from loaded chunks...")
            # Build keyword index from existing chunks without full rebuild
            docs_for_index = []
            for idx, chunk in enumerate(self.chunks):
                docs_for_index.append({
                    'id': idx,
                    'title': chunk['name'],
                    'description': chunk['description']
                })
            
            
            self.keyword_index = InvertedIndex()
            self.keyword_index.build_from_documents(docs_for_index)
            
            # Save just the keyword index
            with open(CODEBASE_KEYWORD_INDEX_PATH, "wb") as f:
                pickle.dump(self.keyword_index, f)

    def search(self, query: str, limit: int = 10, score_threshold: float = 0.01, use_reranking: bool = False, mode: str = "hyde") -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF fusion of semantic and keyword search.
        mode: "hyde" (default, uses description embeddings) or "code" (uses code content embeddings)
        """
        if self.embeddings is None or self.keyword_index is None:
            self.load_index()
        
        # Semantic search
        query_embedding = self.model.encode(query)
        
        # Select embeddings based on mode
        if mode == "code" and self.code_embeddings is not None:
            target_embeddings = self.code_embeddings
        else:
            target_embeddings = self.embeddings
            
        semantic_scores = np.dot(target_embeddings, query_embedding) / (
            np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        semantic_indices = np.argsort(semantic_scores)[::-1]
        
        # 2. Keyword search (BM25)
        keyword_results = self.keyword_index.bm25_search(query, limit=limit*3)
        
        # 3. RRF Fusion
        rrf_scores = defaultdict(float)
        k = DEFAULT_K_VALUE
        
        # Add semantic scores
        for rank, idx in enumerate(semantic_indices):
            rrf_scores[idx] += rrf_score(rank, k)
        
        # Add keyword scores
        for rank, result in enumerate(keyword_results):
            idx = result['id']
            rrf_scores[idx] += rrf_score(rank, k)
        
        # Get candidates based on RRF score threshold
        candidate_count = limit * 3 if use_reranking else limit
        top_indices = heapq.nlargest(candidate_count, rrf_scores, key=rrf_scores.get)
        
        # Filter by score threshold
        top_indices = [idx for idx in top_indices if rrf_scores[idx] >= score_threshold]
        
        if not top_indices:
            return []
        
        # Build initial results
        results = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.chunks):
                print(f"Warning: Index {idx} out of bounds for chunks list (len={len(self.chunks)}). Skipping.")
                continue
                
            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = float(rrf_scores[idx])
            chunk["semantic_score"] = float(semantic_scores[idx])
            # Default score to RRF score for compatibility
            chunk["score"] = chunk["rrf_score"]
            # print(f"DEBUG: Assigned score {chunk['score']} to {chunk['name']}")
            results.append(chunk)
        
        # Apply re-ranking if requested
        if use_reranking and len(results) > 0:
            results = self.rerank(query, results, limit)
        else:
            results = results[:limit]
            
        return results
    
    def rerank(self, query: str, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Re-rank results using a cross-encoder model."""
        if not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [[query, f"{r['description']}\n{r['content']}"] for r in results]
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by rerank scores
        for i, result in enumerate(results):
            result["rerank_score"] = float(rerank_scores[i])
            # Update main score to rerank score
            result["score"] = result["rerank_score"]
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:limit]

def build_codebase_index_command():
    rag = CodebaseRAG()
    rag.build_index()

def search_codebase_command(query: str, limit: int = 5):
    rag = CodebaseRAG()
    results = rag.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['filename']}:{res['name']} (Score: {res['score']:.4f})")
        print(f"   {res['filepath']}")
