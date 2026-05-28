import ast
import os
import pickle
import json
import fnmatch
import time
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import heapq
from .search_utils import PROJECT_ROOT, CACHE_DIR, DEFAULT_K_VALUE, get_llm_client, generate_text
from .keyword_search import InvertedIndex
from .model_loader import get_embedding_model, get_cross_encoder_minilm

# Constants
CODEBASE_INDEX_PATH = os.path.join(CACHE_DIR, "codebase_index.pkl")
CODEBASE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "codebase_embeddings.npy")
CODEBASE_EMBEDDINGS_CODE_PATH = os.path.join(CACHE_DIR, "codebase_embeddings_code.npy")
CODEBASE_KEYWORD_INDEX_PATH = os.path.join(CACHE_DIR, "codebase_keyword_index.pkl")


def rrf_score(rank, k: int = DEFAULT_K_VALUE):
    """Calculate RRF score for a given rank."""
    return 1 / (k + rank)


def _load_readme_context() -> str:
    """Helper to load README content for context."""
    readme_content = ""
    try:
        root_readme = os.path.join(PROJECT_ROOT, "README.md")
        if os.path.exists(root_readme):
            with open(root_readme, "r") as f:
                readme_content += f"\n--- Root README ---\n{f.read()}"

        cli_readme = os.path.join(PROJECT_ROOT, "cli", "README.md")
        if os.path.exists(cli_readme):
            with open(cli_readme, "r") as f:
                readme_content += f"\n--- CLI README ---\n{f.read()}"
    except Exception:
        pass
    return readme_content


def _create_client(api_key: str = None):
    """Create an LLM client based on LLM_PROVIDER env var or explicit key."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()

    if provider == "gemini":
        from google import genai
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            return None, None, None
        return genai.Client(api_key=key), "gemini-2.0-flash", "gemini"
    else:
        from openai import OpenAI
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return None, None, None
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
        return client, "deepseek-v4-flash", "deepseek"


def _llm_generate(prompt: str, client, model: str, provider: str, model_override: str = None) -> str:
    """Unified text generation. Uses model_override if provided (for lite models etc)."""
    m = model_override or model
    if provider == "gemini":
        response = client.models.generate_content(model=m, contents=prompt)
        return response.text.strip()
    else:
        response = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


def rewrite_query(query: str, api_key: str = None) -> str:
    """
    Rewrites a user query to be more suitable for searching function descriptions.
    Focus on expanding the query with related concepts and technical terms.
    """
    client, model, provider = _create_client(api_key)

    if not client:
        return query  # Fallback to original query if no key

    try:
        readme_content = _load_readme_context()

        prompt = f"""You are a helpful assistant for Hoopla. Your task is to rewrite the user query into a decent-sized, technical search query suitable for RAG (Retrieval Augmented Generation) against the codebase.

        Use the following context from the project READMEs to understand the terminology:
        {readme_content}

        User Query: "{query}"

        Rewritten Query (just the query text, no quotes or explanations):"""

        rewritten = _llm_generate(prompt, client, model, provider)
        return rewritten.replace('"', '')
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query


class CodebaseChunker:
    def __init__(self, root_dir: str, api_key: str = None):
        self.root_dir = root_dir
        self.ignore_patterns = self._load_gitignore()
        self._rate_limit_hit = False  # Circuit breaker flag
        self.cached_descriptions = self._load_cached_descriptions()

        self.client, self.model, self.provider = _create_client(api_key)

        if not self.client:
            print("Warning: No LLM API key found. AI descriptions will be disabled.")

    def _load_cached_descriptions(self) -> Dict[str, str]:
        """Load existing descriptions from cache/codebase_data.json to avoid re-generation."""
        cache_path = os.path.join(CACHE_DIR, "codebase_data.json")
        descriptions = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
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

        if filename == "admin_panel_ui.py":
            return True

        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(filename, pattern):
                return True
            if os.path.isdir(path) and fnmatch.fnmatch(rel_path + "/", pattern):
                return True
        return False

    def generate_description(self, code: str, function_name: str) -> str:
        """Generate an AI description for a function with retry logic."""
        if self._rate_limit_hit:
            return f"Function {function_name}"

        if not self.client:
            return f"Function {function_name}"

        max_retries = 5
        base_delay = 2

        # Use flash-lite for Gemini, v4-flash for DeepSeek
        lite_model = "gemini-2.0-flash-lite" if self.provider == "gemini" else "deepseek-v4-flash"

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

                description = _llm_generate(prompt, self.client, self.model, self.provider, model_override=lite_model)

                time.sleep(0.5)

                return description

            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit for {function_name}, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Max retries reached for {function_name}. Disabling AI descriptions for this session.")
                        self._rate_limit_hit = True
                        return f"Function {function_name}"
                else:
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
                    if "admin" in node.name.lower():
                        continue

                    start_line = node.lineno
                    end_line = node.end_lineno
                    code_segment = "\n".join(content.splitlines()[start_line-1:end_line])

                    cache_key = f"{filename}:{node.name}"
                    if cache_key in self.cached_descriptions and len(self.cached_descriptions[cache_key].split()) > 5:
                        ai_description = self.cached_descriptions[cache_key]
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
                    pass

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return chunks

    def walk_and_chunk(self) -> List[Dict[str, Any]]:
        all_chunks = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not self._is_ignored(os.path.join(root, d))]

            for file in files:
                filepath = os.path.join(root, file)
                if self._is_ignored(filepath):
                    continue
                if not file.endswith(".py"):
                    continue

                chunks = self.chunk_file(filepath)
                all_chunks.extend(chunks)
        return all_chunks


class CodebaseRAG:
    def __init__(self, root_dir: str = PROJECT_ROOT, api_key: str = None):
        self.root_dir = root_dir
        self.model = get_embedding_model()
        self._reranker = None  # lazy-loaded only when reranking is used
        self.chunks = []
        self.embeddings = None
        self.code_embeddings = None
        self.keyword_index = None

        self.llm_client, self.llm_model, self.llm_provider = _create_client(api_key)

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_cross_encoder_minilm()
        return self._reranker

    def generate_hypothetical_code(self, query: str) -> str:
        """
        Generates a hypothetical code snippet based on the user query (Actual HyDE).
        """
        if not self.llm_client:
            return query

        try:
            readme_context = _load_readme_context()

            prompt = f"""You are an expert Python developer. Write a hypothetical Python function or code snippet that would answer the following user query.

            Context from Project READMEs:
            {readme_context}

            User Query: "{query}"

            Do not include any explanations or markdown formatting. Just provide the raw Python code that might exist in a codebase to solve this problem.
            
            Example:
            Query: "tell me about hybrid search"
            Response:
            def hybrid_search(self, query: str, limit: int = 10):
                # Perform keyword search
                bm25_results = self.keyword_index.search(query, limit=limit)

                # Perform semantic search
                semantic_results = self.vector_store.search(query, limit=limit)

                # Combine results using RRF
                combined_results = self.rrf_fusion(bm25_results, semantic_results)
                return combined_results

            Generated Hypothetical Code:"""

            return _llm_generate(prompt, self.llm_client, self.llm_model, self.llm_provider)
        except Exception as e:
            print(f"Error generating hypothetical code: {e}")
            return query

    def build_index(self):
        chunker = CodebaseChunker(self.root_dir)
        self.chunks = chunker.walk_and_chunk()

        if not self.chunks:
            print("No chunks found.")
            return

        texts = [c['description'] for c in self.chunks]
        print(f"Generating description embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        code_texts = [f"File: {c['filename']}\nFunction: {c['name']}\n{c['content']}" for c in self.chunks]
        print(f"Generating code embeddings for {len(code_texts)} chunks...")
        self.code_embeddings = self.model.encode(code_texts, show_progress_bar=True)

        print("Building keyword index...")
        docs_for_index = []
        for idx, chunk in enumerate(self.chunks):
            docs_for_index.append({
                'id': idx,
                'title': chunk['name'],
                'description': chunk['description']
            })

        self.keyword_index = InvertedIndex()
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

        if self.keyword_index:
            with open(CODEBASE_KEYWORD_INDEX_PATH, "wb") as f:
                pickle.dump(self.keyword_index, f)

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

        if os.path.exists(CODEBASE_KEYWORD_INDEX_PATH):
            with open(CODEBASE_KEYWORD_INDEX_PATH, "rb") as f:
                self.keyword_index = pickle.load(f)
        else:
            print("Keyword index not found. Building keyword index from loaded chunks...")
            docs_for_index = []
            for idx, chunk in enumerate(self.chunks):
                docs_for_index.append({
                    'id': idx,
                    'title': chunk['name'],
                    'description': chunk['description']
                })

            self.keyword_index = InvertedIndex()
            self.keyword_index.build_from_documents(docs_for_index)

            with open(CODEBASE_KEYWORD_INDEX_PATH, "wb") as f:
                pickle.dump(self.keyword_index, f)

    def search(self, query: str, limit: int = 10, score_threshold: float = 0.01, use_reranking: bool = False, mode: str = "concept") -> List[Dict[str, Any]]:
        if self.embeddings is None or self.keyword_index is None:
            self.load_index()

        search_text = query
        target_embeddings = self.embeddings

        if mode == "hyde":
            print("Generating hypothetical code for HyDE...")
            search_text = self.generate_hypothetical_code(query)
            print(f"Hypothetical Code:\n{search_text[:200]}...")
            if self.code_embeddings is not None:
                target_embeddings = self.code_embeddings
            else:
                print("Warning: Code embeddings missing for HyDE. Falling back to descriptions.")

        elif mode == "simple":
            if self.code_embeddings is not None:
                target_embeddings = self.code_embeddings
            else:
                print("Warning: Code embeddings missing for SimpleRAG. Falling back to descriptions.")

        query_embedding = self.model.encode(search_text)

        semantic_scores = np.dot(target_embeddings, query_embedding) / (
            np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        semantic_indices = np.argsort(semantic_scores)[::-1]

        keyword_results = self.keyword_index.bm25_search(query, limit=limit*3)

        rrf_scores = defaultdict(float)
        k = DEFAULT_K_VALUE

        for rank, idx in enumerate(semantic_indices):
            rrf_scores[idx] += rrf_score(rank, k)

        for rank, result in enumerate(keyword_results):
            idx = result['id']
            rrf_scores[idx] += rrf_score(rank, k)

        candidate_count = limit * 10 if use_reranking else limit
        top_indices = heapq.nlargest(candidate_count, rrf_scores, key=rrf_scores.get)

        top_indices = [idx for idx in top_indices if rrf_scores[idx] >= score_threshold]

        if not top_indices:
            return []

        results = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.chunks):
                print(f"Warning: Index {idx} out of bounds for chunks list (len={len(self.chunks)}). Skipping.")
                continue

            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = float(rrf_scores[idx])
            chunk["semantic_score"] = float(semantic_scores[idx])
            chunk["score"] = chunk["rrf_score"]
            results.append(chunk)

        if use_reranking and len(results) > 0:
            results = self.rerank(query, results, limit)
        else:
            results = results[:limit]

        return results

    def rerank(self, query: str, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Re-rank results using a cross-encoder model."""
        if not results:
            return results

        pairs = [[query, f"{r['description']}\n{r['content']}"] for r in results]

        rerank_scores = self.reranker.predict(pairs)

        for i, result in enumerate(results):
            result["rerank_score"] = float(rerank_scores[i])
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
