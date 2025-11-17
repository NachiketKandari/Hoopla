# RAG & Hybrid Search CLI Toolkit

This project provides a comprehensive Command Line Interface (CLI) tool for advanced movie retrieval and content generation, primarily leveraging Hybrid Search techniques (Keyword and Semantic Search) and Retrieval Augmented Generation (RAG) using the Gemini API. The generation features are specifically tailored to provide information useful for Hoopla users (a movie streaming service).

The toolkit encompasses functions for traditional keyword search (BM25), semantic search, multimodal image search, advanced hybrid search via Reciprocal Rank Fusion (RRF), query enhancement, result reranking, and automated performance evaluation.

## Features

The toolkit is divided into several main components, each accessible via its own CLI module:

### 1. Retrieval Augmented Generation (RAG)

These commands utilize the RRF search results and pass them to an LLM (`gemini-2.0-flash`) for customized content generation:

| Command | Description | Core Functionality |
| --- | --- | --- |
| `rag <query>` | Performs basic RAG (search + generate answer). | The response is tailored for Hoopla users and delivered without markdown/bolding. |
| `summarize <query>` | Provides a concise, information-dense summary by synthesizing information from multiple search results. | Synthesizes information about genre, plot, etc. |
| `citations <query>` | Generates a comprehensive answer that includes citations. | Uses a document ID format like `[1]`, `[2]`, etc., and handles situations where sources may disagree. |
| `question <query>` | Answers specific questions (factual, analytical, or opinion-based) directly and concisely. | Uses only information found in the documents. |

---

### 2. Hybrid Search

The hybrid search module combines Keyword Search (BM25) and Semantic Search outputs to improve relevance.

| Command | Description | Search Mechanism |
| --- | --- | --- |
| `weighted-search <query> --alpha <float> --limit <int>` | Performs a weighted combination search. | The `alpha` value (default 0.5) dictates the balance between BM25 scores and normalized semantic scores. |
| `rrf-search <query> --k <float> --limit <int>` | Uses Reciprocal Rank Fusion (RRF). | Documents are scored based on their rank in the result lists, calculated using the formula $1/(k + rank)$. |

The `rrf-search` command also supports optional advanced features:

* **Query Enhancement (`--enhance`):** Improves the initial query before search using LLM generation methods.
    * `spell`: Fixes obvious spelling errors.
    * `rewrite`: Rewrites the query to be more specific and searchable (like a Google-style query).
    * `expand`: Appends synonyms and related terms to the original query.
* **Result Re-ranking (`--rerank-method`):** Reranks the RRF results using different LLM or model-based methods.
    * `individual`: Reranking by scoring each document individually (0-10) using an LLM prompt.
    * `batch`: Reranking by asking the LLM to return a JSON list of document IDs in order of relevance.
    * `cross_encoder`: Reranking using a pre-trained CrossEncoder model (`cross-encoder/ms-marco-TinyBERT-L2-v2`) to predict relevance scores.
* **Utility:**
    * `normalize <values...>`: Normalizes a list of numerical scores.

---

### 3. Keyword Search

This module implements a standard inverted index and BM25 scoring.

| Command | Description |
| --- | --- |
| `build` | Builds the inverted index, document map, and term frequencies from the movie data, then saves them to disk. |
| `tf <doc_id> <term>` | Gets the term frequency for a specific term in a document. |
| `idf <term>` | Gets the Inverse Document Frequency for a term. |
| `bm25idf <term>` | Gets the BM25 IDF score. |
| `bm25tf <doc_id> <term> [k1] [b]` | Gets the BM25 TF score, allowing customization of K1 (default 1.5) and B (default 0.75) parameters. |
| `bm25search <query> --limit <int>` | Performs search using full BM25 scoring. |
| `search <query> --limit <int>` | Performs a simpler retrieval of documents containing the query tokens. |

---

### 4. Semantic Search

This module handles vector embeddings for search, utilizing the SentenceTransformer model.

| Command | Description | Embedding Type |
| --- | --- | --- |
| `verify` | Verifies the semantic search model loaded and its properties. | N/A |
| `verify_embeddings` | Checks the count and dimensionality of the loaded movie embeddings. | Document |
| `embed_text <query>` | Generates and displays the embedding for a given text input. | N/A |
| `search <query> --limit <int>` | Searches using embeddings generated from the full movie description. | Document |
| `embed_chunks` | Builds and saves chunked embeddings for documents, based on sentence splitting. | Chunk |
| `search_chunked <query> --limit <int>` | Searches using document chunks. The score for a movie is the maximum score achieved by any of its constituent chunks. | Chunk |
| `chunk <query> --chunk-size <int> --overlap <int>` | Utility for query chunking based on word count/overlap. | N/A |
| `semantic_chunk <query> --max-chunk-size <int> --overlap <int>` | Utility for semantic chunking based on sentence structure. | N/A |

---

### 5. Multimodal Capabilities

The toolkit includes functionality to integrate image information into the search pipeline.

| Command | Description |
| --- | --- |
| `image_search <image_path>` | Performs a search against the movie database using an image as the primary query input, leveraging CLIP embeddings for similarity scoring. |
| `verify_image_embedding <image_path>` | Verifies the dimensions of an embedding generated from an image. |
| `describe_image_command <query> --image <image_path>` | Uses an LLM to synthesize visual and textual information from the image and query, rewriting the query to be more effective for movie search (focusing on actors, scenes, or style). |

---

### 6. Evaluation

The Evaluation CLI module runs performance tests against a pre-loaded "golden dataset" of test cases.

| Command | Description |
| --- | --- |
| `evaluation_cli.py --limit <int>` | Loads test cases and executes RRF search against them, calculating and printing evaluation metrics like Precision@k, Recall@k, and F1 Score@k based on the specified limit. |

---

## Setup

This tool relies heavily on local files for caching and external APIs for LLM operations.

* **API Key:** An API key for the Gemini service is required and should be loaded as an environment variable (`gemini_api_key` or `GEMINI_API_KEY`) using a `.env` file.
* **Local Data:** The tool expects movie data (`movies.json`) and evaluation test cases (`golden_dataset.json`) to be available in the defined data paths.
* **Index/Embeddings:** Many commands require cached files (e.g., inverted index, semantic embeddings, chunk embeddings) to be built before use. Users must run `keyword_search_cli.py build` and `semantic_search_cli.py embed_chunks` (or similar commands) initially to create the necessary cache files.