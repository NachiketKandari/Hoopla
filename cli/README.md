# CLI Documentation

This directory contains the Command Line Interface (CLI) scripts for the Hoopla RAG & Hybrid Search Toolkit. Each script exposes specific functionality of the system.

## Available CLIs

### `keyword_search_cli.py`
Handles keyword-based search operations using BM25 and inverted indexes.
- **Build Index:** `python keyword_search_cli.py build`
- **Search:** `python keyword_search_cli.py search "query" --limit 5`
- **BM25 Search:** `python keyword_search_cli.py bm25search "query"`
- **Inspect:** `python keyword_search_cli.py tf <doc_id> <term>`

### `semantic_search_cli.py`
Handles vector-based semantic search using SentenceTransformers.
- **Embed Chunks:** `python semantic_search_cli.py embed_chunks`
- **Search:** `python semantic_search_cli.py search "query"`
- **Chunked Search:** `python semantic_search_cli.py search_chunked "query"`

### `hybrid_search_cli.py`
Combines keyword and semantic search for better retrieval.
- **Weighted Search:** `python hybrid_search_cli.py weighted-search "query" --alpha 0.5`
- **RRF Search:** `python hybrid_search_cli.py rrf-search "query"`

### `augmented_generation_cli.py`
Performs RAG operations to generate answers based on retrieved context.
- **RAG:** `python augmented_generation_cli.py rag "query"`
- **Summarize:** `python augmented_generation_cli.py summarize "query"`

### `multimodal_search_cli.py`
Enables searching using images.
- **Image Search:** `python multimodal_search_cli.py image_search path/to/image.jpg`

### `evaluation_cli.py`
Runs evaluation metrics against a golden dataset.
- **Run Eval:** `python evaluation_cli.py --limit 10`

## Usage
Run any CLI script with `--help` to see the full list of available commands and arguments.
