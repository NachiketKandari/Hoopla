import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np


class ChunkVectorStore:
    """
    Lightweight FAISS-based vector store for chunk-level movie embeddings.

    The store keeps a cosine-similarity-ready FAISS index on disk along with
    parallel metadata so both CLI utilities and the Streamlit app can query
    chunks without having to load 100MB+ numpy arrays into the repo.
    """

    def __init__(self, dimension: int, index_path: str, metadata_path: str) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.metadata: Dict | None = None

    def is_ready(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.metadata_path)

    def count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def _ensure_parent_dir(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def rebuild(self, embeddings: np.ndarray, chunk_metadata: List[Dict]) -> None:
        if embeddings.size == 0 or len(chunk_metadata) == 0:
            raise ValueError("No chunk embeddings were provided to build the store.")

        normalized = self._normalize(embeddings)
        if normalized.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, got {normalized.shape[1]}"
            )

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(normalized)
        self.metadata = {
            "chunks": chunk_metadata,
            "total_chunks": len(chunk_metadata),
        }

        self._ensure_parent_dir()
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def ensure_loaded(self) -> None:
        if self.index is None or self.metadata is None:
            self.load()

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        self.ensure_loaded()
        if self.index is None or self.metadata is None:
            return []

        normalized_query = self._normalize(query_embedding)
        limit = min(top_k, self.index.ntotal)
        if limit <= 0:
            return []
        scores, indices = self.index.search(normalized_query, limit)

        results: List[Tuple[Dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_metadata = self.metadata["chunks"][int(idx)]
            results.append((chunk_metadata, float(score)))
        return results

