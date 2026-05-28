"""Module-level model singletons — each model loads exactly once."""

from sentence_transformers import SentenceTransformer, CrossEncoder
import threading

_lock = threading.Lock()

_embedding_model = None
_clip_model = None
_cross_encoder_minilm = None
_cross_encoder_tinybert = None


def get_embedding_model():
    """Shared all-MiniLM-L6-v2 — ~90MB, used by semantic/search/codebase/readme RAG."""
    global _embedding_model
    if _embedding_model is None:
        with _lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_clip_model():
    """Shared clip-ViT-B-32 — ~600MB, used by multimodal search."""
    global _clip_model
    if _clip_model is None:
        with _lock:
            if _clip_model is None:
                _clip_model = SentenceTransformer("clip-ViT-B-32")
    return _clip_model


def get_cross_encoder_minilm():
    """cross-encoder/ms-marco-MiniLM-L-6-v2 — used by codebase reranking (thinking mode)."""
    global _cross_encoder_minilm
    if _cross_encoder_minilm is None:
        with _lock:
            if _cross_encoder_minilm is None:
                _cross_encoder_minilm = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_minilm


def get_cross_encoder_tinybert():
    """cross-encoder/ms-marco-TinyBERT-L2-v2 — used by hybrid search reranking."""
    global _cross_encoder_tinybert
    if _cross_encoder_tinybert is None:
        with _lock:
            if _cross_encoder_tinybert is None:
                _cross_encoder_tinybert = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    return _cross_encoder_tinybert
