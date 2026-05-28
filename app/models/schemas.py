from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    redirect: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    mode: str = "concept"  # concept, simple, hyde
    thinking_mode: bool = False


class ChatChunk(BaseModel):
    token: str = ""
    done: bool = False


class RAGRequest(BaseModel):
    query: str
    mode: str = "rag"  # rag, summarize, citations, question


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # hybrid, semantic, keyword
    enhance: bool = False
    rerank: str = "none"  # none, cross_encoder, individual, batch
    evaluate: bool = False
    limit: int = 10
    alpha: float = 0.5
    method: str = "rrf"


class MultimodalSearchResponse(BaseModel):
    results: List[dict]
    success: bool
    error: Optional[str] = None


class APIKeyRequest(BaseModel):
    api_key: str


class AdminSessionCreate(BaseModel):
    name: str
    model: str


class AdminMessageRequest(BaseModel):
    session_id: int
    query: str
    token_count: int = 0


class ConversationRequest(BaseModel):
    user_id: int
    mode: Optional[str] = None  # chat, rag, etc.
    include_deleted: bool = False


class DeleteConversationsRequest(BaseModel):
    user_id: int
    mode: str = "all"


class ResetQuotaRequest(BaseModel):
    user_id: int


class OllamaModel(BaseModel):
    name: str
    size: Optional[str] = None
