"""
Memory module for Admin Chatbot.
Handles RAG-based retrieval of chat context to optimize token usage.
"""
import numpy as np
import streamlit as st
from app.admin_database import get_session_messages
from cli.lib.model_loader import get_embedding_model
from typing import List, Dict, Any

class AdminMemory:
    def __init__(self):
        self.model = get_embedding_model()
        
    def retrieve_context(self, query: str, session_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant messages from the chat session using semantic search.
        Always includes the last 2 messages for immediate context.
        """
        messages = get_session_messages(session_id)
        
        if not messages:
            return []
            
        # Separate user and assistant messages that have content
        valid_msgs = [m for m in messages if m['content'].strip()]
        
        if len(valid_msgs) <= limit:
            return valid_msgs
            
        # Always include the very last message (immediate context) if it exists
        # Actually, usually the logical flow needs the last few messages.
        # Let's say last 2 are mandatory.
        recent_context = valid_msgs[-2:]
        searchable_history = valid_msgs[:-2]
        
        if not searchable_history:
            return recent_context
            
        # Embed query
        query_embedding = self.model.encode(query)
        
        # Embed history
        # In a production app, we would cache these embeddings in the DB.
        # For this scale, embedding on the fly is acceptable (fast for <100 msgs).
        history_texts = [m['content'] for m in searchable_history]
        history_embeddings = self.model.encode(history_texts)
        
        # Calculate similarities
        similarities = np.dot(history_embeddings, query_embedding) / (
            np.linalg.norm(history_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top indices
        top_k = min(limit - len(recent_context), len(searchable_history))
        if top_k <= 0:
            return recent_context
            
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Retrieve context messages
        context_msgs = [searchable_history[i] for i in sorted(top_indices)] # Sort by original order
        
        # Combine
        # Return sorted by timestamp (original order)
        final_context = context_msgs + recent_context
        
        return final_context

# Singleton instance to be used in UI
def get_memory_handler():
    if 'admin_memory' not in st.session_state:
        st.session_state.admin_memory = AdminMemory()
    return st.session_state.admin_memory
