"""
Database module for the Admin Chatbot.
Manages a separate SQLite database for admin chat history and sessions.
"""
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Database file path - Separate from main hoopla.db
DB_PATH = Path(__file__).parent.parent / "data" / "admin_chat.db"


def get_db_connection():
    """Get a database connection with WAL mode for concurrent access."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_admin_database():
    """Initialize the admin database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we need to migrate (simple check for now)
    try:
        cursor.execute("SELECT * FROM admin_chat_sessions LIMIT 1")
    except sqlite3.OperationalError:
        # Tables don't exist, create them
        pass
        
    # Admin Chat Sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Admin Chat Messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES admin_chat_sessions(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()


def create_session(name: str, model: str) -> int:
    """Create a new chat session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO admin_chat_sessions (name, model)
        VALUES (?, ?)
    """, (name, model))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def delete_session(session_id: int):
    """Delete a chat session and its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # SQLite should handle cascade delete if enabled, but let's be explicit to be safe
    cursor.execute("DELETE FROM admin_chat_messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM admin_chat_sessions WHERE id = ?", (session_id,))
    
    conn.commit()
    conn.close()


def get_all_sessions() -> List[Dict[str, Any]]:
    """Get all chat sessions ordered by update time."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM admin_chat_sessions 
        ORDER BY updated_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session_messages(session_id: int) -> List[Dict[str, Any]]:
    """Get all messages for a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM admin_chat_messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_message(session_id: int, role: str, content: str, token_count: int = 0):
    """Add a message to a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO admin_chat_messages (session_id, role, content, token_count)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, token_count))
    
    # Update session timestamp
    cursor.execute("""
        UPDATE admin_chat_sessions 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = ?
    """, (session_id,))
    
    conn.commit()
    conn.close()


def update_session_name(session_id: int, new_name: str):
    """Update session name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE admin_chat_sessions 
        SET name = ? 
        WHERE id = ?
    """, (new_name, session_id))
    
    conn.commit()
    conn.close()

# Initialize on import
init_admin_database()
