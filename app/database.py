"""
Database module for managing users, rate limiting, and chat history.
Uses SQLite for persistent storage.
"""
import sqlite3
import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List, Any
import base64
from io import BytesIO
from PIL import Image

# Database file path
DB_PATH = Path(__file__).parent.parent / "data" / "hoopla.db"


def get_db_connection():
    """Get a database connection."""
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            requests_left INTEGER DEFAULT 50,
            last_reset_date DATE DEFAULT CURRENT_DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query_type TEXT NOT NULL,
            query_text TEXT,
            query_image_base64 TEXT,
            response_text TEXT,
            response_results TEXT,
            model_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Conversation history table (simplified for chat and RAG)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('chat', 'rag', 'rag_summarize', 'rag_citations', 'rag_question')),
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            model_type TEXT,
            deleted BOOLEAN DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Add deleted column to existing table if it doesn't exist
    try:
        cursor.execute("ALTER TABLE conversation_history ADD COLUMN deleted BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add is_admin column to users table if it doesn't exist
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Admin chat sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Admin chat messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES admin_chat_sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()


def create_admin_user():
    """Create admin user with hardcoded credentials if it doesn't exist."""
    import bcrypt
    
    admin_username = "admin"
    admin_password = "f20212691@pilani.best-pilani.ac.in"
    
    # Hash password using bcrypt (same as regular users)
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), salt).decode('utf-8')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if admin exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (admin_username,))
    existing_admin = cursor.fetchone()
    
    if not existing_admin:
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, requests_left, last_reset_date, is_admin)
                VALUES (?, ?, 999999, ?, 1)
            """, (admin_username, password_hash, date.today()))
            conn.commit()
            print(f"✅ Admin user '{admin_username}' created successfully")
        except sqlite3.IntegrityError:
            pass  # Admin already exists
    else:
        # Update existing admin to have correct password and is_admin = 1
        cursor.execute("""
            UPDATE users SET password_hash = ?, is_admin = 1, requests_left = 999999
            WHERE username = ?
        """, (password_hash, admin_username))
        conn.commit()
        print(f"✅ Admin user '{admin_username}' updated successfully")
    
    conn.close()


def reset_daily_requests():
    """Reset daily requests for all users if it's a new day."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    today = date.today()
    cursor.execute("""
        UPDATE users 
        SET requests_left = 50, last_reset_date = ?
        WHERE last_reset_date < ?
    """, (today, today))
    
    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str) -> bool:
    """Create a new user. Returns True if successful, False if username exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, requests_left, last_reset_date)
            VALUES (?, ?, 50, ?)
        """, (username, password_hash, date.today()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username. Returns None if not found."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID. Returns None if not found."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def decrement_user_requests(user_id: int) -> bool:
    """Decrement user's request count. Returns True if successful, False if limit reached."""
    # Don't reset here - it's called by get_user_requests_left which is always called before this
    # reset_daily_requests()  # Removed to avoid redundant DB operations
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users 
        SET requests_left = requests_left - 1
        WHERE id = ? AND requests_left > 0
    """, (user_id,))
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success


def get_user_requests_left(user_id: int) -> int:
    """Get remaining requests for a user."""
    reset_daily_requests()  # Reset if needed
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT requests_left FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row['requests_left']
    return 0


def add_chat_history(
    user_id: int,
    query_type: str,
    query_text: Optional[str] = None,
    query_image_base64: Optional[str] = None,
    response_text: Optional[str] = None,
    response_results: Optional[str] = None,
    model_type: Optional[str] = None
):
    """Add a chat history entry."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO chat_history 
        (user_id, query_type, query_text, query_image_base64, response_text, response_results, model_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, query_type, query_text, query_image_base64, response_text, response_results, model_type))
    
    conn.commit()
    conn.close()


def get_chat_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Get chat history for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM chat_history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (user_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def add_conversation(
    user_id: int,
    mode: str,
    query: str,
    response: str,
    model_type: Optional[str] = None
):
    """Add a conversation entry (chat or RAG)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO conversation_history 
        (user_id, mode, query, response, model_type)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, mode, query, response, model_type))
    
    conn.commit()
    conn.close()


def get_conversation_history(user_id: int, mode: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get conversation history for a user. Optionally filter by mode (chat/rag)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if mode:
        cursor.execute("""
            SELECT * FROM conversation_history 
            WHERE user_id = ? AND mode = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, mode, limit))
    else:
        cursor.execute("""
            SELECT * FROM conversation_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_recent_chat_messages(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent chat messages for loading chat history (excludes deleted messages)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT query, response, timestamp FROM conversation_history 
        WHERE user_id = ? AND mode = 'chat' AND deleted = 0
        ORDER BY timestamp ASC 
        LIMIT ?
    """, (user_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def mark_conversations_as_deleted(user_id: int, mode: str = 'chat'):
    """Soft delete conversations for a user by mode (chat/rag/all)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if mode == 'all':
        cursor.execute("""
            UPDATE conversation_history 
            SET deleted = 1
            WHERE user_id = ?
        """, (user_id,))
    else:
        cursor.execute("""
            UPDATE conversation_history 
            SET deleted = 1
            WHERE user_id = ? AND mode = ?
        """, (user_id, mode))
    
    conn.commit()
    conn.close()


def get_all_users() -> List[Dict[str, Any]]:
    """Get all users for admin panel."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, requests_left, last_reset_date, is_admin, created_at 
        FROM users 
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_user_conversations(user_id: int, include_deleted: bool = False) -> List[Dict[str, Any]]:
    """Get all conversations for a specific user (admin function)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if include_deleted:
        cursor.execute("""
            SELECT * FROM conversation_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT * FROM conversation_history 
            WHERE user_id = ? AND deleted = 0
            ORDER BY timestamp DESC
        """, (user_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_db_stats() -> Dict[str, Any]:
    """Get database statistics for admin panel."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total users
    cursor.execute("SELECT COUNT(*) as count FROM users")
    stats['total_users'] = cursor.fetchone()['count']
    
    # Total conversations
    cursor.execute("SELECT COUNT(*) as count FROM conversation_history WHERE deleted = 0")
    stats['total_conversations'] = cursor.fetchone()['count']
    
    # Deleted conversations
    cursor.execute("SELECT COUNT(*) as count FROM conversation_history WHERE deleted = 1")
    stats['deleted_conversations'] = cursor.fetchone()['count']
    
    # Conversations by mode
    cursor.execute("""
        SELECT mode, COUNT(*) as count 
        FROM conversation_history 
        WHERE deleted = 0
        GROUP BY mode
    """)
    stats['conversations_by_mode'] = {row['mode']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    return stats


def create_admin_chat_session(session_name: str, model_name: str) -> int:
    """Create a new admin chat session. Returns session_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO admin_chat_sessions (session_name, model_name)
        VALUES (?, ?)
    """, (session_name, model_name))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return session_id


def get_admin_chat_sessions() -> List[Dict[str, Any]]:
    """Get all admin chat sessions, ordered by most recently updated."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, session_name, model_name, created_at, updated_at
        FROM admin_chat_sessions
        ORDER BY updated_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_admin_chat_messages(session_id: int) -> List[Dict[str, Any]]:
    """Get all messages for a specific admin chat session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, role, content, timestamp
        FROM admin_chat_messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def add_admin_chat_message(session_id: int, role: str, content: str):
    """Add a message to an admin chat session and update session timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Add message
    cursor.execute("""
        INSERT INTO admin_chat_messages (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))
    
    # Update session timestamp
    cursor.execute("""
        UPDATE admin_chat_sessions
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (session_id,))
    
    conn.commit()
    conn.close()


def delete_admin_chat_session(session_id: int):
    """Delete an admin chat session and all its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete messages first
    cursor.execute("DELETE FROM admin_chat_messages WHERE session_id = ?", (session_id,))
    
    # Delete session
    cursor.execute("DELETE FROM admin_chat_sessions WHERE id = ?", (session_id,))
    
    conn.commit()
    conn.close()


def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_file_to_base64(uploaded_file) -> str:
    """Convert an uploaded Streamlit file to base64 string."""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


# Initialize database on import
init_database()
create_admin_user()

