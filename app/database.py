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
    
    conn.commit()
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
    reset_daily_requests()  # Reset if needed
    
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


def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_file_to_base64(uploaded_file) -> str:
    """Convert an uploaded Streamlit file to base64 string."""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


# Initialize database on import
init_database()

