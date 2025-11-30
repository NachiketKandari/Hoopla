"""
Authentication and rate limiting module.
"""
import bcrypt
import streamlit as st
from typing import Optional, Tuple
from app.database import (
    create_user, get_user_by_username, get_user_by_id,
    decrement_user_requests, get_user_requests_left
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except (ValueError, AttributeError) as e:
        # Invalid hash format (e.g., SHA256 instead of bcrypt)
        return False


def register_user(username: str, password: str) -> Tuple[bool, str, Optional[int]]:
    """
    Register a new user.
    Returns (success, message, user_id)
    """
    if not username or not password:
        return False, "Username and password are required", None
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters", None
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters", None
    
    password_hash = hash_password(password)
    success = create_user(username, password_hash)
    
    if success:
        user = get_user_by_username(username)
        return True, "Registration successful! Logging you in...", user['id']
    else:
        return False, "Username already exists", None


def authenticate_user(username: str, password: str) -> Tuple[Optional[int], str]:
    """
    Authenticate a user.
    Returns (user_id, message). user_id is None if authentication failed.
    """
    if not username or not password:
        return None, "Username and password are required"
    
    user = get_user_by_username(username)
    if not user:
        return None, "Invalid username or password"
    
    if not verify_password(password, user['password_hash']):
        return None, "Invalid username or password"
    
    return user['id'], "Login successful"


def check_rate_limit(user_id: int, is_system_api: bool) -> Tuple[bool, int]:
    """
    Check if user can make a request.
    Returns (allowed, requests_left)
    - SYSTEM API (Gemini): 50 requests/day limit (unlimited for admin)
    - User API (Ollama): No limit
    """
    # Check if user is admin - unlimited access
    user = get_user_by_id(user_id)
    if user and user.get('is_admin', 0) == 1:
        return True, 999999  # Unlimited for admin
    
    if not is_system_api:
        # User API has no limit
        return True, -1  # -1 indicates unlimited
    
    requests_left = get_user_requests_left(user_id)
    
    if requests_left > 0:
        return True, requests_left
    else:
        return False, 0


def consume_rate_limit(user_id: int, is_system_api: bool) -> bool:
    """
    Consume a rate limit for a user.
    Returns True if successful, False if limit reached.
    """
    if not is_system_api:
        # User API has no limit
        return True
    
    return decrement_user_requests(user_id)


def get_current_user_id() -> Optional[int]:
    """Get the current logged-in user ID from session state."""
    return st.session_state.get('user_id')


def is_user_logged_in() -> bool:
    """Check if a user is logged in."""
    return get_current_user_id() is not None


def login_user(user_id: int):
    """Set user as logged in."""
    st.session_state.user_id = user_id
    user = get_user_by_id(user_id)
    if user:
        st.session_state.username = user['username']


def logout_user():
    """Logout the current user and clear session state."""
    keys_to_clear = ['user_id', 'username', 'chat_messages', 'chat_loaded', 'thinking_mode']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def is_admin() -> bool:
    """Check if the current user is an admin."""
    user_id = get_current_user_id()
    if not user_id:
        return False
    
    user = get_user_by_id(user_id)
    if not user:
        return False
    
    return user.get('is_admin', 0) == 1
