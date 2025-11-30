import sys
import os
from pathlib import Path
import sqlite3

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.auth import register_user, authenticate_user, consume_rate_limit, check_rate_limit
from app.database import (
    get_user_by_username, add_chat_history, get_chat_history, 
    init_database, get_db_connection, reset_user_requests, get_user_requests_left
)
from cli.lib.codebase_rag import rewrite_query, CodebaseRAG

def setup_clean_db():
    """Reset DB for testing"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username LIKE 'test_%'")
    cursor.execute("DELETE FROM chat_history WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'test_%')")
    conn.commit()
    conn.close()

def test_unique_usernames():
    print("Testing Unique Usernames...")
    username = "test_user_unique"
    password = "password123"
    
    # First registration
    success, msg, _ = register_user(username, password)
    if not success:
        print(f"❌ First registration failed: {msg}")
        return
    print("✅ First registration successful")
    
    # Duplicate registration
    success, msg, _ = register_user(username, password)
    if not success and "already exists" in msg:
        print("✅ Duplicate registration failed as expected")
    else:
        print(f"❌ Duplicate registration should have failed but got: {success}, {msg}")

def test_chat_privacy():
    print("\nTesting Chat Privacy...")
    user1 = "test_user_1"
    user2 = "test_user_2"
    pwd = "password123"
    
    register_user(user1, pwd)
    register_user(user2, pwd)
    
    u1_id, _ = authenticate_user(user1, pwd)
    u2_id, _ = authenticate_user(user2, pwd)
    
    # Add chat for User 1
    add_chat_history(u1_id, "chat", query_text="User 1 Secret")
    
    # Add chat for User 2
    add_chat_history(u2_id, "chat", query_text="User 2 Secret")
    
    # Check User 1 sees only their chat
    chats1 = get_chat_history(u1_id)
    if len(chats1) == 1 and chats1[0]['query_text'] == "User 1 Secret":
        print("✅ User 1 sees only their chat")
    else:
        print(f"❌ User 1 saw: {[c['query_text'] for c in chats1]}")

    # Check User 2 sees only their chat
    chats2 = get_chat_history(u2_id)
    if len(chats2) == 1 and chats2[0]['query_text'] == "User 2 Secret":
        print("✅ User 2 sees only their chat")
    else:
        print(f"❌ User 2 saw: {[c['query_text'] for c in chats2]}")

def test_admin_reset():
    print("\nTesting Admin Reset...")
    username = "test_user_quota"
    pwd = "password123"
    
    register_user(username, pwd)
    u_id, _ = authenticate_user(username, pwd)
    
    # Consume some requests
    consume_rate_limit(u_id, is_system_api=True)
    left_before = get_user_requests_left(u_id)
    if left_before == 49:
        print("✅ Consumed 1 request")
    else:
        print(f"❌ Expected 49 requests left, got {left_before}")
        
    # Reset
    reset_user_requests(u_id)
    left_after = get_user_requests_left(u_id)
    if left_after == 50:
        print("✅ Quota reset successfully")
    else:
        print(f"❌ Expected 50 requests left, got {left_after}")

def test_rag_rewrite():
    print("\nTesting RAG Rewrite...")
    query = "test query"
    rewritten = rewrite_query(query, api_key=None)
    
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}")
    
    if rewritten and isinstance(rewritten, str):
        print("✅ Rewrite returned a string")
    else:
        print(f"❌ Rewrite failed to return string: {rewritten}")

def test_search_modes():
    print("\nTesting Search Modes...")
    rag = CodebaseRAG()
    # Mock embeddings to avoid full load if possible, or just test logic
    # For now, we'll just check if the method accepts the modes without crashing
    
    modes = ["concept", "simple", "hyde"]
    query = "database connection"
    
    for mode in modes:
        try:
            # We don't expect actual results without a full index, but we check for crashes
            # Note: This might print warnings about missing embeddings, which is expected
            rag.search(query, limit=1, mode=mode)
            print(f"✅ Search mode '{mode}' executed without error")
        except Exception as e:
            print(f"❌ Search mode '{mode}' failed: {e}")

def test_registration_validation():
    print("\nTesting Registration Validation...")
    # Test short password
    success, msg, uid = register_user("valid_user", "short")
    if not success and uid is None:
        print("✅ Short password validation passed (returns 3 values)")
    else:
        print(f"❌ Short password validation failed signature check: {success}, {msg}, {uid}")

if __name__ == "__main__":
    setup_clean_db()
    test_unique_usernames()
    test_registration_validation()
    test_chat_privacy()
    test_admin_reset()
    test_rag_rewrite()
    test_search_modes()
