import sys
import os
from pathlib import Path
import sqlite3

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.auth import register_user, authenticate_user
from app.database import (
    get_user_by_username, add_chat_history, get_chat_history, 
    init_database, get_db_connection
)

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

if __name__ == "__main__":
    setup_clean_db()
    test_unique_usernames()
    test_chat_privacy()
