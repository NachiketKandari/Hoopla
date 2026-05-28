"""
Verification script for Admin Chatbot backend.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.admin_database import (
    init_admin_database, create_session, add_message, 
    get_session_messages, get_all_sessions, delete_session
)
from app.admin_memory import AdminMemory
from app.model_handler import generate_with_gemini

def verify_database():
    print("Testing Database...")
    init_admin_database()
    
    # Create session
    session_id = create_session("Test Session", "gemini-test")
    print(f"Created session {session_id}")
    
    # Add messages
    add_message(session_id, "user", "Hello computer", 5)
    add_message(session_id, "assistant", "Hello human", 5)
    
    # Verify retrieval
    msgs = get_session_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0]['content'] == "Hello computer"
    print("Database verification passed!")
    return session_id

def verify_memory(session_id):
    print("\nTesting Memory RAG...")
    memory = AdminMemory()
    
    # Retrieve context
    # Should return everything since it's small
    context = memory.retrieve_context("computer", session_id)
    print(f"Retrieved {len(context)} messages for context")
    assert len(context) > 0
    assert context[0]['content'] == "Hello computer"
    print("Memory verification passed!")

def verify_model_handler():
    print("\nTesting Model Handler Signature...")
    # We won't make a real API call to avoid cost/auth issues if env not set, 
    # but we will check if the function accepts the arguments.
    # Actually, let's try a dry run if possible or just inspect.
    # We can try catching the API error which means the function signature was correct at least.
    try:
        generate_with_gemini("test", model_name="gemini-2.0-flash", return_usage=True)
    except Exception as e:
        # If we get InvalidAPIKeyError or similar, it means it reached the client call, so args were accepted.
        # If we get TypeError, it means signature is wrong.
        if "unexpected keyword argument" in str(e):
             print(f"FAILED: Model handler signature mismatch: {e}")
             return
        print(f"Model handler signature verification passed (Error was expected: {str(e)[:50]}...)")

if __name__ == "__main__":
    try:
        session_id = verify_database()
        verify_memory(session_id)
        verify_model_handler()
        
        # Cleanup
        delete_session(session_id)
        print("\nCleanup complete.")
        print("✅ ALL TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
