"""
Admin Chatbot UI Component.
"""
import streamlit as st
import pandas as pd
from app.admin_database import (
    create_session, get_all_sessions, get_session_messages, 
    add_message, delete_session, update_session_name
)
from app.model_handler import generate_with_gemini
from app.admin_memory import get_memory_handler

def render_admin_chat():
    st.header("🤖 Admin Chatbot")
    st.caption("Private, memory-enhanced chatbot for administrators.")
    
    # Initialize session state for active chat
    if 'admin_active_session_id' not in st.session_state:
        st.session_state.admin_active_session_id = None

    # Create layout: Left (History) - Right (Chat)
    col_hist, col_chat = st.columns([1, 3])

    # --- Left Column: Session Management ---
    with col_hist:
        st.subheader("Records")
        
        if st.button("➕ New Chat", type="primary", help="Start a new chat session"):
            session_id = create_session("New Chat", "gemini-2.0-flash")
            st.session_state.admin_active_session_id = session_id
            st.rerun()
            
        st.divider()
        
        sessions = get_all_sessions()
        
        # Use a container for scrolling if list is long
        with st.container(height=500):
            for session in sessions:
                # Active styling
                label = f"📝 {session['name']}"
                if st.session_state.admin_active_session_id == session['id']:
                    st.info(f"**{label}**")
                else:
                    if st.button(label, key=f"sess_{session['id']}", width=True):
                        st.session_state.admin_active_session_id = session['id']
                        st.rerun()
    
    # --- Right Column: Chat Area ---
    with col_chat:
        if st.session_state.admin_active_session_id is None:
            st.info("Select or create a chat session to begin.")
            
            # Illustration / Instructions
            st.markdown("""
            ### Features
            - **Persistent History**: Chats are saved automatically.
            - **Memory RAG**: intelligently recalls context to save tokens.
            - **Model Selection**: Choose your preferred Gemini model.
            - **Token Tracking**: Monitor your usage.
            """)
            return

        # Load current session data
        session_id = st.session_state.admin_active_session_id
        messages = get_session_messages(session_id)
        
        # --- Top Bar: Config & Stats ---
        # Use a container for the header of the chat area
        with st.container():
            col_model, col_stats, col_del = st.columns([2, 2, 1])
            
            with col_model:
                selected_model = st.selectbox(
                    "Model", 
                    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
                    key="admin_model_selector",
                    label_visibility="collapsed"
                )
                
            with col_stats:
                # Calculate total tokens
                total_tokens = sum(m['token_count'] for m in messages)
                st.caption(f"**Tokens:** {total_tokens}")
                
            with col_del:
                if st.button("🗑️", type="secondary", help="Delete Chat"):
                    delete_session(session_id)
                    st.session_state.admin_active_session_id = None
                    st.rerun()
            
            st.divider()

        # --- Message Display ---
        # Use a container for messages
        with st.container(height=550):
            for msg in messages:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
                    if msg['token_count'] > 0:
                        st.caption(f"Tokens: {msg['token_count']}")

        # --- Chat Input ---
        if prompt := st.chat_input("Message Admin Chatbot..."):
            # 1. Add User Message to UI and DB
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Estimate user tokens crudely
            add_message(session_id, "user", prompt, token_count=len(prompt)//4) 
            
            # 2. Rename session if it's the first message
            if len(messages) == 0:
                try:
                    title = generate_with_gemini(f"Generate a 3-5 word title for this chat start: {prompt}")
                    update_session_name(session_id, title.strip())
                except:
                    update_session_name(session_id, prompt[:20])

            # 3. Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get Memory (RAG Context)
                        memory = get_memory_handler()
                        context_msgs = memory.retrieve_context(prompt, session_id)
                        
                        # Build Prompt with Context
                        context_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in context_msgs])
                        
                        system_prompt = f"""You are a helpful admin assistant.
                        
                        Relevant Context from previous conversation:
                        {context_str}
                        
                        User Query: {prompt}
                        
                        Answer directly. If code is requested, provide it in code blocks."""
                        
                        # Call API
                        api_key = st.session_state.get('custom_api_key')
                        response_text, usage = generate_with_gemini(
                            system_prompt, 
                            api_key=api_key, 
                            model_name=selected_model,
                            return_usage=True
                        )
                        
                        # Display Response
                        st.markdown(response_text)
                        
                        # Display Usage
                        total_usage = usage.get('total_token_count', 0)
                        if total_usage > 0:
                            st.caption(f"Used {total_usage} tokens for this turn.")
                        
                        # Save to DB
                        add_message(session_id, "assistant", response_text, token_count=total_usage)
                        
                        # Rerun to update
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

