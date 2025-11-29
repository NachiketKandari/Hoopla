"""
Admin Panel UI Component
This creates the admin panel interface for viewing users and conversations.
"""
import streamlit as st
import pandas as pd
from app.database import get_all_users, get_user_conversations, get_db_stats, reset_user_requests


def render_admin_panel():
    """Render the admin panel UI."""
    st.header("🔐 Admin Panel")
    st.caption("Administrator dashboard for viewing users, conversations, and database statistics")
    
    # Database Statistics
    st.subheader("📊 Database Statistics")
    stats = get_db_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", stats.get('total_users', 0))
    with col2:
        st.metric("Total Conversations", stats.get('total_conversations', 0))
    with col3:
        st.metric("Deleted Conversations", stats.get('deleted_conversations', 0))
    
    # Conversations by mode
    if stats.get('conversations_by_mode'):
        st.write("**Conversations by Mode:**")
        mode_data = stats['conversations_by_mode']
        cols = st.columns(len(mode_data))
        for idx, (mode, count) in enumerate(mode_data.items()):
            with cols[idx]:
                st.metric(mode.capitalize(), count)
    
    st.divider()
    
    # View All Users
    st.subheader("👥 All Users")
    users = get_all_users()
    
    if users:
        # Create DataFrame for better display
        users_df = pd.DataFrame(users)
        users_df = users_df[['id', 'username', 'requests_left', 'is_admin', 'created_at']]
        st.dataframe(users_df, width="stretch", hide_index=True)
    else:
        st.info("No users found")
    
    st.divider()
    
    # View User Conversations
    st.subheader("💬 View User Conversations")
    
    if users:
        # User selector
        user_options = {f"{u['username']} (ID: {u['id']})": u['id'] for u in users}
        selected_user = st.selectbox("Select a user to view their conversations", options=list(user_options.keys()))
        
        include_deleted = st.checkbox("Include deleted conversations", value=False)
        
        include_deleted = st.checkbox("Include deleted conversations", value=False)
        
        col_actions1, col_actions2 = st.columns(2)
        
        with col_actions1:
            if st.button("Load Conversations", width="stretch"):
                st.session_state.admin_load_conv = True
        
        with col_actions2:
            if st.button("🔄 Reset Quota (to 50)", width="stretch"):
                user_id = user_options[selected_user]
                if reset_user_requests(user_id):
                    st.success(f"Quota reset for user {selected_user}!")
                    st.rerun()
                else:
                    st.error("Failed to reset quota.")

        if st.session_state.get('admin_load_conv', False):
            user_id = user_options[selected_user]
            conversations = get_user_conversations(user_id, include_deleted=include_deleted)
            
            if conversations:
                st.write(f"**Found {len(conversations)} conversation(s)**")
                
                for conv in conversations:
                    status = "🗑️ DELETED" if conv.get('deleted', 0) == 1 else "✅ Active"
                    with st.expander(f"{status} | {conv['mode'].upper()} | {conv['timestamp']}"):
                        st.write(f"**Query:** {conv['query']}")
                        st.write(f"**Response:** {conv['response']}")
                        st.write(f"**Model:** {conv.get('model_type', 'N/A')}")
                        st.write(f"**ID:** {conv['id']}")
            else:
                st.info("No conversations found for this user")
    else:
        st.warning("No users available")
