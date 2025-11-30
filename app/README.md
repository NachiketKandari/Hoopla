# Hoopla Streamlit App

This directory contains the core application logic for the Hoopla web interface.

## Structure

- **`streamlit_app.py`**: The main entry point for the Streamlit application. It handles the UI layout, state management, and routing between different tabs (Chat, RAG, Search, Admin).
- **`auth.py`**: Manages user authentication (registration, login, logout) and rate limiting logic.
- **`database.py`**: Handles all SQLite database interactions, including user management, chat history persistence, and admin queries.
- **`admin_panel_ui.py`**: A dedicated component for the Admin Panel interface, allowing administrators to view stats, manage users, and inspect conversations.
- **`model_handler.py`**: Abstraction layer for interacting with LLMs (Gemini and Ollama). It handles API calls, error handling, and prompt construction.

## Key Features

### Authentication
- **User Registration**: Users can create accounts with unique usernames.
- **Login/Logout**: Secure session management using Streamlit's session state.
- **Rate Limiting**: Regular users are limited to 50 requests per day (when using the System API). Admins have unlimited access.

### Chat & RAG
- **Chat Interface**: A conversational interface to query the codebase.
- **RAG Workflows**: Specialized modes for summarization, citation generation, and direct Q&A.
- **Hybrid Search**: Advanced search capabilities combining keyword (BM25) and semantic (vector) search with RRF fusion.

### Admin Panel
- **Dashboard**: View database statistics (users, conversations).
- **User Management**: Inspect user lists and reset request quotas.
- **Conversation Viewer**: Browse user chat histories (including deleted ones) for debugging or auditing.

## Database Schema
The application uses a local SQLite database (`hoopla.db`) with the following key tables:
- `users`: Stores user credentials (hashed passwords) and quota information.
- `chat_history`: Stores individual chat messages.
- `conversation_history`: Stores structured RAG/Search interactions.
