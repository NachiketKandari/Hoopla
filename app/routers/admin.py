import os
from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import status

from app.dependencies import require_admin
from app.database import (
    get_all_users, get_user_conversations, get_db_stats, reset_user_requests,
    create_admin_chat_session, get_admin_chat_sessions, get_admin_chat_messages,
    add_admin_chat_message, delete_admin_chat_session,
)
from app.admin_memory import AdminMemory
from app.model_handler import generate_with_gemini, generate_with_ollama
from app.templates_config import templates
router = APIRouter()
admin_memory = AdminMemory()


def _get_model_config(request: Request):
    model_type = request.cookies.get("hoopla_model_type", "API")
    api_key = request.cookies.get("hoopla_custom_api_key") or os.getenv("GEMINI_API_KEY")
    ollama_model = request.cookies.get("hoopla_ollama_model")
    return model_type, api_key, ollama_model


@router.get("", response_class=HTMLResponse)
async def admin_panel(request: Request, user: dict = Depends(require_admin)):
    users = get_all_users()
    stats = get_db_stats()
    return templates.TemplateResponse(
        "pages/admin.html", {"request": request, "username": user["username"], "is_admin": True, "users": users, "stats": stats}
    )


@router.get("/conversations/{user_id}", response_class=HTMLResponse)
async def view_conversations(request: Request, user_id: int, user: dict = Depends(require_admin)):
    conversations = get_user_conversations(user_id, include_deleted=True)
    html = ""
    for conv in conversations:
        deleted_badge = '<span class="badge badge-deleted">deleted</span>' if conv.get("deleted") else ""
        html += f"""
        <div class="conv-item">
            <div class="conv-header">
                <span class="conv-mode">{conv['mode']}</span>
                <span class="conv-time">{conv['timestamp']}</span>
                {deleted_badge}
            </div>
            <p><strong>Q:</strong> {conv['query']}</p>
            <p><strong>A:</strong> {conv['response'][:300]}...</p>
        </div>"""
    return HTMLResponse(content=html)


@router.post("/reset-quota", response_class=HTMLResponse)
async def reset_quota(request: Request, user_id: int = Form(...), user: dict = Depends(require_admin)):
    reset_user_requests(user_id)
    return HTMLResponse(content='<div class="success-message">Quota reset to 50 for user #{user_id}</div>')


@router.get("/chatbot", response_class=HTMLResponse)
async def admin_chatbot(request: Request, user: dict = Depends(require_admin)):
    sessions = get_admin_chat_sessions()
    return templates.TemplateResponse(
        "pages/admin_chatbot.html",
        {"request": request, "username": user["username"], "is_admin": True, "sessions": sessions},
    )


@router.post("/chatbot/session", response_class=HTMLResponse)
async def create_session(
    request: Request,
    user: dict = Depends(require_admin),
    name: str = Form(...),
    model: str = Form("gemini-2.0-flash"),
):
    session_id = create_admin_chat_session(name, model)
    return HTMLResponse(
        content=f'<option value="{session_id}" selected>{name} ({model})</option>',
        headers={"HX-Trigger": "sessionCreated"},
    )


@router.post("/chatbot/send", response_class=HTMLResponse)
async def chatbot_send(
    request: Request,
    user: dict = Depends(require_admin),
    session_id: int = Form(...),
    query: str = Form(...),
):
    if not query:
        return HTMLResponse(content="")

    model_type, api_key, ollama_model = _get_model_config(request)

    # Get conversation messages for the session
    messages = get_admin_chat_messages(session_id)

    # Retrieve relevant context via memory RAG
    context_msgs = admin_memory.retrieve_context(query, session_id, limit=5)
    context_str = ""
    for msg in context_msgs:
        context_str += f"[{msg['role']}]: {msg['content']}\n"

    prompt = f"""You are the admin assistant for the Hoopla RAG toolkit.
Use the following conversation context to answer the user's question naturally.

Conversation context:
{context_str}

User: {query}
Assistant:"""

    try:
        if model_type == "local" and ollama_model:
            response_text = generate_with_ollama(prompt, ollama_model)
        else:
            response_text = generate_with_gemini(prompt, api_key=api_key, model_name="gemini-2.0-flash")
    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    add_admin_chat_message(session_id, "user", query)
    add_admin_chat_message(session_id, "assistant", response_text)

    return HTMLResponse(
        content=f"""
        <div class="message user-message"><div class="message-content">{query}</div></div>
        <div class="message assistant-message"><div class="message-content">{response_text}</div></div>"""
    )


@router.post("/chatbot/delete-session", response_class=HTMLResponse)
async def delete_session(
    request: Request,
    user: dict = Depends(require_admin),
    session_id: int = Form(...),
):
    delete_admin_chat_session(session_id)
    return RedirectResponse(url="/admin/chatbot", status_code=status.HTTP_302_FOUND)


@router.get("/chatbot/messages/{session_id}", response_class=HTMLResponse)
async def get_session_messages(request: Request, session_id: int, user: dict = Depends(require_admin)):
    messages = get_admin_chat_messages(session_id)
    html = ""
    for msg in messages:
        html += f"""
        <div class="message {msg['role']}-message">
            <div class="message-content">{msg['content']}</div>
        </div>"""
    return HTMLResponse(content=html)
