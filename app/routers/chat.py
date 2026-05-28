import json
import os
from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends, Query as QueryParam
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import status

from app.dependencies import require_user, consume_rate_limit, check_rate_limit, get_current_user, TEMPLATES_DIR
from app.database import add_conversation, get_recent_chat_messages, mark_conversations_as_deleted
from cli.lib.codebase_rag import CodebaseRAG, rewrite_query

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter()


async def get_api_key(request: Request) -> str | None:
    """Extract API key from session. Returns None for system/local models."""
    model_type = request.cookies.get("hoopla_model_type", "API")
    if model_type == "custom_gemini":
        return request.cookies.get("hoopla_custom_api_key") or os.getenv("GEMINI_API_KEY")
    return None


def get_model_type(request: Request) -> str:
    return request.cookies.get("hoopla_model_type", "API")


def get_ollama_model(request: Request) -> str | None:
    if get_model_type(request) == "local":
        return request.cookies.get("hoopla_ollama_model")
    return None


@router.get("", response_class=HTMLResponse)
async def chat_page(
    request: Request, user: dict = Depends(require_user)
):
    history = get_recent_chat_messages(user["id"], limit=20)
    return templates.TemplateResponse(
        "pages/chat.html",
        {"request": request, "user": user, "history": history},
    )


@router.post("/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    user: dict = Depends(require_user),
    query: str = Form(...),
    mode: str = Form("concept"),
    thinking_mode: str = Form("false"),
):
    is_system_api = get_model_type(request) == "API"
    allowed, msg = check_rate_limit(user["id"], is_system_api)
    if not allowed:
        return HTMLResponse(
            content=f'<div class="error-message">{msg}</div>',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    api_key = await get_api_key(request)
    ollama_model = get_ollama_model(request)

    try:
        rag = CodebaseRAG(api_key=api_key)
        search_query = query
        search_mode = mode

        if mode == "hyde":
            search_query = rag.generate_hypothetical_code(query)
        elif mode == "concept":
            rewritten = rewrite_query(query, api_key)
            if rewritten != query:
                search_query = rewritten
        elif mode == "simple":
            search_query = query

        use_rerank = thinking_mode.lower() == "true"
        results = rag.search(search_query, limit=10, score_threshold=0.01, use_reranking=use_rerank, mode=mode)

        context_str = ""
        for res in results:
            score = res.get("score", 0.0)
            context_str += f"File: {res['filename']}\nFunction: {res['name']}\nScore: {score:.4f}\nCode:\n{res['content']}\n\n"

        prompt = f"""Original Question: {query}
Rewritten Query (for technical retrieval): {search_query}

You are an expert coding assistant for the Hoopla codebase.
Use the following code chunks to answer the user's ORIGINAL QUESTION.
Cite the file and function name when explaining code.
If the code chunks don't contain the answer, say so.

Code Chunks:
{context_str}

Your response:"""

        model_type_str = get_model_type(request)
        response_text = ""

        if model_type_str == "local" and ollama_model:
            from app.model_handler import generate_with_ollama
            response_text = generate_with_ollama(prompt, ollama_model)
        else:
            from app.model_handler import generate_with_gemini
            response_text = generate_with_gemini(prompt, api_key=api_key)

        consume_rate_limit(user["id"], is_system_api)

        model_str = model_type_str
        if model_type_str == "local":
            model_str = f"local:{ollama_model}"
        elif model_type_str == "custom_gemini":
            model_str = "gemini:custom"

        add_conversation(user_id=user["id"], mode="chat", query=query, response=response_text, model_type=model_str)

        results_html = ""
        if results:
            chunks_html = ""
            for i, res in enumerate(results, 1):
                score = res.get("score", 0.0)
                chunks_html += f"""
                <details class="chunk-detail">
                    <summary><strong>{i}. {res['filename']}:{res['name']}</strong> (Score: {score:.4f})</summary>
                    <pre><code class="language-python">{res['content']}</code></pre>
                </details>"""

            results_html = f"""
            <details class="results-expander">
                <summary>📄 Relevant Code Chunks ({len(results)})</summary>
                {chunks_html}
            </details>"""

        return HTMLResponse(
            content=f"""
            <div class="message assistant-message">
                <div class="message-content">{response_text}</div>
                {results_html}
            </div>"""
        )

    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/clear", response_class=HTMLResponse)
async def clear_chat(request: Request, user: dict = Depends(require_user)):
    mark_conversations_as_deleted(user["id"], mode="chat")
    return HTMLResponse(content="")


@router.get("/history", response_class=HTMLResponse)
async def get_history(request: Request, user: dict = Depends(require_user)):
    messages = get_recent_chat_messages(user["id"], limit=20)
    html = ""
    for msg in messages:
        html += f"""
        <div class="message user-message history">
            <div class="message-content">{msg['query']}</div>
        </div>
        <div class="message assistant-message history">
            <div class="message-content">{msg['response']}</div>
        </div>"""
    return HTMLResponse(content=html)
