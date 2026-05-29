from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi import status

from app.dependencies import require_user, check_rate_limit, consume_rate_limit
from app.database import add_conversation
from app.model_handler import generate_response, generate_multidoc_summary, generate_citations, generate_answer, InvalidAPIKeyError
from cli.lib.augmented_generation import get_results
from app.templates_config import templates

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def rag_page(request: Request, user: dict = Depends(require_user)):
    return templates.TemplateResponse(
        "pages/rag.html",
        {"request": request, "username": user["username"], "is_admin": user.get("is_admin", 0) == 1},
    )


@router.post("/generate", response_class=HTMLResponse)
async def generate_rag(
    request: Request,
    user: dict = Depends(require_user),
    query: str = Form(...),
    rag_type: str = Form("rag"),
):
    if not query:
        return HTMLResponse(content='<div class="warning">Please enter a query</div>')

    model_type = request.cookies.get("hoopla_model_type", "API")
    is_system_api = model_type == "API"
    allowed, msg = check_rate_limit(user["id"], is_system_api)
    if not allowed:
        return HTMLResponse(
            content=f'<div class="error-message">{msg}</div>',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    import os
    api_key = request.cookies.get("hoopla_custom_api_key") or os.getenv("GEMINI_API_KEY")
    ollama_model = request.cookies.get("hoopla_ollama_model")

    try:
        results = get_results(query)

        response = ""
        if rag_type == "rag":
            response = generate_response(query, results, model_type, ollama_model, api_key=api_key)
        elif rag_type == "summarize":
            response = generate_multidoc_summary(query, results, model_type, ollama_model, api_key=api_key)
        elif rag_type == "citations":
            response = generate_citations(query, results, model_type, ollama_model, api_key=api_key)
        elif rag_type == "question":
            response = generate_answer(query, results, model_type, ollama_model, api_key=api_key)

        consume_rate_limit(user["id"], is_system_api)

        model_str = model_type
        if model_type == "local":
            model_str = f"local:{ollama_model}"
        elif model_type == "custom_gemini":
            model_str = "gemini:custom"

        mode_map = {"rag": "rag", "summarize": "rag_summarize", "citations": "rag_citations", "question": "rag_question"}
        add_conversation(user_id=user["id"], mode=mode_map.get(rag_type, "rag"), query=query, response=response, model_type=model_str)

        results_html = ""
        for i, res in enumerate(results, 1):
            results_html += f'<li>{i}. {res["title"]}</li>'

        return HTMLResponse(
            content=f"""
            <div class="rag-results">
                <h3>Search Results</h3>
                <ol>{results_html}</ol>
                <h3>Generated Response</h3>
                <div class="response-text">{response}</div>
            </div>"""
        )

    except InvalidAPIKeyError as e:
        return HTMLResponse(
            content=f'''<div class="error-message">
                <strong>API Key Error</strong><br>
                {str(e)}<br><br>
                <small>Set a valid key in your .env file or use the sidebar to configure a custom API key.</small>
            </div>''',
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
