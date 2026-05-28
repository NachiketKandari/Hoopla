import os
import tempfile
import requests
from fastapi import APIRouter, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import status

from app.dependencies import require_user
from cli.lib.hybrid_search import HybridSearch
from cli.lib.search_utils import load_movies, DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA_VALUE, DEFAULT_K_VALUE
from cli.lib.semantic_search import search_chunked_command
from cli.lib.keyword_search import InvertedIndex
from cli.lib.multimodal_search import MultiModalSearch
from cli.lib.query_enhancement import enhance_query
from app.templates_config import templates
router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def search_page(request: Request, user: dict = Depends(require_user)):
    return templates.TemplateResponse(
        "pages/search.html",
        {"request": request, "username": user["username"], "is_admin": user.get("is_admin", 0) == 1},
    )


@router.post("/hybrid", response_class=HTMLResponse)
async def hybrid_search(
    request: Request,
    user: dict = Depends(require_user),
    query: str = Form(...),
    search_type: str = Form("rrf"),
    k_value: float = Form(DEFAULT_K_VALUE),
    alpha_value: float = Form(DEFAULT_ALPHA_VALUE),
    limit: int = Form(DEFAULT_SEARCH_LIMIT),
    enhance: str = Form(""),
    rerank: str = Form(""),
    evaluate: str = Form(""),
):
    if not query:
        return HTMLResponse(content='<div class="warning">Please enter a query</div>')

    try:
        documents = load_movies()
        hs = HybridSearch(documents)

        if enhance:
            query = enhance_query(query, method=enhance)

        if search_type == "rrf":
            results = hs.rrf_search(query, k=k_value, limit=limit)
            if rerank == "cross_encoder":
                from cli.lib.reranking import rerank_cross_encoder
                results = rerank_cross_encoder(query, results, limit)
        else:
            results = hs.weighted_search(query, alpha=alpha_value, limit=limit)

        html = _format_results_html(results, query, enhance, rerank)
        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/semantic", response_class=HTMLResponse)
async def semantic_search(
    request: Request,
    user: dict = Depends(require_user),
    query: str = Form(...),
    limit: int = Form(DEFAULT_SEARCH_LIMIT),
):
    if not query:
        return HTMLResponse(content='<div class="warning">Please enter a query</div>')

    try:
        results = search_chunked_command(query, limit)
        html = ""
        for i, res in enumerate(results, 1):
            html += f"""
            <div class="result-item">
                <strong>{i}. {res['title']}</strong> (Score: {res['score']:.4f})
                <p class="result-desc">{res.get('description', '')}</p>
            </div>"""
        return HTMLResponse(content=f'<div class="search-results">{html}</div>')

    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/keyword", response_class=HTMLResponse)
async def keyword_search(
    request: Request,
    user: dict = Depends(require_user),
    query: str = Form(...),
    limit: int = Form(DEFAULT_SEARCH_LIMIT),
):
    if not query:
        return HTMLResponse(content='<div class="warning">Please enter a query</div>')

    try:
        documents = load_movies()
        idx = InvertedIndex()
        idx.build_from_documents(documents)
        results = idx.bm25_search(query, limit)

        html = ""
        for i, res in enumerate(results, 1):
            html += f"""
            <div class="result-item">
                <strong>{i}. {res['title']}</strong> (Score: {res.get('score', 0):.4f})
                <p class="result-desc">{res.get('description', '')[:200]}</p>
            </div>"""
        return HTMLResponse(content=f'<div class="search-results">{html}</div>')

    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/multimodal", response_class=HTMLResponse)
async def multimodal_search(
    request: Request,
    user: dict = Depends(require_user),
    file: UploadFile = File(...),
):
    try:
        documents = load_movies()
        mms = MultiModalSearch(documents)
        mms.load_or_create_embeddings()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            results = mms.search_with_image(tmp_path)
            html = ""
            for i, res in enumerate(results, 1):
                html += f"""
                <div class="result-item">
                    <strong>{i}. {res['title']}</strong> (Score: {res['score']:.4f})
                    <p class="result-desc">{res.get('description', '')[:200]}</p>
                </div>"""
            return HTMLResponse(content=f'<div class="search-results">{html}</div>')
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return HTMLResponse(
            content=f'<div class="error-message">Error: {str(e)}</div>',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _format_results_html(results: list, query: str, enhance: str = "", rerank: str = "") -> str:
    lines = []
    if enhance:
        lines.append(f'<p class="info">Enhanced query: {query}</p>')
    for i, res in enumerate(results, 1):
        score = res.get('score', res.get('rrf_score', 0))
        lines.append(f"""
        <div class="result-item">
            <strong>{i}. {res['title']}</strong> (Score: {score:.4f})
            <p class="result-desc">{res.get('description', '')[:200]}</p>
        </div>""")
    return f'<div class="search-results">{"".join(lines)}</div>'


@router.get("/ollama-models")
async def ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return JSONResponse(content=models)
    except Exception:
        pass
    return JSONResponse(content=[])
