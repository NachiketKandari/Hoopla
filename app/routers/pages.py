from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.dependencies import require_user, get_current_user, is_admin_user
from app.templates_config import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: dict | None = Depends(get_current_user)):
    if not user:
        return templates.TemplateResponse("pages/login.html", {"request": request})
    is_admin = user.get("is_admin", 0) == 1
    return templates.TemplateResponse(
        "pages/index.html",
        {"request": request, "username": user["username"], "is_admin": is_admin},
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: dict | None = Depends(get_current_user)):
    if user:
        return HTMLResponse(status_code=302, headers={"Location": "/"})
    return templates.TemplateResponse("pages/login.html", {"request": request})


@router.get("/readme", response_class=HTMLResponse)
async def readme_page(request: Request, user: dict = Depends(require_user)):
    readme_path = Path(__file__).parent.parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "README not found."
    return templates.TemplateResponse(
        "pages/readme.html", {"request": request, "username": user["username"], "content": content}
    )


@router.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request, user: dict = Depends(require_user)):
    from cli.lib.search_utils import load_movies
    import json
    movies = load_movies()
    return templates.TemplateResponse(
        "pages/dataset.html",
        {"request": request, "username": user["username"], "movies": movies, "movies_json": json.dumps(movies, indent=2)},
    )
