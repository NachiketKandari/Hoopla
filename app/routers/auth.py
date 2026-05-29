from fastapi import APIRouter, Request, Depends, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import status

from app.dependencies import register_user, authenticate_user
from app.database import get_user_by_id
from app.templates_config import templates
router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("partials/login_form.html", {"request": request})


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user_id, message = authenticate_user(username, password)
    if user_id:
        user = get_user_by_id(user_id)
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="hoopla_user",
            value=str(user_id),
            httponly=True,
            samesite="lax",
            max_age=86400 * 30,  # 30 days
        )
        return response
    return templates.TemplateResponse(
        "partials/login_form.html",
        {"request": request, "error": message},
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    return templates.TemplateResponse("partials/register_form.html", {"request": request})


@router.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    success, message, user_id = register_user(username, password)
    if success:
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="hoopla_user",
            value=str(user_id),
            httponly=True,
            samesite="lax",
            max_age=86400 * 30,
        )
        return response
    return templates.TemplateResponse(
        "partials/register_form.html",
        {"request": request, "error": message},
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("hoopla_user")
    return response
