"""FastAPI dependencies — stateless auth, DB, model singletons."""

import os
from functools import lru_cache
from typing import Optional, Tuple
import bcrypt
from pathlib import Path

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from app.database import (
    get_user_by_username,
    get_user_by_id,
    create_user,
    get_user_requests_left,
    decrement_user_requests,
)
from app.model_handler import InvalidAPIKeyError


TEMPLATES_DIR = Path(__file__).parent / "templates"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, AttributeError):
        return False


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def register_user(username: str, password: str) -> Tuple[bool, str, Optional[int]]:
    if not username or not password:
        return False, "Username and password are required", None
    if len(username) < 3:
        return False, "Username must be at least 3 characters", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters", None
    password_hash = hash_password(password)
    success = create_user(username, password_hash)
    if success:
        user = get_user_by_username(username)
        return True, "Registration successful!", user["id"]
    return False, "Username already exists", None


def authenticate_user(username: str, password: str) -> Tuple[Optional[int], str]:
    if not username or not password:
        return None, "Username and password are required"
    user = get_user_by_username(username)
    if not user:
        return None, "Invalid username or password"
    if not verify_password(password, user["password_hash"]):
        return None, "Invalid username or password"
    return user["id"], "Login successful"


def check_rate_limit(user_id: int, is_system_api: bool) -> Tuple[bool, int]:
    user = get_user_by_id(user_id)
    if user and user.get("is_admin", 0) == 1:
        return True, 999999
    if not is_system_api:
        return True, -1
    requests_left = get_user_requests_left(user_id)
    if requests_left > 0:
        return True, requests_left
    return False, 0


def consume_rate_limit(user_id: int, is_system_api: bool) -> bool:
    if not is_system_api:
        return True
    return decrement_user_requests(user_id)


def is_admin_user(user_id: Optional[int]) -> bool:
    if not user_id:
        return False
    user = get_user_by_id(user_id)
    return bool(user and user.get("is_admin", 0) == 1)


async def get_current_user(hoopla_user: Optional[str] = Cookie(default=None)) -> Optional[dict]:
    if not hoopla_user:
        return None
    try:
        uid = int(hoopla_user)
        user = get_user_by_id(uid)
        if user:
            return dict(user)
    except (ValueError, TypeError):
        pass
    return None


async def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
    return user


async def require_admin(user: dict = Depends(require_user)) -> dict:
    if not user.get("is_admin", 0):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/"})
    return user


def get_model_handler(model_type: str = "API", api_key: str = None, ollama_model: str = None):
    """Returns appropriate model config dict for passing to generation functions."""
    return {
        "model_type": model_type,
        "api_key": api_key or os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key"),
        "ollama_model": ollama_model,
    }
