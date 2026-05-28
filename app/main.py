"""Hoopla FastAPI application — serves the web UI and API endpoints."""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import auth, chat, rag, search, admin, pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hoopla_fastapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database at startup. Models are lazy-loaded on first use."""
    from app.database import init_database, create_admin_user
    init_database()
    create_admin_user()
    logger.info("Database initialized")
    logger.info("Models will load lazily on first request (may take a few seconds for first user)")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Hoopla", lifespan=lifespan, docs_url=None, redoc_url=None)

@app.get("/health")
async def health():
    return {"status": "ok"}

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth.router, prefix="/auth")
app.include_router(chat.router, prefix="/chat")
app.include_router(rag.router, prefix="/rag")
app.include_router(search.router, prefix="/search")
app.include_router(admin.router, prefix="/admin")
app.include_router(pages.router)
