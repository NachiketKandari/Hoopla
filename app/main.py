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
    """Preload heavy models at startup so first request is fast."""
    logger.info("Preloading ML models...")
    try:
        from cli.lib.model_loader import (
            get_embedding_model,
            get_clip_model,
            get_cross_encoder_tinybert,
            get_cross_encoder_minilm,
        )
        get_embedding_model()
        get_cross_encoder_tinybert()
        logger.info("Core models loaded (embedding + tinybert cross-encoder)")
        # CLIP and MiniLM cross-encoder loaded lazily on first use
    except Exception as e:
        logger.warning(f"Model preload issue (non-fatal): {e}")

    from app.database import init_database, create_admin_user
    init_database()
    create_admin_user()
    logger.info("Database initialized")

    yield
    logger.info("Shutting down")


app = FastAPI(title="Hoopla", lifespan=lifespan, docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth.router, prefix="/auth")
app.include_router(chat.router, prefix="/chat")
app.include_router(rag.router, prefix="/rag")
app.include_router(search.router, prefix="/search")
app.include_router(admin.router, prefix="/admin")
app.include_router(pages.router)
