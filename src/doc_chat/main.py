"""Main application module."""
# Prevent LiteLLM from fetching model_prices_and_context_window.json (set before litellm is ever imported)
import os
os.environ.setdefault("LITELLM_MODEL_COST_MAP_URL", "")

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure application loggers (e.g. doc_chat.routers.documents) emit INFO to the terminal
_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    _root.addHandler(_h)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import Base, engine
# Import all models so they are registered with Base.metadata before create_all
from .models import conversation, document, user  # noqa: F401
from .routers import auth, conversations, documents
from .services.rag_service import rag_service
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up application...")

    # Create database tables (users, documents, conversations, etc.)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created or already exist")
    except Exception as e:
        logger.exception("Failed to create database tables: %s", e)
        raise

    # Initialize RAG service
    rag_initialized = await rag_service.initialize()
    if rag_initialized:
        logger.info("RAG service initialized successfully")
    else:
        logger.info("RAG service disabled or failed to initialize")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="Document Chat",
    description="API for document analysis, summarization, and FAQ generation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://0.0.0.0:8000",
        "http://localhost:8000",
        "http://0.0.0.0:3001",
        "http://localhost:3001",
        "http://0.0.0.0:8001",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers first
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(conversations.router)

# Mount static files last
frontend_build_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=frontend_build_dir, html=True), name="static")




