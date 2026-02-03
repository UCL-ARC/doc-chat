"""Main application module."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import Base, engine
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


@app.on_event("startup")
async def startup():
    """Create database tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)




