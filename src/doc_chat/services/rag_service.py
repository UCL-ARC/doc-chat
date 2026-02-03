"""RAG (Retrieval-Augmented Generation) service using FAISS."""
import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import get_settings
from .embeddings import is_ollama_embedding_model, embed_with_ollama

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_sentence_transformer():
    """Lazy import to avoid loading SentenceTransformer when using Ollama."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer


class RAGService:
    """RAG service for document retrieval and augmentation."""

    def __init__(self) -> None:
        """Initialize the RAG service."""
        self._use_ollama = is_ollama_embedding_model(settings.RAG_EMBEDDING_MODEL)
        self.embedding_model: Optional[Any] = None  # SentenceTransformer when not Ollama
        self._embedding_dim: Optional[int] = None  # Set at init (Ollama or from model)
        self._faiss: Any = None  # Lazy-loaded when RAG is enabled to avoid loading FAISS at startup
        self.index: Optional[Any] = None  # faiss.IndexFlatIP when RAG enabled
        self.documents: List[Dict[str, Any]] = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the RAG service.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        if not settings.RAG_ENABLED:
            logger.info("RAG is disabled in configuration")
            return False

        # Lazy-load FAISS only when RAG is enabled (avoids loading at startup when disabled)
        import faiss
        self._faiss = faiss

        try:
            if self._use_ollama:
                # Ollama: no local model load; get dimension by embedding a single string
                sample = await embed_with_ollama(["test"])
                self._embedding_dim = sample.shape[1]
                logger.info(f"RAG using Ollama embeddings, dimension={self._embedding_dim}")
            else:
                # SentenceTransformers (e.g. Hugging Face): load model in thread pool
                loop = asyncio.get_event_loop()
                SentenceTransformer = _get_sentence_transformer()
                self.embedding_model = await loop.run_in_executor(
                    self.executor,
                    lambda: SentenceTransformer(settings.RAG_EMBEDDING_MODEL),
                )
                sample = self.embedding_model.encode(["test"])
                self._embedding_dim = sample.shape[1]

            # Create index directory if it doesn't exist
            index_path = Path(settings.FAISS_INDEX_PATH)
            index_path.mkdir(parents=True, exist_ok=True)

            # Load existing index or create new one
            await self._load_or_create_index()

            self._initialized = True
            logger.info("RAG service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False
    
    async def _encode(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for texts (Ollama or SentenceTransformer)."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim or 0)
        if self._use_ollama:
            return await embed_with_ollama(texts)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            texts,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    async def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        index_file = Path(settings.FAISS_INDEX_PATH) / "index.faiss"
        docs_file = Path(settings.FAISS_INDEX_PATH) / "documents.pkl"

        if index_file.exists() and docs_file.exists():
            try:
                # Load existing index
                loop = asyncio.get_event_loop()
                self.index = await loop.run_in_executor(
                    self.executor,
                    self._faiss.read_index,
                    str(index_file),
                )

                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)

                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")

            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                await self._create_new_index()
        else:
            await self._create_new_index()

    async def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        dimension = self._embedding_dim
        if dimension is None:
            raise ValueError("Embedding dimension not set (RAG not initialized for embeddings)")
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = self._faiss.IndexFlatIP(dimension)
        self.documents = []
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    async def add_document_chunks(
        self,
        document_id: int,
        text_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add document chunks to the RAG index.

        Args:
            document_id: ID of the document
            text_chunks: List of text chunks to add
            metadata: Optional metadata for the document

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._initialized or not self.index:
            logger.warning("RAG service not initialized")
            return False
        if not self._use_ollama and not self.embedding_model:
            logger.warning("RAG embedding model not initialized")
            return False

        try:
            embeddings = await self._encode(text_chunks)
            # Add to FAISS index
            self.index.add(embeddings.astype("float32"))
            
            # Store document metadata
            for i, chunk in enumerate(text_chunks):
                doc_data = {
                    'document_id': document_id,
                    'chunk_index': i,
                    'text': chunk,
                    'metadata': metadata or {}
                }
                self.documents.append(doc_data)
            
            # Save index and documents
            await self._save_index()
            
            logger.info(f"Added {len(text_chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}")
            return False
    
    async def search_similar_chunks(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using the query.
        
        Args:
            query: Search query
            document_ids: Optional list of document IDs to filter by
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with metadata
        """
        if not self._initialized or not self.index:
            logger.warning("RAG service not initialized")
            return []
        if not self._use_ollama and not self.embedding_model:
            logger.warning("RAG embedding model not initialized")
            return []
        if not self.documents:
            logger.warning("No documents in RAG index")
            return []

        try:
            top_k = top_k or settings.RAG_TOP_K
            query_embedding = await self._encode([query])
            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding.astype("float32"), top_k * 2
            )  # Get more to filter
            
            # Filter and format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Filter by document IDs if specified
                    if document_ids and doc['document_id'] not in document_ids:
                        continue
                        
                    result = {
                        'document_id': doc['document_id'],
                        'chunk_index': doc['chunk_index'],
                        'text': doc['text'],
                        'score': float(score),
                        'metadata': doc['metadata']
                    }
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []
    
    async def remove_document(self, document_id: int) -> bool:
        """Remove all chunks for a document from the index.
        
        Args:
            document_id: ID of the document to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._initialized:
            logger.warning("RAG service not initialized")
            return False
            
        try:
            # Filter out documents with the given ID
            old_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc['document_id'] != document_id]
            removed_count = old_count - len(self.documents)
            
            if removed_count > 0:
                # Rebuild index without the removed documents
                await self._rebuild_index()
                logger.info(f"Removed {removed_count} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            return False
    
    async def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from current documents."""
        if not self.documents or self._embedding_dim is None:
            await self._create_new_index()
            return
        text_chunks = [doc["text"] for doc in self.documents]
        embeddings = await self._encode(text_chunks)
        self.index = self._faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        await self._save_index()
    
    async def _save_index(self) -> None:
        """Save the FAISS index and documents to disk."""
        if not self.index:
            return
            
        try:
            index_file = Path(settings.FAISS_INDEX_PATH) / "index.faiss"
            docs_file = Path(settings.FAISS_INDEX_PATH) / "documents.pkl"
            
            # Save in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._faiss.write_index,
                self.index,
                str(index_file)
            )
            
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            'enabled': settings.RAG_ENABLED,
            'initialized': self._initialized,
            'total_chunks': len(self.documents),
            'unique_documents': len(set(doc['document_id'] for doc in self.documents)),
            "embedding_model": settings.RAG_EMBEDDING_MODEL,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def __del__(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global RAG service instance
rag_service = RAGService() 