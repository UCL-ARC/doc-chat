"""RAG (Retrieval-Augmented Generation) service using FAISS."""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """RAG service for document retrieval and augmentation."""
    
    def __init__(self) -> None:
        """Initialize the RAG service."""
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
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
            
        try:
            # Load embedding model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                self.executor,
                self._load_embedding_model
            )
            
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
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model (runs in thread pool)."""
        return SentenceTransformer(settings.RAG_EMBEDDING_MODEL)
    
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
                    faiss.read_index,
                    str(index_file)
                )
                
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                    
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                await self._create_new_index()
        else:
            await self._create_new_index()
    
    async def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
            
        # Get embedding dimension
        sample_embedding = self.embedding_model.encode(["test"])
        dimension = sample_embedding.shape[1]
        
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    async def add_document_chunks(
        self,
        document_id: int,
        text_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document chunks to the RAG index.
        
        Args:
            document_id: ID of the document
            text_chunks: List of text chunks to add
            metadata: Optional metadata for the document
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._initialized or not self.embedding_model or not self.index:
            logger.warning("RAG service not initialized")
            return False
            
        try:
            # Generate embeddings for chunks
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                text_chunks
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
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
        if not self._initialized or not self.embedding_model or not self.index:
            logger.warning("RAG service not initialized")
            return []
            
        if not self.documents:
            logger.warning("No documents in RAG index")
            return []
            
        try:
            top_k = top_k or settings.RAG_TOP_K
            
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                [query]
            )
            
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)  # Get more to filter
            
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
        if not self.embedding_model or not self.documents:
            await self._create_new_index()
            return
            
        # Extract all text chunks
        text_chunks = [doc['text'] for doc in self.documents]
        
        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            text_chunks
        )
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create new index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save updated index
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
                faiss.write_index,
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
            'embedding_model': settings.RAG_EMBEDDING_MODEL,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def __del__(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global RAG service instance
rag_service = RAGService() 