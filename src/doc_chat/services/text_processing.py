"""Text processing utilities for RAG."""
import re
from typing import List, Optional
from ..config import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    preserve_sentences: bool = True
) -> List[str]:
    """Split text into chunks for RAG processing.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk (defaults to config value)
        chunk_overlap: Overlap between chunks (defaults to config value)
        preserve_sentences: Whether to try to preserve sentence boundaries
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
    
    if not text.strip():
        return []
    
    # Clean the text
    text = clean_text(text)
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    if preserve_sentences:
        chunks = _chunk_by_sentences(text, chunk_size, chunk_overlap)
    else:
        chunks = _chunk_by_characters(text, chunk_size, chunk_overlap)
    
    # Filter out very short chunks
    min_chunk_size = max(50, chunk_size // 10)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]
    
    return chunks


def clean_text(text: str) -> str:
    """Clean text for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    return text.strip()


def _chunk_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by sentences while respecting size limits.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = _split_into_sentences(text)
    
    if not sentences:
        return _chunk_by_characters(text, chunk_size, chunk_overlap)
    
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # If adding this sentence would exceed chunk size
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            # Save current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if chunk_overlap > 0 and chunks:
                overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        i += 1
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _chunk_by_characters(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by characters.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at word boundary
        if end < len(text):
            # Look for last space within the chunk
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - chunk_overlap)
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting - can be improved with more sophisticated methods
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences and very short ones
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """Get overlap text from the end of a chunk.
    
    Args:
        text: Source text
        overlap_size: Size of overlap
        
    Returns:
        Overlap text
    """
    if len(text) <= overlap_size:
        return text
    
    # Try to break at word boundary
    start_pos = len(text) - overlap_size
    space_pos = text.find(' ', start_pos)
    
    if space_pos != -1:
        return text[space_pos + 1:]
    else:
        return text[start_pos:] 