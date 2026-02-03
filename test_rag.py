#!/usr/bin/env python3
"""Simple test script for RAG functionality."""
import asyncio
import os
import sys

# Set environment variables for testing BEFORE importing any modules
# Use Ollama so no Hugging Face token is required (run: ollama pull nomic-embed-text)
os.environ['RAG_ENABLED'] = 'true'
os.environ['RAG_EMBEDDING_MODEL'] = 'ollama/nomic-embed-text'
os.environ['FAISS_INDEX_PATH'] = './test_data/faiss_index'

# Add src to path for imports
sys.path.insert(0, 'src')

# Clear the settings cache to ensure new environment variables are loaded
from doc_chat.config import get_settings
get_settings.cache_clear()

from doc_chat.services.rag_service import rag_service
from doc_chat.services.text_processing import chunk_text


async def test_rag():
    """Test RAG functionality."""
    print("Testing RAG functionality...")
    
    # Initialize RAG service
    print("1. Initializing RAG service...")
    success = await rag_service.initialize()
    if not success:
        print("❌ Failed to initialize RAG service")
        return False
    print("✅ RAG service initialized")
    
    # Test text chunking
    print("\n2. Testing text chunking...")
    sample_text = """
    This is a sample document about artificial intelligence.
    AI has many applications in modern technology.
    Machine learning is a subset of AI that focuses on algorithms.
    Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand human language.
    """
    
    chunks = chunk_text(sample_text)
    print(f"✅ Created {len(chunks)} chunks from sample text")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {chunk[:50]}...")
    
    # Test adding document chunks
    print("\n3. Testing document indexing...")
    doc_id = 1
    metadata = {'filename': 'test_document.txt', 'type': 'text'}
    
    success = await rag_service.add_document_chunks(
        document_id=doc_id,
        text_chunks=chunks,
        metadata=metadata
    )
    
    if not success:
        print("❌ Failed to add document chunks")
        return False
    print("✅ Document chunks added to index")
    
    # Test searching
    print("\n4. Testing similarity search...")
    query = "What is machine learning?"
    results = await rag_service.search_similar_chunks(query, top_k=3)
    
    print(f"✅ Found {len(results)} similar chunks for query: '{query}'")
    for i, result in enumerate(results):
        print(f"   Result {i+1} (score: {result['score']:.3f}): {result['text'][:50]}...")
    
    # Test statistics
    print("\n5. Testing service statistics...")
    stats = await rag_service.get_stats()
    print("✅ RAG service statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test document removal
    print("\n6. Testing document removal...")
    success = await rag_service.remove_document(doc_id)
    if not success:
        print("❌ Failed to remove document")
        return False
    print("✅ Document removed from index")
    
    # Verify removal
    final_stats = await rag_service.get_stats()
    print(f"✅ Final chunk count: {final_stats['total_chunks']}")
    
    print("\n🎉 All RAG tests passed!")
    return True


async def test_rag_disabled():
    """Test RAG with disabled configuration."""
    print("\nTesting RAG with disabled configuration...")
    
    # Disable RAG
    os.environ['RAG_ENABLED'] = 'false'
    
    # Clear cache and reload settings
    get_settings.cache_clear()
    
    # Try to initialize
    from doc_chat.services.rag_service import RAGService
    disabled_service = RAGService()
    success = await disabled_service.initialize()
    
    if success:
        print("❌ RAG service should not initialize when disabled")
        return False
    
    print("✅ RAG service correctly disabled")
    return True


if __name__ == "__main__":
    async def main():
        """Run all tests."""
        print("🚀 Starting RAG tests...\n")
        
        # Test enabled RAG
        success1 = await test_rag()
        
        # Test disabled RAG
        success2 = await test_rag_disabled()
        
        if success1 and success2:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    
    # Run the tests
    asyncio.run(main()) 