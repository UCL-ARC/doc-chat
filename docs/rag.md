# RAG (Retrieval-Augmented Generation)

This application includes optional RAG functionality using FAISS for improved document Q&A. RAG enhances responses by finding the most relevant document chunks for each question, rather than using entire documents as context.

## Benefits of RAG

- **Better relevance**: Only the most relevant document sections are used as context
- **Improved accuracy**: Reduces noise from irrelevant document content
- **Faster responses**: Smaller context windows lead to faster LLM processing
- **Better handling of large documents**: Can work with documents that exceed LLM context limits

## Configuration

RAG is disabled by default. To enable it, set the following environment variables:

```bash
# Enable RAG
RAG_ENABLED=true

# Embedding model: Ollama (no HF token) or SentenceTransformers
# Default: ollama/nomic-embed-text (pull with: ollama pull nomic-embed-text)
# Alternative: sentence-transformers/all-MiniLM-L6-v2 (requires Hugging Face)
RAG_EMBEDDING_MODEL=ollama/nomic-embed-text

# Text chunking settings
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50

# Number of similar chunks to retrieve for each query
RAG_TOP_K=5

# Path to store FAISS index
FAISS_INDEX_PATH=./data/faiss_index
```

## Docker Configuration

RAG is pre-configured in `docker-compose.yml` to use **Ollama for embeddings** (no Hugging Face token). The service will automatically:

1. Use Ollama's embedding API (ensure `ollama pull nomic-embed-text` if using the default model)
2. Create text chunks from uploaded documents
3. Build and maintain a FAISS vector index
4. Use semantic search for Q&A queries

## Testing RAG

Run the test script to verify RAG functionality:

```bash
python test_rag.py
```

## RAG Status Endpoint

Check RAG service status via the API:

```bash
curl http://localhost:8001/documents/rag/status
```

Response includes:

- Whether RAG is enabled
- Initialization status
- Number of indexed chunks
- Number of unique documents
- Embedding model in use

## Disabling RAG

To disable RAG, set `RAG_ENABLED=false` or remove the environment variable. The application will fall back to using complete document text for Q&A, which is the original behavior.
