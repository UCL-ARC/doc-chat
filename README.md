# Document Analysis Web Application

A web application for Question Answering over local documents using local Large Language Models (option to use GPT5.2-nano with own API key). Users can upload PDFs and images, get summaries and ask questions from their documents.

## Features

- File upload support for PDFs and images
- Document summarization
- Question answering based on documents
- PostgreSQL database integration

## Prerequisites

- Python 3.9+
- PostgreSQL
- Node.js 16+ and npm (for frontend). Install from [nodejs.org](https://nodejs.org/) or via your package manager (e.g. `brew install node` on macOS).
- uv (Python package manager). Install: `curl -LsSf https://astral.sh/uv/install.sh | sh` (see [uv docs](https://docs.astral.sh/uv/) for other options)
- Ollama

To run with Docker instead of local setup, see the [Docker guide](docs/docker.md).

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd doc-chat
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
RAG_ENABLED=false
DISABLE_AUTH=true
```

5. **Database:** Ensure PostgreSQL is installed and running. Create a database and user for the app, then set `DATABASE_URL` in `.env` to match (e.g. `postgresql+asyncpg://user:password@localhost/dbname`).  
   - **macOS (Homebrew):** `brew install postgresql@16` then `brew services start postgresql@16`. Create DB: `createdb doc_chat` (uses your OS user; use `postgresql+asyncpg://$USER@localhost/doc_chat` if no password).  
   - **Linux:** Install the `postgresql` package for your distro, start the service, then create a DB/user as above.  


6. **First run:** Database tables are created automatically on app startup.

## Local Development with Ollama

For local development, if you want to use local LLMs via Ollama, you'll need to install and run Ollama separately:

### Installing Ollama

1. **macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Windows:**
   Download from [https://ollama.ai/download](https://ollama.ai/download)

### Running Ollama

1. Start the Ollama service:
   ```bash
   ollama serve
   ```
   This will start Ollama on `http://localhost:11434` (the default port the application expects).

2. Pull models (optional - models are auto-downloaded when first used):
   ```bash
   ollama pull gemma3:1b
   ```

### Configuration

When running locally, the application will automatically connect to Ollama at `http://localhost:11434`. When using [Docker](docs/docker.md), it connects to `http://ollama:11434` within the container network.

If you need to use a different Ollama URL, set the environment variable:
```env
OLLAMA_API_BASE_URL=http://your-ollama-host:11434
```

## Running the Application

### Using the startup script

From the project root (with your virtual environment activated and `.env` configured):

```bash
./startup.sh
```

This builds the frontend, copies it into the backend static folder, and starts the API. Default port is 8001; set `PORT` to override.

### Using Docker

For containerized deployment, see **[Docker guide](docs/docker.md)**.

The API will be available at `http://localhost:8001`.

## API Documentation

Once the server is running, you can access:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project uses:

- Ruff for linting
- Black for code formatting
- MyPy for type checking

Run the formatters:

```bash
ruff check .
ruff format .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
