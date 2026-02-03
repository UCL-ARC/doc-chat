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

2. **Install uv and Node.js** (if not already installed):

   **uv (Python package manager):**
   - **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh` then restart your shell or `source $HOME/.local/bin/env` (or add it to your PATH).
   - **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
   - Other options (pip, Homebrew, etc.): [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

   **Node.js and npm (for frontend):**
   - **macOS:** `brew install node` or install from [nodejs.org](https://nodejs.org/).
   - **Linux:** Use your distro’s package manager (e.g. `sudo apt install nodejs npm` on Debian/Ubuntu) or [nodejs.org](https://nodejs.org/).
   - **Windows:** Download the LTS installer from [nodejs.org](https://nodejs.org/).

   Check versions: `uv --version` and `node --version` (Node 16+).

3. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:

```bash
uv pip install -r requirements.txt
```

5. Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
RAG_ENABLED=false
DISABLE_AUTH=true
```

6. **Database:** Ensure PostgreSQL is installed and running, then create a database (and optionally a user/password) for the app. The app creates **tables** on first run; it does **not** create the database or user.

   **Install & start PostgreSQL**
   - **macOS (Homebrew):** `brew install postgresql@16` then `brew services start postgresql@16`
   - **Linux:** Install the `postgresql` (and optionally `postgresql-client`) package for your distro and start the service

   **Option A – Simple (OS user, no password)**  
   Create a database owned by your current OS user (good for local dev):
   ```bash
   createdb doc_chat
   ```
   In `.env` use: `DATABASE_URL=postgresql+asyncpg://YOUR_OS_USERNAME@localhost/doc_chat` (no password; replace `YOUR_OS_USERNAME` with your username, or `$USER` on macOS/Linux).

   **Option B – Dedicated user and password**  
   Create a role and database (e.g. for a shared or production-like setup):
   ```bash
   psql -U postgres -c "CREATE USER doc_chat_user WITH PASSWORD 'your_password';"
   psql -U postgres -c "CREATE DATABASE doc_chat OWNER doc_chat_user;"
   ```
   In `.env` use: `DATABASE_URL=postgresql+asyncpg://doc_chat_user:your_password@localhost/doc_chat`

7. **First run:** Start the app (see *Running the Application*). Database **tables** are created automatically on first startup; the database and user must already exist (step 6).

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
