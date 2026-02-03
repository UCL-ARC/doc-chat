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
- Node.js 16+ and npm (for frontend)
- uv (Python package manager)
- Ollama

Install steps for all platforms (macOS, Linux, Windows) are in **Setup** below. To run with Docker instead of local setup, see the [Docker guide](docs/docker.md).

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

6. **Database:** PostgreSQL is used to store document metadata, parsed text, user settings (e.g. selected model, prompts), and conversation history. Ensure PostgreSQL is installed and running, then create a database (and optionally a user/password) for the app. The app creates **tables** on first run; it does **not** create the database or user.

   **Install & start PostgreSQL**
   - **macOS (Homebrew):** `brew install postgresql@16` then `brew services start postgresql@16`
   - **Linux:** Install the `postgresql` (e.g. `sudo apt install postgresql`) package for your distro and start the service
   - **Windows:** Download the installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/) and run it. During setup, set a password for the `postgres` superuser. The PostgreSQL service usually starts automatically; you can manage it in *Services* (e.g. `services.msc`) or via *pgAdmin*.

   **Create a database and user**
   - **macOS (Homebrew):** Usually your OS user already exists as a role. Run:
     ```bash
     createdb doc_chat
     ```
     In `.env` use: `DATABASE_URL=postgresql+asyncpg://YOUR_OS_USERNAME@localhost/doc_chat` (no password; replace `YOUR_OS_USERNAME` with your username).
   - **Linux:** The PostgreSQL role matching your OS user (e.g. `ubuntu`) often does not exist. Create it first, then the database:
     ```bash
     sudo -u postgres createuser -s $USER
     createdb doc_chat
     ```
     The app connects over TCP and requires a password. Set one for your user, then use it in `.env`:
     ```bash
     sudo -u postgres psql -c "ALTER USER $USER WITH PASSWORD 'dev';"
     ```
     In `.env` use your **actual username** (e.g. `ubuntu`). The `.env` file is not processed by the shell, so `$USER` will not expand—write the name explicitly:
     `DATABASE_URL=postgresql+asyncpg://ubuntu:dev@localhost/doc_chat`
   - **Windows:** Open *SQL Shell (psql)* or a terminal where `psql` is on PATH (e.g. `C:\Program Files\PostgreSQL\16\bin`). Connect as `postgres` (use the password you set during install), then create a dedicated user and database:
     ```sql
     CREATE USER doc_chat_user WITH PASSWORD 'your_password';
     CREATE DATABASE doc_chat OWNER doc_chat_user;
     ```
     In `.env` use: `DATABASE_URL=postgresql+asyncpg://doc_chat_user:your_password@localhost/doc_chat`

7. **First run:** Start the app (see *Running the Application*). Database **tables** are created automatically on first startup; the database and user must already exist (step 6).

## Local LLMs with Ollama

If you want to use local LLMs via Ollama, you'll need to install and run Ollama separately:

### Installing Ollama

- **macOS/Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows:** Download and run the installer from [ollama.ai/download](https://ollama.ai/download)

### Running Ollama

- **macOS/Linux:** In a terminal, run `ollama serve`. This starts Ollama on `http://localhost:11434` (the default port the application expects).
- **Windows:** Ollama may start from the Start menu or system tray after install. To run from a terminal, open PowerShell or CMD and run `ollama serve`.

### Pulling models (optional)

Models are auto-downloaded when first used. To pull in advance, run in a terminal (PowerShell or CMD on Windows):

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

- **macOS/Linux:** `./startup.sh`
- **Windows:** Use WSL (Windows Subsystem for Linux) and run `./startup.sh` there, or run the steps manually: `cd frontend`, `npm install`, `npm run build`, then from the repo root copy `frontend/build` into `src/doc_chat/static`, then `uvicorn src.doc_chat.main:app --host 0.0.0.0 --port 8001`.

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
