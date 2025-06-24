# Document Analysis Web Application

A web application for document analysis. Users can upload PDFs and images, get summaries and ask questions from their documents.

## Features

- User authentication (signup/login)
- File upload support for PDFs and images
- Document summarization
- Question answering based on documents
- Secure password storage
- PostgreSQL database integration

## Tech Stack

- Backend: FastAPI (Python)
- Frontend: React (TypeScript)
- Database: PostgreSQL
- Authentication: JWT
- File Storage: Local filesystem (configurable)

## Prerequisites

- Python 3.9+
- PostgreSQL
- Node.js 16+ (for frontend)
- uv (Python package manager)
- Docker
- Docker Compose

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
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

5. Ensure your PostgreSQL database is running and accessible.

6. **First Run:**
   The database tables will be created automatically on app startup.

7. Run the development server:

   The recommended way to run this application is with Docker Compose (see below).

The API will be available at `http://localhost:8001`

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

2. Pull a model (optional - models are auto-downloaded when first used):
   ```bash
   ollama pull llama3.2:1b
   # or
   ollama pull gemma2:2b
   ```

### Configuration

When running locally, the application will automatically connect to Ollama at `http://localhost:11434`. In Docker, it connects to `http://ollama:11434` within the container network.

If you need to use a different Ollama URL, set the environment variable:
```env
OLLAMA_API_BASE_URL=http://your-ollama-host:11434
```

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

# Running with Docker

This application can be run using Docker and docker-compose, which makes it easy to set up and run on any machine.

## Setup

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google API key (for Gemini)

## Running the Application

### For Standard Users (including M-series Macs)

Build and start all services using the standard compose file:
```bash
docker-compose up --build
```

### For Users with NVIDIA GPUs

To enable GPU acceleration for Ollama, include the `docker-compose.gpu.yml` override file. This will merge the base configuration with the GPU-specific settings.

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Accessing the Application

Once the containers are running, you can access the different parts of the application:
- **Frontend & Backend API:** [`http://localhost:8001`](http://localhost:8001)
- **Database (if direct access is needed):** `localhost:5432`
- **Ollama API (for local models):** [`http://localhost:11434`](http://localhost:11434)

## Stopping the Application

To stop all services:

```bash
docker-compose down
```

To stop and remove all data (including database):

```bash
docker-compose down -v
```

## Development

- The `uploads` directory is mounted as a volume, so uploaded files persist between container restarts
- The `ollama_data` volume stores downloaded LLMs so they are not re-downloaded on every start.
- Database data is persisted in a Docker volume named `postgres_data`
- Environment variables can be configured in the `.env` file

## Troubleshooting

1. If the frontend can't connect to the backend:

   - Check that all containers are running: `docker-compose ps`
   - Check backend logs: `docker-compose logs backend`
   - Ensure the nginx configuration is correct

2. If the backend can't connect to the database:

   - Check database logs: `docker-compose logs db`
   - Ensure database credentials in `.env` match the ones in `docker-compose.yml`

3. For permission issues with uploads:
   - Ensure the `uploads` directory exists and has correct permissions
   - Try: `mkdir -p uploads && chmod 777 uploads`
