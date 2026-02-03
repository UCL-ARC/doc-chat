# Running with Docker

Docker is an alternative to the [local setup in the main README](../README.md); use this guide for containerized deployment. For non-Docker setup (Python, Node, PostgreSQL on the host), see the [README](../README.md).

## Prerequisites

- Docker
- Docker Compose

No need to install Python, Node, or PostgreSQL locally—they run in containers.

## Setup

1. Create a `.env` file in the root directory (same as local setup). Use the same core variables as in the README:

   ```env
   DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
   RAG_ENABLED=false
   DISABLE_AUTH=true
   ```

   When using Docker Compose's `db` service, `DATABASE_URL` is set automatically by the stack; you only need `.env` if you want to override (e.g. `DISABLE_AUTH`) or add optional API keys for cloud LLMs.


2. **Database:** PostgreSQL is used to store document metadata, parsed text, user settings, and conversation history. With Docker, the `db` service provides PostgreSQL and the app creates tables on first run; no manual database creation is needed unless you use an external database.

## Running the Application

### For standard users (including M-series Macs)

Build and start all services using the standard compose file:

```bash
docker-compose up --build
```

### For users with NVIDIA GPUs

To enable GPU acceleration for Ollama, include the `docker-compose.gpu.yml` override file:

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Accessing the application

Once the containers are running, the API will be available at `http://localhost:8001`.

- **Frontend & Backend API:** [http://localhost:8001](http://localhost:8001)
- **API documentation:** Swagger UI at [http://localhost:8001/docs](http://localhost:8001/docs), ReDoc at [http://localhost:8001/redoc](http://localhost:8001/redoc)
- **Database (if direct access is needed):** `localhost:5432`
- **Ollama API (for local models):** [http://localhost:11434](http://localhost:11434). When using Docker, the app connects to `http://ollama:11434` inside the container network; you access Ollama from the host at `http://localhost:11434`.

## Stopping the application

To stop all services:

```bash
docker-compose down
```

To stop and remove all data (including database):

```bash
docker-compose down -v
```

## Development notes

- The `uploads` directory is mounted as a volume, so uploaded files persist between container restarts.
- The `ollama_data` volume stores downloaded LLMs so they are not re-downloaded on every start.
- Database data is persisted in a Docker volume named `postgres_data`.
- Environment variables use the same names as in the README (`DATABASE_URL`, `RAG_ENABLED`, `DISABLE_AUTH`); configure them in the `.env` file when you need to override compose defaults.

## Troubleshooting

1. **Frontend can't connect to the backend**
   - Check that all containers are running: `docker-compose ps`
   - Check backend logs: `docker-compose logs backend`
   - Ensure the nginx configuration is correct

2. **Backend can't connect to the database**
   - Check database logs: `docker-compose logs db`
   - Ensure database credentials in `.env` match the ones in `docker-compose.yml`

3. **Permission issues with uploads**
   - Ensure the `uploads` directory exists and has correct permissions
   - Try: `mkdir -p uploads && chmod 777 uploads`
