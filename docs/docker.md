# Running with Docker

This application can be run using Docker and Docker Compose, which makes it easy to set up and run on any machine.

## Prerequisites

- Docker
- Docker Compose

## Setup

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:

   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google API key (for Gemini)

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

Once the containers are running:

- **Frontend & Backend API:** [http://localhost:8001](http://localhost:8001)
- **Database (if direct access is needed):** `localhost:5432`
- **Ollama API (for local models):** [http://localhost:11434](http://localhost:11434)

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
- Environment variables can be configured in the `.env` file.

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
