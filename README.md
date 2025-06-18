# Document Analysis Web Application

A modern web application for document analysis, summarization, and FAQ generation. Users can upload PDFs and images, get summaries, ask questions, and generate FAQs from their documents.

## Features

- User authentication (signup/login)
- File upload support for PDFs and images
- Document summarization
- Question answering based on documents
- FAQ generation
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
cd participatory-ai-for-workshops
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

```bash
uvicorn src.participatory_ai_for_workshops.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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

1. Build and start all services:

   ```bash
   docker-compose up --build
   ```

2. Access the application:
   - Frontend: <http://localhost>
   - Backend API: <http://localhost:8000>
   - Database: localhost:5432 (if you need direct access)

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
