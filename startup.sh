#!/bin/bash
set -e

# 1. Build the frontend
cd frontend
npm install
npm run build

# 2. Copy the build output to the backend's static directory
cd ..
ls -l frontend/build

rm -rf src/participatory_ai_for_workshops/static
mkdir -p src/participatory_ai_for_workshops/static
cp -r frontend/build/* src/participatory_ai_for_workshops/static/

# 3. Start FastAPI (Uvicorn) on the port Azure expects
PORT=${PORT:-8000}

# Free the port if already in use (Linux/macOS)
if lsof -i :$PORT; then
  echo "Port $PORT is in use. Killing process..."
  lsof -ti :$PORT | xargs kill -9
fi

uvicorn src.participatory_ai_for_workshops.main:app --host 0.0.0.0 --port $PORT