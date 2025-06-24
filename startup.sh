#!/bin/bash
set -e

# 1. Build the frontend
cd frontend
npm install
npm run build

# 2. Copy the build output to the backend's static directory
cd ..
ls -l frontend/build

rm -rf src/doc_chat/static
mkdir -p src/doc_chat/static
cp -r frontend/build/* src/doc_chat/static/

# 3. Start FastAPI (Uvicorn) on the port Azure expects
PORT=${PORT:-8001}

# Free the port if already in use (Linux/macOS)
if lsof -i :$PORT; then
  echo "Port $PORT is in use. Killing process..."
  lsof -ti :$PORT | xargs kill -9
fi

uvicorn src.doc_chat.main:app --host 0.0.0.0 --port $PORT