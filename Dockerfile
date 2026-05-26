# CodeLens backend (FastAPI + ChromaDB + local embeddings + Gemini).
FROM python:3.13-slim

# git is required by gitpython to clone repositories for indexing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv installer for fast, reproducible dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install Python dependencies first (cached unless requirements change).
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-bake the default embedding model into the image so the container boots
# fast and works without network access to the model hub at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code.
COPY src ./src
COPY config ./config

# Writable dirs for the vector store / registry and logs.
RUN mkdir -p data logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
