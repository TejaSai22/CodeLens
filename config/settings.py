"""
Configuration settings for CodeLens application.
Loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


class Settings:
    """Application settings and configuration."""

    # Cloud Inference Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Optional API key for the backend. When empty, auth is disabled (local/dev).
    # When set, clients must send a matching `X-API-Key` header.
    API_KEY = os.getenv("API_KEY", "")

    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Local sentence-transformers model
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # Gemini model

    # ChromaDB Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # Chunking Configuration
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval Configuration
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # CORS Configuration (comma-separated list of allowed frontend origins)
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://localhost:8501,http://localhost:8080",
        ).split(",")
        if origin.strip()
    ]

    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY is not set in the environment variables.")

    @classmethod
    def get_config_summary(cls):
        """Get a summary of current configuration."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "chroma_persist_dir": cls.CHROMA_PERSIST_DIR,
            "max_chunk_size": cls.MAX_CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "retrieval_top_k": cls.RETRIEVAL_TOP_K,
            "cors_origins": cls.CORS_ORIGINS
        }


# Create singleton instance
settings = Settings()
