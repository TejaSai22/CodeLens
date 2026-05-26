"""
Embedder module for generating text embeddings using local sentence-transformers.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    """Wrapper for local sentence-transformers embeddings."""

    def __init__(self, model: str = None):
        """
        Initialize Embedder with local model.

        Args:
            model: Sentence-transformers model name
        """
        self.model_name = model or settings.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")

        # Load local sentence-transformers model
        # Common models: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
        self.model = SentenceTransformer(self.model_name)
        self.batch_size = 32  # Batch size for local processing

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            # Encode texts using sentence-transformers
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            print(f"Error embedding texts: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """
        Convenience method for embedding single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.

        Returns:
            Embedding dimension
        """
        # Get dimension from model
        return self.model.get_sentence_embedding_dimension()
