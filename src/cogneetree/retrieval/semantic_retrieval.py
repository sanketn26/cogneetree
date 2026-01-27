"""Embedding models for semantic retrieval."""

from typing import Any, Union, Optional
import numpy as np
import logging
from cogneetree.core.interfaces import EmbeddingModelABC

logger = logging.getLogger(__name__)


class SentenceTransformerModel(EmbeddingModelABC):
    """Wrapper for sentence-transformers model."""

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = "cpu"):
        """Initialize embedding model."""
        self.model_name = model_name
        self.device = device
        self.model: Any = None
        self._load()

    def _load(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic retrieval will fail.")
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    def encode(self, text: str) -> Any:
        """Get embedding for text."""
        if not text or not text.strip():
            return np.zeros(768)
        if self.model:
            return self.model.encode(text, convert_to_tensor=False)
        return np.zeros(768)

    def similarity(self, emb1: Any, emb2: Any) -> float:
        """Cosine similarity between two embeddings."""
        emb1 = np.array(emb1) if not isinstance(emb1, np.ndarray) else emb1
        emb2 = np.array(emb2) if not isinstance(emb2, np.ndarray) else emb2
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class OpenAIEmbeddingModel(EmbeddingModelABC):
    """Wrapper for OpenAI embeddings."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI support requires: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode(self, text: str) -> Any:
        if not text or not text.strip():
            return []
        
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def similarity(self, emb1: Any, emb2: Any) -> float:
        # Use numpy for cosine similarity
        import numpy as np
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


# Check for backward compatibility with old class name
EmbeddingModel = SentenceTransformerModel
