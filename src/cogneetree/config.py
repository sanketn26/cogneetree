"""Simple configuration - just defaults."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration with simple defaults."""

    # Retrieval settings
    use_semantic: bool = False  # Use embeddings in retrieval?
    max_results: int = 5  # Max items to retrieve
    min_score: float = 0.0  # Minimum relevance score

    # Embedding model (if semantic enabled)
    embedding_model: str = "microsoft/codebert-base"
    openai_api_key: Optional[str] = None

    # Storage settings (use ContextManager.with_storage() for custom backends)
    storage_path: Optional[str] = None  # Optional: file path for future storage backends

    # Cascading propagation settings
    propagation_threshold: float = 0.3  # Minimum relevance score for a decision/learning to propagate

    @classmethod
    def default(cls) -> "Config":
        """Get default config."""
        return cls()

    @classmethod
    def semantic(cls) -> "Config":
        """Get config with semantic retrieval enabled."""
        return cls(use_semantic=True)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create from dict."""
        return cls(
            use_semantic=data.get("use_semantic", False),
            max_results=data.get("max_results", 5),
            min_score=data.get("min_score", 0.0),
            embedding_model=data.get("embedding_model", "microsoft/codebert-base"),
            openai_api_key=data.get("openai_api_key"),
            storage_path=data.get("storage_path"),
            propagation_threshold=data.get("propagation_threshold", 0.3),
        )
