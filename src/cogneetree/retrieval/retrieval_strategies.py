"""Simple retriever - tag-based and optional semantic."""

from typing import List, Dict, Any, Optional
from cogneetree.core.interfaces import ContextStorageABC, EmbeddingModelABC
from cogneetree.core.models import ContextItem
from cogneetree.retrieval.tag_normalization import normalize_tag


class Retriever:
    """Simple retriever - fast and easy to understand."""

    def __init__(
        self,
        storage: ContextStorageABC,
        use_semantic: bool = False,
        embedding_model: Optional[EmbeddingModelABC] = None,
    ):
        self.storage = storage
        self.use_semantic = use_semantic
        self.embedding_model = embedding_model

        if use_semantic and not self.embedding_model:
            try:
                from cogneetree.retrieval.semantic_retrieval import SentenceTransformerModel
                self.embedding_model = SentenceTransformerModel()
            except ImportError:
                # Semantic retrieval dependencies not installed, skipping
                pass

    def retrieve(
        self,
        query_tags: List[str],
        query_description: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context.

        Args:
            query_tags: Tags from current task
            query_description: Task description
            max_results: Max items to return

        Returns:
            List of items with scores
        """
        # Normalize tags
        normalized_tags = [normalize_tag(tag) for tag in query_tags]

        # Get all items matching tags
        items = self.storage.get_items_by_tags(normalized_tags)
        if not items:
            return []

        # Local embedding cache to avoid recomputation within this retrieval call
        embedding_cache: Dict[str, Any] = {}

        # Score items
        scored = self._score_items(items, normalized_tags, query_description, embedding_cache)

        # Sort and limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max_results]

    def _score_items(
        self,
        items: List[ContextItem],
        query_tags: List[str],
        query_description: str,
        embedding_cache: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Score items by relevance. Uses embedding_cache to avoid recomputation."""
        scored = []

        for item in items:
            # Tag matching score (0-1)
            matching_tags = [tag for tag in item.tags if normalize_tag(tag) in query_tags]
            tag_score = len(matching_tags) / max(len(query_tags), 1) if query_tags else 0

            score = tag_score
            score_breakdown = {"tags": tag_score}

            # Semantic score if enabled
            if self.use_semantic and self.embedding_model and query_description:
                semantic_score = self._semantic_score(item, query_description, embedding_cache)
                score = (tag_score + semantic_score) / 2
                score_breakdown["semantic"] = semantic_score

            scored.append(
                {
                    "item": item,
                    "score": score,
                    "breakdown": score_breakdown,
                    "matched_tags": matching_tags,
                }
            )

        return scored

    def _semantic_score(
        self,
        item: ContextItem,
        query_description: str,
        embedding_cache: Dict[str, Any],
    ) -> float:
        """Calculate semantic similarity score with embedding caching."""
        if not self.embedding_model:
            return 0.0

        try:
            import numpy as np

            # Get or compute query embedding (cached by description)
            if "query" not in embedding_cache:
                embedding_cache["query"] = self.embedding_model.encode(query_description)
            query_emb = embedding_cache["query"]

            # Get or compute item embedding (cached by item id)
            item_id = id(item)
            if item_id not in embedding_cache:
                # Combine content and tags, handling empty tags
                tags_str = f" {' '.join(item.tags)}" if item.tags else ""
                item_text = f"{item.content}{tags_str}"
                embedding_cache[item_id] = self.embedding_model.encode(item_text)
            item_emb = embedding_cache[item_id]

            # Cosine similarity
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(item_emb)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = float(np.dot(query_emb, item_emb) / (norm1 * norm2))
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except (ImportError, AttributeError, ValueError):
            # Catch expected errors: missing numpy, encoding issues, API errors
            return 0.0
