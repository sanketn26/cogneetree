"""Simple retriever - tag-based and optional semantic."""

from typing import List, Dict, Any, Optional
from cogneetree.core.context_storage import ContextStorage, ContextItem
from cogneetree.retrieval.tag_normalization import normalize_tag


class Retriever:
    """Simple retriever - fast and easy to understand."""

    def __init__(self, storage: ContextStorage, use_semantic: bool = False):
        self.storage = storage
        self.use_semantic = use_semantic
        self.embedding_model = None

        if use_semantic:
            try:
                from cogneetree.retrieval.semantic_retrieval import EmbeddingModel
                self.embedding_model = EmbeddingModel()
            except (ImportError, Exception):
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

        # Score items
        scored = self._score_items(items, normalized_tags, query_description)

        # Sort and limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max_results]

    def _score_items(
        self, items: List[ContextItem], query_tags: List[str], query_description: str
    ) -> List[Dict[str, Any]]:
        """Score items by relevance."""
        scored = []

        for item in items:
            # Tag matching score (0-1)
            matching_tags = [tag for tag in item.tags if normalize_tag(tag) in query_tags]
            tag_score = len(matching_tags) / max(len(query_tags), 1) if query_tags else 0

            score = tag_score
            score_breakdown = {"tags": tag_score}

            # Semantic score if enabled
            if self.use_semantic and self.embedding_model and query_description:
                semantic_score = self._semantic_score(item, query_description)
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

    def _semantic_score(self, item: ContextItem, query_description: str) -> float:
        """Calculate semantic similarity score."""
        if not self.embedding_model:
            return 0.0

        try:
            query_emb = self.embedding_model.encode(query_description)
            item_text = f"{item.content} {' '.join(item.tags)}"
            item_emb = self.embedding_model.encode(item_text)

            # Cosine similarity
            import numpy as np
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(item_emb)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = float(np.dot(query_emb, item_emb) / (norm1 * norm2))
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception:
            return 0.0
