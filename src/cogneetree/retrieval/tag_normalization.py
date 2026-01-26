"""Tag normalization - simple lowercase and strip."""


SYNONYMS = {
    "auth": ["authentication", "jwt", "bearer-token"],
    "kafka": ["kafka-client", "message-queue", "messaging"],
    "error": ["error-handling", "exception", "fault-tolerance"],
    "log": ["logging", "audit", "tracing"],
}


def normalize_tag(tag: str) -> str:
    """Normalize tag to lowercase canonical form."""
    if not tag:
        return ""
    
    normalized = tag.lower().strip()
    
    # Check if it's a synonym
    for canonical, synonyms in SYNONYMS.items():
        if normalized in synonyms:
            return canonical
    
    return normalized


def get_related_tags(tag: str) -> list:
    """Get related tags."""
    normalized = normalize_tag(tag)
    return SYNONYMS.get(normalized, [])
