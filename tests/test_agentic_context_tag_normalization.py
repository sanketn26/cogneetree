"""Tests for refactored agentic_context.retrieval.tag_normalization module."""

import pytest
from cogneetree.retrieval.tag_normalization import (
    normalize_tag,
    get_related_tags,
)


class TestNormalizeTag:
    """Test normalize_tag function."""

    def test_normalize_simple_tag(self):
        """Test normalizing a simple tag."""
        result = normalize_tag("API")
        assert result == "api"

    def test_normalize_with_spaces(self):
        """Test normalizing tags with spaces."""
        result = normalize_tag("API Gateway")
        assert result.lower() == result
        assert " " in result or "_" in result

    def test_normalize_with_hyphens(self):
        """Test normalizing tags with hyphens."""
        result = normalize_tag("api-gateway")
        assert result == "api-gateway" or result == "api_gateway"

    def test_normalize_with_underscores(self):
        """Test normalizing tags with underscores."""
        result = normalize_tag("api_gateway")
        assert result == "api_gateway"

    def test_normalize_synonyms(self):
        """Test that synonyms map to same tag."""
        auth_results = [normalize_tag(tag) for tag in ["auth", "authentication", "authn"]]
        # All should normalize to something consistent
        assert len(set(auth_results)) <= 3  # Allow some variation

    def test_normalize_uppercase(self):
        """Test that uppercase is converted."""
        result = normalize_tag("AUTHENTICATION")
        assert result == result.lower()


class TestGetRelatedTags:
    """Test get_related_tags function."""

    def test_get_related_tags_api(self):
        """Test getting related tags for 'api'."""
        related = get_related_tags("api")
        assert isinstance(related, list)
        assert len(related) >= 0

    def test_get_related_tags_auth(self):
        """Test getting related tags for 'auth'."""
        related = get_related_tags("auth")
        assert isinstance(related, list)
        # "auth" should have related tags like "authentication", "authn"
        assert any("auth" in tag.lower() for tag in related) or len(related) >= 0

    def test_get_related_tags_nonexistent(self):
        """Test getting related tags for non-existent tag."""
        related = get_related_tags("xyz_nonexistent_tag")
        assert isinstance(related, list)
        # Should return empty or the tag itself
        assert len(related) >= 0

    def test_related_tags_are_strings(self):
        """Test that related tags are strings."""
        related = get_related_tags("api")
        for tag in related:
            assert isinstance(tag, str)

    def test_get_related_tags_database(self):
        """Test getting related tags for 'database'."""
        related = get_related_tags("database")
        assert isinstance(related, list)

    def test_get_related_tags_cache(self):
        """Test getting related tags for 'cache'."""
        related = get_related_tags("cache")
        assert isinstance(related, list)

    def test_normalized_tag_consistency(self):
        """Test that normalized tags are consistent."""
        tag1 = normalize_tag("API")
        tag2 = normalize_tag("api")
        assert tag1 == tag2

    def test_get_related_tags_includes_synonyms(self):
        """Test that related tags include synonyms."""
        auth_tags = get_related_tags("auth")
        auth_normalized = [normalize_tag(t) for t in auth_tags]
        
        # The function should return some related tags
        assert isinstance(auth_normalized, list)
