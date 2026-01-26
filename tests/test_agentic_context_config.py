"""Tests for Config class."""

import pytest
from cogneetree.config import Config


class TestConfig:
    """Test Config dataclass."""

    def test_config_defaults(self):
        """Test default Config values."""
        config = Config()
        assert config.use_semantic is False
        assert config.max_results == 5
        assert config.min_score == 0.0

    def test_config_custom_init(self):
        """Test Config with custom values."""
        config = Config(use_semantic=True, max_results=10, min_score=0.3)
        assert config.use_semantic is True
        assert config.max_results == 10
        assert config.min_score == 0.3

    def test_config_default_classmethod(self):
        """Test Config.default() class method."""
        config = Config.default()
        assert isinstance(config, Config)
        assert config.use_semantic is False

    def test_config_semantic_classmethod(self):
        """Test Config.semantic() class method."""
        config = Config.semantic()
        assert config.use_semantic is True

    def test_config_from_dict_full(self):
        """Test Config.from_dict with all parameters."""
        data = {
            "use_semantic": True,
            "max_results": 20,
            "min_score": 0.5,
            "embedding_model": "model-v2"
        }
        config = Config.from_dict(data)
        assert config.use_semantic is True
        assert config.max_results == 20

    def test_config_from_dict_partial(self):
        """Test Config.from_dict with partial parameters."""
        config = Config.from_dict({"max_results": 15})
        assert config.max_results == 15
        assert config.use_semantic is False

    def test_config_from_dict_empty(self):
        """Test Config.from_dict with empty dict."""
        config = Config.from_dict({})
        assert config.max_results == 5
