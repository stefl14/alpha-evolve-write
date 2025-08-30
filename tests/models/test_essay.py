"""Tests for Essay model."""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from alpha_evolve_essay.models import Essay


class TestEssayModel:
    """Test cases for Essay model validation and behavior."""

    def test_essay_creation_with_minimal_fields_succeeds(self):
        """Test that Essay can be created with just content and prompt."""
        essay = Essay(
            content="This is a test essay about AI.",
            prompt="Write about AI"
        )
        
        assert essay.content == "This is a test essay about AI."
        assert essay.prompt == "Write about AI"
        assert isinstance(essay.id, UUID)
        assert isinstance(essay.created_at, datetime)
        assert essay.created_at.tzinfo == timezone.utc
        assert essay.version == 1
        assert essay.parent_id is None
        assert essay.metadata == {}
        assert essay.word_count == 7  # "This is a test essay about AI"

    def test_essay_creation_with_all_fields_succeeds(self):
        """Test that Essay accepts all optional fields correctly."""
        metadata = {"mode": "creative", "temperature": 0.8}
        parent_uuid = UUID("12345678-1234-5678-9012-123456789012")
        
        essay = Essay(
            content="Detailed essay content here.",
            prompt="Write a creative essay",
            version=3,
            parent_id=parent_uuid,
            metadata=metadata
        )
        
        assert essay.content == "Detailed essay content here."
        assert essay.prompt == "Write a creative essay"
        assert essay.version == 3
        assert essay.parent_id == parent_uuid
        assert essay.metadata == metadata
        assert essay.word_count == 4  # "Detailed essay content here"

    def test_essay_word_count_calculation_handles_extra_whitespace(self):
        """Test word count calculation with content that has extra whitespace."""
        # Content with extra whitespace should still count words correctly
        essay = Essay(content="  hello  world  ", prompt="Test prompt")
        assert essay.word_count == 2
        
        essay = Essay(content="\n\t hello   \n\t world  \n", prompt="Test prompt")
        assert essay.word_count == 2

    def test_essay_word_count_calculation_handles_punctuation(self):
        """Test word count calculation with punctuation and special characters."""
        essay = Essay(
            content="Hello, world! This is a test. How are you?",
            prompt="Test prompt"
        )
        # Should count: Hello, world, This, is, a, test, How, are, you = 9 words
        assert essay.word_count == 9

    def test_essay_content_validation_rejects_empty_string(self):
        """Test that Essay rejects empty content string."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            Essay(content="", prompt="Test prompt")

    def test_essay_prompt_validation_rejects_empty_string(self):
        """Test that Essay rejects empty prompt string."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            Essay(content="Valid content", prompt="")

    def test_essay_version_validation_rejects_zero_or_negative(self):
        """Test that Essay version must be positive integer."""
        with pytest.raises(ValueError, match="Version must be positive"):
            Essay(content="Valid content", prompt="Valid prompt", version=0)
            
        with pytest.raises(ValueError, match="Version must be positive"):
            Essay(content="Valid content", prompt="Valid prompt", version=-1)

    def test_essay_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to empty dictionary when not provided."""
        essay = Essay(content="Content", prompt="Prompt")
        assert essay.metadata == {}
        assert isinstance(essay.metadata, dict)

    def test_essay_id_is_unique_across_instances(self):
        """Test that each Essay instance gets a unique ID."""
        essay1 = Essay(content="Content 1", prompt="Prompt 1")
        essay2 = Essay(content="Content 2", prompt="Prompt 2")
        
        assert essay1.id != essay2.id
        assert isinstance(essay1.id, UUID)
        assert isinstance(essay2.id, UUID)

    def test_essay_created_at_is_utc_timezone(self):
        """Test that created_at timestamp is in UTC timezone."""
        essay = Essay(content="Content", prompt="Prompt")
        assert essay.created_at.tzinfo == timezone.utc

    def test_essay_equality_comparison_works_correctly(self):
        """Test that Essay instances can be compared for equality."""
        essay1 = Essay(content="Same content", prompt="Same prompt")
        essay2 = Essay(content="Same content", prompt="Same prompt")
        essay3 = Essay(content="Different content", prompt="Same prompt")
        
        # Same content should be equal (ignoring auto-generated fields)
        assert essay1.content == essay2.content
        assert essay1.prompt == essay2.prompt
        # But different instances have different IDs
        assert essay1.id != essay2.id
        
        # Different content should not be equal
        assert essay1.content != essay3.content

    def test_essay_json_serialization_works(self):
        """Test that Essay can be serialized to and from JSON."""
        original = Essay(
            content="Test content",
            prompt="Test prompt",
            version=2,
            metadata={"key": "value"}
        )
        
        # Serialize to dict
        data = original.model_dump()
        assert data["content"] == "Test content"
        assert data["prompt"] == "Test prompt"
        assert data["version"] == 2
        assert data["metadata"] == {"key": "value"}
        assert "id" in data
        assert "created_at" in data
        assert "word_count" in data
        
        # Deserialize from dict
        restored = Essay(**data)
        assert restored.content == original.content
        assert restored.prompt == original.prompt
        assert restored.version == original.version
        assert restored.metadata == original.metadata
        assert restored.id == original.id