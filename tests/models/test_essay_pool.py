"""Tests for EssayPool model."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from alpha_evolve_essay.models import Essay, EssayPool, EvaluationResult


class TestEssayPoolModel:
    """Test cases for EssayPool model validation and behavior."""

    def test_essay_pool_creation_with_minimal_fields_succeeds(self):
        """Test that EssayPool can be created with required fields."""
        pool = EssayPool(name="Test Pool")

        assert pool.name == "Test Pool"
        assert isinstance(pool.id, UUID)
        assert isinstance(pool.created_at, datetime)
        assert pool.created_at.tzinfo == UTC
        assert pool.essays == []
        assert pool.generation_count == 0
        assert pool.metadata == {}

    def test_essay_pool_creation_with_all_fields_succeeds(self):
        """Test that EssayPool accepts all optional fields."""
        essay1 = Essay(content="First essay", prompt="Test prompt 1")
        essay2 = Essay(content="Second essay", prompt="Test prompt 2")
        metadata = {"mode": "creative", "max_generations": 5}

        pool = EssayPool(
            name="Creative Pool",
            essays=[essay1, essay2],
            generation_count=3,
            metadata=metadata,
        )

        assert pool.name == "Creative Pool"
        assert len(pool.essays) == 2
        assert pool.essays[0] == essay1
        assert pool.essays[1] == essay2
        assert pool.generation_count == 3
        assert pool.metadata == metadata

    def test_essay_pool_name_validation_rejects_empty_string(self):
        """Test that pool name cannot be empty."""
        with pytest.raises(ValueError, match="Pool name cannot be empty"):
            EssayPool(name="")

        with pytest.raises(ValueError, match="Pool name cannot be empty"):
            EssayPool(name="   ")

    def test_essay_pool_generation_count_validation_accepts_non_negative(self):
        """Test that generation_count accepts non-negative values."""
        pool_zero = EssayPool(name="Test", generation_count=0)
        assert pool_zero.generation_count == 0

        pool_positive = EssayPool(name="Test", generation_count=5)
        assert pool_positive.generation_count == 5

    def test_essay_pool_generation_count_validation_rejects_negative(self):
        """Test that generation_count rejects negative values."""
        with pytest.raises(ValueError, match="Generation count cannot be negative"):
            EssayPool(name="Test", generation_count=-1)

    def test_essay_pool_add_essay_method(self):
        """Test adding essays to the pool."""
        pool = EssayPool(name="Test Pool")
        essay1 = Essay(content="First essay", prompt="Test prompt 1")
        essay2 = Essay(content="Second essay", prompt="Test prompt 2")

        # Add first essay
        pool.add_essay(essay1)
        assert len(pool.essays) == 1
        assert pool.essays[0] == essay1

        # Add second essay
        pool.add_essay(essay2)
        assert len(pool.essays) == 2
        assert pool.essays[1] == essay2

    def test_essay_pool_add_essay_prevents_duplicates(self):
        """Test that adding the same essay twice doesn't create duplicates."""
        pool = EssayPool(name="Test Pool")
        essay = Essay(content="Test essay", prompt="Test prompt")

        pool.add_essay(essay)
        pool.add_essay(essay)  # Add same essay again

        assert len(pool.essays) == 1
        assert pool.essays[0] == essay

    def test_essay_pool_remove_essay_method(self):
        """Test removing essays from the pool."""
        essay1 = Essay(content="First essay", prompt="Test prompt 1")
        essay2 = Essay(content="Second essay", prompt="Test prompt 2")
        pool = EssayPool(name="Test Pool", essays=[essay1, essay2])

        # Remove first essay
        removed = pool.remove_essay(essay1.id)
        assert removed == essay1
        assert len(pool.essays) == 1
        assert pool.essays[0] == essay2

        # Remove second essay
        removed = pool.remove_essay(essay2.id)
        assert removed == essay2
        assert len(pool.essays) == 0

    def test_essay_pool_remove_essay_nonexistent_returns_none(self):
        """Test removing non-existent essay returns None."""
        pool = EssayPool(name="Test Pool")
        non_existent_id = uuid4()

        removed = pool.remove_essay(non_existent_id)
        assert removed is None

    def test_essay_pool_get_essay_by_id_method(self):
        """Test retrieving essay by ID."""
        essay1 = Essay(content="First essay", prompt="Test prompt 1")
        essay2 = Essay(content="Second essay", prompt="Test prompt 2")
        pool = EssayPool(name="Test Pool", essays=[essay1, essay2])

        # Get existing essay
        retrieved = pool.get_essay(essay1.id)
        assert retrieved == essay1

        # Get non-existent essay
        non_existent_id = uuid4()
        retrieved = pool.get_essay(non_existent_id)
        assert retrieved is None

    def test_essay_pool_get_top_essays_method(self):
        """Test getting top essays based on evaluation scores."""
        # Create essays with different scores
        essay1 = Essay(content="Essay 1", prompt="Test")
        essay2 = Essay(content="Essay 2", prompt="Test")
        essay3 = Essay(content="Essay 3", prompt="Test")

        # Create evaluation results for reference (in real implementation would be stored)
        _eval1 = EvaluationResult(essay_id=essay1.id, overall_score=85.0)
        _eval2 = EvaluationResult(essay_id=essay2.id, overall_score=92.0)
        _eval3 = EvaluationResult(essay_id=essay3.id, overall_score=78.0)

        pool = EssayPool(name="Test Pool", essays=[essay1, essay2, essay3])

        # Mock evaluation results (in real implementation, these would be stored/retrieved)
        evaluation_scores = {
            essay1.id: 85.0,
            essay2.id: 92.0,
            essay3.id: 78.0,
        }

        # Get top 2 essays - should return essay2 (92.0) and essay1 (85.0)
        top_essays = pool.get_top_essays(2, evaluation_scores)
        assert len(top_essays) == 2
        assert top_essays[0] == essay2  # Highest score
        assert top_essays[1] == essay1  # Second highest

    def test_essay_pool_get_top_essays_with_no_evaluations(self):
        """Test getting top essays when no evaluation scores available."""
        essay1 = Essay(content="Essay 1", prompt="Test")
        essay2 = Essay(content="Essay 2", prompt="Test")
        pool = EssayPool(name="Test Pool", essays=[essay1, essay2])

        # With no evaluation scores, should return essays in original order
        top_essays = pool.get_top_essays(1, {})
        assert len(top_essays) == 1
        assert top_essays[0] == essay1  # First essay in pool

    def test_essay_pool_get_top_essays_limit_exceeds_pool_size(self):
        """Test getting top essays when limit exceeds number of essays."""
        essay1 = Essay(content="Essay 1", prompt="Test")
        pool = EssayPool(name="Test Pool", essays=[essay1])

        # Request more essays than available
        top_essays = pool.get_top_essays(5, {essay1.id: 85.0})
        assert len(top_essays) == 1  # Should return only available essay
        assert top_essays[0] == essay1

    def test_essay_pool_clear_method(self):
        """Test clearing all essays from pool."""
        essay1 = Essay(content="Essay 1", prompt="Test")
        essay2 = Essay(content="Essay 2", prompt="Test")
        pool = EssayPool(name="Test Pool", essays=[essay1, essay2])

        pool.clear()
        assert len(pool.essays) == 0

    def test_essay_pool_size_property(self):
        """Test pool size property."""
        pool = EssayPool(name="Test Pool")
        assert pool.size == 0

        essay1 = Essay(content="Essay 1", prompt="Test")
        essay2 = Essay(content="Essay 2", prompt="Test")
        pool.add_essay(essay1)
        pool.add_essay(essay2)

        assert pool.size == 2

    def test_essay_pool_is_empty_property(self):
        """Test pool is_empty property."""
        pool = EssayPool(name="Test Pool")
        assert pool.is_empty is True

        essay = Essay(content="Essay", prompt="Test")
        pool.add_essay(essay)
        assert pool.is_empty is False

        pool.clear()
        assert pool.is_empty is True

    def test_essay_pool_increment_generation_method(self):
        """Test incrementing generation count."""
        pool = EssayPool(name="Test Pool")
        assert pool.generation_count == 0

        pool.increment_generation()
        assert pool.generation_count == 1

        pool.increment_generation()
        assert pool.generation_count == 2

    def test_essay_pool_json_serialization_works(self):
        """Test that EssayPool can be serialized to and from JSON."""
        essay1 = Essay(content="Essay 1", prompt="Test 1")
        essay2 = Essay(content="Essay 2", prompt="Test 2")
        metadata = {"mode": "creative"}

        original = EssayPool(
            name="Creative Pool",
            essays=[essay1, essay2],
            generation_count=3,
            metadata=metadata,
        )

        # Serialize to dict
        data = original.model_dump()
        assert data["name"] == "Creative Pool"
        assert len(data["essays"]) == 2
        assert data["generation_count"] == 3
        assert data["metadata"] == metadata
        assert "id" in data
        assert "created_at" in data

        # Deserialize from dict
        restored = EssayPool(**data)
        assert restored.name == original.name
        assert len(restored.essays) == len(original.essays)
        assert restored.essays[0].content == original.essays[0].content
        assert restored.essays[1].content == original.essays[1].content
        assert restored.generation_count == original.generation_count
        assert restored.metadata == original.metadata

    def test_essay_pool_filter_essays_by_version(self):
        """Test filtering essays by version number."""
        # Create essays with different versions
        essay1 = Essay(content="Essay v1", prompt="Test", version=1)
        essay2 = Essay(content="Essay v2", prompt="Test", version=2)
        essay3 = Essay(content="Essay v3", prompt="Test", version=1)

        pool = EssayPool(name="Test Pool", essays=[essay1, essay2, essay3])

        # Filter for version 1
        v1_essays = pool.filter_by_version(1)
        assert len(v1_essays) == 2
        assert essay1 in v1_essays
        assert essay3 in v1_essays
        assert essay2 not in v1_essays

        # Filter for version 2
        v2_essays = pool.filter_by_version(2)
        assert len(v2_essays) == 1
        assert essay2 in v2_essays

    def test_essay_pool_filter_essays_by_parent(self):
        """Test filtering essays by parent ID."""
        parent_essay = Essay(content="Parent", prompt="Test")
        child1 = Essay(content="Child 1", prompt="Test", parent_id=parent_essay.id)
        child2 = Essay(content="Child 2", prompt="Test", parent_id=parent_essay.id)
        unrelated = Essay(content="Unrelated", prompt="Test")

        pool = EssayPool(
            name="Test Pool", essays=[parent_essay, child1, child2, unrelated]
        )

        # Filter for children of parent_essay
        children = pool.filter_by_parent(parent_essay.id)
        assert len(children) == 2
        assert child1 in children
        assert child2 in children
        assert parent_essay not in children
        assert unrelated not in children

        # Filter for non-existent parent
        no_children = pool.filter_by_parent(uuid4())
        assert len(no_children) == 0
