"""Tests for EvaluationResult model."""

import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from alpha_evolve_essay.models import EvaluationResult


class TestEvaluationResultModel:
    """Test cases for EvaluationResult model validation and behavior."""

    def test_evaluation_result_creation_with_minimal_fields_succeeds(self):
        """Test that EvaluationResult can be created with required fields."""
        essay_id = uuid4()
        result = EvaluationResult(
            essay_id=essay_id,
            overall_score=85.5
        )
        
        assert result.essay_id == essay_id
        assert result.overall_score == 85.5
        assert isinstance(result.id, UUID)
        assert isinstance(result.evaluated_at, datetime)
        assert result.evaluated_at.tzinfo == timezone.utc
        assert result.criteria_scores == {}
        assert result.feedback == ""
        assert result.evaluator == "unknown"

    def test_evaluation_result_creation_with_all_fields_succeeds(self):
        """Test that EvaluationResult accepts all optional fields."""
        essay_id = uuid4()
        criteria_scores = {
            "coherence": 90.0,
            "grammar": 85.5,
            "style": 78.0,
            "factual_accuracy": 92.5
        }
        
        result = EvaluationResult(
            essay_id=essay_id,
            overall_score=86.5,
            criteria_scores=criteria_scores,
            feedback="Well-written essay with minor grammatical issues.",
            evaluator="gpt-4"
        )
        
        assert result.essay_id == essay_id
        assert result.overall_score == 86.5
        assert result.criteria_scores == criteria_scores
        assert result.feedback == "Well-written essay with minor grammatical issues."
        assert result.evaluator == "gpt-4"

    def test_evaluation_result_overall_score_validation_accepts_valid_range(self):
        """Test that overall_score accepts values in valid range 0-100."""
        essay_id = uuid4()
        
        # Test boundary values
        result_min = EvaluationResult(essay_id=essay_id, overall_score=0.0)
        assert result_min.overall_score == 0.0
        
        result_max = EvaluationResult(essay_id=essay_id, overall_score=100.0)
        assert result_max.overall_score == 100.0
        
        result_mid = EvaluationResult(essay_id=essay_id, overall_score=50.5)
        assert result_mid.overall_score == 50.5

    def test_evaluation_result_overall_score_validation_rejects_invalid_range(self):
        """Test that overall_score rejects values outside 0-100 range."""
        essay_id = uuid4()
        
        with pytest.raises(ValueError, match="Overall score must be between 0 and 100"):
            EvaluationResult(essay_id=essay_id, overall_score=-0.1)
            
        with pytest.raises(ValueError, match="Overall score must be between 0 and 100"):
            EvaluationResult(essay_id=essay_id, overall_score=100.1)

    def test_evaluation_result_criteria_scores_validation_accepts_valid_scores(self):
        """Test that criteria_scores accepts valid score values."""
        essay_id = uuid4()
        criteria_scores = {
            "coherence": 0.0,
            "grammar": 50.5,
            "style": 100.0
        }
        
        result = EvaluationResult(
            essay_id=essay_id,
            overall_score=50.0,
            criteria_scores=criteria_scores
        )
        assert result.criteria_scores == criteria_scores

    def test_evaluation_result_criteria_scores_validation_rejects_invalid_scores(self):
        """Test that criteria_scores rejects invalid score values."""
        essay_id = uuid4()
        
        with pytest.raises(ValueError, match="All criteria scores must be between 0 and 100"):
            EvaluationResult(
                essay_id=essay_id,
                overall_score=50.0,
                criteria_scores={"coherence": -0.1}
            )
            
        with pytest.raises(ValueError, match="All criteria scores must be between 0 and 100"):
            EvaluationResult(
                essay_id=essay_id,
                overall_score=50.0,
                criteria_scores={"grammar": 100.1}
            )

    def test_evaluation_result_feedback_defaults_to_empty_string(self):
        """Test that feedback defaults to empty string when not provided."""
        essay_id = uuid4()
        result = EvaluationResult(essay_id=essay_id, overall_score=75.0)
        assert result.feedback == ""
        assert isinstance(result.feedback, str)

    def test_evaluation_result_evaluator_defaults_to_unknown(self):
        """Test that evaluator defaults to 'unknown' when not provided."""
        essay_id = uuid4()
        result = EvaluationResult(essay_id=essay_id, overall_score=75.0)
        assert result.evaluator == "unknown"

    def test_evaluation_result_id_is_unique_across_instances(self):
        """Test that each EvaluationResult instance gets a unique ID."""
        essay_id = uuid4()
        result1 = EvaluationResult(essay_id=essay_id, overall_score=75.0)
        result2 = EvaluationResult(essay_id=essay_id, overall_score=80.0)
        
        assert result1.id != result2.id
        assert isinstance(result1.id, UUID)
        assert isinstance(result2.id, UUID)

    def test_evaluation_result_evaluated_at_is_utc_timezone(self):
        """Test that evaluated_at timestamp is in UTC timezone."""
        essay_id = uuid4()
        result = EvaluationResult(essay_id=essay_id, overall_score=75.0)
        assert result.evaluated_at.tzinfo == timezone.utc

    def test_evaluation_result_computed_average_score_calculation(self):
        """Test that average criteria score is computed correctly."""
        essay_id = uuid4()
        criteria_scores = {
            "coherence": 80.0,
            "grammar": 90.0,
            "style": 70.0
        }
        
        result = EvaluationResult(
            essay_id=essay_id,
            overall_score=85.0,
            criteria_scores=criteria_scores
        )
        
        # Average should be (80 + 90 + 70) / 3 = 80.0
        assert result.average_criteria_score == 80.0

    def test_evaluation_result_computed_average_score_with_empty_criteria(self):
        """Test average criteria score computation with no criteria."""
        essay_id = uuid4()
        result = EvaluationResult(essay_id=essay_id, overall_score=85.0)
        # Should return 0.0 when no criteria scores
        assert result.average_criteria_score == 0.0

    def test_evaluation_result_json_serialization_works(self):
        """Test that EvaluationResult can be serialized to and from JSON."""
        essay_id = uuid4()
        criteria_scores = {"coherence": 85.5, "grammar": 90.0}
        
        original = EvaluationResult(
            essay_id=essay_id,
            overall_score=87.75,
            criteria_scores=criteria_scores,
            feedback="Good essay with minor issues",
            evaluator="claude-3"
        )
        
        # Serialize to dict
        data = original.model_dump()
        assert data["essay_id"] == essay_id  # UUID remains as UUID object in model_dump()
        assert data["overall_score"] == 87.75
        assert data["criteria_scores"] == criteria_scores
        assert data["feedback"] == "Good essay with minor issues"
        assert data["evaluator"] == "claude-3"
        assert "id" in data
        assert "evaluated_at" in data
        assert "average_criteria_score" in data
        
        # Deserialize from dict (essay_id is already UUID)
        restored = EvaluationResult(**data)
        assert restored.essay_id == original.essay_id
        assert restored.overall_score == original.overall_score
        assert restored.criteria_scores == original.criteria_scores
        assert restored.feedback == original.feedback
        assert restored.evaluator == original.evaluator

    def test_evaluation_result_comparison_and_ranking(self):
        """Test that EvaluationResult can be compared by overall score."""
        essay_id = uuid4()
        result1 = EvaluationResult(essay_id=essay_id, overall_score=75.0)
        result2 = EvaluationResult(essay_id=essay_id, overall_score=85.0)
        result3 = EvaluationResult(essay_id=essay_id, overall_score=85.0)
        
        # Test comparison methods
        assert result1.overall_score < result2.overall_score
        assert result2.overall_score == result3.overall_score
        assert result2.overall_score > result1.overall_score