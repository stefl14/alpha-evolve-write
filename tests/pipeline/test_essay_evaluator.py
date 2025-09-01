"""Tests for EssayEvaluator component."""

import pytest

from alpha_evolve_essay.models import Essay, GenerationConfig
from alpha_evolve_essay.pipeline import EssayEvaluator
from alpha_evolve_essay.services import MockLLMService


class TestEssayEvaluator:
    """Test cases for EssayEvaluator component."""

    @pytest.fixture
    def evaluator(self) -> EssayEvaluator:
        """Create EssayEvaluator with MockLLMService."""
        llm_service = MockLLMService(seed=42)
        return EssayEvaluator(llm_service)

    @pytest.fixture
    def config(self) -> GenerationConfig:
        """Create basic GenerationConfig for evaluation."""
        return GenerationConfig(
            model_name="mock-model",
            prompt_template="Evaluate this essay: {content}",
            temperature=0.3,  # Lower temperature for consistent evaluation
            max_tokens=1000,
        )

    @pytest.fixture
    def sample_essay(self) -> Essay:
        """Create a sample essay for testing."""
        return Essay(
            content="This is a well-structured essay about artificial intelligence. "
            "It explores the implications of AI in modern society, discussing both "
            "benefits and challenges. The argument is coherent and well-supported.",
            prompt="Write about artificial intelligence",
            version=1,
        )

    def test_essay_evaluator_initialization_succeeds(self):
        """Test that EssayEvaluator can be initialized with LLM service."""
        llm_service = MockLLMService()
        evaluator = EssayEvaluator(llm_service)

        assert evaluator is not None
        assert evaluator.llm_service is llm_service
        assert evaluator.default_criteria is not None
        assert len(evaluator.default_criteria) == 5

    def test_default_criteria_weights_sum_to_one(self, evaluator: EssayEvaluator):
        """Test that default criteria weights sum to 1.0."""
        total_weight = sum(evaluator.default_criteria.values())
        assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision

    def test_default_criteria_includes_expected_criteria(
        self, evaluator: EssayEvaluator
    ):
        """Test that default criteria includes all expected evaluation criteria."""
        expected_criteria = [
            "coherence",
            "depth",
            "creativity",
            "clarity",
            "engagement",
        ]

        assert set(evaluator.default_criteria.keys()) == set(expected_criteria)

        # All weights should be positive
        for weight in evaluator.default_criteria.values():
            assert weight > 0

    @pytest.mark.asyncio
    async def test_evaluate_essay_with_default_criteria_succeeds(
        self, evaluator: EssayEvaluator, config: GenerationConfig, sample_essay: Essay
    ):
        """Test evaluating essay with default criteria."""
        result = await evaluator.evaluate_essay(sample_essay, config)

        assert result is not None
        assert result.essay_id == sample_essay.id
        assert 0 <= result.overall_score <= 100
        assert len(result.criteria_scores) == 5
        assert len(result.feedback) > 0

        # Check all expected criteria are present
        expected_criteria = evaluator.default_criteria.keys()
        assert set(result.criteria_scores.keys()) == set(expected_criteria)

        # All scores should be valid
        for score in result.criteria_scores.values():
            assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_evaluate_essay_with_custom_criteria_succeeds(
        self, evaluator: EssayEvaluator, config: GenerationConfig, sample_essay: Essay
    ):
        """Test evaluating essay with custom criteria."""
        custom_criteria = {"coherence": 0.4, "creativity": 0.3, "clarity": 0.3}

        result = await evaluator.evaluate_essay(sample_essay, config, custom_criteria)

        assert result is not None
        assert len(result.criteria_scores) == 3
        assert set(result.criteria_scores.keys()) == set(custom_criteria.keys())

    @pytest.mark.asyncio
    async def test_evaluate_essay_validates_criteria_weights(
        self, evaluator: EssayEvaluator, config: GenerationConfig, sample_essay: Essay
    ):
        """Test that criteria weights must sum to 1.0."""
        invalid_criteria = {
            "coherence": 0.3,
            "creativity": 0.4,  # Total = 0.7, should fail
        }

        with pytest.raises(ValueError, match="Criteria weights must sum to 1.0"):
            await evaluator.evaluate_essay(sample_essay, config, invalid_criteria)

    @pytest.mark.asyncio
    async def test_evaluate_essay_includes_metadata(
        self, evaluator: EssayEvaluator, config: GenerationConfig, sample_essay: Essay
    ):
        """Test that evaluation result includes proper metadata."""
        result = await evaluator.evaluate_essay(sample_essay, config)

        assert "evaluator_model" in result.metadata
        assert "evaluation_mode" in result.metadata
        assert "essay_version" in result.metadata
        assert "word_count" in result.metadata

        assert result.metadata["evaluator_model"] == config.model_name
        assert result.metadata["evaluation_mode"] == config.mode
        assert result.metadata["essay_version"] == sample_essay.version
        assert result.metadata["word_count"] == sample_essay.word_count

    @pytest.mark.asyncio
    async def test_evaluate_essays_batch_with_multiple_essays_succeeds(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test batch evaluation of multiple essays."""
        essays = [
            Essay(content="First essay content", prompt="Test", version=1),
            Essay(content="Second essay content", prompt="Test", version=1),
            Essay(content="Third essay content", prompt="Test", version=1),
        ]

        results = await evaluator.evaluate_essays_batch(essays, config)

        assert len(results) == 3

        # Check each result
        essay_ids = {essay.id for essay in essays}
        result_ids = {result.essay_id for result in results}
        assert result_ids == essay_ids

        for result in results:
            assert 0 <= result.overall_score <= 100
            assert len(result.criteria_scores) == 5

    @pytest.mark.asyncio
    async def test_evaluate_essays_batch_with_empty_list_returns_empty(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test batch evaluation with empty essay list."""
        results = await evaluator.evaluate_essays_batch([], config)
        assert results == []

    @pytest.mark.asyncio
    async def test_evaluate_essays_batch_with_custom_criteria_succeeds(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test batch evaluation with custom criteria."""
        essays = [
            Essay(content="Essay 1", prompt="Test", version=1),
            Essay(content="Essay 2", prompt="Test", version=1),
        ]

        custom_criteria = {"coherence": 0.5, "creativity": 0.5}

        results = await evaluator.evaluate_essays_batch(essays, config, custom_criteria)

        assert len(results) == 2
        for result in results:
            assert len(result.criteria_scores) == 2
            assert set(result.criteria_scores.keys()) == {"coherence", "creativity"}

    @pytest.mark.asyncio
    async def test_compare_essays_succeeds(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test comparing two essays directly."""
        essay1 = Essay(
            content="This is a comprehensive and well-structured essay with deep analysis.",
            prompt="Test topic",
            version=1,
        )
        essay2 = Essay(
            content="This is a shorter essay with basic points.",
            prompt="Test topic",
            version=1,
        )

        comparison = await evaluator.compare_essays(essay1, essay2, config)

        assert comparison is not None
        assert "winner_id" in comparison
        assert "loser_id" in comparison
        assert "winner_letter" in comparison
        assert "confidence" in comparison
        assert "reasoning" in comparison
        assert "essay1_id" in comparison
        assert "essay2_id" in comparison

        # Winner should be one of the essays
        assert comparison["winner_id"] in [essay1.id, essay2.id]
        assert comparison["loser_id"] in [essay1.id, essay2.id]
        assert comparison["winner_id"] != comparison["loser_id"]

        # Confidence should be valid percentage
        assert 0 <= comparison["confidence"] <= 100

        # Winner letter should be A or B
        assert comparison["winner_letter"] in ["A", "B"]

    @pytest.mark.asyncio
    async def test_compare_essays_tracks_original_essay_ids(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test that comparison tracks original essay IDs correctly."""
        essay1 = Essay(content="Essay A content", prompt="Test", version=1)
        essay2 = Essay(content="Essay B content", prompt="Test", version=1)

        comparison = await evaluator.compare_essays(essay1, essay2, config)

        assert comparison["essay1_id"] == essay1.id
        assert comparison["essay2_id"] == essay2.id

    def test_evaluation_prompt_includes_all_criteria(self, evaluator: EssayEvaluator):
        """Test that evaluation prompt includes all specified criteria."""
        essay = Essay(content="Test content", prompt="Test", version=1)
        config = GenerationConfig(model_name="test", prompt_template="test")

        criteria = {"coherence": 0.5, "creativity": 0.5}
        prompt = evaluator._create_evaluation_prompt(essay, criteria, config)

        assert "coherence" in prompt.lower()
        assert "creativity" in prompt.lower()
        assert essay.content in prompt
        assert "SCORES:" in prompt
        assert "FEEDBACK:" in prompt

    def test_comparison_prompt_includes_both_essays(self, evaluator: EssayEvaluator):
        """Test that comparison prompt includes both essays."""
        essay1 = Essay(content="First essay", prompt="Test", version=1)
        essay2 = Essay(content="Second essay", prompt="Test", version=1)
        config = GenerationConfig(model_name="test", prompt_template="test")

        prompt = evaluator._create_comparison_prompt(essay1, essay2, config)

        assert essay1.content in prompt
        assert essay2.content in prompt
        assert "Essay A:" in prompt
        assert "Essay B:" in prompt
        assert "WINNER:" in prompt
        assert "CONFIDENCE:" in prompt

    def test_parse_evaluation_response_handles_well_formatted_response(
        self, evaluator: EssayEvaluator
    ):
        """Test parsing well-formatted evaluation response."""
        criteria = {"coherence": 0.5, "creativity": 0.5}
        response = """SCORES:
- Coherence: 85
- Creativity: 75

FEEDBACK:
This essay demonstrates good coherence with clear structure.
The creativity could be enhanced with more original examples."""

        scores, feedback = evaluator._parse_evaluation_response(response, criteria)

        assert scores["coherence"] == 85.0
        assert scores["creativity"] == 75.0
        assert "coherence" in feedback.lower()
        assert "creativity" in feedback.lower()

    def test_parse_evaluation_response_handles_malformed_response(
        self, evaluator: EssayEvaluator
    ):
        """Test parsing malformed evaluation response with fallbacks."""
        criteria = {"coherence": 0.6, "depth": 0.4}
        response = "This is a poorly formatted response without proper structure."

        scores, feedback = evaluator._parse_evaluation_response(response, criteria)

        # Should provide fallback scores
        assert "coherence" in scores
        assert "depth" in scores
        assert all(0 <= score <= 100 for score in scores.values())
        assert len(feedback) > 0

    def test_parse_comparison_response_handles_well_formatted_response(
        self, evaluator: EssayEvaluator
    ):
        """Test parsing well-formatted comparison response."""
        essay1 = Essay(content="Essay A", prompt="Test", version=1)
        essay2 = Essay(content="Essay B", prompt="Test", version=1)

        response = """WINNER: B
CONFIDENCE: 75
REASONING:
Essay B demonstrates superior analysis and clearer arguments."""

        result = evaluator._parse_comparison_response(response, essay1, essay2)

        assert result["winner_letter"] == "B"
        assert result["winner_id"] == essay2.id
        assert result["loser_id"] == essay1.id
        assert result["confidence"] == 75
        assert "superior analysis" in result["reasoning"]

    def test_parse_comparison_response_handles_malformed_response(
        self, evaluator: EssayEvaluator
    ):
        """Test parsing malformed comparison response with fallbacks."""
        essay1 = Essay(content="Essay A", prompt="Test", version=1)
        essay2 = Essay(content="Essay B", prompt="Test", version=1)

        response = "This is a malformed comparison response."

        result = evaluator._parse_comparison_response(response, essay1, essay2)

        assert result["winner_letter"] in ["A", "B"]
        assert result["winner_id"] in [essay1.id, essay2.id]
        assert 0 <= result["confidence"] <= 100
        assert len(result["reasoning"]) > 0

    @pytest.mark.asyncio
    async def test_evaluator_concurrent_evaluation_performance(
        self, evaluator: EssayEvaluator, config: GenerationConfig
    ):
        """Test that concurrent evaluation works efficiently."""
        # Create multiple essays for concurrent evaluation
        essays = [
            Essay(
                content=f"Essay {i} content with unique analysis",
                prompt="Test",
                version=1,
            )
            for i in range(5)
        ]

        results = await evaluator.evaluate_essays_batch(essays, config)

        assert len(results) == 5

        # All results should be valid
        for result in results:
            assert 0 <= result.overall_score <= 100
            assert len(result.criteria_scores) == 5
            assert len(result.feedback) > 0
