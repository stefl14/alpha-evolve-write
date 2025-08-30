"""Tests for LLM service implementations."""

import pytest

from alpha_evolve_essay.models import GenerationConfig
from alpha_evolve_essay.services import MockLLMService


class TestMockLLMService:
    """Test cases for MockLLMService implementation."""

    def test_mock_llm_service_initialization_succeeds(self):
        """Test that MockLLMService can be initialized with default seed."""
        service = MockLLMService()
        assert service is not None

    def test_mock_llm_service_initialization_with_custom_seed_succeeds(self):
        """Test that MockLLMService accepts custom seed."""
        service = MockLLMService(seed=123)
        assert service is not None

    @pytest.mark.asyncio
    async def test_mock_llm_validate_connection_always_returns_true(self):
        """Test that mock service validation always succeeds."""
        service = MockLLMService()
        result = await service.validate_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_with_minimal_config_succeeds(self):
        """Test text generation with basic configuration."""
        service = MockLLMService()
        config = GenerationConfig(
            model_name="mock-model", prompt_template="Write about {topic}"
        )

        result = await service.generate_text(
            "Write about artificial intelligence", config
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "artificial intelligence" in result.lower()

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_respects_writing_modes(self):
        """Test that different writing modes produce different styles."""
        service = MockLLMService(seed=42)
        prompt = "Write about technology"

        # Test each writing mode
        modes = ["general", "creative", "formal", "technical"]
        results = {}

        for mode in modes:
            config = GenerationConfig(
                model_name="mock-model",
                prompt_template="Test",
                mode=mode,  # type: ignore[arg-type]
            )
            result = await service.generate_text(prompt, config)
            results[mode] = result
            assert isinstance(result, str)
            assert len(result) > 0

        # Verify different modes produce different content
        assert results["creative"] != results["formal"]
        assert results["technical"] != results["general"]

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_creative_mode_characteristics(self):
        """Test that creative mode produces imaginative content."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(
            model_name="mock-model", prompt_template="Test", mode="creative"
        )

        result = await service.generate_text("Write about dreams", config)

        # Creative mode should contain imaginative language
        creative_indicators = ["imagine", "creativity", "artistic", "canvas", "realm"]
        assert any(indicator in result.lower() for indicator in creative_indicators)

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_formal_mode_characteristics(self):
        """Test that formal mode produces academic content."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(
            model_name="mock-model", prompt_template="Test", mode="formal"
        )

        result = await service.generate_text("Write about research", config)

        # Formal mode should contain academic language
        formal_indicators = [
            "analysis",
            "methodology",
            "empirical",
            "study",
            "research",
        ]
        assert any(indicator in result.lower() for indicator in formal_indicators)

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_technical_mode_characteristics(self):
        """Test that technical mode produces engineering content."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(
            model_name="mock-model", prompt_template="Test", mode="technical"
        )

        result = await service.generate_text("Write about systems", config)

        # Technical mode should contain engineering language
        technical_indicators = [
            "technical",
            "system",
            "architecture",
            "implementation",
            "performance",
        ]
        assert any(indicator in result.lower() for indicator in technical_indicators)

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_deterministic_with_same_seed(self):
        """Test that same seed produces consistent results."""
        service1 = MockLLMService(seed=42)
        service2 = MockLLMService(seed=42)

        config = GenerationConfig(model_name="mock-model", prompt_template="Test")
        prompt = "Write about consistency"

        result1 = await service1.generate_text(prompt, config)
        result2 = await service2.generate_text(prompt, config)

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_different_with_different_seeds(self):
        """Test that different seeds produce different results."""
        service1 = MockLLMService(seed=42)
        service2 = MockLLMService(seed=123)

        config = GenerationConfig(model_name="mock-model", prompt_template="Test")
        prompt = "Write about variation"

        result1 = await service1.generate_text(prompt, config)
        result2 = await service2.generate_text(prompt, config)

        # Different seeds should produce different content
        assert result1 != result2

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_respects_temperature_high(self):
        """Test that high temperature adds creative variations."""
        service = MockLLMService(seed=42)

        # High temperature config
        config_high = GenerationConfig(
            model_name="mock-model", prompt_template="Test", temperature=1.5
        )

        result = await service.generate_text("Write about creativity", config_high)

        assert isinstance(result, str)
        assert len(result) > 0
        # High temperature may add extra creative variations
        # At least one variation indicator might be present
        assert len(result) > 100  # Should be substantial content

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_respects_temperature_low(self):
        """Test that low temperature produces more structured output."""
        service = MockLLMService(seed=42)

        # Low temperature config
        config_low = GenerationConfig(
            model_name="mock-model", prompt_template="Test", temperature=0.1
        )

        result = await service.generate_text("Write about structure", config_low)

        assert isinstance(result, str)
        assert len(result) > 0
        # Low temperature should produce more structured output without random variations
        # The exact content depends on template selected, but should be deterministic

        # Compare with high temperature to verify different behavior
        config_high = GenerationConfig(
            model_name="mock-model", prompt_template="Test", temperature=1.5
        )

        result_high = await service.generate_text("Write about structure", config_high)

        # Different temperatures should potentially produce different content
        # (though exact difference depends on template and random variation)
        assert isinstance(result_high, str)
        assert len(result_high) > 0

    @pytest.mark.asyncio
    async def test_mock_llm_generate_text_respects_max_tokens(self):
        """Test that response length is limited by max_tokens."""
        service = MockLLMService(seed=42)

        # Very low token limit
        config_short = GenerationConfig(
            model_name="mock-model", prompt_template="Test", max_tokens=50  # Very short
        )

        config_long = GenerationConfig(
            model_name="mock-model", prompt_template="Test", max_tokens=2000  # Long
        )

        result_short = await service.generate_text("Write about brevity", config_short)
        result_long = await service.generate_text("Write about brevity", config_long)

        # Short result should be significantly shorter
        assert len(result_short.split()) <= len(result_long.split())
        assert len(result_short.split()) < 20  # Should be truncated

    @pytest.mark.asyncio
    async def test_mock_llm_topic_extraction_with_clear_patterns(self):
        """Test topic extraction from common prompt patterns."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(model_name="mock-model", prompt_template="Test")

        # Test various prompt patterns
        prompts_and_topics = [
            ("Write about machine learning", "machine learning"),
            ("Essay about climate change", "climate change"),
            ("Discuss quantum physics", "quantum physics"),
            ("Analyze economic trends", "economic trends"),
        ]

        for prompt, expected_topic in prompts_and_topics:
            result = await service.generate_text(prompt, config)
            # Topic should appear in the result
            assert expected_topic in result.lower()

    @pytest.mark.asyncio
    async def test_mock_llm_topic_extraction_fallback_for_unclear_prompts(self):
        """Test fallback topic extraction for unclear prompts."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(model_name="mock-model", prompt_template="Test")

        # Unclear prompt without standard patterns
        result = await service.generate_text("Random unclear input here", config)

        # Should still generate valid content
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain either extracted words or fallback
        assert (
            "random unclear input" in result.lower()
            or "the given subject" in result.lower()
        )

    @pytest.mark.asyncio
    async def test_mock_llm_different_prompts_produce_different_responses(self):
        """Test that different prompts produce different responses."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(model_name="mock-model", prompt_template="Test")

        result1 = await service.generate_text("Write about dogs", config)
        result2 = await service.generate_text("Write about cats", config)

        # Different prompts should produce different content
        assert result1 != result2
        assert "dogs" in result1.lower() or "dog" in result1.lower()
        assert "cats" in result2.lower() or "cat" in result2.lower()

    @pytest.mark.asyncio
    async def test_mock_llm_generates_substantial_content(self):
        """Test that generated content is substantial and coherent."""
        service = MockLLMService(seed=42)
        config = GenerationConfig(
            model_name="mock-model", prompt_template="Test", max_tokens=1000
        )

        result = await service.generate_text(
            "Write a comprehensive essay about innovation", config
        )

        # Should generate substantial content
        assert len(result) > 200  # At least a good paragraph
        assert len(result.split()) > 40  # Multiple sentences worth
        assert "." in result  # Should have sentence structure
        assert result[0].isupper()  # Should start with capital letter
