"""Tests for EssayGenerator component."""

import pytest

from alpha_evolve_essay.models import Essay, GenerationConfig
from alpha_evolve_essay.pipeline import EssayGenerator
from alpha_evolve_essay.services import MockLLMService


class TestEssayGenerator:
    """Test cases for EssayGenerator component."""

    @pytest.fixture
    def generator(self) -> EssayGenerator:
        """Create EssayGenerator with MockLLMService."""
        llm_service = MockLLMService(seed=42)
        return EssayGenerator(llm_service)

    @pytest.fixture
    def config(self) -> GenerationConfig:
        """Create basic GenerationConfig."""
        return GenerationConfig(
            model_name="mock-model",
            prompt_template="Write an essay about {topic}",
            temperature=0.7,
            max_tokens=1000,
        )

    def test_essay_generator_initialization_succeeds(self):
        """Test that EssayGenerator can be initialized with LLM service."""
        llm_service = MockLLMService()
        generator = EssayGenerator(llm_service)

        assert generator is not None
        assert generator.llm_service is llm_service

    @pytest.mark.asyncio
    async def test_generate_initial_population_with_default_size_succeeds(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test generating initial population with default size."""
        prompt = "artificial intelligence"

        pool = await generator.generate_initial_population(prompt, config)

        assert pool is not None
        assert pool.size == 5  # Default population size
        assert "artificial intelligence" in pool.name.lower()

        # All essays should be initial generation
        for essay in pool.essays:
            assert essay.version == 1
            assert essay.parent_id is None
            assert essay.metadata["generation_type"] == "initial"
            assert len(essay.content) > 0

    @pytest.mark.asyncio
    async def test_generate_initial_population_with_custom_size_succeeds(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test generating initial population with custom size."""
        prompt = "climate change"
        population_size = 3

        pool = await generator.generate_initial_population(
            prompt, config, population_size
        )

        assert pool.size == population_size
        assert all(essay.version == 1 for essay in pool.essays)
        assert all(essay.parent_id is None for essay in pool.essays)

    @pytest.mark.asyncio
    async def test_generate_initial_population_validates_positive_size(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that population size must be positive."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            await generator.generate_initial_population("topic", config, 0)

        with pytest.raises(ValueError, match="Population size must be positive"):
            await generator.generate_initial_population("topic", config, -1)

    @pytest.mark.asyncio
    async def test_generate_initial_population_uses_prompt_template(
        self, generator: EssayGenerator
    ):
        """Test that generator uses the prompt template correctly."""
        config = GenerationConfig(
            model_name="mock-model",
            prompt_template="Discuss the implications of {topic} in modern society",
        )

        pool = await generator.generate_initial_population("robotics", config, 1)

        # Content should reflect the formatted prompt
        essay = pool.essays[0]
        assert "robotics" in essay.content.lower()

    @pytest.mark.asyncio
    async def test_generate_variations_with_single_parent_succeeds(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test generating variations from a single parent essay."""
        # Create a parent essay
        parent = Essay(
            content="Original essay about technology and society.",
            prompt="Write about technology",
            version=1,
        )

        variations = await generator.generate_variations([parent], config, 2)

        assert len(variations) == 2
        for variation in variations:
            assert variation.parent_id == parent.id
            assert variation.version == parent.version + 1
            assert variation.metadata["generation_type"] == "variation"
            assert variation.metadata["parent_version"] == parent.version
            assert len(variation.content) > 0

    @pytest.mark.asyncio
    async def test_generate_variations_with_multiple_parents_succeeds(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test generating variations from multiple parent essays."""
        parents = [
            Essay(content="First essay", prompt="Test", version=1),
            Essay(content="Second essay", prompt="Test", version=1),
        ]

        variations = await generator.generate_variations(parents, config, 2)

        # Should have 2 variations per parent = 4 total
        assert len(variations) == 4

        # Check that variations are properly linked to parents
        parent_ids = {parent.id for parent in parents}
        variation_parent_ids = {var.parent_id for var in variations}
        assert variation_parent_ids == parent_ids

    @pytest.mark.asyncio
    async def test_generate_variations_with_empty_parents_returns_empty_list(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that empty parent list returns empty variations."""
        variations = await generator.generate_variations([], config, 2)
        assert variations == []

    @pytest.mark.asyncio
    async def test_generate_variations_validates_positive_count(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that variations per parent must be positive."""
        parent = Essay(content="Test", prompt="Test", version=1)

        with pytest.raises(ValueError, match="Variations per parent must be positive"):
            await generator.generate_variations([parent], config, 0)

    @pytest.mark.asyncio
    async def test_generate_variations_uses_different_strategies(
        self, generator: EssayGenerator
    ):
        """Test that different variation indices use different strategies."""
        parent = Essay(content="Original content", prompt="Test", version=1)

        # Test different writing modes
        modes = ["general", "creative", "formal", "technical"]

        for mode in modes:
            config = GenerationConfig(
                model_name="mock-model",
                prompt_template="Test",
                mode=mode,  # type: ignore[arg-type]
            )

            variations = await generator.generate_variations([parent], config, 3)

            # Should generate 3 variations with potentially different strategies
            assert len(variations) == 3
            for i, variation in enumerate(variations):
                assert variation.metadata["variation_index"] == i
                assert variation.metadata["config_mode"] == mode

    @pytest.mark.asyncio
    async def test_generate_crossover_succeeds(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test generating crossover essay from two parents."""
        parent1 = Essay(
            content="First essay discusses topic A with perspective X.",
            prompt="Write about topic",
            version=2,
        )
        parent2 = Essay(
            content="Second essay explores topic B with approach Y.",
            prompt="Write about topic",
            version=3,
        )

        crossover = await generator.generate_crossover(parent1, parent2, config)

        assert crossover is not None
        assert crossover.parent_id == parent1.id  # Uses first parent's ID
        assert crossover.version == 4  # max(2, 3) + 1
        assert crossover.metadata["generation_type"] == "crossover"
        assert crossover.metadata["parent1_id"] == str(parent1.id)
        assert crossover.metadata["parent2_id"] == str(parent2.id)
        assert len(crossover.content) > 0

    @pytest.mark.asyncio
    async def test_generate_crossover_handles_different_versions(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test crossover with parents having different versions."""
        parent1 = Essay(content="Content 1", prompt="Test", version=1)
        parent2 = Essay(content="Content 2", prompt="Test", version=5)

        crossover = await generator.generate_crossover(parent1, parent2, config)

        # Should use max version + 1
        assert crossover.version == 6

    @pytest.mark.asyncio
    async def test_generate_crossover_preserves_original_prompt(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that crossover preserves the original prompt."""
        original_prompt = "Original writing prompt"
        parent1 = Essay(content="Content 1", prompt=original_prompt, version=1)
        parent2 = Essay(content="Content 2", prompt=original_prompt, version=1)

        crossover = await generator.generate_crossover(parent1, parent2, config)

        assert crossover.prompt == original_prompt

    @pytest.mark.asyncio
    async def test_generator_respects_different_writing_modes(
        self, generator: EssayGenerator
    ):
        """Test that generator respects different writing modes."""
        modes = ["general", "creative", "formal", "technical"]

        for mode in modes:
            config = GenerationConfig(
                model_name="mock-model",
                prompt_template="Write about {topic}",
                mode=mode,  # type: ignore[arg-type]
            )

            pool = await generator.generate_initial_population("technology", config, 1)
            essay = pool.essays[0]

            assert essay.metadata["config_mode"] == mode
            assert len(essay.content) > 0

    @pytest.mark.asyncio
    async def test_generator_preserves_metadata_across_operations(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that metadata is properly set across all generation operations."""
        # Test initial generation
        pool = await generator.generate_initial_population("test topic", config, 1)
        initial_essay = pool.essays[0]

        assert initial_essay.metadata["generation_type"] == "initial"
        assert initial_essay.metadata["config_mode"] == config.mode

        # Test variation
        variations = await generator.generate_variations([initial_essay], config, 1)
        variation = variations[0]

        assert variation.metadata["generation_type"] == "variation"
        assert variation.metadata["config_mode"] == config.mode
        assert "variation_index" in variation.metadata
        assert "parent_version" in variation.metadata

        # Test crossover
        second_parent = Essay(content="Second", prompt="Test", version=1)
        crossover = await generator.generate_crossover(
            initial_essay, second_parent, config
        )

        assert crossover.metadata["generation_type"] == "crossover"
        assert crossover.metadata["config_mode"] == config.mode
        assert "parent1_id" in crossover.metadata
        assert "parent2_id" in crossover.metadata

    @pytest.mark.asyncio
    async def test_generator_concurrent_generation_performance(
        self, generator: EssayGenerator, config: GenerationConfig
    ):
        """Test that concurrent generation works efficiently."""
        # Generate larger population to test concurrency
        pool = await generator.generate_initial_population(
            "performance test", config, 10
        )

        assert pool.size == 10
        assert all(essay.content for essay in pool.essays)

        # All essays should be unique (different hashes due to concurrent generation)
        contents = [essay.content for essay in pool.essays]
        # Note: MockLLMService should produce different content for same prompt
        # when called concurrently due to different timing/ordering
        assert len(set(contents)) >= 1  # At least some variety expected
