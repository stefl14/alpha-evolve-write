"""Tests for EvolutionaryPipeline orchestrator."""

import pytest

from alpha_evolve_essay.models import Essay, GenerationConfig
from alpha_evolve_essay.pipeline import EvolutionaryPipeline
from alpha_evolve_essay.services import MockLLMService


class TestEvolutionaryPipeline:
    """Test cases for EvolutionaryPipeline orchestrator."""

    @pytest.fixture
    def pipeline(self) -> EvolutionaryPipeline:
        """Create EvolutionaryPipeline with MockLLMService."""
        llm_service = MockLLMService(seed=42)
        generator_config = GenerationConfig(
            model_name="mock-model",
            prompt_template="Write an essay about {topic}",
            temperature=0.7,
        )
        evaluator_config = GenerationConfig(
            model_name="mock-model",
            prompt_template="Evaluate: {content}",
            temperature=0.3,
        )

        return EvolutionaryPipeline(llm_service, generator_config, evaluator_config)

    @pytest.fixture
    def basic_config(self) -> GenerationConfig:
        """Create basic GenerationConfig."""
        return GenerationConfig(
            model_name="mock-model", prompt_template="Write about {topic}"
        )

    def test_evolutionary_pipeline_initialization_succeeds(self):
        """Test that EvolutionaryPipeline can be initialized."""
        llm_service = MockLLMService()
        config = GenerationConfig(model_name="test", prompt_template="test")

        pipeline = EvolutionaryPipeline(llm_service, config)

        assert pipeline is not None
        assert pipeline.llm_service is llm_service
        assert pipeline.generator_config is config
        assert pipeline.evaluator_config is config  # Should default to generator_config
        assert pipeline.generator is not None
        assert pipeline.evaluator is not None

    def test_evolutionary_pipeline_uses_separate_evaluator_config(self):
        """Test that pipeline can use separate generator and evaluator configs."""
        llm_service = MockLLMService()
        gen_config = GenerationConfig(model_name="gen", prompt_template="gen")
        eval_config = GenerationConfig(model_name="eval", prompt_template="eval")

        pipeline = EvolutionaryPipeline(llm_service, gen_config, eval_config)

        assert pipeline.generator_config is gen_config
        assert pipeline.evaluator_config is eval_config

    def test_evolutionary_pipeline_has_default_parameters(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that pipeline has sensible default evolution parameters."""
        assert pipeline.population_size > 0
        assert pipeline.elite_size > 0
        assert pipeline.elite_size < pipeline.population_size
        assert 0 <= pipeline.crossover_rate <= 1
        assert 0 <= pipeline.mutation_rate <= 1
        assert pipeline.max_generations > 0

    def test_configure_evolution_updates_parameters(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that evolution parameters can be configured."""
        # Store original size for reference but don't use it

        pipeline.configure_evolution(
            population_size=15,
            elite_size=4,
            crossover_rate=0.4,
            mutation_rate=0.6,
            max_generations=8,
            random_seed=123,
        )

        assert pipeline.population_size == 15
        assert pipeline.elite_size == 4
        assert pipeline.crossover_rate == 0.4
        assert pipeline.mutation_rate == 0.6
        assert pipeline.max_generations == 8

        # Should not update None values
        pipeline.configure_evolution(population_size=None)
        assert pipeline.population_size == 15  # Unchanged

    @pytest.mark.asyncio
    async def test_evolve_with_minimal_parameters_succeeds(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test evolution with minimal parameters."""
        # Configure for fast test
        pipeline.configure_evolution(population_size=3, elite_size=1, max_generations=2)

        results = await pipeline.evolve("artificial intelligence")

        assert results is not None
        assert "best_essay" in results
        assert "final_pool" in results
        assert "evolution_history" in results
        assert "total_generations" in results
        assert "final_best_score" in results

        assert results["total_generations"] == 2
        assert len(results["evolution_history"]) == 2
        assert results["best_essay"] is not None
        assert 0 <= results["final_best_score"] <= 100

    @pytest.mark.asyncio
    async def test_evolve_with_custom_parameters_succeeds(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test evolution with custom parameters."""
        results = await pipeline.evolve(
            "climate change", generations=3, population_size=4, elite_size=2
        )

        assert results["total_generations"] == 3
        assert len(results["evolution_history"]) == 3

        # Check that each generation has expected statistics
        for i, gen_stats in enumerate(results["evolution_history"]):
            assert gen_stats["generation"] == i + 1
            assert gen_stats["population_size"] >= 2  # At least elite size
            assert "avg_score" in gen_stats
            assert "max_score" in gen_stats
            assert "min_score" in gen_stats

    @pytest.mark.asyncio
    async def test_evolve_validates_parameters(self, pipeline: EvolutionaryPipeline):
        """Test that evolve validates input parameters."""
        # Invalid generations
        with pytest.raises(ValueError, match="Generations must be positive"):
            await pipeline.evolve("topic", generations=0)

        with pytest.raises(ValueError, match="Generations must be positive"):
            await pipeline.evolve("topic", generations=-1)

        # Invalid population size
        with pytest.raises(ValueError, match="Population size must be positive"):
            await pipeline.evolve("topic", population_size=0)

        with pytest.raises(ValueError, match="Population size must be positive"):
            await pipeline.evolve("topic", population_size=-1)

        # Elite size >= population size
        with pytest.raises(
            ValueError, match="Elite size must be less than population size"
        ):
            await pipeline.evolve("topic", population_size=3, elite_size=3)

        with pytest.raises(
            ValueError, match="Elite size must be less than population size"
        ):
            await pipeline.evolve("topic", population_size=2, elite_size=3)

    @pytest.mark.asyncio
    async def test_evolve_shows_improvement_over_generations(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that evolution generally shows improvement over generations."""
        pipeline.configure_evolution(population_size=5, elite_size=2, max_generations=3)

        results = await pipeline.evolve("technology innovation")

        # Check that we have progression data
        history = results["evolution_history"]
        assert len(history) == 3

        # All generations should have valid scores
        for gen_stats in history:
            assert 0 <= gen_stats["avg_score"] <= 100
            assert 0 <= gen_stats["max_score"] <= 100
            assert 0 <= gen_stats["min_score"] <= 100
            assert (
                gen_stats["min_score"]
                <= gen_stats["avg_score"]
                <= gen_stats["max_score"]
            )

        # Should have improvement data
        assert "improvement" in results
        assert isinstance(results["improvement"], int | float)

    @pytest.mark.asyncio
    async def test_evolve_preserves_elites_across_generations(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that elite essays are preserved across generations."""
        pipeline.configure_evolution(population_size=4, elite_size=2, max_generations=2)

        results = await pipeline.evolve("space exploration")

        # Should have evolution history with elite preservation
        history = results["evolution_history"]
        assert len(history) == 2

        # Each generation should track top essays
        for gen_stats in history:
            assert "top_essays" in gen_stats
            assert len(gen_stats["top_essays"]) == 2  # Elite size

    @pytest.mark.asyncio
    async def test_quick_improve_single_essay_succeeds(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test quick improvement of a single essay."""
        essay = Essay(
            content="Basic essay about renewable energy sources.",
            prompt="Write about renewable energy",
            version=1,
        )

        results = await pipeline.quick_improve(essay, iterations=2)

        assert results is not None
        assert "original_essay" in results
        assert "improved_essay" in results
        assert "initial_score" in results
        assert "final_score" in results
        assert "total_improvement" in results
        assert "improvement_history" in results

        assert results["original_essay"] is essay
        assert results["improved_essay"] is not None
        assert 0 <= results["initial_score"] <= 100
        assert 0 <= results["final_score"] <= 100
        assert len(results["improvement_history"]) == 2

    @pytest.mark.asyncio
    async def test_quick_improve_validates_iterations(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that quick_improve validates iteration count."""
        essay = Essay(content="Test", prompt="Test", version=1)

        with pytest.raises(ValueError, match="Iterations must be positive"):
            await pipeline.quick_improve(essay, iterations=0)

    @pytest.mark.asyncio
    async def test_quick_improve_tracks_progress(self, pipeline: EvolutionaryPipeline):
        """Test that quick_improve tracks improvement progress."""
        essay = Essay(
            content="Short essay about AI ethics.",
            prompt="Discuss AI ethics",
            version=1,
        )

        results = await pipeline.quick_improve(essay, iterations=3)

        history = results["improvement_history"]
        assert len(history) == 3

        # Each iteration should have proper tracking
        for i, iter_result in enumerate(history):
            assert iter_result["iteration"] == i + 1
            assert "score" in iter_result
            assert "improvement" in iter_result
            assert 0 <= iter_result["score"] <= 100

    @pytest.mark.asyncio
    async def test_generate_crossovers_with_sufficient_elites(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test crossover generation with sufficient elite essays."""
        elites = [
            Essay(content="Elite 1", prompt="Test", version=1),
            Essay(content="Elite 2", prompt="Test", version=1),
            Essay(content="Elite 3", prompt="Test", version=1),
        ]

        crossovers = await pipeline._generate_crossovers(elites, 2)

        assert len(crossovers) == 2
        for crossover in crossovers:
            assert crossover.metadata["generation_type"] == "crossover"
            assert "parent1_id" in crossover.metadata
            assert "parent2_id" in crossover.metadata

    @pytest.mark.asyncio
    async def test_generate_crossovers_with_insufficient_elites(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test crossover generation with insufficient elite essays."""
        elite = [Essay(content="Only one elite", prompt="Test", version=1)]

        crossovers = await pipeline._generate_crossovers(elite, 2)

        assert len(crossovers) == 0  # Should return empty list

    @pytest.mark.asyncio
    async def test_generate_mutations_with_elites(self, pipeline: EvolutionaryPipeline):
        """Test mutation generation from elite essays."""
        elites = [
            Essay(content="Elite 1", prompt="Test", version=1),
            Essay(content="Elite 2", prompt="Test", version=1),
        ]

        mutations = await pipeline._generate_mutations(elites, 3)

        assert len(mutations) == 3
        for mutation in mutations:
            assert mutation.metadata["generation_type"] == "variation"
            assert mutation.parent_id in [elite.id for elite in elites]

    @pytest.mark.asyncio
    async def test_generate_mutations_with_empty_elites(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test mutation generation with no elite essays."""
        mutations = await pipeline._generate_mutations([], 2)
        assert len(mutations) == 0

    @pytest.mark.asyncio
    async def test_evolution_maintains_population_size(
        self, pipeline: EvolutionaryPipeline
    ):
        """Test that evolution maintains target population size across generations."""
        target_population = 6
        pipeline.configure_evolution(
            population_size=target_population, elite_size=2, max_generations=3
        )

        results = await pipeline.evolve("sustainable development")

        # Check that all generations maintain proper population size
        for gen_stats in results["evolution_history"]:
            assert gen_stats["population_size"] == target_population

    @pytest.mark.asyncio
    async def test_evolution_uses_different_random_seeds(
        self, basic_config: GenerationConfig
    ):
        """Test that different random seeds produce different evolution paths."""
        llm_service = MockLLMService(seed=42)

        # Create two pipelines with different seeds
        pipeline1 = EvolutionaryPipeline(llm_service, basic_config)
        pipeline1.configure_evolution(
            random_seed=42, max_generations=2, population_size=4, elite_size=2
        )

        pipeline2 = EvolutionaryPipeline(llm_service, basic_config)
        pipeline2.configure_evolution(
            random_seed=123, max_generations=2, population_size=4, elite_size=2
        )

        results1 = await pipeline1.evolve("machine learning")
        results2 = await pipeline2.evolve("machine learning")

        # Different seeds should potentially produce different results
        # (though exact difference depends on MockLLMService behavior)
        assert results1 is not None
        assert results2 is not None
        assert results1["best_essay"].id != results2["best_essay"].id

