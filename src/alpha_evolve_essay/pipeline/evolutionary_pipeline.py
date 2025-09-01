"""Main evolutionary pipeline orchestrator for iterative essay improvement."""

import asyncio
import random
from typing import Any

from ..models import Essay, EssayPool, GenerationConfig
from ..services import LLMService
from .essay_evaluator import EssayEvaluator
from .essay_generator import EssayGenerator


class EvolutionaryPipeline:
    """Orchestrates the evolutionary essay improvement process.

    The EvolutionaryPipeline implements the AlphaEvolve approach for essays:
    1. Generate initial population
    2. Evaluate essays
    3. Select top performers
    4. Generate variations and crossovers
    5. Repeat for multiple generations
    """

    def __init__(
        self,
        llm_service: LLMService,
        generator_config: GenerationConfig,
        evaluator_config: GenerationConfig | None = None,
    ) -> None:
        """Initialize the evolutionary pipeline.

        Args:
            llm_service: The LLM service for generation and evaluation
            generator_config: Configuration for essay generation
            evaluator_config: Configuration for evaluation (uses generator_config if None)
        """
        self.llm_service = llm_service
        self.generator_config = generator_config
        self.evaluator_config = evaluator_config or generator_config

        # Initialize components
        self.generator = EssayGenerator(llm_service)
        self.evaluator = EssayEvaluator(llm_service)

        # Evolution parameters
        self.population_size = 10
        self.elite_size = 3
        self.crossover_rate = 0.3
        self.mutation_rate = 0.5
        self.max_generations = 5

        # Random seed for reproducible evolution
        self.random = random.Random(42)

    async def evolve(
        self,
        prompt: str,
        generations: int | None = None,
        population_size: int | None = None,
        elite_size: int | None = None,
    ) -> dict[str, Any]:
        """Run the complete evolutionary process.

        Args:
            prompt: The writing prompt/topic
            generations: Number of generations to evolve (uses default if None)
            population_size: Size of population per generation (uses default if None)
            elite_size: Number of top essays to keep (uses default if None)

        Returns:
            Dictionary containing evolution results and statistics

        Raises:
            ValueError: If parameters are invalid
            LLMServiceError: If generation or evaluation fails
        """
        # Use provided parameters or defaults
        generations = generations if generations is not None else self.max_generations
        population_size = (
            population_size if population_size is not None else self.population_size
        )
        elite_size = elite_size if elite_size is not None else self.elite_size

        # Validate parameters
        if generations <= 0:
            raise ValueError("Generations must be positive")
        if population_size <= 0:
            raise ValueError("Population size must be positive")
        if elite_size >= population_size:
            raise ValueError("Elite size must be less than population size")

        evolution_history = []
        current_pool = None

        # Generate initial population
        print(
            f"🧬 Starting evolution with {generations} generations, population {population_size}"
        )
        current_pool = await self.generator.generate_initial_population(
            prompt, self.generator_config, population_size
        )

        # Evolution loop
        for generation in range(generations):
            print(f"🔄 Generation {generation + 1}/{generations}")

            # Evaluate current population
            evaluations = await self.evaluator.evaluate_essays_batch(
                current_pool.essays, self.evaluator_config
            )

            # Create evaluation scores mapping
            evaluation_scores = {
                eval_result.essay_id: eval_result.overall_score
                for eval_result in evaluations
            }

            # Get generation statistics
            scores = list(evaluation_scores.values())
            generation_stats = {
                "generation": generation + 1,
                "population_size": len(current_pool.essays),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "top_essays": current_pool.get_top_essays(
                    elite_size, evaluation_scores
                ),
            }
            evolution_history.append(generation_stats)

            print(
                f"📊 Avg: {generation_stats['avg_score']:.1f}, "
                f"Max: {generation_stats['max_score']:.1f}, "
                f"Min: {generation_stats['min_score']:.1f}"
            )

            # If this is the final generation, don't create next generation
            if generation == generations - 1:
                break

            # Evolution step: create next generation
            current_pool = await self._evolve_generation(
                current_pool, evaluation_scores, population_size, elite_size
            )
            current_pool.increment_generation()

        # Prepare final results
        final_evaluations = await self.evaluator.evaluate_essays_batch(
            current_pool.essays, self.evaluator_config
        )
        final_scores = {
            eval_result.essay_id: eval_result.overall_score
            for eval_result in final_evaluations
        }

        best_essay = current_pool.get_top_essays(1, final_scores)[0]

        results = {
            "best_essay": best_essay,
            "final_pool": current_pool,
            "final_evaluations": final_evaluations,
            "evolution_history": evolution_history,
            "total_generations": generations,
            "final_best_score": max(final_scores.values()),
            "improvement": max(final_scores.values())
            - evolution_history[0]["max_score"],  # type: ignore[operator]
        }

        print(
            f"✅ Evolution complete! Best score: {results['final_best_score']:.1f} "
            f"(+{results['improvement']:+.1f})"
        )

        return results

    async def _evolve_generation(
        self,
        current_pool: EssayPool,
        evaluation_scores: dict,
        target_population_size: int,
        elite_size: int,
    ) -> EssayPool:
        """Create the next generation through selection, crossover, and mutation.

        Args:
            current_pool: Current generation pool
            evaluation_scores: Evaluation scores for selection
            target_population_size: Target size for new generation
            elite_size: Number of elite essays to preserve

        Returns:
            New EssayPool for the next generation
        """
        # Select elite essays (best performers)
        elite_essays = current_pool.get_top_essays(elite_size, evaluation_scores)

        # Calculate how many new essays to generate
        remaining_slots = target_population_size - len(elite_essays)
        crossover_count = int(remaining_slots * self.crossover_rate)
        mutation_count = remaining_slots - crossover_count

        new_essays = list(elite_essays)  # Start with elites

        # Generate crossovers
        if crossover_count > 0:
            crossovers = await self._generate_crossovers(elite_essays, crossover_count)
            new_essays.extend(crossovers)

        # Generate mutations (variations)
        if mutation_count > 0:
            mutations = await self._generate_mutations(elite_essays, mutation_count)
            new_essays.extend(mutations)

        # Create new pool
        pool_name = f"Generation {current_pool.generation_count + 1}"
        new_pool = EssayPool(
            name=pool_name,
            essays=new_essays,
            generation_count=current_pool.generation_count,
        )

        return new_pool

    async def _generate_crossovers(
        self, elite_essays: list[Essay], count: int
    ) -> list[Essay]:
        """Generate crossover essays from elite parents.

        Args:
            elite_essays: List of top-performing essays
            count: Number of crossovers to generate

        Returns:
            List of crossover essays
        """
        if len(elite_essays) < 2:
            return []

        crossover_tasks = []
        for _ in range(count):
            # Select two random parents from elites
            parent1, parent2 = self.random.sample(elite_essays, 2)
            task = self.generator.generate_crossover(
                parent1, parent2, self.generator_config
            )
            crossover_tasks.append(task)

        crossovers = await asyncio.gather(*crossover_tasks)
        return crossovers

    async def _generate_mutations(
        self, elite_essays: list[Essay], count: int
    ) -> list[Essay]:
        """Generate mutations (variations) from elite essays.

        Args:
            elite_essays: List of top-performing essays
            count: Number of mutations to generate

        Returns:
            List of mutated essays
        """
        if not elite_essays:
            return []

        # Distribute mutations across elite essays
        selected_parents = []

        # Select parents for mutation (with possible repetition)
        for _ in range(count):
            parent = self.random.choice(elite_essays)
            selected_parents.append(parent)

        # Generate variations
        mutations = await self.generator.generate_variations(
            selected_parents, self.generator_config, 1
        )

        return mutations[:count]  # Ensure exact count

    def configure_evolution(
        self,
        population_size: int | None = None,
        elite_size: int | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        max_generations: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Configure evolution parameters.

        Args:
            population_size: Size of population per generation
            elite_size: Number of top essays to preserve
            crossover_rate: Fraction of new generation from crossovers
            mutation_rate: Fraction of new generation from mutations
            max_generations: Maximum generations to evolve
            random_seed: Random seed for reproducible evolution
        """
        if population_size is not None:
            self.population_size = population_size
        if elite_size is not None:
            self.elite_size = elite_size
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        if max_generations is not None:
            self.max_generations = max_generations
        if random_seed is not None:
            self.random = random.Random(random_seed)

    async def quick_improve(self, essay: Essay, iterations: int = 3) -> dict[str, Any]:
        """Quickly improve a single essay through iterative refinement.

        Args:
            essay: The essay to improve
            iterations: Number of improvement iterations

        Returns:
            Dictionary with improvement results
        """
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        current_essay = essay
        improvement_history = []

        # Evaluate initial essay
        initial_eval = await self.evaluator.evaluate_essay(
            current_essay, self.evaluator_config
        )

        print(f"🎯 Quick improve: {iterations} iterations")
        print(f"📊 Initial score: {initial_eval.overall_score:.1f}")

        for i in range(iterations):
            # Generate variations
            variations = await self.generator.generate_variations(
                [current_essay], self.generator_config, 3
            )

            # Evaluate all candidates (current + variations)
            all_candidates = [current_essay] + variations
            evaluations = await self.evaluator.evaluate_essays_batch(
                all_candidates, self.evaluator_config
            )

            # Find best candidate
            best_eval = max(evaluations, key=lambda e: e.overall_score)
            best_essay = next(
                essay for essay in all_candidates if essay.id == best_eval.essay_id
            )

            improvement_history.append(
                {
                    "iteration": i + 1,
                    "score": best_eval.overall_score,
                    "improvement": best_eval.overall_score - initial_eval.overall_score,
                }
            )

            print(
                f"🔄 Iteration {i + 1}: {best_eval.overall_score:.1f} "
                f"(+{best_eval.overall_score - initial_eval.overall_score:+.1f})"
            )

            current_essay = best_essay

        return {
            "original_essay": essay,
            "improved_essay": current_essay,
            "initial_score": initial_eval.overall_score,
            "final_score": improvement_history[-1]["score"],
            "total_improvement": improvement_history[-1]["improvement"],
            "improvement_history": improvement_history,
        }

