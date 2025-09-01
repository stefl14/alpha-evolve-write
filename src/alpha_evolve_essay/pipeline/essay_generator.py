"""Essay generation component for the evolutionary pipeline."""

import asyncio
from uuid import UUID

from ..models import Essay, EssayPool, GenerationConfig
from ..services import LLMService


class EssayGenerator:
    """Generates and varies essays using LLM services.

    The EssayGenerator is responsible for creating initial essay populations
    and generating variations of existing essays during the evolutionary process.
    It uses configurable prompts and LLM settings to produce diverse content.
    """

    def __init__(self, llm_service: LLMService) -> None:
        """Initialize the essay generator.

        Args:
            llm_service: The LLM service to use for text generation
        """
        self.llm_service = llm_service

    async def generate_initial_population(
        self, prompt: str, config: GenerationConfig, population_size: int = 5
    ) -> EssayPool:
        """Generate an initial population of essays for a given prompt.

        Args:
            prompt: The writing prompt or topic
            config: Generation configuration settings
            population_size: Number of essays to generate

        Returns:
            EssayPool containing the generated essays

        Raises:
            ValueError: If population_size is not positive
            LLMServiceError: If generation fails
        """
        if population_size <= 0:
            raise ValueError("Population size must be positive")

        # Format the prompt template
        formatted_prompt = config.format_prompt(topic=prompt)

        # Generate essays concurrently for efficiency
        tasks = [
            self._generate_single_essay(formatted_prompt, config, version=1)
            for _ in range(population_size)
        ]

        essays = await asyncio.gather(*tasks)

        # Create pool with generated essays
        pool_name = f"Initial Population - {prompt[:50]}"
        pool = EssayPool(name=pool_name, essays=essays)

        return pool

    async def generate_variations(
        self,
        parent_essays: list[Essay],
        config: GenerationConfig,
        variations_per_parent: int = 2,
    ) -> list[Essay]:
        """Generate variations of existing essays.

        Creates new essay versions by prompting the LLM to modify,
        improve, or extend the parent essays.

        Args:
            parent_essays: Essays to create variations from
            config: Generation configuration settings
            variations_per_parent: Number of variations per parent essay

        Returns:
            List of generated essay variations

        Raises:
            ValueError: If variations_per_parent is not positive
            LLMServiceError: If generation fails
        """
        if variations_per_parent <= 0:
            raise ValueError("Variations per parent must be positive")

        if not parent_essays:
            return []

        # Create variation tasks for all parent essays
        tasks = []
        for parent in parent_essays:
            for i in range(variations_per_parent):
                task = self._generate_variation(parent, config, i)
                tasks.append(task)

        variations = await asyncio.gather(*tasks)
        return variations

    async def generate_crossover(
        self, parent1: Essay, parent2: Essay, config: GenerationConfig
    ) -> Essay:
        """Generate a crossover essay combining elements from two parents.

        Creates a new essay that incorporates ideas, style, or content
        from both parent essays.

        Args:
            parent1: First parent essay
            parent2: Second parent essay
            config: Generation configuration settings

        Returns:
            New essay combining elements from both parents

        Raises:
            LLMServiceError: If generation fails
        """
        # Create crossover prompt combining both essays
        crossover_prompt = self._create_crossover_prompt(parent1, parent2, config)

        # Generate the crossover essay
        content = await self.llm_service.generate_text(crossover_prompt, config)

        # Determine the next version number
        next_version = max(parent1.version, parent2.version) + 1

        # Create essay with both parents referenced (use first parent's ID)
        essay = Essay(
            content=content,
            prompt=parent1.prompt,  # Use original prompt
            parent_id=parent1.id,
            version=next_version,
            metadata={
                "generation_type": "crossover",
                "parent1_id": str(parent1.id),
                "parent2_id": str(parent2.id),
                "config_mode": config.mode,
            },
        )

        return essay

    async def _generate_single_essay(
        self,
        prompt: str,
        config: GenerationConfig,
        version: int = 1,
        parent_id: UUID | None = None,
    ) -> Essay:
        """Generate a single essay from a prompt.

        Args:
            prompt: The formatted prompt for generation
            config: Generation configuration
            version: Version number for the essay
            parent_id: ID of parent essay if this is a variation

        Returns:
            Generated Essay instance
        """
        content = await self.llm_service.generate_text(prompt, config)

        essay = Essay(
            content=content,
            prompt=prompt,
            parent_id=parent_id,
            version=version,
            metadata={
                "generation_type": "initial" if parent_id is None else "variation",
                "config_mode": config.mode,
            },
        )

        return essay

    async def _generate_variation(
        self, parent: Essay, config: GenerationConfig, variation_index: int
    ) -> Essay:
        """Generate a single variation of a parent essay.

        Args:
            parent: Parent essay to create variation from
            config: Generation configuration
            variation_index: Index of this variation (for metadata)

        Returns:
            Generated essay variation
        """
        # Create variation prompt
        variation_prompt = self._create_variation_prompt(
            parent, config, variation_index
        )

        # Generate the variation
        content = await self.llm_service.generate_text(variation_prompt, config)

        # Create variation essay
        essay = Essay(
            content=content,
            prompt=parent.prompt,  # Keep original prompt
            parent_id=parent.id,
            version=parent.version + 1,
            metadata={
                "generation_type": "variation",
                "variation_index": variation_index,
                "config_mode": config.mode,
                "parent_version": parent.version,
            },
        )

        return essay

    def _create_variation_prompt(
        self, parent: Essay, config: GenerationConfig, variation_index: int
    ) -> str:
        """Create a prompt for generating essay variations.

        Args:
            parent: Parent essay to vary
            config: Generation configuration
            variation_index: Index of variation for diversity

        Returns:
            Formatted prompt for variation generation
        """
        # Different variation strategies based on writing mode and index
        strategies = {
            "general": [
                "Rewrite this essay with a different perspective:",
                "Expand on the key ideas in this essay:",
                "Restructure this essay with better organization:",
            ],
            "creative": [
                "Transform this essay with more imaginative language:",
                "Add creative metaphors and examples to this essay:",
                "Rewrite this essay with a more artistic style:",
            ],
            "formal": [
                "Make this essay more academic and rigorous:",
                "Add scholarly analysis to this essay:",
                "Restructure this essay with formal academic style:",
            ],
            "technical": [
                "Add more technical details to this essay:",
                "Restructure this essay with systematic analysis:",
                "Enhance this essay with implementation specifics:",
            ],
        }

        mode_strategies = strategies.get(config.mode, strategies["general"])
        strategy = mode_strategies[variation_index % len(mode_strategies)]

        return f"{strategy}\n\n{parent.content}\n\nImproved version:"

    def _create_crossover_prompt(
        self, parent1: Essay, parent2: Essay, config: GenerationConfig
    ) -> str:
        """Create a prompt for crossover generation.

        Args:
            parent1: First parent essay
            parent2: Second parent essay
            config: Generation configuration

        Returns:
            Formatted prompt for crossover generation
        """
        crossover_instruction = (
            "Combine the best ideas, arguments, and writing style from these two essays "
            "into a new, cohesive essay that incorporates strengths from both:\n\n"
            f"Essay 1:\n{parent1.content}\n\n"
            f"Essay 2:\n{parent2.content}\n\n"
            "Combined essay incorporating the best from both:"
        )

        return crossover_instruction
