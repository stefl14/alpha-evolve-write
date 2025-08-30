"""LLM service interfaces and implementations for essay generation."""

import hashlib
import random
from abc import ABC, abstractmethod

from ..models import GenerationConfig


class LLMService(ABC):
    """Abstract base class for LLM services.

    Defines the interface for generating text using Large Language Models.
    Implementations can target different providers (OpenAI, Anthropic, etc.)
    or provide mock functionality for testing.
    """

    @abstractmethod
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: The input prompt for text generation
            config: Configuration settings for the generation

        Returns:
            The generated text response

        Raises:
            LLMServiceError: If generation fails
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the LLM service is accessible.

        Returns:
            True if the service is available, False otherwise
        """
        pass


class MockLLMService(LLMService):
    """Mock LLM service for development and testing.

    Provides deterministic responses based on prompt hashing to avoid API costs
    during development. Responses vary by writing mode and simulate realistic
    essay content.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the mock service.

        Args:
            seed: Random seed for consistent deterministic responses
        """
        self._seed = seed
        self._random = random.Random(seed)
        self._response_templates = {
            "general": [
                "This essay explores the topic of {topic} through a balanced perspective. "
                "The subject presents multiple facets worth examining. "
                "First, we must consider the foundational aspects that shape our understanding. "
                "Furthermore, the implications extend beyond immediate considerations. "
                "In conclusion, {topic} represents a complex subject requiring thoughtful analysis.",
                "When examining {topic}, several key themes emerge. "
                "The historical context provides valuable insights into current perspectives. "
                "Additionally, contemporary research offers new frameworks for understanding. "
                "These developments suggest that our approach to {topic} must evolve. "
                "Therefore, a comprehensive view requires integration of multiple viewpoints.",
                "The concept of {topic} deserves careful consideration from multiple angles. "
                "Modern understanding builds upon established foundations while embracing innovation. "
                "Through systematic analysis, we can identify patterns and relationships. "
                "These insights contribute to a more comprehensive worldview. "
                "Ultimately, studying {topic} enriches our perspective on related subjects.",
                "Understanding {topic} requires both depth and breadth of analysis. "
                "Various perspectives contribute to a holistic understanding of the subject. "
                "Evidence from multiple sources supports key arguments and conclusions. "
                "The complexity of {topic} demands careful evaluation of competing viewpoints. "
                "This thorough approach ensures a well-rounded examination of the issues.",
            ],
            "creative": [
                "Imagine a world where {topic} transforms everything we know. "
                "The possibilities stretch beyond conventional boundaries, inviting exploration. "
                "Through vivid imagery and metaphorical thinking, new insights emerge. "
                "Each paragraph weaves together imagination and insight, creating a tapestry of ideas. "
                "This creative exploration of {topic} opens doors to unexpected revelations.",
                "In the realm of {topic}, creativity knows no bounds. "
                "Like an artist painting with words, we craft meaning from possibility. "
                "The canvas of imagination allows for bold strokes of thought. "
                "Through this creative lens, {topic} becomes a living, breathing entity. "
                "Such artistic exploration reveals hidden dimensions of understanding.",
            ],
            "formal": [
                "This paper presents a systematic analysis of {topic}. "
                "The methodology employed ensures rigorous examination of relevant factors. "
                "Empirical evidence supports the primary arguments presented herein. "
                "The analysis demonstrates significant implications for theoretical frameworks. "
                "These findings contribute meaningfully to the scholarly discourse on {topic}.",
                "The present study investigates {topic} through established academic protocols. "
                "Literature review reveals substantial research gaps requiring attention. "
                "Methodological considerations ensure validity and reliability of conclusions. "
                "Results indicate statistically significant patterns relevant to {topic}. "
                "Recommendations for future research are provided based on these findings.",
            ],
            "technical": [
                "Technical analysis of {topic} requires systematic decomposition of core components. "
                "Implementation details must address scalability and performance considerations. "
                "Architecture design follows established engineering principles and best practices. "
                "Error handling and edge cases receive particular attention in the specification. "
                "The proposed solution demonstrates measurable improvements in relevant metrics.",
                "System design for {topic} incorporates modular architecture principles. "
                "Performance benchmarks indicate optimal resource utilization patterns. "
                "Integration protocols ensure compatibility with existing infrastructure. "
                "Security considerations address potential vulnerabilities and mitigation strategies. "
                "Documentation provides comprehensive implementation guidelines for developers.",
            ],
        }

    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate mock text based on prompt and configuration.

        Uses prompt hashing and writing mode to select appropriate template,
        then applies variations based on temperature setting.

        Args:
            prompt: The input prompt for text generation
            config: Configuration settings including mode and temperature

        Returns:
            Generated text response appropriate to the writing mode
        """
        # Create deterministic hash from prompt for consistent responses
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        hash_seed = int(prompt_hash[:8], 16)

        # Select template based on mode
        templates = self._response_templates.get(
            config.mode, self._response_templates["general"]
        )

        # Use seed and hash to determine template index deterministically
        template_index = (self._seed + hash_seed) % len(templates)
        template = templates[template_index]

        # Create random generator for other variations
        local_random = random.Random(self._seed * 1000 + hash_seed)

        # Extract topic from prompt (simple heuristic)
        topic = self._extract_topic(prompt)

        # Format template with topic
        base_response = template.format(topic=topic)

        # Apply temperature-based variation
        if config.temperature > 0.8:
            # High temperature: add creative variations
            variations = [
                " Moreover, unconventional approaches yield surprising insights.",
                " Interestingly, alternative perspectives challenge traditional assumptions.",
                " Notably, emerging trends suggest paradigm shifts ahead.",
            ]
            if local_random.random() < 0.7:
                base_response += local_random.choice(variations)
        elif config.temperature < 0.3:
            # Low temperature: more structured, less variation
            base_response = base_response.replace("Furthermore, ", "Additionally, ")
            base_response = base_response.replace("Therefore, ", "Consequently, ")

        # Respect max_tokens by truncating if needed
        words = base_response.split()
        if len(words) > config.max_tokens // 4:  # Rough words to tokens ratio
            words = words[: config.max_tokens // 4]
            base_response = " ".join(words)

        return base_response

    async def validate_connection(self) -> bool:
        """Mock validation always succeeds.

        Returns:
            Always True for mock service
        """
        return True

    def _extract_topic(self, prompt: str) -> str:
        """Extract topic from prompt using simple heuristics.

        Args:
            prompt: The input prompt text

        Returns:
            Extracted topic string
        """
        # Look for common prompt patterns
        lower_prompt = prompt.lower()

        # Pattern: "write about X" or "essay about X"
        for phrase in ["write about ", "essay about ", "discuss ", "analyze "]:
            if phrase in lower_prompt:
                topic_start = lower_prompt.find(phrase) + len(phrase)
                topic_end = lower_prompt.find(" ", topic_start + 20)
                if topic_end == -1:
                    topic_end = len(lower_prompt)
                return prompt[topic_start:topic_end].strip()

        # Fallback: use first few words
        words = prompt.split()
        if len(words) >= 3:
            return " ".join(words[:3])

        return "the given subject"


class LLMServiceError(Exception):
    """Exception raised by LLM services for generation errors."""

    pass

