"""Essay evaluation component for the evolutionary pipeline."""

import asyncio
import re
from typing import Any

from ..models import Essay, EvaluationResult, GenerationConfig
from ..services import LLMService


class EssayEvaluator:
    """Evaluates essays using LLM-based assessment.

    The EssayEvaluator provides objective scoring of essays across multiple
    criteria including coherence, creativity, depth, and overall quality.
    It uses structured prompts to obtain consistent evaluations.
    """

    def __init__(self, llm_service: LLMService) -> None:
        """Initialize the essay evaluator.

        Args:
            llm_service: The LLM service to use for evaluation
        """
        self.llm_service = llm_service

        # Default evaluation criteria with weights
        self.default_criteria = {
            "coherence": 0.25,  # Logical flow and structure
            "depth": 0.25,  # Thoroughness and insight
            "creativity": 0.20,  # Originality and innovation
            "clarity": 0.15,  # Clear communication
            "engagement": 0.15,  # Reader interest and appeal
        }

    async def evaluate_essay(
        self,
        essay: Essay,
        config: GenerationConfig,
        criteria: dict[str, float] | None = None,
    ) -> EvaluationResult:
        """Evaluate a single essay across multiple criteria.

        Args:
            essay: The essay to evaluate
            config: Generation configuration for evaluation LLM
            criteria: Custom criteria weights (uses defaults if None)

        Returns:
            EvaluationResult with scores and feedback

        Raises:
            ValueError: If criteria weights don't sum to 1.0
            LLMServiceError: If evaluation fails
        """
        if criteria is None:
            criteria = self.default_criteria

        # Validate criteria weights
        total_weight = sum(criteria.values())
        if not (0.95 <= total_weight <= 1.05):  # Allow small floating point errors
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")

        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(essay, criteria, config)

        # Get evaluation from LLM
        evaluation_response = await self.llm_service.generate_text(
            evaluation_prompt, config
        )

        # Parse structured response
        criteria_scores, feedback = self._parse_evaluation_response(
            evaluation_response, criteria
        )

        # Calculate overall score using weighted average
        overall_score = sum(
            score * criteria[criterion] for criterion, score in criteria_scores.items()
        )

        # Create evaluation result
        result = EvaluationResult(
            essay_id=essay.id,
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            feedback=feedback,
            metadata={
                "evaluator_model": config.model_name,
                "evaluation_mode": config.mode,
                "essay_version": essay.version,
                "word_count": essay.word_count,
            },
        )

        return result

    async def evaluate_essays_batch(
        self,
        essays: list[Essay],
        config: GenerationConfig,
        criteria: dict[str, float] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple essays concurrently.

        Args:
            essays: List of essays to evaluate
            config: Generation configuration for evaluation LLM
            criteria: Custom criteria weights (uses defaults if None)

        Returns:
            List of EvaluationResult objects

        Raises:
            ValueError: If criteria weights don't sum to 1.0
            LLMServiceError: If evaluation fails
        """
        if not essays:
            return []

        # Evaluate all essays concurrently
        tasks = [self.evaluate_essay(essay, config, criteria) for essay in essays]

        results = await asyncio.gather(*tasks)
        return results

    async def compare_essays(
        self, essay1: Essay, essay2: Essay, config: GenerationConfig
    ) -> dict[str, Any]:
        """Compare two essays directly and determine which is better.

        Args:
            essay1: First essay to compare
            essay2: Second essay to compare
            config: Generation configuration for comparison LLM

        Returns:
            Dictionary with comparison results including winner and reasoning

        Raises:
            LLMServiceError: If comparison fails
        """
        comparison_prompt = self._create_comparison_prompt(essay1, essay2, config)

        comparison_response = await self.llm_service.generate_text(
            comparison_prompt, config
        )

        # Parse comparison response
        comparison_result = self._parse_comparison_response(
            comparison_response, essay1, essay2
        )

        return comparison_result

    def _create_evaluation_prompt(
        self, essay: Essay, criteria: dict[str, float], config: GenerationConfig
    ) -> str:
        """Create structured prompt for essay evaluation.

        Args:
            essay: Essay to evaluate
            criteria: Evaluation criteria and weights
            config: Generation configuration

        Returns:
            Formatted evaluation prompt
        """
        criteria_descriptions = {
            "coherence": "logical flow, structure, and organization",
            "depth": "thoroughness, insight, and analytical depth",
            "creativity": "originality, innovation, and unique perspectives",
            "clarity": "clear communication and readability",
            "engagement": "reader interest and compelling presentation",
        }

        criteria_list = []
        for criterion, weight in criteria.items():
            description = criteria_descriptions.get(
                criterion, f"{criterion} (custom criterion)"
            )
            criteria_list.append(
                f"- {criterion.title()}: {description} (weight: {weight:.2f})"
            )

        prompt = f"""Evaluate the following essay objectively using the specified criteria.
Provide scores from 0-100 for each criterion, then calculate the overall weighted score.

Essay to evaluate:
{essay.content}

Evaluation criteria:
{chr(10).join(criteria_list)}

Please respond in this exact format:
SCORES:
- Coherence: [0-100 score]
- Depth: [0-100 score]
- Creativity: [0-100 score]
- Clarity: [0-100 score]
- Engagement: [0-100 score]

FEEDBACK:
[Detailed constructive feedback explaining the scores and suggesting improvements]

Be objective, specific, and constructive in your evaluation."""

        return prompt

    def _create_comparison_prompt(
        self, essay1: Essay, essay2: Essay, config: GenerationConfig
    ) -> str:
        """Create prompt for comparing two essays.

        Args:
            essay1: First essay to compare
            essay2: Second essay to compare
            config: Generation configuration

        Returns:
            Formatted comparison prompt
        """
        prompt = f"""Compare these two essays objectively and determine which is better overall.
Consider coherence, depth, creativity, clarity, and engagement.

Essay A:
{essay1.content}

Essay B:
{essay2.content}

Please respond in this exact format:
WINNER: [A or B]
CONFIDENCE: [0-100 indicating how confident you are in this judgment]
REASONING:
[Detailed explanation of why the winning essay is better, comparing specific strengths and weaknesses]

Be objective and specific in your comparison."""

        return prompt

    def _parse_evaluation_response(
        self, response: str, criteria: dict[str, float]
    ) -> tuple[dict[str, float], str]:
        """Parse structured evaluation response from LLM.

        Args:
            response: Raw LLM response with scores and feedback
            criteria: Expected criteria for validation

        Returns:
            Tuple of (criteria_scores, feedback)
        """
        criteria_scores = {}
        feedback = ""

        # Extract scores section
        scores_match = re.search(r"SCORES:(.*?)(?=FEEDBACK:|$)", response, re.DOTALL)
        if scores_match:
            scores_text = scores_match.group(1)

            # Parse individual scores
            for criterion in criteria.keys():
                pattern = rf"{criterion}:\s*(\d+)"
                match = re.search(pattern, scores_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    criteria_scores[criterion] = max(0.0, min(100.0, score))
                else:
                    # Fallback: assign neutral score if parsing fails
                    criteria_scores[criterion] = 75.0

        # Extract feedback section
        feedback_match = re.search(r"FEEDBACK:(.*?)$", response, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            feedback = "Evaluation completed."

        # Ensure all criteria have scores
        for criterion in criteria.keys():
            if criterion not in criteria_scores:
                criteria_scores[criterion] = 75.0  # Default neutral score

        return criteria_scores, feedback

    def _parse_comparison_response(
        self, response: str, essay1: Essay, essay2: Essay
    ) -> dict[str, Any]:
        """Parse comparison response from LLM.

        Args:
            response: Raw LLM comparison response
            essay1: First essay (A)
            essay2: Second essay (B)

        Returns:
            Dictionary with comparison results
        """
        # Extract winner
        winner_match = re.search(r"WINNER:\s*([AB])", response, re.IGNORECASE)
        winner_letter = winner_match.group(1).upper() if winner_match else "A"

        winner_essay = essay1 if winner_letter == "A" else essay2
        loser_essay = essay2 if winner_letter == "A" else essay1

        # Extract confidence
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response)
        confidence = int(confidence_match.group(1)) if confidence_match else 50

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:(.*?)$", response, re.DOTALL)
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "No reasoning provided."
        )

        return {
            "winner_id": winner_essay.id,
            "loser_id": loser_essay.id,
            "winner_letter": winner_letter,
            "confidence": max(0, min(100, confidence)),
            "reasoning": reasoning,
            "essay1_id": essay1.id,
            "essay2_id": essay2.id,
        }
