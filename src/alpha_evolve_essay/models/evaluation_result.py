"""EvaluationResult data model for essay scoring and feedback."""

from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict


class EvaluationResult(BaseModel):
    """Represents the result of evaluating an essay.
    
    Contains overall scores, detailed criteria scores, feedback text,
    and metadata about the evaluation process.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the evaluation")
    essay_id: UUID = Field(..., description="ID of the essay that was evaluated")
    overall_score: float = Field(..., description="Overall score from 0.0 to 100.0")
    criteria_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores for individual evaluation criteria"
    )
    feedback: str = Field(default="", description="Textual feedback about the essay")
    evaluator: str = Field(default="unknown", description="Name or ID of the evaluator (human/LLM)")
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when evaluation was performed"
    )

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score_range(cls, v: float) -> float:
        """Validate that overall score is between 0.0 and 100.0.
        
        Args:
            v: The overall score to validate
            
        Returns:
            The validated overall score
            
        Raises:
            ValueError: If score is outside valid range
        """
        if not (0.0 <= v <= 100.0):
            raise ValueError("Overall score must be between 0 and 100")
        return v

    @field_validator("criteria_scores")
    @classmethod
    def validate_criteria_scores_range(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all criteria scores are between 0.0 and 100.0.
        
        Args:
            v: The criteria scores dictionary to validate
            
        Returns:
            The validated criteria scores
            
        Raises:
            ValueError: If any score is outside valid range
        """
        for criterion, score in v.items():
            if not (0.0 <= score <= 100.0):
                raise ValueError("All criteria scores must be between 0 and 100")
        return v

    @computed_field  # type: ignore[misc]
    @property
    def average_criteria_score(self) -> float:
        """Calculate and return the average of all criteria scores.
        
        Returns:
            The average criteria score, or 0.0 if no criteria scores
        """
        if not self.criteria_scores:
            return 0.0
        
        total_score = sum(self.criteria_scores.values())
        return total_score / len(self.criteria_scores)

    model_config = ConfigDict(
        # Use enum values for serialization
        use_enum_values=True,
        # Validate assignment for computed fields
        validate_assignment=True
    )