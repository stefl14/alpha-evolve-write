"""EssayPool data model for managing collections of essays."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .essay import Essay


class EssayPool(BaseModel):
    """A collection of essays with management and ranking capabilities.

    EssayPool manages groups of essays during the evolutionary process,
    providing methods for selection, filtering, and ranking based on
    evaluation scores.
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the pool"
    )
    name: str = Field(..., description="Human-readable name for the pool")
    essays: list[Essay] = Field(
        default_factory=list, description="Collection of essays in the pool"
    )
    generation_count: int = Field(
        default=0, description="Number of evolution generations"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when pool was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the pool"
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that pool name is not empty or whitespace only.

        Args:
            v: The pool name to validate

        Returns:
            The validated pool name

        Raises:
            ValueError: If pool name is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Pool name cannot be empty")
        return v

    @field_validator("generation_count")
    @classmethod
    def validate_generation_count_non_negative(cls, v: int) -> int:
        """Validate that generation_count is non-negative.

        Args:
            v: The generation count to validate

        Returns:
            The validated generation count

        Raises:
            ValueError: If generation count is negative
        """
        if v < 0:
            raise ValueError("Generation count cannot be negative")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size(self) -> int:
        """Get the number of essays in the pool.

        Returns:
            The number of essays in the pool
        """
        return len(self.essays)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_empty(self) -> bool:
        """Check if the pool is empty.

        Returns:
            True if the pool has no essays, False otherwise
        """
        return len(self.essays) == 0

    def add_essay(self, essay: Essay) -> None:
        """Add an essay to the pool, avoiding duplicates.

        Args:
            essay: The essay to add to the pool
        """
        # Check if essay is already in pool (by ID)
        for existing_essay in self.essays:
            if existing_essay.id == essay.id:
                return  # Don't add duplicate

        self.essays.append(essay)

    def remove_essay(self, essay_id: UUID) -> Essay | None:
        """Remove an essay from the pool by ID.

        Args:
            essay_id: The ID of the essay to remove

        Returns:
            The removed essay if found, None otherwise
        """
        for i, essay in enumerate(self.essays):
            if essay.id == essay_id:
                return self.essays.pop(i)
        return None

    def get_essay(self, essay_id: UUID) -> Essay | None:
        """Get an essay from the pool by ID.

        Args:
            essay_id: The ID of the essay to retrieve

        Returns:
            The essay if found, None otherwise
        """
        for essay in self.essays:
            if essay.id == essay_id:
                return essay
        return None

    def get_top_essays(
        self, limit: int, evaluation_scores: dict[UUID, float]
    ) -> list[Essay]:
        """Get the top-scoring essays from the pool.

        Args:
            limit: Maximum number of essays to return
            evaluation_scores: Dictionary mapping essay IDs to scores

        Returns:
            List of essays sorted by score (highest first), limited by limit
        """
        if not evaluation_scores:
            # If no scores available, return first essays up to limit
            return self.essays[:limit]

        # Sort essays by their evaluation scores (highest first)
        scored_essays = []
        for essay in self.essays:
            score = evaluation_scores.get(essay.id, 0.0)
            scored_essays.append((score, essay))

        # Sort by score (descending) and extract essays
        scored_essays.sort(key=lambda x: x[0], reverse=True)
        return [essay for _score, essay in scored_essays[:limit]]

    def clear(self) -> None:
        """Remove all essays from the pool."""
        self.essays.clear()

    def increment_generation(self) -> None:
        """Increment the generation count by one."""
        self.generation_count += 1

    def filter_by_version(self, version: int) -> list[Essay]:
        """Filter essays by version number.

        Args:
            version: The version number to filter by

        Returns:
            List of essays with the specified version
        """
        return [essay for essay in self.essays if essay.version == version]

    def filter_by_parent(self, parent_id: UUID) -> list[Essay]:
        """Filter essays by parent ID.

        Args:
            parent_id: The parent essay ID to filter by

        Returns:
            List of essays that have the specified parent ID
        """
        return [essay for essay in self.essays if essay.parent_id == parent_id]

    model_config = ConfigDict(
        # Use enum values for serialization
        use_enum_values=True,
        # Validate assignment for computed fields
        validate_assignment=True,
    )
