"""Essay data model for the AlphaEvolve Essay system."""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Essay(BaseModel):
    """Represents an essay with content, metadata, and tracking information.

    The Essay model stores the core content along with generation metadata,
    versioning information, and automatically computed fields like word count.
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the essay"
    )
    content: str = Field(..., description="The essay content text")
    prompt: str = Field(
        ..., description="The original prompt used to generate this essay"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when essay was created",
    )
    version: int = Field(default=1, description="Version number for tracking evolution")
    parent_id: UUID | None = Field(
        default=None, description="ID of parent essay if this is a variation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about generation"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty or whitespace only.

        Args:
            v: The content string to validate

        Returns:
            The validated content string

        Raises:
            ValueError: If content is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        """Validate that prompt is not empty or whitespace only.

        Args:
            v: The prompt string to validate

        Returns:
            The validated prompt string

        Raises:
            ValueError: If prompt is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator("version")
    @classmethod
    def validate_version_positive(cls, v: int) -> int:
        """Validate that version is a positive integer.

        Args:
            v: The version number to validate

        Returns:
            The validated version number

        Raises:
            ValueError: If version is not positive
        """
        if v <= 0:
            raise ValueError("Version must be positive")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        """Calculate and return the word count of the essay content.

        Returns:
            The number of words in the content
        """
        if not self.content.strip():
            return 0
        # Split on whitespace and filter out empty strings
        words = [word for word in re.split(r"\s+", self.content.strip()) if word]
        return len(words)

    model_config = ConfigDict(
        # Use enum values for serialization
        use_enum_values=True,
        # Validate assignment for computed fields
        validate_assignment=True,
    )
