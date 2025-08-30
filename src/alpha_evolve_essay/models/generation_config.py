"""GenerationConfig data model for LLM generation settings."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# Valid writing modes
WritingMode = Literal["general", "creative", "formal", "technical"]


class GenerationConfig(BaseModel):
    """Configuration settings for LLM text generation.

    Contains model parameters, prompt templates, and generation settings
    used to control how essays are generated and evolved.
    """

    model_name: str = Field(..., description="Name of the LLM model to use")
    prompt_template: str = Field(
        ..., description="Template string for prompts with placeholders"
    )
    temperature: float = Field(
        default=0.7, description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    top_p: float = Field(
        default=1.0, description="Nucleus sampling parameter (0.0-1.0)"
    )
    frequency_penalty: float = Field(
        default=0.0, description="Frequency penalty (-2.0 to 2.0)"
    )
    presence_penalty: float = Field(
        default=0.0, description="Presence penalty (-2.0 to 2.0)"
    )
    mode: WritingMode = Field(default="general", description="Writing mode/style")
    additional_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific parameters"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name_not_empty(cls, v: str) -> str:
        """Validate that model_name is not empty or whitespace only.

        Args:
            v: The model name to validate

        Returns:
            The validated model name

        Raises:
            ValueError: If model name is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template_not_empty(cls, v: str) -> str:
        """Validate that prompt_template is not empty or whitespace only.

        Args:
            v: The prompt template to validate

        Returns:
            The validated prompt template

        Raises:
            ValueError: If prompt template is empty or whitespace only
        """
        if not v or not v.strip():
            raise ValueError("Prompt template cannot be empty")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature_range(cls, v: float) -> float:
        """Validate that temperature is between 0.0 and 2.0.

        Args:
            v: The temperature value to validate

        Returns:
            The validated temperature

        Raises:
            ValueError: If temperature is outside valid range
        """
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens_positive(cls, v: int) -> int:
        """Validate that max_tokens is positive.

        Args:
            v: The max tokens value to validate

        Returns:
            The validated max tokens

        Raises:
            ValueError: If max tokens is not positive
        """
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p_range(cls, v: float) -> float:
        """Validate that top_p is between 0.0 and 1.0.

        Args:
            v: The top-p value to validate

        Returns:
            The validated top-p value

        Raises:
            ValueError: If top-p is outside valid range
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")
        return v

    @field_validator("frequency_penalty")
    @classmethod
    def validate_frequency_penalty_range(cls, v: float) -> float:
        """Validate that frequency_penalty is between -2.0 and 2.0.

        Args:
            v: The frequency penalty to validate

        Returns:
            The validated frequency penalty

        Raises:
            ValueError: If frequency penalty is outside valid range
        """
        if not (-2.0 <= v <= 2.0):
            raise ValueError("Frequency penalty must be between -2.0 and 2.0")
        return v

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty_range(cls, v: float) -> float:
        """Validate that presence_penalty is between -2.0 and 2.0.

        Args:
            v: The presence penalty to validate

        Returns:
            The validated presence penalty

        Raises:
            ValueError: If presence penalty is outside valid range
        """
        if not (-2.0 <= v <= 2.0):
            raise ValueError("Presence penalty must be between -2.0 and 2.0")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_cost_per_1k_tokens(self) -> float:
        """Estimate the cost per 1000 tokens for this model configuration.

        Returns:
            Estimated cost in USD per 1000 tokens
        """
        # Rough cost estimates (input + output average)
        model_costs = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.015,
            "gpt-3.5-turbo": 0.002,
            "claude-3-opus": 0.075,
            "claude-3-sonnet": 0.015,
            "claude-3-haiku": 0.001,
        }

        # Try exact match first, then fallback to partial match
        for model, cost in model_costs.items():
            if model in self.model_name.lower():
                return cost

        # Default for unknown models (assume expensive)
        return 0.03

    def format_prompt(self, **kwargs: Any) -> str:
        """Format the prompt template with provided keyword arguments.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            The formatted prompt string

        Raises:
            KeyError: If required template variables are missing
        """
        return self.prompt_template.format(**kwargs)

    model_config = ConfigDict(
        # Use enum values for serialization
        use_enum_values=True,
        # Validate assignment for computed fields
        validate_assignment=True,
    )
