"""Tests for GenerationConfig model."""

import pytest

from alpha_evolve_essay.models import GenerationConfig


class TestGenerationConfigModel:
    """Test cases for GenerationConfig model validation and behavior."""

    def test_generation_config_creation_with_minimal_fields_succeeds(self):
        """Test that GenerationConfig can be created with required fields."""
        config = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Write an essay about: {topic}"
        )

        assert config.model_name == "gpt-3.5-turbo"
        assert config.prompt_template == "Write an essay about: {topic}"
        assert config.temperature == 0.7  # Default value
        assert config.max_tokens == 1000  # Default value
        assert config.top_p == 1.0  # Default value
        assert config.frequency_penalty == 0.0  # Default value
        assert config.presence_penalty == 0.0  # Default value
        assert config.mode == "general"  # Default value
        assert config.additional_params == {}  # Default value

    def test_generation_config_creation_with_all_fields_succeeds(self):
        """Test that GenerationConfig accepts all optional fields."""
        additional_params = {"seed": 42, "custom_param": "value"}

        config = GenerationConfig(
            model_name="gpt-4",
            prompt_template="Creative writing task: {topic}\nStyle: {style}",
            temperature=0.8,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            mode="creative",
            additional_params=additional_params,
        )

        assert config.model_name == "gpt-4"
        assert (
            config.prompt_template == "Creative writing task: {topic}\nStyle: {style}"
        )
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.mode == "creative"
        assert config.additional_params == additional_params

    def test_generation_config_model_name_validation_rejects_empty_string(self):
        """Test that model_name cannot be empty."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            GenerationConfig(model_name="", prompt_template="Test prompt")

        with pytest.raises(ValueError, match="Model name cannot be empty"):
            GenerationConfig(model_name="   ", prompt_template="Test prompt")

    def test_generation_config_prompt_template_validation_rejects_empty_string(self):
        """Test that prompt_template cannot be empty."""
        with pytest.raises(ValueError, match="Prompt template cannot be empty"):
            GenerationConfig(model_name="gpt-3.5-turbo", prompt_template="")

        with pytest.raises(ValueError, match="Prompt template cannot be empty"):
            GenerationConfig(model_name="gpt-3.5-turbo", prompt_template="   ")

    def test_generation_config_temperature_validation_accepts_valid_range(self):
        """Test that temperature accepts values in valid range 0.0-2.0."""
        # Test boundary values
        config_min = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", temperature=0.0
        )
        assert config_min.temperature == 0.0

        config_max = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", temperature=2.0
        )
        assert config_max.temperature == 2.0

        config_mid = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", temperature=1.5
        )
        assert config_mid.temperature == 1.5

    def test_generation_config_temperature_validation_rejects_invalid_range(self):
        """Test that temperature rejects values outside 0.0-2.0 range."""
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", temperature=-0.1
            )

        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", temperature=2.1
            )

    def test_generation_config_max_tokens_validation_accepts_positive_values(self):
        """Test that max_tokens accepts positive integer values."""
        config = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", max_tokens=1
        )
        assert config.max_tokens == 1

        config = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", max_tokens=4000
        )
        assert config.max_tokens == 4000

    def test_generation_config_max_tokens_validation_rejects_non_positive(self):
        """Test that max_tokens rejects zero or negative values."""
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", max_tokens=0
            )

        with pytest.raises(ValueError, match="Max tokens must be positive"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", max_tokens=-1
            )

    def test_generation_config_top_p_validation_accepts_valid_range(self):
        """Test that top_p accepts values in valid range 0.0-1.0."""
        config_min = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", top_p=0.0
        )
        assert config_min.top_p == 0.0

        config_max = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test", top_p=1.0
        )
        assert config_max.top_p == 1.0

    def test_generation_config_top_p_validation_rejects_invalid_range(self):
        """Test that top_p rejects values outside 0.0-1.0 range."""
        with pytest.raises(ValueError, match="Top-p must be between 0.0 and 1.0"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", top_p=-0.1
            )

        with pytest.raises(ValueError, match="Top-p must be between 0.0 and 1.0"):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", top_p=1.1
            )

    def test_generation_config_penalty_validation_accepts_valid_range(self):
        """Test that frequency and presence penalties accept valid range -2.0 to 2.0."""
        config = GenerationConfig(
            model_name="gpt-3.5-turbo",
            prompt_template="Test",
            frequency_penalty=-2.0,
            presence_penalty=2.0,
        )
        assert config.frequency_penalty == -2.0
        assert config.presence_penalty == 2.0

    def test_generation_config_penalty_validation_rejects_invalid_range(self):
        """Test that penalties reject values outside -2.0 to 2.0 range."""
        with pytest.raises(
            ValueError, match="Frequency penalty must be between -2.0 and 2.0"
        ):
            GenerationConfig(
                model_name="gpt-3.5-turbo",
                prompt_template="Test",
                frequency_penalty=-2.1,
            )

        with pytest.raises(
            ValueError, match="Presence penalty must be between -2.0 and 2.0"
        ):
            GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", presence_penalty=2.1
            )

    def test_generation_config_mode_validation_accepts_valid_modes(self):
        """Test that mode accepts valid writing modes."""
        valid_modes = ["general", "creative", "formal", "technical"]

        for mode in valid_modes:
            config = GenerationConfig(
                model_name="gpt-3.5-turbo", prompt_template="Test", mode=mode
            )
            assert config.mode == mode

    def test_generation_config_mode_validation_rejects_invalid_modes(self):
        """Test that mode rejects invalid writing modes."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerationConfig(
                model_name="gpt-3.5-turbo",
                prompt_template="Test",
                mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_generation_config_prompt_template_format_validation(self):
        """Test prompt template format string validation."""
        # Valid template with placeholders
        config = GenerationConfig(
            model_name="gpt-3.5-turbo",
            prompt_template="Write about {topic} in {style} style",
        )

        # Should be able to format with valid keys
        formatted = config.format_prompt(topic="AI", style="academic")
        assert formatted == "Write about AI in academic style"

    def test_generation_config_prompt_template_missing_keys_raises_error(self):
        """Test that formatting with missing keys raises KeyError."""
        config = GenerationConfig(
            model_name="gpt-3.5-turbo",
            prompt_template="Write about {topic} in {style} style",
        )

        with pytest.raises(KeyError):
            config.format_prompt(topic="AI")  # Missing 'style' key

    def test_generation_config_prompt_template_extra_keys_ignored(self):
        """Test that extra keys in format_prompt are ignored."""
        config = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Write about {topic}"
        )

        formatted = config.format_prompt(topic="AI", extra_key="ignored")
        assert formatted == "Write about AI"

    def test_generation_config_json_serialization_works(self):
        """Test that GenerationConfig can be serialized to and from JSON."""
        original = GenerationConfig(
            model_name="gpt-4",
            prompt_template="Write about {topic}",
            temperature=0.8,
            max_tokens=1500,
            mode="creative",
            additional_params={"seed": 123},
        )

        # Serialize to dict
        data = original.model_dump()
        assert data["model_name"] == "gpt-4"
        assert data["prompt_template"] == "Write about {topic}"
        assert data["temperature"] == 0.8
        assert data["max_tokens"] == 1500
        assert data["mode"] == "creative"
        assert data["additional_params"] == {"seed": 123}

        # Deserialize from dict
        restored = GenerationConfig(**data)
        assert restored.model_name == original.model_name
        assert restored.prompt_template == original.prompt_template
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens
        assert restored.mode == original.mode
        assert restored.additional_params == original.additional_params

    def test_generation_config_equality_comparison_works(self):
        """Test that GenerationConfig instances can be compared for equality."""
        config1 = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test template"
        )
        config2 = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test template"
        )
        config3 = GenerationConfig(model_name="gpt-4", prompt_template="Test template")

        # Same configuration should be equal
        assert config1.model_name == config2.model_name
        assert config1.prompt_template == config2.prompt_template
        assert config1.temperature == config2.temperature

        # Different configuration should not be equal
        assert config1.model_name != config3.model_name

    def test_generation_config_cost_estimation_property(self):
        """Test estimated cost per 1000 tokens property."""
        gpt35_config = GenerationConfig(
            model_name="gpt-3.5-turbo", prompt_template="Test"
        )

        gpt4_config = GenerationConfig(model_name="gpt-4", prompt_template="Test")

        # GPT-4 should be more expensive than GPT-3.5-turbo
        assert (
            gpt4_config.estimated_cost_per_1k_tokens
            > gpt35_config.estimated_cost_per_1k_tokens
        )
        assert gpt35_config.estimated_cost_per_1k_tokens > 0.0
