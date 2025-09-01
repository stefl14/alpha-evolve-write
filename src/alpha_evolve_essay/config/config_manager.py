"""Configuration management system for AlphaEvolve Essay Writer."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class SystemConfig(BaseModel):
    """System-level configuration settings."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    debug: bool = False
    log_level: str = "INFO"
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    cache_enabled: bool = True
    cache_ttl: int = 3600

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v

    @field_validator("max_concurrent_requests")
    @classmethod
    def validate_max_concurrent_requests(cls, v: int) -> int:
        """Validate max concurrent requests is positive."""
        if v <= 0:
            raise ValueError("Max concurrent requests must be positive")
        return v

    @field_validator("request_timeout")
    @classmethod
    def validate_request_timeout(cls, v: float) -> float:
        """Validate request timeout is positive."""
        if v <= 0.0:
            raise ValueError("Request timeout must be positive")
        return v

    @field_validator("cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL is positive."""
        if v <= 0:
            raise ValueError("Cache TTL must be positive")
        return v


class EvolutionConfig(BaseModel):
    """Evolution algorithm configuration settings."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    population_size: int = 10
    elite_size: int = 3
    crossover_rate: float = 0.3
    mutation_rate: float = 0.5
    max_generations: int = 5
    random_seed: int = 42

    @field_validator("population_size")
    @classmethod
    def validate_population_size(cls, v: int) -> int:
        """Validate population size is positive."""
        if v <= 0:
            raise ValueError("Population size must be positive")
        return v

    @field_validator("elite_size")
    @classmethod
    def validate_elite_size(cls, v: int, info) -> int:
        """Validate elite size is positive and less than population size."""
        if v <= 0:
            raise ValueError("Elite size must be positive")

        # Check against population_size if it's available in the context
        if hasattr(info, "data") and "population_size" in info.data:
            population_size = info.data["population_size"]
            if v >= population_size:
                raise ValueError("Elite size must be less than population size")

        return v

    @field_validator("crossover_rate")
    @classmethod
    def validate_crossover_rate(cls, v: float) -> float:
        """Validate crossover rate is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Crossover rate must be between 0 and 1")
        return v

    @field_validator("mutation_rate")
    @classmethod
    def validate_mutation_rate(cls, v: float) -> float:
        """Validate mutation rate is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Mutation rate must be between 0 and 1")
        return v

    @field_validator("max_generations")
    @classmethod
    def validate_max_generations(cls, v: int) -> int:
        """Validate max generations is positive."""
        if v <= 0:
            raise ValueError("Max generations must be positive")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate cross-field constraints after model initialization."""
        if self.elite_size >= self.population_size:
            raise ValueError("Elite size must be less than population size")


class ConfigManager:
    """Central configuration manager for the AlphaEvolve Essay Writer system."""

    def __init__(
        self,
        system_config: SystemConfig | None = None,
        evolution_config: EvolutionConfig | None = None,
    ) -> None:
        """Initialize ConfigManager with optional custom configurations.

        Args:
            system_config: Custom system configuration (uses defaults if None)
            evolution_config: Custom evolution configuration (uses defaults if None)
        """
        self.system = system_config or SystemConfig()
        self.evolution = evolution_config or EvolutionConfig()

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "ConfigManager":
        """Load configuration from a JSON file.

        Args:
            file_path: Path to the configuration JSON file

        Returns:
            ConfigManager instance with loaded configuration
        """
        file_path = Path(file_path)

        if not file_path.exists():
            # Return default configuration if file doesn't exist
            return cls()

        try:
            with open(file_path) as f:
                config_data = json.load(f)

            system_data = config_data.get("system", {})
            evolution_data = config_data.get("evolution", {})

            system_config = SystemConfig(**system_data)
            evolution_config = EvolutionConfig(**evolution_data)

            return cls(system_config, evolution_config)

        except (json.JSONDecodeError, ValueError, TypeError):
            # Return default configuration if file is invalid
            return cls()

    @classmethod
    def load_from_environment(cls, prefix: str = "AEW") -> "ConfigManager":
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: AEW)

        Returns:
            ConfigManager instance with environment-based configuration
        """
        system_data = {}
        evolution_data = {}

        # System configuration from environment
        system_env_map = {
            f"{prefix}_SYSTEM_DEBUG": ("debug", lambda x: x.lower() == "true"),
            f"{prefix}_SYSTEM_LOG_LEVEL": ("log_level", str),
            f"{prefix}_SYSTEM_MAX_CONCURRENT_REQUESTS": (
                "max_concurrent_requests",
                int,
            ),
            f"{prefix}_SYSTEM_REQUEST_TIMEOUT": ("request_timeout", float),
            f"{prefix}_SYSTEM_CACHE_ENABLED": (
                "cache_enabled",
                lambda x: x.lower() == "true",
            ),
            f"{prefix}_SYSTEM_CACHE_TTL": ("cache_ttl", int),
        }

        for env_key, (config_key, converter) in system_env_map.items():
            if env_key in os.environ:
                try:
                    system_data[config_key] = converter(os.environ[env_key])
                except (ValueError, TypeError):
                    # Skip invalid environment values
                    pass

        # Evolution configuration from environment
        evolution_env_map = {
            f"{prefix}_EVOLUTION_POPULATION_SIZE": ("population_size", int),
            f"{prefix}_EVOLUTION_ELITE_SIZE": ("elite_size", int),
            f"{prefix}_EVOLUTION_CROSSOVER_RATE": ("crossover_rate", float),
            f"{prefix}_EVOLUTION_MUTATION_RATE": ("mutation_rate", float),
            f"{prefix}_EVOLUTION_MAX_GENERATIONS": ("max_generations", int),
            f"{prefix}_EVOLUTION_RANDOM_SEED": ("random_seed", int),
        }

        for env_key, (config_key, converter) in evolution_env_map.items():
            if env_key in os.environ:
                try:
                    evolution_data[config_key] = converter(os.environ[env_key])
                except (ValueError, TypeError):
                    # Skip invalid environment values
                    pass

        system_config = SystemConfig(**system_data)
        evolution_config = EvolutionConfig(**evolution_data)

        return cls(system_config, evolution_config)

    def save_to_file(self, file_path: str | Path) -> None:
        """Save current configuration to a JSON file.

        Args:
            file_path: Path where configuration should be saved
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = self.to_dict()

        with open(file_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_merged_config(self, override_manager: "ConfigManager") -> "ConfigManager":
        """Merge this configuration with another, using override values where available.

        Args:
            override_manager: ConfigManager with override values

        Returns:
            New ConfigManager with merged configuration
        """
        # Convert both to dictionaries
        base_data = self.to_dict()
        override_data = override_manager.to_dict()

        # Deep merge dictionaries
        merged_data = base_data.copy()
        for section_key, section_data in override_data.items():
            if section_key in merged_data:
                merged_data[section_key].update(section_data)
            else:
                merged_data[section_key] = section_data

        return self.from_dict(merged_data)

    def validate_configuration(self) -> None:
        """Validate the entire configuration for consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Individual model validation is handled by Pydantic
        # Additional cross-model validation can be added here

        # Validate evolution constraints
        if self.evolution.elite_size >= self.evolution.population_size:
            raise ValueError("Elite size must be less than population size")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary representation.

        Returns:
            Dictionary containing all configuration data
        """
        return {
            "system": self.system.model_dump(),
            "evolution": self.evolution.model_dump(),
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ConfigManager":
        """Create ConfigManager from dictionary representation.

        Args:
            config_dict: Dictionary containing configuration data

        Returns:
            ConfigManager instance created from dictionary
        """
        system_data = config_dict.get("system", {})
        evolution_data = config_dict.get("evolution", {})

        system_config = SystemConfig(**system_data)
        evolution_config = EvolutionConfig(**evolution_data)

        return cls(system_config, evolution_config)

    def update_system_config(self, **kwargs: Any) -> None:
        """Update system configuration with new values.

        Args:
            **kwargs: System configuration fields to update
        """
        current_data = self.system.model_dump()
        current_data.update(kwargs)
        self.system = SystemConfig(**current_data)

    def update_evolution_config(self, **kwargs: Any) -> None:
        """Update evolution configuration with new values.

        Args:
            **kwargs: Evolution configuration fields to update

        Raises:
            ValueError: If updated configuration is invalid
        """
        current_data = self.evolution.model_dump()
        current_data.update(kwargs)
        new_config = EvolutionConfig(**current_data)

        # Additional validation for cross-field constraints
        if new_config.elite_size >= new_config.population_size:
            raise ValueError("Elite size must be less than population size")

        self.evolution = new_config
