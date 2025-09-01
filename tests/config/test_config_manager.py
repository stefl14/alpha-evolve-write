"""Tests for ConfigManager and configuration system."""

import os
import tempfile

import pytest

from alpha_evolve_essay.config import ConfigManager, EvolutionConfig, SystemConfig


class TestSystemConfig:
    """Test cases for SystemConfig data model."""

    def test_system_config_initialization_with_defaults_succeeds(self):
        """Test SystemConfig initialization with default values."""
        config = SystemConfig()

        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30.0
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600

    def test_system_config_initialization_with_custom_values_succeeds(self):
        """Test SystemConfig initialization with custom values."""
        config = SystemConfig(
            debug=True,
            log_level="DEBUG",
            max_concurrent_requests=20,
            request_timeout=60.0,
            cache_enabled=False,
            cache_ttl=7200,
        )

        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.max_concurrent_requests == 20
        assert config.request_timeout == 60.0
        assert config.cache_enabled is False
        assert config.cache_ttl == 7200

    def test_system_config_validates_log_level(self):
        """Test that SystemConfig validates log level values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = SystemConfig(log_level=level)
            assert config.log_level == level

        with pytest.raises(ValueError, match="Log level must be one of"):
            SystemConfig(log_level="INVALID")

    def test_system_config_validates_positive_values(self):
        """Test that SystemConfig validates positive numeric values."""
        with pytest.raises(
            ValueError, match="Max concurrent requests must be positive"
        ):
            SystemConfig(max_concurrent_requests=0)

        with pytest.raises(ValueError, match="Request timeout must be positive"):
            SystemConfig(request_timeout=0.0)

        with pytest.raises(ValueError, match="Cache TTL must be positive"):
            SystemConfig(cache_ttl=0)


class TestEvolutionConfig:
    """Test cases for EvolutionConfig data model."""

    def test_evolution_config_initialization_with_defaults_succeeds(self):
        """Test EvolutionConfig initialization with default values."""
        config = EvolutionConfig()

        assert config.population_size == 10
        assert config.elite_size == 3
        assert config.crossover_rate == 0.3
        assert config.mutation_rate == 0.5
        assert config.max_generations == 5
        assert config.random_seed == 42

    def test_evolution_config_initialization_with_custom_values_succeeds(self):
        """Test EvolutionConfig initialization with custom values."""
        config = EvolutionConfig(
            population_size=20,
            elite_size=5,
            crossover_rate=0.4,
            mutation_rate=0.6,
            max_generations=10,
            random_seed=123,
        )

        assert config.population_size == 20
        assert config.elite_size == 5
        assert config.crossover_rate == 0.4
        assert config.mutation_rate == 0.6
        assert config.max_generations == 10
        assert config.random_seed == 123

    def test_evolution_config_validates_population_size(self):
        """Test that EvolutionConfig validates population size."""
        with pytest.raises(ValueError, match="Population size must be positive"):
            EvolutionConfig(population_size=0)

        with pytest.raises(ValueError, match="Population size must be positive"):
            EvolutionConfig(population_size=-1)

    def test_evolution_config_validates_elite_size(self):
        """Test that EvolutionConfig validates elite size constraints."""
        with pytest.raises(ValueError, match="Elite size must be positive"):
            EvolutionConfig(elite_size=0)

        with pytest.raises(
            ValueError, match="Elite size must be less than population size"
        ):
            EvolutionConfig(population_size=5, elite_size=5)

        with pytest.raises(
            ValueError, match="Elite size must be less than population size"
        ):
            EvolutionConfig(population_size=3, elite_size=4)

    def test_evolution_config_validates_rates(self):
        """Test that EvolutionConfig validates rate values."""
        with pytest.raises(ValueError, match="Crossover rate must be between 0 and 1"):
            EvolutionConfig(crossover_rate=-0.1)

        with pytest.raises(ValueError, match="Crossover rate must be between 0 and 1"):
            EvolutionConfig(crossover_rate=1.1)

        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            EvolutionConfig(mutation_rate=-0.1)

        with pytest.raises(ValueError, match="Mutation rate must be between 0 and 1"):
            EvolutionConfig(mutation_rate=1.1)

    def test_evolution_config_validates_max_generations(self):
        """Test that EvolutionConfig validates max generations."""
        with pytest.raises(ValueError, match="Max generations must be positive"):
            EvolutionConfig(max_generations=0)

        with pytest.raises(ValueError, match="Max generations must be positive"):
            EvolutionConfig(max_generations=-1)


class TestConfigManager:
    """Test cases for ConfigManager."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"system": {"debug": true}, "evolution": {"population_size": 15}}')
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def invalid_config_file(self):
        """Create temporary invalid config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": "json"')  # Invalid JSON
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_config_manager_initialization_with_defaults_succeeds(self):
        """Test ConfigManager initialization with default configurations."""
        manager = ConfigManager()

        assert manager.system is not None
        assert manager.evolution is not None
        assert isinstance(manager.system, SystemConfig)
        assert isinstance(manager.evolution, EvolutionConfig)

    def test_config_manager_load_from_file_succeeds(self, temp_config_file):
        """Test loading configuration from JSON file."""
        manager = ConfigManager.load_from_file(temp_config_file)

        assert manager.system.debug is True
        assert manager.evolution.population_size == 15
        # Other values should use defaults
        assert manager.system.log_level == "INFO"
        assert manager.evolution.elite_size == 3

    def test_config_manager_load_from_nonexistent_file_uses_defaults(self):
        """Test loading from nonexistent file uses default configurations."""
        manager = ConfigManager.load_from_file("nonexistent.json")

        assert isinstance(manager.system, SystemConfig)
        assert isinstance(manager.evolution, EvolutionConfig)
        assert manager.system.debug is False
        assert manager.evolution.population_size == 10

    def test_config_manager_load_from_invalid_file_uses_defaults(
        self, invalid_config_file
    ):
        """Test loading from invalid JSON file uses default configurations."""
        manager = ConfigManager.load_from_file(invalid_config_file)

        assert isinstance(manager.system, SystemConfig)
        assert isinstance(manager.evolution, EvolutionConfig)

    def test_config_manager_save_to_file_succeeds(self):
        """Test saving configuration to JSON file."""
        manager = ConfigManager()
        manager.system.debug = True
        manager.evolution.population_size = 20

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager.save_to_file(temp_path)

            # Verify file was created and contains expected data
            loaded_manager = ConfigManager.load_from_file(temp_path)
            assert loaded_manager.system.debug is True
            assert loaded_manager.evolution.population_size == 20

        finally:
            os.unlink(temp_path)

    def test_config_manager_load_from_environment_variables_succeeds(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        env_vars = {
            "AEW_SYSTEM_DEBUG": "true",
            "AEW_SYSTEM_LOG_LEVEL": "DEBUG",
            "AEW_EVOLUTION_POPULATION_SIZE": "25",
            "AEW_EVOLUTION_ELITE_SIZE": "6",
        }

        # Temporarily set environment variables
        original_values = {}
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            manager = ConfigManager.load_from_environment()

            assert manager.system.debug is True
            assert manager.system.log_level == "DEBUG"
            assert manager.evolution.population_size == 25
            assert manager.evolution.elite_size == 6

        finally:
            # Restore original environment
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_config_manager_get_merged_config_succeeds(self):
        """Test merging configurations from multiple sources."""
        # Create base manager
        base_manager = ConfigManager()
        base_manager.system.debug = False
        base_manager.evolution.population_size = 10

        # Create override manager
        override_manager = ConfigManager()
        override_manager.system.debug = True
        override_manager.system.log_level = "DEBUG"

        # Merge configurations
        merged = base_manager.get_merged_config(override_manager)

        assert merged.system.debug is True  # Overridden
        assert merged.system.log_level == "DEBUG"  # Overridden
        assert merged.evolution.population_size == 10  # From base

    def test_config_manager_validate_configuration_succeeds(self):
        """Test configuration validation with valid settings."""
        manager = ConfigManager()

        # Should not raise any exceptions
        manager.validate_configuration()

        # Test with valid custom values
        manager.evolution.population_size = 15
        manager.evolution.elite_size = 4
        manager.validate_configuration()

    def test_config_manager_validate_configuration_fails_with_invalid_settings(self):
        """Test configuration validation with invalid settings."""
        manager = ConfigManager()

        # Create invalid evolution config that bypasses individual validation
        # by constructing it with compatible values first, then testing validation
        try:
            # This should fail due to Pydantic's validation
            manager.evolution.elite_size = manager.evolution.population_size
            raise AssertionError("Expected validation error")
        except ValueError as e:
            assert "Elite size must be less than population size" in str(e)

    def test_config_manager_to_dict_returns_complete_config(self):
        """Test converting ConfigManager to dictionary representation."""
        manager = ConfigManager()
        manager.system.debug = True
        manager.evolution.population_size = 15

        config_dict = manager.to_dict()

        assert "system" in config_dict
        assert "evolution" in config_dict
        assert config_dict["system"]["debug"] is True
        assert config_dict["evolution"]["population_size"] == 15

    def test_config_manager_from_dict_creates_valid_manager(self):
        """Test creating ConfigManager from dictionary representation."""
        config_dict = {
            "system": {"debug": True, "log_level": "DEBUG"},
            "evolution": {"population_size": 20, "elite_size": 5},
        }

        manager = ConfigManager.from_dict(config_dict)

        assert manager.system.debug is True
        assert manager.system.log_level == "DEBUG"
        assert manager.evolution.population_size == 20
        assert manager.evolution.elite_size == 5

    def test_config_manager_update_system_config_succeeds(self):
        """Test updating system configuration dynamically."""
        manager = ConfigManager()

        original_debug = manager.system.debug
        manager.update_system_config(debug=True, log_level="DEBUG")

        assert manager.system.debug is True
        assert manager.system.log_level == "DEBUG"
        assert manager.system.debug != original_debug

    def test_config_manager_update_evolution_config_succeeds(self):
        """Test updating evolution configuration dynamically."""
        manager = ConfigManager()

        original_size = manager.evolution.population_size
        manager.update_evolution_config(population_size=25, elite_size=7)

        assert manager.evolution.population_size == 25
        assert manager.evolution.elite_size == 7
        assert manager.evolution.population_size != original_size

    def test_config_manager_update_evolution_config_validates_constraints(self):
        """Test that updating evolution config validates constraints."""
        manager = ConfigManager()

        with pytest.raises(
            ValueError, match="Elite size must be less than population size"
        ):
            manager.update_evolution_config(population_size=5, elite_size=5)
