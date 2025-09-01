# AlphaEvolve Essay Writer

An evolutionary essay writing application inspired by DeepMind's AlphaEvolve approach. This system uses Large Language Models (LLMs) to iteratively improve essays through generation, evaluation, selection, and variation cycles.

## Features

🧬 **Evolutionary Algorithm**: Implements AlphaEvolve-inspired iterative improvement
📝 **Multi-Mode Writing**: Supports general, creative, formal, and technical writing styles  
🎯 **Multi-Criteria Evaluation**: Essays evaluated on coherence, depth, creativity, clarity, and engagement
🚀 **Concurrent Processing**: Efficient parallel generation and evaluation
💰 **Cost Management**: MockLLMService for development without API costs
📊 **Comprehensive Analytics**: Detailed evolution statistics and progress tracking
🔬 **Test-Driven Development**: 139 tests with 99% coverage

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd alpha-evolve-essay

# Install dependencies with uv
make install

# Activate virtual environment
source .venv/bin/activate
```

### Basic Usage

```python
from alpha_evolve_essay.models import GenerationConfig
from alpha_evolve_essay.pipeline import EvolutionaryPipeline
from alpha_evolve_essay.services import MockLLMService

# Initialize the system
llm_service = MockLLMService(seed=42)
config = GenerationConfig(
    model_name="mock-model",
    prompt_template="Write an essay about {topic}",
    temperature=0.7
)

# Create pipeline
pipeline = EvolutionaryPipeline(llm_service, config)

# Run evolution
results = await pipeline.evolve(
    prompt="artificial intelligence in education",
    generations=5,
    population_size=10,
    elite_size=3
)

# Access results
best_essay = results["best_essay"]
final_score = results["final_best_score"]
improvement = results["improvement"]
```

### Quick Essay Improvement

```python
from alpha_evolve_essay.models import Essay

# Improve a single essay
essay = Essay(
    content="Basic essay about renewable energy...",
    prompt="Write about renewable energy"
)

improvement_results = await pipeline.quick_improve(
    essay, 
    iterations=3
)

improved_essay = improvement_results["improved_essay"]
total_improvement = improvement_results["total_improvement"]
```

## Architecture

### Core Components

- **EssayGenerator**: Creates initial populations and generates variations/crossovers
- **EssayEvaluator**: Evaluates essays using structured LLM prompts with multiple criteria
- **EvolutionaryPipeline**: Main orchestrator coordinating the evolutionary process
- **MockLLMService**: Cost-free LLM service for development and testing

### Data Models

- **Essay**: Core essay model with content, metadata, and versioning
- **EssayPool**: Collection management with ranking and filtering capabilities
- **EvaluationResult**: Structured evaluation results with criteria scores and feedback
- **GenerationConfig**: LLM configuration with validation and cost estimation

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make ci

# Run specific test file
uv run pytest tests/pipeline/test_evolutionary_pipeline.py
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all quality checks
make ci
```

### Development Tools

- **Python 3.13** with modern type annotations
- **uv** for fast dependency management
- **pytest** for comprehensive testing
- **black** + **ruff** for code formatting and linting
- **mypy** for static type checking
- **Docker** support for containerized development

## Configuration

### Writing Modes

The system supports four writing modes:

- **General**: Balanced, analytical writing
- **Creative**: Imaginative, artistic expression
- **Formal**: Academic, scholarly style
- **Technical**: Engineering, systematic analysis

### Evolution Parameters

Configure the evolutionary process:

```python
pipeline.configure_evolution(
    population_size=15,      # Essays per generation
    elite_size=4,           # Top essays to preserve
    crossover_rate=0.3,     # Fraction from crossovers
    mutation_rate=0.5,      # Fraction from mutations
    max_generations=8,      # Evolution cycles
    random_seed=42          # Reproducible results
)
```

### Evaluation Criteria

Default evaluation weights:

- **Coherence** (25%): Logical flow and structure
- **Depth** (25%): Thoroughness and insight
- **Creativity** (20%): Originality and innovation
- **Clarity** (15%): Clear communication
- **Engagement** (15%): Reader interest

Custom criteria can be provided:

```python
custom_criteria = {
    "coherence": 0.4,
    "creativity": 0.6
}
```

## Testing

The project maintains high test coverage with comprehensive test suites:

- **139 tests** across all components
- **99% code coverage**
- **Async/await testing** with pytest-asyncio
- **Deterministic testing** with MockLLMService
- **Integration tests** for full pipeline workflows

## API Cost Management

### MockLLMService

For development and testing:

```python
# Deterministic responses based on prompt hashing
mock_service = MockLLMService(seed=42)

# Supports all writing modes
# Respects temperature and token limits
# Zero API costs
```

### Production LLM Integration

The abstract `LLMService` interface supports:

- OpenAI GPT models
- Anthropic Claude models  
- Custom LLM providers
- Cost estimation and tracking

## Contributing

1. **Follow TDD**: Write tests before implementation
2. **Maintain Coverage**: Keep test coverage above 95%
3. **Code Quality**: All CI checks must pass
4. **Documentation**: Update docstrings and README
5. **Conventional Commits**: Use conventional commit format

### Commit Process

```bash
# Stage changes
git add .

# Create conventional commit
make commit

# The system will guide you through:
# - Type selection (feat, fix, docs, etc.)
# - Scope specification
# - Description and body
```

## License

[Your License Here]

## Acknowledgments

Inspired by DeepMind's AlphaEvolve approach to iterative improvement through evolutionary algorithms.

---

**Status**: Core evolutionary pipeline complete ✅  
**Coverage**: 99% test coverage across 139 tests  
**Quality**: All linting, formatting, and type checking passes