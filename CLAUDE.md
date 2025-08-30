# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaEvolve Essay Writer is an AI-powered iterative essay writing application inspired by DeepMind's AlphaEvolve. The system uses LLMs to evolutionarily improve essay drafts through generation, evaluation, selection, and variation cycles.

## Architecture

```
src/alpha_evolve_essay/
├── __init__.py           # Main entry point
├── core/                 # Core evolutionary pipeline
├── evaluation/          # Essay scoring and ranking
├── mcp_server/          # Model Context Protocol integration
└── models/              # Data models and types

tests/                   # Mirror source structure for tests
├── conftest.py          # Pytest fixtures and configuration
├── test_main.py         # Entry point tests
├── core/                # Core module tests
├── evaluation/          # Evaluation tests
├── mcp_server/          # MCP server tests
└── models/              # Model tests
```

## Development Commands

Essential commands for development workflow:

```bash
# Setup and Installation
make install              # Install dependencies with uv
make install-dev          # Install development dependencies

# Testing (TDD Workflow)
make test                 # Run all tests
make test-cov            # Run tests with coverage report
pytest tests/specific_test.py  # Run specific test file
pytest tests/core/ -v    # Run tests in specific module with verbose output

# Code Quality
make lint                # Run ruff linter
make format              # Format with black + ruff fixes  
make type-check          # Run mypy type checking
make ci                  # Run full CI pipeline (lint + type-check + test-cov)

# Development
make run                 # Run the application
make clean               # Clean build artifacts
make build               # Build package
make commit              # Interactive commitizen commit (use manual if fails)

# Docker
make docker-build        # Build Docker image
make docker-run          # Run in Docker container
```

## Test-Driven Development

This project follows strict TDD practices:

1. **Red**: Write failing tests first
2. **Green**: Write minimal code to pass tests  
3. **Refactor**: Improve code while keeping tests passing

### Testing Strategy

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete evolutionary cycles
- **Mock External APIs**: Use fixtures for OpenAI API and MCP servers

### Key Testing Patterns

```python
# Use provided fixtures in conftest.py
def test_something(mock_openai_client, sample_essay_prompt):
    # Test implementation
    pass

# Async testing
@pytest.mark.asyncio
async def test_async_function():
    # Async test implementation
    pass
```

## Code Style and Standards

### Type Annotations
- **Required**: Type hints on all functions, methods, and class attributes
- **Return Types**: Always specify return types, including `-> None`
- **Complex Types**: Use `typing` module for complex types
  ```python
  from typing import List, Dict, Optional, Union
  
  def process_essays(essays: List[Essay]) -> Dict[str, float]:
      """Process essays and return scores."""
  ```

### Docstrings (Google Style)
- **Required**: All public functions, classes, and modules
- **Format**: Google-style docstrings for consistency with type hints
  ```python
  def generate_essay(prompt: str, mode: str = "general") -> Essay:
      """Generate an essay from the given prompt.
      
      Args:
          prompt: The essay prompt or topic
          mode: Writing mode (creative, general, formal, technical)
          
      Returns:
          Essay object containing generated content and metadata
          
      Raises:
          EssayGenerationError: If generation fails
          ValueError: If mode is invalid
      """
  ```

### Import Organization  
- **Order**: Standard library → Third party → Local imports
- **Style**: Absolute imports preferred, relative only within packages
- **Grouping**: Separate groups with blank lines
  ```python
  import os
  from typing import List, Dict
  
  import openai
  from pydantic import BaseModel
  
  from alpha_evolve_essay.models import Essay
  from alpha_evolve_essay.core.exceptions import EssayError
  ```

### Code Structure
- **Line Length**: 88 characters (Black default)  
- **Naming**: 
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- **Error Handling**: Use specific exception types, never bare `except:`
  ```python
  try:
      result = api_call()
  except OpenAIError as e:
      logger.error(f"OpenAI API failed: {e}")
      raise EssayGenerationError(f"Generation failed: {e}") from e
  ```

### Testing Style
- **Test Names**: Descriptive, following `test_<what>_<condition>_<expected>`
  ```python
  def test_generate_essay_with_creative_mode_returns_essay_object():
  def test_evaluate_essay_with_invalid_criteria_raises_value_error():
  ```
- **Fixtures**: Use descriptive fixture names, avoid generic names
- **Assertions**: One logical assertion per test, use specific assertion methods
- **Mocking**: Mock at the service boundary, not internal implementation details

### Development Practices  
- **Single Responsibility**: Functions should do one thing well
- **Immutability**: Prefer immutable data structures, use dataclasses/Pydantic
- **Logging**: Use structured logging with appropriate levels
  ```python
  logger.info("Starting essay generation", extra={"prompt_length": len(prompt)})
  ```
- **Configuration**: Environment variables with sensible defaults
- **Async/Await**: Use async for I/O operations (API calls, file operations)

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional (with defaults)
MAX_GENERATIONS=10
INITIAL_DRAFT_COUNT=5
TOP_K_SELECTION=3
DEFAULT_MODE=general
```

## API Cost Management

**IMPORTANT**: This project can incur significant OpenAI API costs if not managed properly. Follow these practices:

### Development Without API Costs

1. **Use Mocks for All Tests**: Never make real API calls in tests
   ```python
   # All tests should use mock_openai_client fixture
   def test_essay_generation(mock_openai_client):
       # Test logic without API calls
   ```

2. **Environment Variables for Testing**:
   ```bash
   # Set in test environment
   USE_MOCK_LLM=true
   OPENAI_API_KEY=test_key_not_real
   ```

3. **Mock LLM Service**: Create a mock service that returns predefined responses
   ```python
   class MockLLMService:
       def generate_essay(self, prompt: str) -> str:
           return "Mock essay response for testing"
   ```

### Cost Control Strategies

- **Rate Limiting**: Implement request throttling
- **Small Models**: Use cheaper models (gpt-3.5-turbo) for development
- **Short Generations**: Limit max_tokens in development
- **Caching**: Cache API responses during development
- **Local Development Flag**: 
  ```python
  if os.getenv("DEVELOPMENT_MODE") == "true":
      return mock_llm_response()
  ```

### Real API Testing

Only use real API calls when:
- Testing final integration before deployment
- Validating specific model behavior
- Performance testing with small datasets

**Budget Alert**: Set OpenAI usage alerts and billing limits!

## Key Components

### Evolutionary Pipeline
- **Draft Generation**: Create initial essay variants
- **Evaluation**: Score essays using LLM judges or metrics
- **Selection**: Choose top-performing drafts
- **Variation**: Generate new drafts from selected parents
- **Iteration**: Repeat cycle for specified generations

### MCP Integration
- **Academic Search**: Query papers from Semantic Scholar/CrossRef
- **Web Search**: Real-time information retrieval
- **Tool Calling**: LLMs can invoke external tools during generation

### Evaluation Modes
- **LLM Judge**: Use GPT-4 to score essay quality
- **Pairwise Tournament**: ELO-based ranking system
- **Multi-criteria**: Score on coherence, grammar, style, factual accuracy

## Development Workflow

### TDD Process
1. **Red**: Write failing test first
2. **Green**: Write minimal code to pass test
3. **Refactor**: Improve code while keeping tests passing
4. **Repeat**: Continue cycle for each feature

### Daily Workflow
1. **Start with Tests**: Always write tests before implementation
2. **Single Test Focus**: Run specific tests, not whole suite (`pytest tests/test_specific.py::test_function`)
3. **Type Check Early**: Run `make type-check` after code changes
4. **Lint Frequently**: Run `make format lint` to catch issues early
5. **Test Coverage**: Maintain >90% coverage, check with `make test-cov`

### Quality Gates
- **Before Commits**: Always run `make ci` (lint + type-check + test-cov)
- **Code Review**: Ensure Google-style docstrings and type hints
- **Integration**: Test with Docker before pushing (`make docker-build && make docker-run`)

### Commit Standards
Use conventional commits via `make commit` or manual format:
- `feat:` new features
- `fix:` bug fixes  
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code restructuring
- `test:` adding tests
- `chore:` maintenance tasks

## Writing Modes

The system supports multiple essay writing modes:

- **Creative**: Focus on narrative techniques and literary style
- **General**: Balanced approach for general topics
- **Formal**: Academic writing with structured arguments
- **Technical**: Specialized technical documentation style

Each mode has distinct prompts, evaluation criteria, and LLM instructions.

## Common Patterns

### Async/Await
Most LLM interactions are asynchronous. Use proper async patterns:

```python
async def generate_essay(prompt: str) -> str:
    response = await client.chat.completions.create(...)
    return response.choices[0].message.content
```

### Error Handling
Wrap API calls with proper exception handling:

```python
try:
    result = await openai_client.generate(...)
except OpenAIError as e:
    logger.error(f"OpenAI API error: {e}")
    raise EssayGenerationError(f"Failed to generate essay: {e}")
```

### Configuration Management
Use environment variables with sensible defaults:

```python
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "10"))
```