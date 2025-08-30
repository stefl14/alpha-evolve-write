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

- **Type Hints**: Required on all functions and methods
- **Line Length**: 88 characters (Black default)
- **Import Order**: Enforced by ruff/isort
- **Docstrings**: Required for public functions and classes
- **Error Handling**: Use specific exception types, avoid bare `except:`

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

1. **Start with Tests**: Always write tests before implementation
2. **Use Make Commands**: Leverage Makefile for consistency
3. **Check Coverage**: Maintain high test coverage (>90%)
4. **Lint Early**: Run `make format lint type-check` frequently
5. **Docker Testing**: Test in containerized environment before commits
6. **Conventional Commits**: Use `make commit` for commitizen or manual format:
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