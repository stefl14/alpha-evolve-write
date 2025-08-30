.PHONY: help install install-dev test test-cov lint format type-check clean build run docker-build docker-run commit

# Default target
help:
	@echo "Available commands:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  lint          - Run linter (ruff)"
	@echo "  format        - Format code (black + ruff)"
	@echo "  type-check    - Run type checker (mypy)"
	@echo "  clean         - Clean build artifacts"
	@echo "  build         - Build the package"
	@echo "  run           - Run the application"
	@echo "  commit        - Interactive commitizen commit"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"

# Installation
install:
	uv sync

install-dev: install
	@echo "Development dependencies included in sync"

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/alpha_evolve_essay --cov-report=term-missing --cov-report=html

# Code quality
lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff check . --fix

type-check:
	uv run mypy src/

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Run
run:
	uv run alpha-evolve-essay

# Git and commits
commit:
	uv run cz commit

# Docker
docker-build:
	docker build -t alpha-evolve-essay .

docker-run:
	docker run --rm -it \
		--env-file .env \
		-v $(PWD):/app \
		alpha-evolve-essay

# Convenience targets
ci: lint type-check test-cov
all: format lint type-check test-cov build