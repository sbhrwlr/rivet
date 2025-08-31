.PHONY: help install test lint format type-check clean dev-install example

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	uv sync

dev-install: ## Install with development dependencies
	uv sync --dev

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ --cov=flowt --cov-report=term-missing

lint: ## Run linting
	uv run ruff check flowt

format: ## Format code
	uv run ruff format flowt

type-check: ## Run type checking
	uv run mypy flowt

check: lint type-check test ## Run all checks (lint, type-check, test)

example: ## Run the basic pipeline example
	uv run python examples/basic_pipeline.py

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv build

publish: ## Publish to PyPI (requires authentication)
	uv publish

lock: ## Update the lock file
	uv lock

add: ## Add a new dependency (usage: make add PACKAGE=package_name)
	uv add $(PACKAGE)

add-dev: ## Add a new development dependency (usage: make add-dev PACKAGE=package_name)
	uv add --dev $(PACKAGE)