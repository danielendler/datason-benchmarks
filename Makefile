# DataSON Benchmarks Makefile
# Simple commands for development workflow

.PHONY: workflows test lint clean install

# Generate GitHub Actions workflows from Python models
workflows:
	@echo "🔄 Generating GitHub Actions workflows..."
	@python -m tools.gen_workflows
	@echo "✅ Workflows generated successfully"

# Run all tests
test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v

# Run workflow generator tests specifically
test-workflows:
	@echo "🧪 Running workflow generator tests..."
	@python -m pytest tests/test_workflow_generator.py -v

# Run linting and formatting
lint:
	@echo "🔍 Running linters..."
	@python -m ruff check . --fix
	@python -m ruff format .

# Install development dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@pip install ruff pytest ruamel.yaml

# Clean up generated and temporary files
clean:
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Validate generated workflows
validate-workflows: workflows
	@echo "🔍 Validating generated workflows..."
	@python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/ci.yml', '.github/workflows/benchmarks.yml']]"
	@echo "✅ All workflows are valid YAML"

# Development setup
setup: install
	@echo "🚀 Setting up development environment..."
	@pip install pre-commit
	@pre-commit install
	@echo "✅ Development environment ready"

# Help
help:
	@echo "DataSON Benchmarks Development Commands:"
	@echo ""
	@echo "  make workflows        Generate GitHub Actions workflows from Python models"
	@echo "  make test            Run all tests"
	@echo "  make test-workflows  Run workflow generator tests"
	@echo "  make lint            Run linting and formatting"
	@echo "  make validate-workflows  Generate and validate workflows"
	@echo "  make install         Install dependencies"
	@echo "  make setup           Set up development environment"
	@echo "  make clean           Clean up temporary files"
	@echo "  make help            Show this help message"
	@echo ""
	@echo "To edit workflows:"
	@echo "  1. Edit tools/gen_workflows.py (or the model)"
	@echo "  2. Run 'make workflows'"
	@echo "  3. Commit the changes"