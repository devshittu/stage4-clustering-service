.PHONY: help test test-unit test-integration test-e2e test-docker test-coverage clean

help:
	@echo "Stage 4 Clustering Service - Test Commands"
	@echo ""
	@echo "Local Testing:"
	@echo "  make test              - Run all tests locally"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-e2e          - Run end-to-end tests only"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "Docker Testing:"
	@echo "  make test-docker       - Run tests in Docker (unit tests)"
	@echo "  make test-docker-all   - Run all tests in Docker"
	@echo "  make test-docker-integration - Run integration tests in Docker"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Run linters (flake8, black check)"
	@echo "  make format            - Format code with black and isort"
	@echo "  make type-check        - Run mypy type checking"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Clean test artifacts and cache"

# Local testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-coverage:
	pytest tests/ -v \
		--cov=src \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=80

test-quick:
	pytest tests/unit/ -v -m "unit and not slow"

# Docker testing
test-docker:
	docker compose -f docker-compose.test.yml up --build test-runner

test-docker-integration:
	docker compose -f docker-compose.test.yml up --build test-integration

test-docker-all:
	docker compose -f docker-compose.test.yml up --build test-all

test-docker-clean:
	docker compose -f docker-compose.test.yml down -v
	docker rmi clustering-test-runner clustering-test-integration clustering-test-all 2>/dev/null || true

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	black src/ tests/ --check --line-length=100

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

type-check:
	mypy src/ --strict --ignore-missing-imports

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf .temp/*.md 2>/dev/null || true

clean-all: clean test-docker-clean
	rm -rf data/clusters/*.jsonl
	rm -rf logs/*.log

# Development
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-simple.txt
	pip install pytest pytest-cov pytest-asyncio pytest-mock black flake8 mypy isort

# CI/CD
ci-test:
	pytest tests/ -v \
		--cov=src \
		--cov-report=xml \
		--cov-fail-under=80 \
		--junitxml=junit.xml \
		-m "not gpu and not requires_stage3"
