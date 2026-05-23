.PHONY: help install test lint typecheck clean run-server run-client docker-build docker-up

help:
	@echo "ASTRA — Async Scalable Training & Research Architecture"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies (core + dev)"
	@echo "  make test          - Run unit tests with pytest"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run ruff linter"
	@echo "  make typecheck     - Run mypy type checker"
	@echo "  make clean         - Clean generated files and caches"
	@echo "  make run-server    - Start the FL API server (dev mode)"
	@echo "  make run-client    - Run an FL client"
	@echo "  make docker-build  - Build all Docker images"
	@echo "  make docker-up     - Start all services via docker-compose"
	@echo "  make fmt           - Auto-format code with ruff"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=astra --cov-report=term-missing

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

typecheck:
	mypy src/astra/

clean:
	rm -rf __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/

run-server:
	uvicorn astra.app.server_api:app --reload --host 0.0.0.0 --port 8000

run-client:
	python -m astra.client.cli --server http://localhost:8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d
