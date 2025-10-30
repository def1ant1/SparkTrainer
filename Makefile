# Makefile for SparkTrainer development tasks

.PHONY: help install install-dev test lint format clean docs setup-pre-commit lock-deps

help:
	@echo "SparkTrainer Development Commands"
	@echo "=================================="
	@echo "make install          - Install production dependencies"
	@echo "make install-dev      - Install development dependencies"
	@echo "make test             - Run tests with coverage"
	@echo "make lint             - Run linters (flake8, mypy)"
	@echo "make format           - Format code with black and isort"
	@echo "make clean            - Clean build artifacts"
	@echo "make docs             - Build documentation"
	@echo "make setup-pre-commit - Install pre-commit hooks"
	@echo "make lock-deps        - Generate requirements lockfile"
	@echo "make security-check   - Run security scans"

install:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov=backend --cov-report=html --cov-report=term --cov-fail-under=80

test-quick:
	pytest tests/ -v --tb=short

lint:
	flake8 backend/ src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 backend/ src/ tests/ --count --max-complexity=10 --max-line-length=127 --statistics
	mypy backend/ src/ --ignore-missing-imports --no-strict-optional

format:
	black backend/ src/ tests/
	isort backend/ src/ tests/

format-check:
	black --check backend/ src/ tests/
	isort --check-only backend/ src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html
	@echo "Documentation built! Open docs/_build/html/index.html"

setup-pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed!"

lock-deps:
	pip-compile requirements.txt -o requirements.lock
	pip-compile backend/requirements.txt -o backend/requirements.lock
	pip-compile requirements-dev.txt -o requirements-dev.lock
	@echo "Lockfiles generated!"

security-check:
	safety check
	pip-audit
	bandit -r backend/ src/

run-backend:
	python backend/app.py

run-worker:
	celery -A backend.celery_app.celery worker --loglevel=info --concurrency=2

run-flower:
	celery -A backend.celery_app.celery flower --port=5555

run-frontend:
	cd frontend && npm run dev

docker-up:
	docker-compose up -d postgres redis mlflow

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

init-db:
	cd backend && python init_db.py --sample-data
