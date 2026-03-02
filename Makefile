SHELL := /bin/sh

# =============================================================================
# Configuration
# =============================================================================
PKG := ids-evaluation-framework
IMAGE := niklassandhu/ids-eval-framework:latest
DOCKER ?= docker

# Default configuration file (can be overridden: make dataset CONFIG=my_config.yml)
CONFIG ?= run_config/example.config.yml

# Optional flags for evaluate command
TRAIN_ONLY ?=
FORCE_TRAIN ?=
FORCE_MODEL ?=

# Build evaluate flags string
EVAL_FLAGS :=
ifdef TRAIN_ONLY
	EVAL_FLAGS += --train-only
endif
ifdef FORCE_TRAIN
	EVAL_FLAGS += --force-train
endif
ifdef FORCE_MODEL
	EVAL_FLAGS += --force-model
endif

# =============================================================================
# Targets
# =============================================================================
.PHONY: help setup test lint lint-fix format typecheck check \
        version dataset evaluate \
        docker-build docker-push docker-version docker-dataset docker-evaluate \
        clean clean-all

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo "Makefile: help page for $(PKG)"
	@echo
	@echo "Usage:"
	@echo "  make <target> [OPTIONS]"
	@echo
	@echo "Options:"
	@echo "  CONFIG=<path>       Configuration file (default: $(CONFIG))"
	@echo "  TRAIN_ONLY=1        Only train models, skip testing"
	@echo "  FORCE_TRAIN=1       Force retraining, ignore saved models"
	@echo "  FORCE_MODEL=1       Load saved models without config validation"
	@echo
	@echo "Development:"
	@echo "  setup               Install dependencies and set up environment"
	@echo "  test                Run test suite"
	@echo "  lint                Check code with ruff"
	@echo "  lint-fix            Fix linting issues automatically"
	@echo "  format              Format code with black"
	@echo "  typecheck           Run type checking with mypy"
	@echo "  check               Run all checks (lint + typecheck)"
	@echo
	@echo "Native Execution:"
	@echo "  version             Show framework version"
	@echo "  dataset             Run data ingestion pipeline"
	@echo "  evaluate            Run evaluation pipeline"
	@echo
	@echo "Docker:"
	@echo "  docker-build        Build Docker image"
	@echo "  docker-push         Push Docker image to registry"
	@echo "  docker-version      Show version (via Docker)"
	@echo "  docker-dataset      Run data ingestion (via Docker Compose)"
	@echo "  docker-evaluate     Run evaluation (via Docker Compose)"
	@echo
	@echo "Cleanup:"
	@echo "  clean               Remove output files (processed data, reports, models)"
	@echo "  clean-all           Remove all generated files including cache"
	@echo
	@echo "Examples:"
	@echo "  make dataset CONFIG=run_config/my_config.yml"
	@echo "  make evaluate CONFIG=run_config/my_config.yml"
	@echo "  make evaluate CONFIG=run_config/my_config.yml FORCE_TRAIN=1"
	@echo "  make evaluate CONFIG=run_config/my_config.yml FORCE_MODEL=1"
	@echo "  make evaluate TRAIN_ONLY=1"

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------
setup:
	uv sync

test:
	uv run -m pytest -q

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run black .

typecheck:
	uv run mypy src/

check: lint typecheck

# -----------------------------------------------------------------------------
# Native Execution
# -----------------------------------------------------------------------------
version:
	uv run ids-eval version

dataset:
	uv run ids-eval dataset $(CONFIG)

evaluate:
	uv run ids-eval evaluate $(CONFIG) $(EVAL_FLAGS)

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build:
	$(DOCKER) build -t $(IMAGE) .

# Note: docker-push destination needs to be set up with appropriate permissions (e.g., Docker Hub) if you want to push your own image.
docker-push: docker-build
	$(DOCKER) push $(IMAGE)

docker-version:
	$(DOCKER) compose run --rm ids-eval version

docker-dataset:
	$(DOCKER) compose run --rm ids-eval dataset $(CONFIG)

docker-evaluate:
	$(DOCKER) compose run --rm ids-eval evaluate $(CONFIG) $(EVAL_FLAGS)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	rm -rf out/processed_datasets/*
	rm -rf out/reports/*
	rm -rf out/saved_models/*

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf **/__pycache__/
	rm -rf *.egg-info/
	rm -rf logs/*.log*
