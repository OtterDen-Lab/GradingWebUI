SHELL := /bin/sh

PYTHON ?= python3
UV ?= uv
APP_MODULE ?= grading_web_ui.web_api.main:app
HOST ?= 127.0.0.1
PORT ?= 8765
DB_DIR ?= .tmp
DB_PATH ?= $(DB_DIR)/grading.db
DOCKER_COMPOSE ?= docker compose -f docker/web-grading/docker-compose.yml
DOCKER_COMPOSE_PROD ?= docker compose -f docker/web-grading/docker-compose.prod.yml
DOCKER_IMAGE ?= ghcr.io/otterden-lab/gradingwebui:latest
DOCKER_ENV_FILE ?= /etc/grading-web/web.env
DOCKER_LOCAL_ENV_FILE ?= docker/web-grading/.env
DEPLOY_ENV_VALIDATOR ?= scripts/validate_deploy_env.py
QUIZGEN_PATH ?= /Users/ssogden/repos/teaching/QuizGeneration

.PHONY: help install install-quizgen-local run run-reload test test-api \
	docker-build docker-up docker-up-build docker-restart-web docker-logs docker-down docker-ps docker-shell \
	docker-prod-pull docker-prod-up docker-prod-down docker-prod-logs docker-prod-ps \
	validate-local-env validate-prod-env deploy deploy-prod

help:
	@echo "Targets:"
	@echo "  install            Install editable package + dev dependencies"
	@echo "  install-quizgen-local Install local QuizGeneration into current env"
	@echo "  run                Run API locally (uvicorn, no reload)"
	@echo "  run-reload         Run API locally with uvicorn --reload"
	@echo "  test               Run full pytest suite"
	@echo "  test-api           Run API tests only"
	@echo "  docker-build       Build Docker image(s)"
	@echo "  docker-up          Start Docker services in background"
	@echo "  docker-up-build    Start Docker services and rebuild first"
	@echo "  docker-restart-web Restart only the web container"
	@echo "  docker-logs        Tail logs for all services"
	@echo "  docker-down        Stop Docker services"
	@echo "  docker-ps          Show Docker service status"
	@echo "  docker-shell       Open shell in running web container"
	@echo "  docker-prod-pull   Pull production image tag"
	@echo "  docker-prod-up     Start production container(s)"
	@echo "  docker-prod-down   Stop production container(s)"
	@echo "  docker-prod-logs   Tail production logs"
	@echo "  docker-prod-ps     Show production service status"
	@echo "  validate-local-env Validate docker/web-grading/.env for local deploy"
	@echo "  validate-prod-env  Validate server env file for production deploy"
	@echo "  deploy             Validate env + rebuild/redeploy local Docker stack"
	@echo "  deploy-prod        Validate env + pull/redeploy production image stack"

install:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) pip install -e ".[dev]"; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

install-quizgen-local:
	@if [ ! -d "$(QUIZGEN_PATH)" ]; then \
		echo "QuizGeneration path not found: $(QUIZGEN_PATH)"; \
		exit 1; \
	fi
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) pip install -e "$(QUIZGEN_PATH)"; \
	else \
		$(PYTHON) -m pip install -e "$(QUIZGEN_PATH)"; \
	fi

run:
	@mkdir -p $(DB_DIR)
	GRADING_DB_PATH=$(DB_PATH) $(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

run-reload:
	@mkdir -p $(DB_DIR)
	GRADING_DB_PATH=$(DB_PATH) $(PYTHON) -m uvicorn $(APP_MODULE) --reload --host $(HOST) --port $(PORT)

test:
	pytest -q

test-api:
	pytest -q tests/test_api.py

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-up-build:
	$(DOCKER_COMPOSE) up -d --build

docker-restart-web:
	$(DOCKER_COMPOSE) restart web

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-down:
	$(DOCKER_COMPOSE) down

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-shell:
	$(DOCKER_COMPOSE) exec web /bin/sh

docker-prod-pull:
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) pull

docker-prod-up:
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) up -d

docker-prod-down:
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) down

docker-prod-logs:
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) logs -f

docker-prod-ps:
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) ps

validate-local-env:
	@$(PYTHON) $(DEPLOY_ENV_VALIDATOR) $(DOCKER_LOCAL_ENV_FILE)

validate-prod-env:
	@$(PYTHON) $(DEPLOY_ENV_VALIDATOR) $(DOCKER_ENV_FILE) --require-prod-pair

deploy: validate-local-env
	$(DOCKER_COMPOSE) up -d --build
	$(DOCKER_COMPOSE) ps

deploy-prod: validate-prod-env
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) pull
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) up -d
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE_PROD) ps
