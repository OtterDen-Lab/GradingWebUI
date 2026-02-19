SHELL := /bin/sh

PYTHON ?= python3
APP_MODULE ?= grading_web_ui.web_api.main:app
HOST ?= 127.0.0.1
PORT ?= 8765
DB_DIR ?= .tmp
DB_PATH ?= $(DB_DIR)/grading.db

DOCKERFILE ?= docker/web-grading/Dockerfile
DOCKER_BUILD_CONTEXT ?= .
DOCKER_IMAGE ?= autograder-web-grading:local
DOCKER_ENV_FILE ?= /etc/grading-web/web.env
DOCKER_COMPOSE ?= docker compose -f docker/web-grading/docker-compose.prod.yml
DEPLOY_ENV_VALIDATOR ?= scripts/validate_deploy_env.py

.PHONY: help dev run image deploy validate-env

help:
	@echo "Targets:"
	@echo "  make dev"
	@echo "    Run local FastAPI server with local DB path."
	@echo "  make image [DOCKER_IMAGE=autograder-web-grading:local]"
	@echo "    Build Docker image locally."
	@echo "  make deploy [DOCKER_ENV_FILE=/etc/grading-web/web.env] [DOCKER_IMAGE=autograder-web-grading:local]"
	@echo "    Validate env, build image, and deploy via production compose file."

dev:
	@mkdir -p $(DB_DIR)
	GRADING_DB_PATH=$(DB_PATH) $(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

# Backward-compatible alias.
run: dev

image:
	docker build -f $(DOCKERFILE) -t $(DOCKER_IMAGE) $(DOCKER_BUILD_CONTEXT)

validate-env:
	@if [ ! -f "$(DOCKER_ENV_FILE)" ]; then \
		echo "Missing env file: $(DOCKER_ENV_FILE)"; \
		exit 1; \
	fi
	@$(PYTHON) $(DEPLOY_ENV_VALIDATOR) $(DOCKER_ENV_FILE) --require-prod-pair

deploy: validate-env image
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE) up -d
	GRADING_WEB_IMAGE=$(DOCKER_IMAGE) GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE) $(DOCKER_COMPOSE) ps
