SHELL := /bin/sh

PYTHON ?= python3
APP_MODULE ?= grading_web_ui.web_api.main:app
HOST ?= 127.0.0.1
PORT ?= 8765
DB_DIR ?= .tmp
DB_PATH ?= $(DB_DIR)/grading.db

DOCKERFILE ?= docker/web-grading/Dockerfile
DOCKER_BUILD_CONTEXT ?= .
RUN_IMAGE ?= autograder-web-grading:local
RUN_ENV_FILE ?= .env
DEPLOY_ENV_FILE ?= /etc/grading-web/web.env
REGISTRY_IMAGE ?= samogden/webgraderui
PUBLISH_VERSION ?= v0.8.1
DEPLOY_TAG ?= latest
PUBLISH_PLATFORMS ?= linux/amd64,linux/arm64
DOCKER_COMPOSE ?= docker compose -f docker/web-grading/docker-compose.prod.yml
DEPLOY_ENV_VALIDATOR ?= scripts/validate_deploy_env.py

# Allow:
#   make publish v0.8.1
#   make deploy v0.8.1
ifneq ($(filter publish,$(firstword $(MAKECMDGOALS))),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    PUBLISH_VERSION := $(word 2,$(MAKECMDGOALS))
    $(eval $(word 2,$(MAKECMDGOALS)):;@:)
  endif
endif
ifneq ($(filter deploy,$(firstword $(MAKECMDGOALS))),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    DEPLOY_TAG := $(word 2,$(MAKECMDGOALS))
    $(eval $(word 2,$(MAKECMDGOALS)):;@:)
  endif
endif

.PHONY: help debug dev run image publish deploy validate-env

help:
	@echo "Targets:"
	@echo "  make debug"
	@echo "    Run local FastAPI server with local DB path (old make run behavior)."
	@echo "  make run [RUN_IMAGE=autograder-web-grading:local] [RUN_ENV_FILE=.env]"
	@echo "    Build local Docker image and run it via production compose."
	@echo "  make publish [vX.Y.Z] [REGISTRY_IMAGE=samogden/webgraderui]"
	@echo "    Build multi-arch image and push both :vX.Y.Z and :latest."
	@echo "  make deploy [vX.Y.Z] [DEPLOY_ENV_FILE=/etc/grading-web/web.env]"
	@echo "    Pull and run REGISTRY_IMAGE tag (defaults to :latest)."
	@echo "  make image [RUN_IMAGE=autograder-web-grading:local]"
	@echo "    Build local Docker image only."

debug:
	@mkdir -p $(DB_DIR)
	GRADING_DB_PATH=$(DB_PATH) $(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

# Backward-compatible alias.
dev: debug

run: image
	@$(MAKE) validate-env ENV_FILE="$(RUN_ENV_FILE)"
	GRADING_WEB_IMAGE=$(RUN_IMAGE) GRADING_WEB_ENV_FILE=$(RUN_ENV_FILE) $(DOCKER_COMPOSE) up -d
	GRADING_WEB_IMAGE=$(RUN_IMAGE) GRADING_WEB_ENV_FILE=$(RUN_ENV_FILE) $(DOCKER_COMPOSE) ps

image:
	docker build -f $(DOCKERFILE) -t $(RUN_IMAGE) $(DOCKER_BUILD_CONTEXT)

validate-env:
	@if [ -z "$(ENV_FILE)" ]; then \
		echo "Missing ENV_FILE value"; \
		exit 1; \
	fi
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "Missing env file: $(ENV_FILE)"; \
		exit 1; \
	fi
	@$(PYTHON) $(DEPLOY_ENV_VALIDATOR) $(ENV_FILE) --require-prod-pair

publish:
	docker buildx build \
		--platform $(PUBLISH_PLATFORMS) \
		-f $(DOCKERFILE) \
		-t $(REGISTRY_IMAGE):$(PUBLISH_VERSION) \
		-t $(REGISTRY_IMAGE):latest \
		--push \
		$(DOCKER_BUILD_CONTEXT)

deploy:
	@$(MAKE) validate-env ENV_FILE="$(DEPLOY_ENV_FILE)"
	GRADING_WEB_IMAGE=$(REGISTRY_IMAGE):$(DEPLOY_TAG) GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE) $(DOCKER_COMPOSE) pull
	GRADING_WEB_IMAGE=$(REGISTRY_IMAGE):$(DEPLOY_TAG) GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE) $(DOCKER_COMPOSE) up -d
	GRADING_WEB_IMAGE=$(REGISTRY_IMAGE):$(DEPLOY_TAG) GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE) $(DOCKER_COMPOSE) ps
