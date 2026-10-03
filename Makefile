SHELL := /bin/sh

PYTHON ?= python3
ifeq ($(shell id -u),0)
  SUDO ?=
else
  SUDO ?= sudo
endif
APP_MODULE ?= grading_web_ui.web_api.main:app
HOST ?= 127.0.0.1
PORT ?= 8765
DB_DIR ?= .tmp
DB_PATH ?= $(DB_DIR)/grading.db
DEBUG_BOOTSTRAP_ADMIN_USERNAME ?= admin
DEBUG_BOOTSTRAP_ADMIN_PASSWORD ?= admin
DEBUG_BOOTSTRAP_ADMIN_EMAIL ?= admin@example.com

DOCKERFILE ?= docker/web-grading/Dockerfile
DOCKER_BUILD_CONTEXT ?= .
RUN_IMAGE ?= autograder-web-grading:local
RUN_ENV_FILE ?= .env
DEPLOY_ENV_FILE ?= /etc/grading-web/web.env
REGISTRY_IMAGE ?= samogden/webgraderui
PROJECT_VERSION ?= $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -n 1)
PUBLISH_VERSION ?= v$(PROJECT_VERSION)
DEPLOY_TAG ?= latest
PUBLISH_PLATFORMS ?= linux/amd64
DOCKER_COMPOSE ?= docker compose -f docker/web-grading/docker-compose.prod.yml
DEPLOY_ENV_VALIDATOR ?= scripts/validate_deploy_env.py
LXC_APP_DIR ?= /opt/grading-web
LXC_STATE_DIR ?= /srv/grading-web
LXC_SERVICE_USER ?= grading-web

# Allow:
#   make publish v0.8.1
#   make deploy v0.8.1
#   make lxc-deploy v0.8.1
ifneq ($(filter publish,$(firstword $(MAKECMDGOALS))),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    PUBLISH_VERSION := $(word 2,$(MAKECMDGOALS))
    $(eval $(word 2,$(MAKECMDGOALS)):;@:)
  endif
endif
ifneq ($(filter deploy deploy-from-backup,$(firstword $(MAKECMDGOALS))),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    DEPLOY_TAG := $(word 2,$(MAKECMDGOALS))
    $(eval $(word 2,$(MAKECMDGOALS)):;@:)
  endif
endif
ifneq ($(filter lxc-deploy,$(firstword $(MAKECMDGOALS))),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    LXC_DEPLOY_TAG := $(word 2,$(MAKECMDGOALS))
    $(eval $(word 2,$(MAKECMDGOALS)):;@:)
  endif
endif

.PHONY: help debug dev run image publish deploy validate-env backup-db backup deploy-from-backup lxc-install lxc-deploy lxc-backup lxc-restore lxc-enable-backups

help:
	@echo "Targets:"
	@echo "  make debug"
	@echo "    Run local FastAPI server with local DB path and a default admin/admin bootstrap account."
	@echo "    Override with DEBUG_BOOTSTRAP_ADMIN_USERNAME/PASSWORD/EMAIL if needed."
	@echo "  make run [RUN_IMAGE=autograder-web-grading:local] [RUN_ENV_FILE=.env]"
	@echo "    Build local Docker image and run it via production compose."
	@echo "  make publish [vX.Y.Z] [REGISTRY_IMAGE=samogden/webgraderui]"
	@echo "    Build and push :vX.Y.Z and :latest (default platform: linux/amd64)."
	@echo "    Optional multi-arch: PUBLISH_PLATFORMS=linux/amd64,linux/arm64"
	@echo "  make deploy [vX.Y.Z] [DEPLOY_ENV_FILE=/etc/grading-web/web.env]"
	@echo "    Pull and run REGISTRY_IMAGE tag (defaults to :latest)."
	@echo "  make image [RUN_IMAGE=autograder-web-grading:local]"
	@echo "    Build local Docker image only."
	@echo "  make backup-db BACKUP_FILE=/absolute/path/grading-YYYYMMDD.db"
	@echo "    Create a verified SQLite backup from DB_PATH (for non-Docker installs)."
	@echo "  make backup BACKUP_DIR=/absolute/path/outside-the-server"
	@echo "    Snapshot the running Docker database and copy it to external storage."
	@echo "  make deploy-from-backup BACKUP_FILE=/path/to/grading-backup.db [vX.Y.Z]"
	@echo "    Deploy a release, replace its fresh database with a verified backup, and start it."
	@echo "  make lxc-install"
	@echo "    Install the native systemd service; requires a mounted LXC state volume."
	@echo "  make lxc-deploy [vX.Y.Z]"
	@echo "    Deploy the current native checkout, or switch to an exact release tag first."
	@echo "  make lxc-backup BACKUP_DIR=/path/on/off-host-storage"
	@echo "    Create a verified backup from a native LXC installation."
	@echo "  make lxc-restore BACKUP_FILE=/path/to/grading-backup.db"
	@echo "    Restore into an empty native LXC state volume."
	@echo "  make lxc-enable-backups"
	@echo "    Enable the daily native backup timer after configuring backup.env."

debug:
	@mkdir -p $(DB_DIR)
	GRADING_DB_PATH=$(DB_PATH) \
	GRADING_BOOTSTRAP_ADMIN_USERNAME=$(DEBUG_BOOTSTRAP_ADMIN_USERNAME) \
	GRADING_BOOTSTRAP_ADMIN_PASSWORD=$(DEBUG_BOOTSTRAP_ADMIN_PASSWORD) \
	GRADING_BOOTSTRAP_ADMIN_EMAIL=$(DEBUG_BOOTSTRAP_ADMIN_EMAIL) \
	$(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT)

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

deploy-from-backup:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Missing BACKUP_FILE=/path/to/grading-backup.db"; \
		exit 1; \
	fi
	@$(MAKE) deploy DEPLOY_TAG="$(DEPLOY_TAG)" DEPLOY_ENV_FILE="$(DEPLOY_ENV_FILE)"
	GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE) scripts/restore_docker_backup.sh --backup-file "$(BACKUP_FILE)"

backup-db:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Missing BACKUP_FILE=/absolute/path/grading-YYYYMMDD.db"; \
		exit 1; \
	fi
	GRADING_DB_PATH=$(DB_PATH) $(PYTHON) scripts/backup_db.py --output "$(BACKUP_FILE)"

backup:
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "Missing BACKUP_DIR=/absolute/path/on-external-or-synced-storage"; \
		exit 1; \
	fi
	GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE) scripts/backup_docker.sh --backup-dir "$(BACKUP_DIR)"

lxc-install:
	$(SUDO) scripts/install_lxc_service.sh --app-dir "$(LXC_APP_DIR)" --state-dir "$(LXC_STATE_DIR)" --service-user "$(LXC_SERVICE_USER)"

lxc-deploy:
	$(SUDO) scripts/deploy_lxc_service.sh --app-dir "$(LXC_APP_DIR)" --state-dir "$(LXC_STATE_DIR)" --service-user "$(LXC_SERVICE_USER)" $(if $(LXC_DEPLOY_TAG),--tag "$(LXC_DEPLOY_TAG)")

lxc-backup:
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "Missing BACKUP_DIR=/path/on/off-host-storage"; \
		exit 1; \
	fi
	$(SUDO) scripts/backup_lxc.sh --app-dir "$(LXC_APP_DIR)" --state-dir "$(LXC_STATE_DIR)" --backup-dir "$(BACKUP_DIR)"

lxc-restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Missing BACKUP_FILE=/path/to/grading-backup.db"; \
		exit 1; \
	fi
	$(SUDO) scripts/restore_lxc_backup.sh --app-dir "$(LXC_APP_DIR)" --state-dir "$(LXC_STATE_DIR)" --service-user "$(LXC_SERVICE_USER)" --backup-file "$(BACKUP_FILE)"

lxc-enable-backups:
	$(SUDO) systemctl enable --now grading-web-backup.timer
