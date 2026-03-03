#!/usr/bin/env bash
set -euo pipefail

# LMSInterface is now consumed as an external pinned dependency in pyproject.toml.
# No vendoring sync is required during pre-commit.
exit 0
