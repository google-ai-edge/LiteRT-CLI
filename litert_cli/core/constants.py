"""Constants for the LiteRT CLI."""

from __future__ import annotations

import os

# Environment variable names
ENV_LITERT_CLI_ROOT: str = "LITERT_CLI_ROOT"
ENV_LITERT_CLI_ANDROID_ROOT: str = "LITERT_CLI_ANDROID_ROOT"

# Default values
_DEFAULT_CLI_ROOT: str = "/tmp/litert-cli"
_DEFAULT_CLI_ANDROID_ROOT: str = "/data/local/tmp/litert-cli"

# Resolved configuration values
LITERT_CLI_ROOT: str = os.environ.get(ENV_LITERT_CLI_ROOT, _DEFAULT_CLI_ROOT)
LITERT_CLI_ANDROID_ROOT: str = os.environ.get(
    ENV_LITERT_CLI_ANDROID_ROOT, _DEFAULT_CLI_ANDROID_ROOT
)
