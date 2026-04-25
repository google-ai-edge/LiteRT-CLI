#!/bin/bash
# LiteRT CLI Demo Script for OSS (GitHub)
set -e

COMMAND=${1:-"--all"}
export COMMAND

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_oss"
export LITERT_CLI_ANDROID_ROOT="/data/local/tmp/litert-cli-oss"

# Clean up directory
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
python3 -m venv venv_litert_cli_oss
source venv_litert_cli_oss/bin/activate

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p $MODEL_DIR

# Test data directory
export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
pip install -e "$REPO_ROOT[test,lm,convert,visualize]"
# Or install from pip
# pip install litert-cli

function generate_test_inputs() {
    echo "Generating test input files..."
    python3 "$TEST_DATA_DIR/generate_test_inputs.py" "$LITERT_CLI_ROOT"
}

# Run test commands in the same shell.
source "$SCRIPT_DIR/litert_cli_test_commands.sh"
