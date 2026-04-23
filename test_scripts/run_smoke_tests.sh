#!/bin/bash
set -ex

# Get current repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Using REPO_ROOT: $REPO_ROOT"

# Portable test runner
export LITERT_CLI_SMOKE_TEST_ROOT="/tmp/litert_cli_smoke_test"
rm -rf "$LITERT_CLI_SMOKE_TEST_ROOT"
mkdir -p "$LITERT_CLI_SMOKE_TEST_ROOT"
cd "$LITERT_CLI_SMOKE_TEST_ROOT"
pwd

# Create venv
python3 -m venv venv_smoke_test
source venv_smoke_test/bin/activate

# Install package with optional dependencies
cd "$REPO_ROOT"
pip install -e .[test,lm]

# Set PYTHONPATH dynamically to resolve google3 and litert_cli imports
G3_PARENT="$(cd "$REPO_ROOT/../../../../" && pwd)"
export PYTHONPATH="$G3_PARENT:$PYTHONPATH"

# Run a subset of CLI commands to verify they load
litert --help
litert download --help
litert quantize --help
litert run --help
litert benchmark --help
litert lm --help
litert clean --help

# Find and run all tests automatically
for test_file in $(find . -name "*_test.py"); do
  echo "Running test: $test_file"
  python3 "$test_file"
done
