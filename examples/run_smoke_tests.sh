#!/bin/bash
# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

# Upgrade setuptools and pip to satisfy PEP 517 build requirements
pip install -i https://pypi.org/simple --upgrade pip setuptools wheel

# Install package with optional dependencies
cd "$REPO_ROOT"
pip install -i https://pypi.org/simple -e .[test,lm]

# Set PYTHONPATH dynamically to resolve internal root and litert_cli imports
INTERNAL_ROOT="$(cd "$REPO_ROOT/../../../../" && pwd)"
export PYTHONPATH="$INTERNAL_ROOT:$PYTHONPATH"

# Run a subset of CLI commands to verify they load
litert --help
litert download --help
litert quantize --help
litert run --help
litert benchmark --help
litert lm --help
litert clean --help

# Find and run all tests automatically in litert_cli directory
for test_file in $(find litert_cli -name "*_test.py"); do
  echo "Running test: $test_file"
  python3 "$test_file"
done
