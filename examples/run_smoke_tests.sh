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

# Smoke test for a LiteRT CLI source checkout: installs the package (with the
# `test` and `lm` extras) from pypi.org into a fresh venv, checks that every
# command loads its --help, and runs the unit tests with pytest.
#
# Must be run from the GitHub (exported) layout, where the package lives in
# `<repo>/litert_cli/`.
set -ex

# Get current repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Using REPO_ROOT: $REPO_ROOT"

if [[ ! -f "$REPO_ROOT/litert_cli/__init__.py" ]]; then
  set +x
  echo "ERROR: $REPO_ROOT/litert_cli/ not found." >&2
  echo "This script needs the GitHub layout of LiteRT CLI (the package in" >&2
  echo "litert_cli/). From google3, export it first with the" >&2
  echo "g3folder_to_gitfolder workflow in copy.bara.sky." >&2
  exit 1
fi

# Portable test runner
export LITERT_CLI_SMOKE_TEST_ROOT="${LITERT_CLI_SMOKE_TEST_ROOT:-${TMPDIR:-/tmp}/litert_cli_smoke_test}"
rm -rf "$LITERT_CLI_SMOKE_TEST_ROOT"
mkdir -p "$LITERT_CLI_SMOKE_TEST_ROOT"
cd "$LITERT_CLI_SMOKE_TEST_ROOT"
pwd

# Create venv
python3 -m venv venv_smoke_test
source venv_smoke_test/bin/activate

# Upgrade setuptools and pip to satisfy PEP 517 build requirements
pip install -i https://pypi.org/simple --upgrade pip setuptools wheel

# Install package with optional dependencies (`test` brings in pytest).
cd "$REPO_ROOT"
pip install -i https://pypi.org/simple -e ".[test,lm]"

# Check that every registered command loads. `lm --help` is forwarded to the
# native litert-lm binary, which the `lm` extra above installs.
litert --help
for cmd in $(python3 -c "from litert_cli import litert; print(' '.join(litert.cli.list_commands(None)))"); do
  litert "$cmd" --help
done

# Run all unit tests. pytest exits non-zero (code 5) if no tests are collected.
python3 -m pytest litert_cli
