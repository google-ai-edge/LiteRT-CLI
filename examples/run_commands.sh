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

# LiteRT CLI Command Demo Driver Script
#
# Usage:
#   1. Run all command demo scripts sequentially (default):
#      ./examples/run_commands.sh [ --all ]
#
#   2. Run specific command demo scripts (comma-separated):
#      ./examples/run_commands.sh download,compile,quantize
set -e

TARGET_COMMANDS="--all"
GCP_PROJECT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      TARGET_COMMANDS="--all"
      shift
      ;;
    --gcp-project)
      GCP_PROJECT="$2"
      shift 2
      ;;
    --gcp-project=*)
      GCP_PROJECT="${1#*=}"
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      TARGET_COMMANDS="$1"
      shift
      ;;
  esac
done

export TARGET_COMMANDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITERT_CLI_SHARED_VENV="true"

echo "Running LiteRT CLI command demo(s) with shared virtual environment..."

if [ "$TARGET_COMMANDS" == "--all" ]; then
  echo "Executing all command test scripts under commands/ directory..."
  for test_script in "$SCRIPT_DIR/commands/"*_test.sh; do
    echo -e "\n=================================================================="
    echo ">>> Executing $test_script..."
    echo -e "==================================================================\n"
    if [[ "$test_script" == *"benchmark_gcp_test"* ]]; then
      if [[ -n "$GCP_PROJECT" ]]; then
        bash "$test_script" --gcp-project "$GCP_PROJECT"
      else
        echo -e "\n[Skipping $test_script: --gcp-project is required for GCP benchmarks]\n"
      fi
    else
      bash "$test_script"
    fi
  done
  echo -e "\nAll LiteRT CLI command demos completed successfully!"
else
  # Split comma-separated string into array
  IFS=',' read -ra CMD_ARRAY <<< "$TARGET_COMMANDS"

  for cmd in "${CMD_ARRAY[@]}"; do
    target_script="$SCRIPT_DIR/commands/${cmd}_test.sh"
    if [ ! -f "$target_script" ]; then
      echo -e "\nError: Command test script not found: $target_script"
      echo "Available command scripts under commands/:"
      ls -1 "$SCRIPT_DIR/commands/"*_test.sh | sed -e "s#.*/##" -e "s#_test.sh##" | sed -e "s/^/  - /"
      exit 1
    fi

    echo -e "\n=================================================================="
    echo ">>> Executing $target_script..."
    echo -e "==================================================================\n"
    if [[ "$target_script" == *"benchmark_gcp_test"* ]]; then
      if [[ -n "$GCP_PROJECT" ]]; then
        bash "$target_script" --gcp-project "$GCP_PROJECT"
      else
        echo -e "\n[Skipping $target_script: --gcp-project is required for GCP benchmarks]\n"
      fi
    else
      bash "$target_script"
    fi
  done
  echo -e "\nLiteRT CLI command demo(s) for '$TARGET_COMMANDS' completed successfully!"
fi
