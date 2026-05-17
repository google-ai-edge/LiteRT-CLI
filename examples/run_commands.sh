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

TARGET_COMMANDS=${1:-"--all"}
export TARGET_COMMANDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITERT_CLI_SHARED_VENV="true"

echo "Running LiteRT CLI command demo(s) with shared virtual environment..."

if [ "$TARGET_COMMANDS" == "--all" ] || [ -z "$1" ]; then
  echo "Executing all command test scripts under commands/ directory..."
  for test_script in "$SCRIPT_DIR/commands/"*_test.sh; do
    if [[ "$test_script" == *"benchmark_gcp_test"* ]]; then
      echo -e "\n[Note: $(basename "$test_script") is excluded from default execution. Please run it manually.]\n"
      continue
    fi
    echo -e "\n=================================================================="
    echo ">>> Executing $test_script..."
    echo -e "==================================================================\n"
    bash "$test_script"
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
      bash "$target_script"
  done
  echo -e "\nLiteRT CLI command demo(s) for '$TARGET_COMMANDS' completed successfully!"
fi
