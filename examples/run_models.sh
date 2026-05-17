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

# LiteRT CLI Model Demo Driver Script
#
# Usage:
#   1. Run all model demos sequentially (default):
#      ./examples/run_models.sh [ --all ]
#
#   2. Run specific model demos (comma-separated):
#      ./examples/run_models.sh efficientnet,resnet,yamnet
set -e

TARGET_MODELS=${1:-"--all"}
export TARGET_MODELS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITERT_CLI_SHARED_VENV="true"

echo "Running LiteRT CLI model demo(s) with shared virtual environment..."

if [ "$TARGET_MODELS" == "--all" ] || [ -z "$1" ]; then
  echo "Executing all model test scripts under models/ directory..."
  for model_script in "$SCRIPT_DIR/models/"*.sh; do
    echo -e "\n=================================================================="
    echo ">>> Executing $model_script..."
    echo -e "==================================================================\n"
    bash "$model_script"
  done
  echo -e "\nAll LiteRT CLI model demos completed successfully!"
else
  # Split comma-separated string into array
  IFS=',' read -ra MODEL_ARRAY <<< "$TARGET_MODELS"

  for model in "${MODEL_ARRAY[@]}"; do
    target_script="$SCRIPT_DIR/models/${model}.sh"
    if [ ! -f "$target_script" ]; then
      echo -e "\nError: Model test script not found: $target_script"
      echo "Available model scripts under models/:"
      ls -1 "$SCRIPT_DIR/models/"*.sh | sed -e "s#.*/##" -e "s#.sh##" | sed -e "s/^/  - /"
      exit 1
    fi

    echo -e "\n=================================================================="
    echo ">>> Executing $target_script..."
    echo -e "==================================================================\n"
    bash "$target_script"
  done
  echo -e "\nLiteRT CLI model demo(s) for '$TARGET_MODELS' completed successfully!"
fi
