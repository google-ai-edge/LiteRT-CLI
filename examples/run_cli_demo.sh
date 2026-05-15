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

# LiteRT CLI Comprehensive E2E Demo Script
#
# Usage:
#   Run all command and model demo scripts sequentially:
#   ./examples/run_cli_demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo ">>> LiteRT CLI Comprehensive E2E Demo"
echo "=================================================================="

echo -e "\n>>> 1. Running Command-Specific Demos..."
bash "$SCRIPT_DIR/run_commands.sh" --all

echo -e "\n>>> 2. Running Model-Specific Demos..."
bash "$SCRIPT_DIR/run_models.sh" --all

echo -e "\n=================================================================="
echo ">>> All LiteRT CLI E2E Demos Completed Successfully!"
echo "=================================================================="
