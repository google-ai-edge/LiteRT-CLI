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

# LiteRT CLI Quantize Commands Test Script
set -e

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_quantize_test"

# Source shared utilities
source "$SCRIPT_DIR/../models/utils.sh"

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Quantize Commands Demo & Test Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment using UV with Python 3.13
echo -e "${YELLOW}Creating Python virtual environment with UV...${NC}"
UV_INDEX_URL=https://pypi.org/simple uv venv --clear --python=3.13 --seed
source .venv/bin/activate

export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli with quantize extra from source using UV
echo -e "${YELLOW}Installing litert-cli with quantize extra...${NC}"
uv pip install -e "$REPO_ROOT[quantize]"

# --- 1. Built-in Quantization Recipes ---
echo -e "\n${BLUE}${BOLD}--- 1. Built-in Quantization Recipes ---${NC}"

# 1.1 Dynamic INT8 Quantization (Default)
run_case "Quantize: Dynamic INT8 (Default dynamic_wi8_afp32)" \
    litert quantize "$TEST_DATA_DIR/dummy_cv_model.tflite" --output "$MODEL_DIR/dummy_dynamic.tflite"

# 1.2 Weight-Only INT8 Quantization
run_case "Quantize: Weight-Only INT8 (weight_only_wi8_afp32)" \
    litert quantize "$TEST_DATA_DIR/dummy_cv_model.tflite" --recipe weight_only_wi8_afp32 --output "$MODEL_DIR/dummy_weight_only.tflite"

# 1.3 Static W8A8 Quantization (Requires calibration data)
run_case "Quantize: Static W8A8 with Calibration Data" \
    litert quantize "$TEST_DATA_DIR/dummy_cv_model.tflite" --recipe static_wi8_ai8 --calibration-data "$TEST_DATA_DIR/mobilenet_v3_calib_data.py" --output "$MODEL_DIR/dummy_static.tflite"


# --- 2. Custom JSON Recipes ---
echo -e "\n${BLUE}${BOLD}--- 2. Custom JSON Recipes ---${NC}"

run_case "Quantize: Custom JSON Recipe" \
    litert quantize "$TEST_DATA_DIR/dummy_cv_model.tflite" --custom-recipe "$TEST_DATA_DIR/quantize_recipe.json" --output "$MODEL_DIR/dummy_custom.tflite"


# --- 3. Error Rejection Verification ---
echo -e "\n${BLUE}${BOLD}--- 3. Error Rejection Verification ---${NC}"

run_case "Quantize: Static W8A8 without Calibration Data (Verify UsageError)" \
    bash -c "litert quantize '$TEST_DATA_DIR/dummy_cv_model.tflite' --recipe static_wi8_ai8 --output '/tmp/fail.tflite' 2>&1 | grep -q -e '--calibration-data is required'"


# --- Summary Report ---
print_summary_report "Quantize Commands"
