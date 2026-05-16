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

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "quantize_test" "Quantize Commands Demo" "quantize"


# --- 1. Built-in Quantization Recipes ---
echo -e "\n${BLUE}${BOLD}--- 1. Built-in Quantization Recipes ---${NC}"

# 1.1 Dynamic INT8 Quantization (Default)
run_case "Quantize: Dynamic INT8 (Default dynamic_wi8_afp32)" \
    litert quantize dummy_cv_model.tflite --output models/dummy_dynamic.tflite

# 1.2 Weight-Only INT8 Quantization
run_case "Quantize: Weight-Only INT8 (weight_only_wi8_afp32)" \
    litert quantize dummy_cv_model.tflite --recipe weight_only_wi8_afp32 --output models/dummy_weight_only.tflite

# 1.3 Static W8A8 Quantization (Requires calibration data)
run_case "Quantize: Static W8A8 with Calibration Data" \
    litert quantize dummy_cv_model.tflite --recipe static_wi8_ai8 --calibration-data dummy_calib_data.py --output models/dummy_static.tflite


# --- 2. Custom JSON Recipes ---
echo -e "\n${BLUE}${BOLD}--- 2. Custom JSON Recipes ---${NC}"

run_case "Quantize: Custom JSON Recipe" \
    litert quantize dummy_cv_model.tflite --custom-recipe quantize_recipe.json --output models/dummy_custom.tflite


# --- 3. Error Rejection Verification ---
echo -e "\n${BLUE}${BOLD}--- 3. Error Rejection Verification ---${NC}"

run_case "Quantize: Static W8A8 without Calibration Data (Verify UsageError)" \
    bash -c "litert quantize 'dummy_cv_model.tflite' --recipe static_wi8_ai8 --output '/tmp/fail.tflite' 2>&1 | grep -q -e '--calibration-data is required'"



# --- Summary Report ---
print_summary_report "Quantize Commands"
