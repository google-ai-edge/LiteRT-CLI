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

# LiteRT CLI ResNet Demo & Test Script
set -e

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI ResNet Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_resnet"

# Source shared utilities
source "$SCRIPT_DIR/utils.sh"


# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv_resnet
source venv_resnet/bin/activate

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

# Test data directory
export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"




# --- 1. Convert PyTorch ResNet18 model to LiteRT ---
run_case "Convert: PyTorch ResNet18 to LiteRT" \
    litert convert "$TEST_DATA_DIR/resnet18.py" --output "$MODEL_DIR/resnet18"

# Verify the converted model exists
RESNET_TFLITE="$MODEL_DIR/resnet18/resnet18.tflite"
if [ ! -f "$RESNET_TFLITE" ]; then
    echo -e "${RED}Error: Converted model not found at $RESNET_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the ResNet18 model ---
run_case "Quantize: ResNet18 Dynamic Range INT8" \
    litert quantize "$RESNET_TFLITE" --type int8_dynamic --output "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite"

run_case "Quantize: ResNet18 Weight-Only INT8" \
    litert quantize "$RESNET_TFLITE" --type int8_weight_only --output "$MODEL_DIR/resnet18/resnet18_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: ResNet18 FP32 on Desktop (CPU)" \
    litert run "$RESNET_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$RESNET_TFLITE"; then
    run_case "Run: ResNet18 FP32 on Desktop (GPU)" \
        litert run "$RESNET_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi


run_case "Run: ResNet18 Dynamic INT8 on Desktop (CPU)" \
    litert run "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: ResNet18 FP32 on Android (CPU)" \
        litert run "$RESNET_TFLITE" --android --cpu --iterations 1

    run_case "Run: ResNet18 FP32 on Android (GPU)" \
        litert run "$RESNET_TFLITE" --android --gpu --iterations 1

    run_case "Run: ResNet18 Dynamic INT8 on Android (CPU)" \
        litert run "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Android) ---
if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: ResNet18 FP32 on Android (CPU)" \
        litert benchmark "$RESNET_TFLITE" --android

    run_case "Benchmark: ResNet18 FP32 on Android (GPU)" \
        litert benchmark "$RESNET_TFLITE" --android --gpu

    run_case "Benchmark: ResNet18 Dynamic INT8 on Android" \
        litert benchmark "$MODEL_DIR/resnet18/resnet18_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks (litert benchmark only supports Android/GCP).${NC}"
fi


# --- 5. Compile (AOT Compilation) ---
# TODO: Add this back when we fix the NPU compile issue.
# run_case "Compile: ResNet18 FP32 for Qualcomm sm8750 NPU" \
#     litert compile "$RESNET_TFLITE" --target sm8750 --output-dir "$MODEL_DIR/resnet18"

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$RESNET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all


# --- Summary Report ---
print_summary_report "ResNet"

