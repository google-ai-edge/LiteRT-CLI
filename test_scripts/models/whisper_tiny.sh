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

# LiteRT CLI Whisper-Tiny Demo & Test Script
set -e


echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Whisper-Tiny ASR Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_whisper_tiny"

# Source shared utilities
source "$SCRIPT_DIR/utils.sh"


# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv_whisper_tiny
source venv_whisper_tiny/bin/activate

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

# Test data directory
export TEST_DATA_DIR="$REPO_ROOT/litert_cli/test_data"

# Install litert-cli from source
echo -e "${YELLOW}Installing litert-cli from source...${NC}"
pip install -e "$REPO_ROOT"

# Upgrade pip and setuptools to ensure build-system requirements (like
# setuptools>=61.0) can be met
echo -e "${YELLOW}Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# --- 1. Download Whisper-Tiny model ---
run_case "Download: Whisper-Tiny from HuggingFace" \
    litert download litert-community/whisper-tiny --file "whisper_tiny_30s_f32.tflite" --output "$MODEL_DIR/whisper_tiny"

# Verify the downloaded model exists
WHISPER_TFLITE="$MODEL_DIR/whisper_tiny/whisper_tiny_30s_f32.tflite"
if [ ! -f "$WHISPER_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $WHISPER_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the Whisper-Tiny model ---
run_case "Quantize: Whisper-Tiny Dynamic Range INT8" \
    litert quantize "$WHISPER_TFLITE" --type int8_dynamic --output "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite"

run_case "Quantize: Whisper-Tiny Weight-Only INT8" \
    litert quantize "$WHISPER_TFLITE" --type int8_weight_only --output "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: Whisper-Tiny FP32 on Desktop (CPU)" \
    litert run "$WHISPER_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$WHISPER_TFLITE"; then
    run_case "Run: Whisper-Tiny FP32 on Desktop (GPU)" \
        litert run "$WHISPER_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi


run_case "Run: Whisper-Tiny Dynamic INT8 on Desktop (CPU)" \
    litert run "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: Whisper-Tiny FP32 on Android (CPU)" \
        litert run "$WHISPER_TFLITE" --android --cpu --iterations 1

    run_case "Run: Whisper-Tiny FP32 on Android (GPU)" \
        litert run "$WHISPER_TFLITE" --android --gpu --iterations 1

    run_case "Run: Whisper-Tiny Dynamic INT8 on Android (CPU)" \
        litert run "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Android) ---
if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: Whisper-Tiny FP32 on Android (CPU)" \
        litert benchmark "$WHISPER_TFLITE" --android

    run_case "Benchmark: Whisper-Tiny FP32 on Android (GPU)" \
        litert benchmark "$WHISPER_TFLITE" --android --gpu

    run_case "Benchmark: Whisper-Tiny Dynamic INT8 on Android" \
        litert benchmark "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
fi

# # --- 5. Compile (AOT Compilation) ---
# run_case "Compile: Whisper-Tiny FP32 for Qualcomm sm8750 NPU" \
#     litert compile "$WHISPER_TFLITE" --target sm8750 --output-dir "$MODEL_DIR/whisper_tiny"
# run_case "Compile: Whisper-Tiny FP32 for MediaTek MT6993 NPU" \
#     litert compile "$WHISPER_TFLITE" --target MT6993 --output-dir "$MODEL_DIR/whisper_tiny"

# # --- 6. Benchnark compiled model ---
# # Enable those use cases, or change to your own targets, if you have connected those android
# # devices through NPU.

# run_case "Run Qualcomm compiled Whisper-Tiny" \
#   litert run "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_Qualcomm_SM8750.tflite" --android --npu
# run_case "Benchmark Qualcomm compiled Whisper-Tiny" \
#   litert benchmark "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_Qualcomm_SM8750.tflite" --android --npu

# run_case "Run MediaTek compiled Whisper-Tiny" \
#    litert run "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_MediaTek_MT6993.tflite" --android --npu
# run_case "Benchmark MediaTek compiled Whisper-Tiny" \
#    litert benchmark "$MODEL_DIR/whisper_tiny/whisper_tiny_30s_MediaTek_MT6993.tflite" --android --npu



# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$WHISPER_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all


# --- Summary Report ---
print_summary_report "Whisper-Tiny"
