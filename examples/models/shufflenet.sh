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

# LiteRT CLI ShuffleNetV2 Demo & Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "shufflenet" "ShuffleNetV2 Demo Script"

# --- 1. Download ShuffleNetV2 model ---
run_case "Download: ShuffleNetV2 from HuggingFace" \
    litert download litert-community/shufflenet_v2_x1_0 --file "*.tflite" --output "models/shufflenet"

# Verify the downloaded model exists using the exact repository naming convention
SHUFFLENET_TFLITE="models/shufflenet/shufflenet_v2_x1_0.tflite"
if [ ! -f "$SHUFFLENET_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $SHUFFLENET_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the ShuffleNet model ---
run_case "Quantize: ShuffleNet Dynamic Range INT8" \
    litert quantize "$SHUFFLENET_TFLITE" --recipe dynamic_wi8_afp32 --output "models/shufflenet/shufflenet_v2_x1_0_int8_dynamic.tflite"

run_case "Quantize: ShuffleNet Weight-Only INT8" \
    litert quantize "$SHUFFLENET_TFLITE" --recipe weight_only_wi8_afp32 --output "models/shufflenet/shufflenet_v2_x1_0_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: ShuffleNet FP32 on Desktop (CPU)" \
    litert run "$SHUFFLENET_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$SHUFFLENET_TFLITE"; then
    run_case "Run: ShuffleNet FP32 on Desktop (GPU)" \
        litert run "$SHUFFLENET_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi

run_case "Run: ShuffleNet Dynamic INT8 on Desktop (CPU)" \
    litert run "models/shufflenet/shufflenet_v2_x1_0_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: ShuffleNet FP32 on Android (CPU)" \
        litert run "$SHUFFLENET_TFLITE" --android --cpu --iterations 1

    run_case "Run: ShuffleNet FP32 on Android (GPU)" \
        litert run "$SHUFFLENET_TFLITE" --android --gpu --cpu --iterations 1
   
    # If you have Android devices with NPU connected, enable those use cases.
    run_case "Run: ShuffleNet FP32 on Android (NPU)" \
        litert run "$SHUFFLENET_TFLITE" --android --npu --iterations 1

    run_case "Run: ShuffleNet Dynamic INT8 on Android (CPU)" \
        litert run "models/shufflenet/shufflenet_v2_x1_0_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Desktop & Android) ---
run_case "Benchmark: ShuffleNet FP32 on Desktop (CPU)" \
    litert benchmark "$SHUFFLENET_TFLITE" --desktop --cpu

run_case "Benchmark: ShuffleNet Dynamic INT8 on Desktop (CPU)" \
    litert benchmark "models/shufflenet/shufflenet_v2_x1_0_int8_dynamic.tflite" --desktop --cpu

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: ShuffleNet FP32 on Android (CPU)" \
        litert benchmark "$SHUFFLENET_TFLITE" --android

    run_case "Benchmark: ShuffleNet FP32 on Android (GPU)" \
        litert benchmark "$SHUFFLENET_TFLITE" --android --gpu

    # If you have Android devices with NPU connected, enable those use cases.
    run_case "Benchmark: ShuffleNet FP32 on Android (NPU)" \
        litert benchmark "$SHUFFLENET_TFLITE" --android --npu

    run_case "Benchmark: ShuffleNet Dynamic INT8 on Android" \
        litert benchmark "models/shufflenet/shufflenet_v2_x1_0_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
fi

# --- 5. Compile (AOT Compilation) ---
if [[ "$(uname)" == "Linux" ]]; then
    run_case "Compile: ShuffleNet FP32 for Qualcomm sm8750 NPU" \
        litert compile "$SHUFFLENET_TFLITE" --target sm8750 --output-dir "models/shufflenet"
else
    echo -e "\n${YELLOW}Skipping offline AOT compilation on non-Linux platform ($(uname)).${NC}"
fi

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$SHUFFLENET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all

# --- Summary Report ---
print_summary_report "ShuffleNet"