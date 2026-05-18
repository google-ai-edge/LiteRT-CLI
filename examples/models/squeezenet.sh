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

# LiteRT CLI SqueezeNet1_1 Demo & Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "squeezenet" "SqueezeNet1_1 Demo Script"

# --- 1. Download SqueezeNet1_1 model ---
run_case "Download: SqueezeNet1_1 from HuggingFace" \
    litert download litert-community/squeezenet1_1 --file "*.tflite" --output "models/squeezenet"

# Verify the downloaded model exists
SQUEEZENET_TFLITE="models/squeezenet/squeezenet1_1.tflite"
if [ ! -f "$SQUEEZENET_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $SQUEEZENET_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the SqueezeNet model ---
run_case "Quantize: SqueezeNet Dynamic Range INT8" \
    litert quantize "$SQUEEZENET_TFLITE" --recipe dynamic_wi8_afp32 --output "models/squeezenet/squeezenet1_1_int8_dynamic.tflite"

run_case "Quantize: SqueezeNet Weight-Only INT8" \
    litert quantize "$SQUEEZENET_TFLITE" --recipe weight_only_wi8_afp32 --output "models/squeezenet/squeezenet1_1_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: SqueezeNet FP32 on Desktop (CPU)" \
    litert run "$SQUEEZENET_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$SQUEEZENET_TFLITE"; then
    run_case "Run: SqueezeNet FP32 on Desktop (GPU)" \
        litert run "$SQUEEZENET_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi

run_case "Run: SqueezeNet Dynamic INT8 on Desktop (CPU)" \
    litert run "models/squeezenet/squeezenet1_1_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: SqueezeNet FP32 on Android (CPU)" \
        litert run "$SQUEEZENET_TFLITE" --android --cpu --iterations 1

    run_case "Run: SqueezeNet FP32 on Android (GPU)" \
        litert run "$SQUEEZENET_TFLITE" --android --gpu --iterations 1
   
    # If you have Android devices with NPU connected, enable those use cases for MacOS
    run_case "Run: SqueezeNet FP32 on Android (NPU)" \
        litert run "$SQUEEZENET_TFLITE" --android --npu --iterations 1

    run_case "Run: SqueezeNet Dynamic INT8 on Android (CPU)" \
        litert run "models/squeezenet/squeezenet1_1_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Desktop & Android) ---
run_case "Benchmark: SqueezeNet FP32 on Desktop (CPU)" \
    litert benchmark "$SQUEEZENET_TFLITE" --desktop --cpu

run_case "Benchmark: SqueezeNet Dynamic INT8 on Desktop (CPU)" \
    litert benchmark "models/squeezenet/squeezenet1_1_int8_dynamic.tflite" --desktop --cpu

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: SqueezeNet FP32 on Android (CPU)" \
        litert benchmark "$SQUEEZENET_TFLITE" --android

    run_case "Benchmark: SqueezeNet FP32 on Android (GPU)" \
        litert benchmark "$SQUEEZENET_TFLITE" --android --gpu

    # If you have Android devices with NPU connected, enable those use cases for MacOS
    run_case "Benchmark: SqueezeNet FP32 on Android (NPU)" \
        litert benchmark "$SQUEEZENET_TFLITE" --android --npu

    run_case "Benchmark: SqueezeNet Dynamic INT8 on Android" \
        litert benchmark "models/squeezenet/squeezenet1_1_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
fi

# --- 5. Compile (AOT Compilation) ---
if [[ "$(uname)" == "Linux" ]]; then
    run_case "Compile: SqueezeNet FP32 for Qualcomm sm8750 NPU" \
        litert compile "$SQUEEZENET_TFLITE" --target sm8750 --output-dir "models/squeezenet"
else
    echo -e "\n${YELLOW}Skipping offline AOT compilation on non-Linux platform ($(uname)).${NC}"
fi

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$SQUEEZENET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all

# --- Summary Report ---
print_summary_report "SqueezeNet"