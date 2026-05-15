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

# LiteRT CLI YamNet Demo & Test Script
set -e


# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "yamnet" "YamNet Demo Script"





# --- 1. Download YamNet model ---
run_case "Download: YamNet TFLite model" \
    litert download "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite" --output "models/yamnet"

YAMNET_TFLITE="models/yamnet/lite-model_yamnet_classification_tflite_1.tflite"
if [ ! -f "$YAMNET_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $YAMNET_TFLITE${NC}"
    exit 1
fi


# --- 2. Quantize the YamNet model ---
run_case "Quantize: YamNet Dynamic Range INT8" \
    litert quantize "$YAMNET_TFLITE" --recipe dynamic_wi8_afp32 --output "models/yamnet/yamnet_int8_dynamic.tflite"

run_case "Quantize: YamNet Weight-Only INT8" \
    litert quantize "$YAMNET_TFLITE" --recipe weight_only_wi8_afp32 --output "models/yamnet/yamnet_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: YamNet FP32 on Desktop (CPU)" \
    litert run "$YAMNET_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$YAMNET_TFLITE"; then
    run_case "Run: YamNet FP32 on Desktop (GPU)" \
        litert run "$YAMNET_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi


run_case "Run: YamNet Dynamic INT8 on Desktop (CPU)" \
    litert run "models/yamnet/yamnet_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: YamNet FP32 on Android (CPU)" \
        litert run "$YAMNET_TFLITE" --android --cpu --iterations 1
    
    # Works on Qualcomm NPU SM8750, but not GPU.
    # run_case "Run: YamNet FP32 on Android (GPU)" \
    #    litert run "$YAMNET_TFLITE" --android --gpu --iterations 1

    run_case "Run: YamNet Dynamic INT8 on Android (CPU)" \
        litert run "models/yamnet/yamnet_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Android) ---
if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: YamNet FP32 on Android (CPU)" \
        litert benchmark "$YAMNET_TFLITE" --android

    # Works on Qualcomm NPU SM8750, but not GPU.
    # run_case "Benchmark: YamNet FP32 on Android (GPU)" \
    #    litert benchmark "$YAMNET_TFLITE" --android --gpu

    run_case "Benchmark: YamNet Dynamic INT8 on Android" \
        litert benchmark "models/yamnet/yamnet_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks (litert benchmark only supports Android/GCP).${NC}"
fi


# --- 5. Compile (AOT Compilation) ---
# TODO: Add this back when we fix the NPU compile issue.
# run_case "Compile: YamNet FP32 for Qualcomm sm8750 NPU" \
#     litert compile "$YAMNET_TFLITE" --target sm8750 --output-dir "models/yamnet"

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$YAMNET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all


# --- Summary Report ---
print_summary_report "YamNet"

