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

# LiteRT CLI EfficientNet Demo & Test Script
set -e


# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "efficientnet" "EfficientNet Demo Script"


# --- 1. Download EfficientNet-B1 model ---
run_case "Download: EfficientNet-B1 from HuggingFace" \
    litert download litert-community/efficientnet_b1 --file "*.tflite" --output "models/efficientnet"

# Verify the downloaded model exists
EFFICIENTNET_TFLITE="models/efficientnet/efficientnet_b1.tflite"
if [ ! -f "$EFFICIENTNET_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $EFFICIENTNET_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the EfficientNet model ---
run_case "Quantize: EfficientNet Dynamic Range INT8" \
    litert quantize "$EFFICIENTNET_TFLITE" --recipe dynamic_wi8_afp32 --output "models/efficientnet/efficientnet_b1_int8_dynamic.tflite"

run_case "Quantize: EfficientNet Weight-Only INT8" \
    litert quantize "$EFFICIENTNET_TFLITE" --recipe weight_only_wi8_afp32 --output "models/efficientnet/efficientnet_b1_int8_weight_only.tflite"

# --- 3. Run Inference (Desktop & Android) ---
run_case "Run: EfficientNet FP32 on Desktop (CPU)" \
    litert run "$EFFICIENTNET_TFLITE" --desktop --cpu --iterations 1

if has_desktop_gpu "$EFFICIENTNET_TFLITE"; then
    run_case "Run: EfficientNet FP32 on Desktop (GPU)" \
        litert run "$EFFICIENTNET_TFLITE" --desktop --gpu --iterations 1
else
    echo -e "\n${YELLOW}Desktop GPU delegate is not supported. Skipping Desktop GPU run.${NC}"
fi


run_case "Run: EfficientNet Dynamic INT8 on Desktop (CPU)" \
    litert run "models/efficientnet/efficientnet_b1_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: EfficientNet FP32 on Android (CPU)" \
        litert run "$EFFICIENTNET_TFLITE" --android --cpu --iterations 1

    run_case "Run: EfficientNet FP32 on Android (GPU)" \
        litert run "$EFFICIENTNET_TFLITE" --android --gpu --iterations 1

    # If you have Android devices with NPU connected, enable those use cases.
    # run_case "Run: EfficientNet FP32 on Android (NPU)" \
    #     litert run "$EFFICIENTNET_TFLITE" --android --npu --iterations 1

    run_case "Run: EfficientNet Dynamic INT8 on Android (CPU)" \
        litert run "models/efficientnet/efficientnet_b1_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Android) ---
if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    run_case "Benchmark: EfficientNet FP32 on Android (CPU)" \
        litert benchmark "$EFFICIENTNET_TFLITE" --android

    run_case "Benchmark: EfficientNet FP32 on Android (GPU)" \
        litert benchmark "$EFFICIENTNET_TFLITE" --android --gpu

    # If you have Android devices with NPU connected, enable those use cases.
    # run_case "Benchmark: EfficientNet FP32 on Android (NPU)" \
    #     litert benchmark "$EFFICIENTNET_TFLITE" --android --npu

    run_case "Benchmark: EfficientNet Dynamic INT8 on Android" \
        litert benchmark "models/efficientnet/efficientnet_b1_int8_dynamic.tflite" --android
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
fi


# --- 5. Compile (AOT Compilation) ---
if [[ "$(uname)" == "Linux" ]]; then
    run_case "Compile: EfficientNet FP32 for Qualcomm sm8750 NPU" \
        litert compile "$EFFICIENTNET_TFLITE" --target sm8750 --output-dir "models/efficientnet"
    run_case "Compile: EfficientNet FP32 for MediaTek MT6993 NPU" \
        litert compile "$EFFICIENTNET_TFLITE" --target MT6993 --output-dir "models/efficientnet"
else
    echo -e "\n${YELLOW}Skipping offline AOT compilation on non-Linux platform ($(uname)).${NC}"
fi


# --- 6. Benchnark compiled model ---
# Enable those use cases, or change to your own targets, if you have connected those android
# devices through NPU.
#
# run_case "Run Qualcomm compiled EfficientNet" \
#   litert run "models/efficientnet/efficientnet_b1_Qualcomm_SM8750.tflite" --android --npu
# run_case "Benchmark Qualcomm compiled EfficientNet" \
#   litert benchmark "models/efficientnet/efficientnet_b1_Qualcomm_SM8750.tflite" --android --npu

# run_case "Run MediaTek compiled EfficientNet" \
#    litert run "models/efficientnet/efficientnet_b1_MediaTek_MT6993.tflite" --android --npu
# run_case "Benchmark MediaTek compiled EfficientNet" \
#    litert benchmark "models/efficientnet/efficientnet_b1_MediaTek_MT6993.tflite" --android --npu

# --- 7. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$EFFICIENTNET_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all


# --- Summary Report ---
print_summary_report "EfficientNet"
