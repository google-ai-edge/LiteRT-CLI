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

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "whisper_tiny" "Whisper-Tiny ASR Demo Script"

# --- 1. Download Whisper-Tiny model ---
run_case "Download: Whisper-Tiny from HuggingFace" \
    litert download litert-community/whisper-tiny --file "whisper_tiny_30s_f32.tflite" --output "models/whisper_tiny"

# Verify the downloaded model exists
WHISPER_TFLITE="models/whisper_tiny/whisper_tiny_30s_f32.tflite"
if [ ! -f "$WHISPER_TFLITE" ]; then
    echo -e "${RED}Error: Downloaded model not found at $WHISPER_TFLITE${NC}"
    exit 1
fi

# --- 2. Quantize the Whisper-Tiny model ---
run_case "Quantize: Whisper-Tiny Dynamic Range INT8" \
    litert quantize "$WHISPER_TFLITE" --recipe dynamic_wi8_afp32 --output "models/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite"

run_case "Quantize: Whisper-Tiny Weight-Only INT8" \
    litert quantize "$WHISPER_TFLITE" --recipe weight_only_wi8_afp32 --output "models/whisper_tiny/whisper_tiny_30s_int8_weight_only.tflite"

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
    litert run "models/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --desktop --cpu --iterations 1

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android inference...${NC}"
    run_case "Run: Whisper-Tiny FP32 on Android (CPU)" \
        litert run "$WHISPER_TFLITE" --android --cpu --iterations 1

    run_case "Run: Whisper-Tiny FP32 on Android (GPU)" \
        litert run "$WHISPER_TFLITE" --android --gpu --iterations 1

    run_case "Run: Whisper-Tiny Dynamic INT8 on Android (CPU)" \
        litert run "models/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --android --cpu --iterations 1
fi

# --- 4. Benchmark (Desktop & Android) ---
echo -e "\n${GREEN}Running Desktop benchmarks...${NC}"

run_case "Benchmark: Whisper-Tiny FP32 on Desktop (CPU) - Encode" \
    litert benchmark "$WHISPER_TFLITE" --desktop --cpu --signature-key "encode"

run_case "Benchmark: Whisper-Tiny FP32 on Desktop (CPU) - Decode" \
    litert benchmark "$WHISPER_TFLITE" --desktop --cpu --signature-key "decode"

if has_desktop_gpu "$WHISPER_TFLITE"; then
    run_case "Benchmark: Whisper-Tiny FP32 on Desktop (GPU) - Decode" \
        litert benchmark "$WHISPER_TFLITE" --desktop --gpu --signature-key "decode"
fi

run_case "Benchmark: Whisper-Tiny Dynamic INT8 on Desktop (CPU) - Decode" \
    litert benchmark "models/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --desktop --cpu --signature-key "decode"

if has_android_device; then
    echo -e "\n${GREEN}Android device detected. Running Android benchmarks...${NC}"
    
    run_case "Benchmark: Whisper-Tiny FP32 on Android (CPU) - Encode" \
        litert benchmark "$WHISPER_TFLITE" --android --cpu --signature-key "encode"

    run_case "Benchmark: Whisper-Tiny FP32 on Android (CPU) - Decode" \
        litert benchmark "$WHISPER_TFLITE" --android --cpu --signature-key "decode"

    run_case "Benchmark: Whisper-Tiny FP32 on Android (GPU) - Decode" \
        litert benchmark "$WHISPER_TFLITE" --android --gpu --signature-key "decode"

    run_case "Benchmark: Whisper-Tiny Dynamic INT8 on Android - Decode" \
        litert benchmark "models/whisper_tiny/whisper_tiny_30s_int8_dynamic.tflite" --android --signature-key "decode"
else
    echo -e "\n${YELLOW}No Android device detected. Skipping benchmarks on Android.${NC}"
fi

# --- 5. Compile (AOT Compilation) ---
if [[ "$(uname)" == "Linux" ]]; then
    run_case "Compile: Whisper-Tiny FP32 for Qualcomm sm8750 NPU" \
        litert compile "$WHISPER_TFLITE" --target sm8750 --output-dir "models/whisper_tiny"
else
    echo -e "\n${YELLOW}Skipping offline AOT compilation on non-Linux platform ($(uname)).${NC}"
fi

# --- 6. Visualize (Model Explorer) ---
run_case "Visualize: Launch Model Explorer in the background" \
    litert visualize "$WHISPER_TFLITE"

run_case "Visualize: Stop all Model Explorer servers" \
    litert visualize --stop-all

# --- Summary Report ---
print_summary_report "Whisper-Tiny"