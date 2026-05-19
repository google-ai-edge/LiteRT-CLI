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

# LiteRT CLI SmolLM2 LLM Demo & Test Script
set -e


# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "smollm2" "SmolLM2 LLM Demo Script"

# --- Variables ---
MODEL_DIR="models/smollm2"
MODEL_INT4_DIR="models/smollm2_int4"
MODEL_PATH="${MODEL_DIR}/model.litertlm"
MODEL_INT4_PATH="${MODEL_INT4_DIR}/model.litertlm"
PRESET_PATH="${REPO_ROOT}/examples/models/presets/default.py"

# --- 1. Convert HuggingFace Model ---
run_case "Convert: HuggingFace SmolLM2 FP32" \
    litert convert HuggingFaceTB/SmolLM2-360M-Instruct --output "$MODEL_DIR"

# --- 2. Convert with Weight-Only INT4 Quantization ---
run_case "Convert: HuggingFace SmolLM2 INT4" \
    litert convert HuggingFaceTB/SmolLM2-360M-Instruct --quantize-recipe weight_only_wi4_afp32 --output "$MODEL_INT4_DIR"

# --- 3. Run SmolLM2 FP32 Model ---
run_case "Run SmolLM2 FP32: Generative inference" \
    litert lm run "$MODEL_PATH" --prompt="What is the capital of France?"

# --- 4a. Run SmolLM2: Sampler configuration test ---
run_case "Run SmolLM2: Sampler configuration" \
    litert lm run "$MODEL_PATH" \
        --prompt="What is the capital of France?" \
        --max-num-tokens 1024 \
        --top-k 40 \
        --top-p 0.95 \
        --temperature 0.7 \
        --seed 42 \
        --backend cpu

# --- 4b. Run SmolLM2: Speculative decoding ---
run_case "Run SmolLM2: Speculative decoding (Auto)" \
    litert lm run "$MODEL_PATH" \
        --prompt="What is the capital of France?" \
        --enable-speculative-decoding auto \
        --backend cpu

# --- 5. Run SmolLM2 Raw Prompt ---
run_case "Run SmolLM2 Raw: No template" \
    litert lm run "$MODEL_PATH" \
        --no-template \
        --prompt="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

# --- 6. Import Model to Registry ---
run_case "Import SmolLM2: Register model alias" \
    litert-lm import "$MODEL_PATH" smollm2-alias

# --- 7. List Models ---
run_case "List Models: All imported LiteRT-LM models" \
    litert-lm list

# --- 8. Run SmolLM2 with System Preset ---
run_case "Run SmolLM2 with Preset: System instruction" \
    litert-lm run smollm2-alias --preset "$PRESET_PATH" --prompt="What is the capital of France?"

# --- 9. Benchmark SmolLM2 FP32 ---
run_case "Benchmark SmolLM2 FP32: Local generation" \
    litert lm benchmark "$MODEL_PATH" -p 128 -d 128

# --- 10. Benchmark SmolLM2 INT4 ---
run_case "Benchmark SmolLM2 INT4: Local generation" \
    litert lm benchmark "$MODEL_INT4_PATH" -p 128 -d 128

# --- 11. Delete Model ---
run_case "Delete Model: Remove registered alias" \
    litert lm delete smollm2-alias

# --- Summary Report ---
print_summary_report "SmolLM2"