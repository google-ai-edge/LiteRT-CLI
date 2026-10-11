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

# LiteRT CLI MiniCPM5 LLM Demo & Test Script
#
# Converts openbmb/MiniCPM5-1B with INT8 dynamic quantization. MiniCPM5 ships
# as a `llama` architecture; the CLI detects it from the model id and bundles
# the canonical LiteRT-LM MiniCPM5 config (chat template, thinking, tool calls).
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "minicpm5" "MiniCPM5 LLM Demo Script"

# --- Variables ---
HF_MODEL="openbmb/MiniCPM5-1B"
MODEL_DIR="models/minicpm5"
MODEL_PATH="${MODEL_DIR}/model.litertlm"
PRESET_PATH="${REPO_ROOT}/examples/models/presets/tool_calling_demo.py"

# Runs `litert lm run` and checks that the output contains a pattern.
# Usage: run_and_expect <pattern> <litert lm run args...>
function run_and_expect() {
    local pattern="$1"
    shift
    local out
    out="$(litert lm run "$@" 2>&1)" || { echo "$out"; return 1; }
    echo "$out"
    if ! grep -qiE "$pattern" <<< "$out"; then
        echo "Expected output to match /$pattern/"
        return 1
    fi
}

# --- 1. Convert HuggingFace Model with INT8 Dynamic Quantization ---
run_case "Convert: HuggingFace ${HF_MODEL} INT8" \
    litert convert "$HF_MODEL" --quantize dynamic_wi8_afp32 --output "$MODEL_DIR"

# --- 2. Run with thinking disabled ---
run_case "Run MiniCPM5: Thinking disabled" \
    run_and_expect "paris" "$MODEL_PATH" \
        --prompt="What is the capital of France? Answer in one word." \
        --thinking false

# --- 3. Run with thinking enabled ---
run_case "Run MiniCPM5: Thinking enabled" \
    run_and_expect "391" "$MODEL_PATH" \
        --prompt="What is 17 times 23?" \
        --thinking true

# --- 4. Run with Tool Calling Preset ---
run_case "Run MiniCPM5: Tool calling with python preset" \
    run_and_expect "22" "$MODEL_PATH" \
        --preset "$PRESET_PATH" \
        --prompt="What is the weather in Tokyo?" \
        --thinking false

# --- 5. Benchmark MiniCPM5 INT8 ---
run_case "Benchmark MiniCPM5 INT8: Local generation" \
    litert lm benchmark "$MODEL_PATH" -p 128 -d 128

# --- Summary Report ---
print_summary_report "MiniCPM5"
