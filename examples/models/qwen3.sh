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

# LiteRT CLI Qwen3 LLM Demo & Test Script
#
# Converts Qwen/Qwen3-0.6B with INT8 dynamic quantization and checks that the
# bundled LiteRT-LM config (chat template, thinking, tool calls) works.
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "qwen3" "Qwen3 LLM Demo Script"

# --- Variables ---
HF_MODEL="Qwen/Qwen3-0.6B"
MODEL_DIR="models/qwen3"
MODEL_PATH="${MODEL_DIR}/model.litertlm"
PRESET_PATH="${REPO_ROOT}/examples/models/presets/tool_calling_demo.py"

# --- 1. Convert HuggingFace Model with INT8 Dynamic Quantization ---
# The canonical LiteRT-LM config for Qwen3 is bundled automatically
# (--llm-metadata auto).
run_case "Convert: HuggingFace ${HF_MODEL} INT8" \
    litert convert "$HF_MODEL" --quantize dynamic_wi8_afp32 --output "$MODEL_DIR"

# --- 2. Run with thinking disabled ---
run_case "Run Qwen3: Thinking disabled" \
    run_and_expect "paris" "$MODEL_PATH" \
        --prompt="What is the capital of France? Answer in one word." \
        --thinking false

# --- 3. Run with thinking enabled ---
run_case "Run Qwen3: Thinking enabled" \
    run_and_expect "391" "$MODEL_PATH" \
        --prompt="What is 17 times 23?" \
        --thinking true

# --- 4. Run with Tool Calling Preset ---
run_case "Run Qwen3: Tool calling with python preset" \
    run_and_expect "22" "$MODEL_PATH" \
        --preset "$PRESET_PATH" \
        --prompt="What is the weather in Tokyo?" \
        --thinking false

# --- 5. Benchmark Qwen3 INT8 ---
run_case "Benchmark Qwen3 INT8: Local generation" \
    litert lm benchmark "$MODEL_PATH" -p 128 -d 128

# --- Summary Report ---
print_summary_report "Qwen3"
