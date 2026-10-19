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

# LiteRT CLI LM Commands Test Script
#
# Runs `litert lm run` and `litert lm benchmark` with Hugging Face repo ids: a
# transformers checkpoint is converted once into the LiteRT CLI cache, and a
# repo with a .litertlm file is passed to litert-lm.
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "lm_test" "LM Commands Demo" "lm,convert"

# Runs `litert lm run` and checks that it fails with output matching a pattern
# (case-insensitive extended regex). Use with run_case.
# Usage: run_and_expect_failure <pattern> <litert lm run args...>
function run_and_expect_failure() {
    local pattern="$1"
    shift
    local out
    if out="$(litert lm run "$@" 2>&1)"; then
        echo "$out"
        echo "Expected a non-zero exit code"
        return 1
    fi
    echo "$out"
    if ! grep -qiE "$pattern" <<< "$out"; then
        echo "Expected output to match /$pattern/"
        return 1
    fi
}

# --- Variables ---
HF_MODEL="Qwen/Qwen3-0.6B"
PROMPT="What is the capital of France? Answer in one word."

# --- 1. Hugging Face checkpoint without a .litertlm file ---
echo -e "\n${BLUE}${BOLD}--- 1. Checkpoint: ${HF_MODEL} ---${NC}"

# 1.1 Convert once (--reconvert replaces a bundle cached by an earlier run)
run_case "LM Run: ${HF_MODEL} converted from the checkpoint" \
    run_and_expect "paris" "$HF_MODEL" \
        --prompt="$PROMPT" \
        --thinking false \
        --reconvert

# 1.2 Run again from the cached bundle, without converting
run_case "LM Run: ${HF_MODEL} from the cached bundle" \
    run_and_expect "using the bundle converted on" "$HF_MODEL" \
        --prompt="$PROMPT" \
        --thinking false

# 1.3 Benchmark resolves the repo id the same way
run_case "LM Benchmark: ${HF_MODEL}" \
    litert lm benchmark "$HF_MODEL" -p 128 -d 128

# --- 2. Repo with a .litertlm file ---
echo -e "\n${BLUE}${BOLD}--- 2. Repo with a .litertlm file ---${NC}"

# 2.1 Passed to litert-lm with --from-huggingface-repo, without converting
run_case "LM Run: litert-community/SmolLM2-135M-Instruct" \
    run_and_expect "found single \.litertlm file" \
        litert-community/SmolLM2-135M-Instruct \
        --prompt="$PROMPT"

# --- 3. Unsupported checkpoint ---
echo -e "\n${BLUE}${BOLD}--- 3. Unsupported checkpoint ---${NC}"

# 3.1 A non-CausalLM architecture fails before the weights are downloaded
run_case "LM Run: bert-base-uncased (Verify Non-CausalLM Rejection)" \
    run_and_expect_failure "CausalLM" google-bert/bert-base-uncased \
        --prompt="$PROMPT"

# --- Summary Report ---
print_summary_report "LM Commands"
