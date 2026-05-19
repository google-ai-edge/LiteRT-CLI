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

# LiteRT CLI Gemma4 LLM Demo & Test Script
set -e


# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "gemma4" "Gemma4 LLM Demo Script"

# --- 1. Convert and run HuggingFace Model google/gemma-4-E2B-it ---
run_case "Convert: HuggingFace google/gemma-4-E2B-it" \
    litert convert google/gemma-4-E2B-it --output "models/gemma4"

run_case "Run converted Gemma4 model google/gemma-4-E2B-it" \
    litert lm run models/gemma4/model.litertlm --prompt="What is the capital of France?"

# --- 2. Download and Run existing Gemma4 Model from HuggingFace ---
run_case "Run Gemma4: Generative inference with custom prompt" \
    litert lm run --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --prompt="What is the capital of France?"

# --- 3. Benchmark Gemma4 LLM Model ---
run_case "Benchmark Gemma4: Local benchmark of LLM generation" \
    litert lm benchmark gemma-4-E2B-it.litertlm --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm -p 128 -d 128

# --- Summary Report ---
print_summary_report "Gemma4"

