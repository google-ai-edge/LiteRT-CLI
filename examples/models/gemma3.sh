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

# LiteRT CLI Gemma3 LLM Demo & Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "gemma3" "Gemma3 LLM Demo Script"

# --- Part 1: Convert from HuggingFace Hub, then Run & Benchmark ---
# Convert HuggingFace Model google/gemma-3-1b-it
run_case "Convert: HuggingFace google/gemma-3-1b-it" \
    litert convert google/gemma-3-1b-it --output "models/gemma3_converted"

# Run the converted model
run_case "Run Gemma3: Converted Model" \
    litert lm run "models/gemma3_converted/model.litertlm" --prompt="What is the capital of France?"

# Benchmark the converted model
run_case "Benchmark Gemma3: Converted Model" \
    litert lm benchmark "models/gemma3_converted/model.litertlm" -p 128 -d 128


# --- Part 2: Directly Download, Run, and Benchmark Pre-converted Model ---
# Run pre-converted Gemma3 directly from huggingface repo
run_case "Run Gemma3: Direct HuggingFace execution" \
    litert lm run \
        --from-huggingface-repo=litert-community/Gemma3-1B-IT \
        gemma3-1b-it-int4.litertlm \
        --prompt="What is the capital of France?"

# Benchmark pre-converted Gemma3 directly
run_case "Benchmark Gemma3: Direct HuggingFace execution" \
    litert lm benchmark \
        --from-huggingface-repo=litert-community/Gemma3-1B-IT \
        gemma3-1b-it-int4.litertlm \
        -p 128 -d 128

# --- Summary Report ---
print_summary_report "Gemma3"
