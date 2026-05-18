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

# --- 1. Convert HuggingFace Model HuggingFaceTB/SmolLM2-360M-Instruct ---
run_case "Convert: HuggingFace HuggingFaceTB/SmolLM2-360M-Instruct" \
    litert convert HuggingFaceTB/SmolLM2-360M-Instruct --output "models/smollm2"

# --- 2. Run SmolLM2 Generative LLM Model ---
run_case "Run SmolLM2: Generative inference with custom prompt" \
    litert lm run models/smollm2/model.litertlm --prompt="What is the capital of France?"

# --- 3. Benchmark SmolLM2 LLM Model ---
run_case "Benchmark SmolLM2: Local benchmark of LLM generation" \
    litert lm benchmark models/smollm2/model.litertlm -p 128 -d 128

# --- Summary Report ---
print_summary_report "SmolLM2"