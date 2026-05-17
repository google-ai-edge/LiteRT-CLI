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

# LiteRT CLI Qwen Demo & Test Script
set -e

# Source shared utilities relative to script
source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"

setup_test_env "qwen1_5" "Qwen1.5 Demo Script"


# --- 1. Convert Qwen 1.5 Model ---
run_case "Convert Qwen1.5-0.5B-Chat from HuggingFace to LiteRT" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output "models/qwen"

QWEN_DIR="models/qwen"
if [ ! -d "$QWEN_DIR" ]; then
    echo -e "${RED}Error: Converted model output directory not found at $QWEN_DIR${NC}"
    exit 1
fi

# --- 2. Run Inference on converted model using LM commands ---
run_case "Run Inference on Qwen1.5-0.5B-Chat with Prompt" \
    litert lm run "$QWEN_DIR/model.litertlm" --prompt "What is LiteRT? Answer in one sentence."

# --- Summary Report ---
print_summary_report "Qwen1.5"
