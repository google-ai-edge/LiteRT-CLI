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

# --- Environment Setup ---
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export LITERT_CLI_ROOT="/tmp/litert_cli_qwen1_5"

# Source shared utilities
source "$SCRIPT_DIR/utils.sh"

echo -e "${BLUE}${BOLD}==================================================================${NC}"
echo -e "${BLUE}${BOLD}>>> LiteRT CLI Qwen1.5 Demo Script${NC}"
echo -e "${BLUE}${BOLD}==================================================================${NC}"

# Clean up and create work directory
echo -e "\n${YELLOW}Setting up workspace at: $LITERT_CLI_ROOT...${NC}"
rm -rf "$LITERT_CLI_ROOT"
mkdir -p "$LITERT_CLI_ROOT"
cd "$LITERT_CLI_ROOT"

# Create Python virtual environment using UV with Python 3.13
echo -e "${YELLOW}Creating Python virtual environment with UV...${NC}"
uv venv --clear --python=3.13
source .venv/bin/activate

# Create output directories
export MODEL_DIR="$LITERT_CLI_ROOT/models"
mkdir -p "$MODEL_DIR"

# Install litert-cli with convert and lm extras from source using UV
echo -e "${YELLOW}Installing litert-cli with convert and lm extras...${NC}"
uv pip install -e "$REPO_ROOT[convert,lm]"

# --- 1. Convert Qwen 1.5 Model ---
run_case "Convert Qwen1.5-0.5B-Chat from HuggingFace to LiteRT" \
    litert convert Qwen/Qwen1.5-0.5B-Chat --output "$MODEL_DIR/qwen"

QWEN_DIR="$MODEL_DIR/qwen"
if [ ! -d "$QWEN_DIR" ]; then
    echo -e "${RED}Error: Converted model output directory not found at $QWEN_DIR${NC}"
    exit 1
fi

# --- 2. Run Inference on converted model using LM commands ---
run_case "Run Inference on Qwen1.5-0.5B-Chat with Prompt" \
    litert lm run "$QWEN_DIR" --prompt "What is LiteRT? Answer in one sentence."

# --- Summary Report ---
print_summary_report "Qwen1.5"
